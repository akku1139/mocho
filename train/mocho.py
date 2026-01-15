import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

@torch.jit.script
def sru_compute(x, ufr, c_initial):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    L, B, D = x.shape
    u, f, r = torch.chunk(ufr, 3, dim=-1)

    f_gate = torch.sigmoid(f)
    r_gate = torch.sigmoid(r)

    if L == 1:
        u_s, f_s, r_s, x_s = u[0], f_gate[0], r_gate[0], x[0]
        c_new = f_s * c_initial + (1.0 - f_s) * u_s
        h_new = r_s * torch.tanh(c_new) + (1.0 - r_s) * x_s
        return h_new.unsqueeze(0), c_new

    c_stack = torch.empty_like(x)
    c = c_initial
    for t in range(L):
        # f_gateが「過去の保持率」なので、1.0に近いほど長く覚える
        c = f_gate[t] * c + (1.0 - f_gate[t]) * u[t]
        c_stack[t] = c

    hs = r_gate * torch.tanh(c_stack) + (1.0 - r_gate) * x
    return hs, c

class SRULayer(nn.Module):
    def __init__(self, n_embd, layer_id, n_layer):
        super().__init__()
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.ln = nn.LayerNorm(n_embd)
        self.n_layer = n_layer
        self.dropout = nn.Dropout(0.1)

        with torch.no_grad():
            self.w_ufr.weight.data.normal_(std=0.001)
            self.w_ufr.bias.data.zero_()
            self.w_ufr.bias.data[2*n_embd : 3*n_embd].fill_(2.0)

    def forward(self, x, c=None):
        residual = x
        x_norm = self.ln(x)
        if c is None:
            c = torch.zeros(x.size(1), x.size(2), device=x.device, dtype=x.dtype)

        ufr = self.w_ufr(x_norm)
        hs, last_c = sru_compute(x_norm, ufr, c)

        return residual + self.dropout(hs), last_c

class Mocho(nn.Module):
    def __init__(self, vocab_size=6003, n_embd=768, n_layer=10):
        super().__init__()
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # レイヤーIDを渡すように変更
        self.layers = nn.ModuleList([SRULayer(n_embd, i, n_layer) for i in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # 修正3: applyを使わず、個別に初期化を管理する
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(self, idx, c_states=None):
        x = self.token_emb(idx)

        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)

        x = self.final_ln(x)
        return self.lm_head(x) * 0.3, new_states
