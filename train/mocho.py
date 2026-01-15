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
    def __init__(self, n_embd):
        super().__init__()
        # 修正1: biasをTrueにし、忘却ゲートを初期化できるようにする
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, c=None):
        residual = x
        x_norm = self.ln(x)
        if c is None:
            c = torch.zeros(x.size(1), x.size(2), device=x.device, dtype=x.dtype)

        ufr = self.w_ufr(x_norm)
        hs, last_c = sru_compute(x_norm, ufr, c)
        return residual + hs, last_c

class Mocho(nn.Module):
    def __init__(self, vocab_size=6003, n_embd=768, n_layer=10):
        super().__init__()
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([SRULayer(n_embd) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                # 修正2: 忘却ゲート(f)の初期バイアスを 1.0 に設定
                # これをしないと、学習初期に情報をすべて忘れてしまい「に」ループに陥る
                if module.out_features == 3 * self.n_embd:
                    nn.init.zeros_(module.bias) # 一旦全部0
                    with torch.no_grad():
                        # [u(0:D), f(D:2D), r(2D:3D)] の f 部分を 1.0 に
                        module.bias[self.n_embd : 2*self.n_embd].fill_(1.0)
                else:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, c_states=None):
        x = self.token_emb(idx)
        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)
        x = self.final_ln(x)
        return self.lm_head(x), new_states
