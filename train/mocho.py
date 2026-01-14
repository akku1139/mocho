import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

@torch.jit.script
def sru_compute(ufr, c_initial, x_norm):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    L, B, D = x_norm.shape
    u, f, r = torch.chunk(ufr, 3, dim=-1)

    f_gate = torch.sigmoid(f)
    r_gate = torch.sigmoid(r)

    if L == 1:
        # u[0]に直接tanhを適用して、爆発を抑えつつ表現力を安定させる
        u_s = torch.tanh(u[0])
        c_new = f_gate[0] * c_initial + (1.0 - f_gate[0]) * u_s
        h_new = r_gate[0] * torch.tanh(c_new) + (1.0 - r_gate[0]) * x_norm[0]
        return h_new.unsqueeze(0), c_new

    c_stack = torch.empty_like(x_norm)
    c = c_initial
    for t in range(L):
        # f_gateの初期値を高めに（忘却しにくく）することで、初期の学習を安定化
        u_t = torch.tanh(u[t])
        c = f_gate[t] * c + (1.0 - f_gate[t]) * u_t
        c_stack[t] = c

    hs = r_gate * torch.tanh(c_stack) + (1.0 - r_gate) * x_norm
    return hs, c

class SRULayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # バイアスを持たせ、忘却ゲートの初期値を制御しやすくする
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.ln = nn.LayerNorm(n_embd)
        self._init_bias()

    def _init_bias(self):
        # 忘却ゲート(f)のバイアスを 1.0〜2.0 程度にして、最初は「覚える」側に倒す
        # [u, f, r] の順なので、真ん中のスライス
        nn.init.constant_(self.w_ufr.bias[self.ln.normalized_shape[0] : 2*self.ln.normalized_shape[0]], 1.5)

    def forward(self, x, c=None):
        residual = x
        x_norm = self.ln(x)

        if c is None:
            c = torch.zeros(x.size(1), x.size(2), device=x.device, dtype=x.dtype)

        ufr = self.w_ufr(x_norm)
        # 修正: residual + hs ではなく、SRU内部で入力をバイパスしているので
        # ここでは hs 自体が residual を含むような形にする（Highway Network的）
        hs, last_c = sru_compute(ufr, c, x_norm)

        # residual(入力) + hs(変換された残差)
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
                # 忘却ゲート以外は0初期化
                if module.out_features != self.n_embd * 3:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, c_states=None):
        # idx: (L, B) assuming .t() from trainer
        x = self.token_emb(idx)

        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)

        x = self.final_ln(x)
        return self.lm_head(x), new_states
