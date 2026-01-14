import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

import torch
import torch.nn.functional as F

@torch.jit.script
def sru_compute(x, ufr, c_initial):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    L, B, D = x.shape
    u, f, r = torch.chunk(ufr, 3, dim=-1)

    # Sigmoid -> Hard-Sigmoid に変更
    # 重みのスケール感を維持しつつ高速化
    f_gate = F.hardsigmoid(f)
    r_gate = F.hardsigmoid(r)

    # 1トークンずつの処理
    u_s, f_s, r_s, x_s = u[0], f_gate[0], r_gate[0], x[0]

    c_new = f_s * c_initial + (1.0 - f_s) * u_s

    # Tanh -> 軽い近似関数に変更
    # ここでは Tanh と挙動が近く、より軽量な計算式を採用
    # x / (1 + |x|) は tanh(x) の非常に軽量な近似です
    h_new = r_s * (c_new / (1.0 + torch.abs(c_new))) + (1.0 - r_s) * x_s

    return h_new.unsqueeze(0), c_new

class SRULayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, c):
        # x: (L, B, D), c: (B, D)
        residual = x
        x_norm = self.ln(x)

        # 線形変換
        ufr = self.w_ufr(x_norm)
        u, f, r = torch.chunk(ufr, 3, dim=-1)

        # 1. Sigmoid -> Hard-Sigmoid に置換
        f_gate = F.hardsigmoid(f)
        r_gate = F.hardsigmoid(r)

        c_prev = c
        c_stack = []

        # ONNX用にシンボリックなループ
        for t in range(x.size(0)):
            ft = f_gate[t]
            ut = u[t]
            c_curr = ft * c_prev + (1.0 - ft) * ut
            c_stack.append(c_curr.unsqueeze(0))
            c_prev = c_curr

        c_all = torch.cat(c_stack, dim=0)

        # 2. Tanh -> Softsign 近似に置換
        # torch.tanh(c_all) を c_all / (1.0 + |c_all|) で代用
        tanh_approx = c_all / (1.0 + torch.abs(c_all))
        hs = r_gate * tanh_approx + (1.0 - r_gate) * x_norm

        # 最後に residual を足す
        return residual + hs, c_prev

class MochoONNX(nn.Module):
    def __init__(self, vocab_size=6003, n_embd=512, n_layer=6):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([SRULayer(n_embd) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, c_states):
        # idx: (L, B) -> Embedding -> (L, B, D)
        x = self.token_emb(idx)

        new_states = []
        for i in range(self.n_layer):
            x, c_out = self.layers[i](x, c_states[i])
            new_states.append(c_out)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        # ONNXの出力として、タプルではなくリストやスタックしたテンソルを返すのが無難
        return logits, torch.stack(new_states)
