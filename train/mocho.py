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

    c_stack = torch.empty_like(x_norm)
    c = c_initial
    for t in range(L):
        # 修正: u に tanh をかけない。
        # c の更新に生の u を使うことで、情報の流入を最大化する。
        c = f_gate[t] * c + (1.0 - f_gate[t]) * u[t]
        c_stack[t] = c

    # 出力時のみ tanh をかけて非線形性を出す
    hs = r_gate * torch.tanh(c_stack) + (1.0 - r_gate) * x_norm
    return hs, c

class SRULayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.ln = nn.LayerNorm(n_embd)
        self._init_bias()

    def _init_bias(self):
        # 忘却ゲートのバイアス。1.5は高すぎた可能性があるため 0.5 程度に。
        # [u, f, r] のうち f の部分
        n_embd = self.ln.normalized_shape[0]
        nn.init.constant_(self.w_ufr.bias[n_embd : 2*n_embd], 0.5) 
        # 残りの u, r のバイアスは 0 でOK
        nn.init.constant_(self.w_ufr.bias[0 : n_embd], 0.0)
        nn.init.constant_(self.w_ufr.bias[2*n_embd : 3*n_embd], 0.0)

    def forward(self, x, c=None):
        # x_norm を使って SRU の演算を行う
        x_norm = self.ln(x)

        if c is None:
            c = torch.zeros(x.size(1), x.size(2), device=x.device, dtype=x.dtype)

        ufr = self.w_ufr(x_norm)
        hs, last_c = sru_compute(ufr, c, x_norm)

        # 修正3: SRU内部のHighway(1-r)ですでにxの成分が混合されているため、
        # ここでは単純に hs を返すか、あるいは residual + (hs - x_norm) にします。
        # 最も安定するのは Pre-LN ResNet 形式のこれです：
        return x + hs, last_c

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
