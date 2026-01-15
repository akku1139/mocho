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
        # 修正1: u[t]への直接のtanhを外し、情報の流入を確保
        # (SRUの標準的な実装に合わせる)
        c = f_gate[t] * c + (1.0 - f_gate[t]) * u[t]
        c_stack[t] = c

    # 修正2: 内部でHighway Network(x_normのブレンド)を完結させる
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
        hs, last_c = sru_compute(ufr, c, x_norm)

        # 修正3: hsはすでにx_norm(入力)をHighway的に含んでいるため、
        # 外側の残差接続は x (正規化前) を足すだけに留めるか、
        # もしくは hs の計算から x_norm を引くなどの調整が必要。
        # 最も安定するのは、以下の形式です：
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
