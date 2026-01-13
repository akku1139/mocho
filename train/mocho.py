import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def sru_compute(x, ufr, c_initial):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    L, B, D = x.shape
    u, f, r = torch.chunk(ufr, 3, dim=-1)

    # Sigmoidをここで一括で行う（L > 1 の時に爆速になる）
    f_gate = torch.sigmoid(f)
    r_gate = torch.sigmoid(r)

    if L == 1:
        # 生成時は極限までシンプルに
        # [1, B, D] -> [B, D] に絞って計算
        u_s, f_s, r_s, x_s = u[0], f_gate[0], r_gate[0], x[0]
        c_new = f_s * c_initial + (1.0 - f_s) * u_s
        h_new = r_s * torch.tanh(c_new) + (1.0 - r_s) * x_s
        return h_new.unsqueeze(0), c_new

    # Prefill時はメモリ確保してループ
    c_stack = torch.empty_like(x)
    c = c_initial
    for t in range(L):
        c = f_gate[t] * c + (1.0 - f_gate[t]) * u[t]
        c_stack[t] = c

    hs = r_gate * torch.tanh(c_stack) + (1.0 - r_gate) * x
    return hs, c

class SRULayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # SRUの重み
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Pre-LayerNorm用
        self.ln = nn.LayerNorm(n_embd)

        # 初期化: GPTの慣例に従う
        nn.init.normal_(self.w_ufr.weight, std=0.02)

    def forward(self, x, c=None):
        # 1. Pre-LayerNorm (入力にLNをかける)
        x_norm = self.ln(x)

        if c is None:
            c = torch.zeros(x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)

        # 2. SRU計算
        ufr = self.w_ufr(x_norm)
        hs, last_c = sru_compute(x_norm, ufr, c)

        # 3. 残差接続 (入力を足す) + GELUはSRU内部のtanhがあるので不要な場合が多い
        # SRUの設計上、hsにはすでに(1-r)*xが含まれていますが、
        # 深いネットワークでは外側に残差接続を持たせるのが安定の秘訣です。
        return x + hs, last_c

class Mocho(nn.Module):
    def __init__(self, vocab_size=6003, n_embd=512, n_layer=6):
        super().__init__()
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([SRULayer(n_embd) for _ in range(n_layer)])
        # 最終層の後のLayerNorm (Pre-LN方式では必須)
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight Tying
        self.lm_head.weight = self.token_emb.weight

        # 全体的な重み初期化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, c_states=None):
        x = self.token_emb(idx)

        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)

        # 最後のLN
        x = self.final_ln(x)
        return self.lm_head(x), new_states
