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
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)
        nn.init.normal_(self.w_ufr.weight, std=0.02)

    def forward(self, x, c=None):
        # x: (L, B, D)
        residual = x  # Pre-LNの残差用に保存
        x_norm = self.ln(x)

        if c is None:
            # size(1)がバッチサイズ、size(2)が次元数
            c = torch.zeros(x.size(1), x.size(2), device=x.device, dtype=x.dtype)

        ufr = self.w_ufr(x_norm)
        hs, last_c = sru_compute(x_norm, ufr, c)

        # SRU内部で x_norm がブレンドされているので
        # ここでは「正規化前の入力」を足して、勾配の通り道を確保する
        return residual + hs, last_c

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
        # idx が (B, L) で来ても (L, B) で来ても対応できるようにする
        # 今回の学習ループでは x.t() しているので (L, B) になっているはず
        x = self.token_emb(idx) # x: (L, B, D)

        # もし token_emb の後に (B, L, D) になっていたら permute(1, 0, 2) が必要
        # しかし、idx が (L, B) なら x は自動的に (L, B, D) になります。

        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)

        x = self.final_ln(x)
        return self.lm_head(x), new_states
