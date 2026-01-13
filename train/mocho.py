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
        if c is None:
            c = torch.zeros(x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)

        # 1. 重み計算（ここまでは並列で行ける）
        ufr = self.w_ufr(x)

        # 2. 残りの全演算を1つのJIT関数に丸投げ
        hs, last_c = sru_compute(x, ufr, c)

        return self.ln(F.gelu(hs)), last_c

class Mocho(nn.Module):
    def __init__(self, vocab_size=6003, n_embd=512, n_layer=6):
        super().__init__()
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([SRULayer(n_embd) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, c_states=None):
        x = self.token_emb(idx)
        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)
        return self.lm_head(x), new_states
