import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def sru_compute(x, u, f, r, c_initial):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    L, B, D = x.shape
    c = c_initial
    cs = []
    for t in range(L):
        c = f[t] * c + (1.0 - f[t]) * u[t]
        cs.append(c)

    c_stack = torch.stack(cs)
    hs = r * torch.tanh(c_stack) + (1.0 - r) * x
    return hs, c

class SRULayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, c=None):
        if c is None:
            c = torch.zeros(x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)

        ufr = self.w_ufr(x)
        u, f, r = torch.chunk(ufr, 3, dim=-1)
        f, r = torch.sigmoid(f), torch.sigmoid(r)

        hs, last_c = sru_compute(x, u, f, r, c)
        return self.ln(F.gelu(hs)), last_c

class Mocho(nn.Module):
    def __init__(self, vocab_size=6003, n_embd=512, n_layer=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([SRULayer(n_embd) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, c_states=None):
        x = self.token_emb(idx)
        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)
        return self.lm_head(x), new_states
