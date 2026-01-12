from tinygrad import Tensor, nn

class SRULayer:
    def __init__(self, n_embd):
        # 3つのゲート用重み (u, f, r)
        self.w_ufr = Tensor.glorot_uniform(n_embd, 3 * n_embd)
        self.ln = nn.LayerNorm(n_embd)

    def __call__(self, x, c=None):
        # x: (Sequence_Length, Batch_Size, Embedding_Dim)
        L, B, D = x.shape
        if c is None: c = Tensor.zeros(B, D)

        # 重い行列演算（並列化可能部分）
        ufr = x.reshape(-1, D).dot(self.w_ufr).reshape(L, B, 3 * D)
        u, f, r = ufr.chunk(3, dim=-1)

        # ゲート適用
        f, r = f.sigmoid(), r.sigmoid()

        # 再帰計算（JITにより最適化される）
        cs, hs = [], []
        for t in range(L):
            c = f[t] * c + (1.0 - f[t]) * u[t]
            h = r[t] * c.tanh() + (1.0 - r[t]) * x[t]
            cs.append(c)
            hs.append(h)

        # 活性化とLayerNorm
        return self.ln(Tensor.stack(*hs).gelu()), cs[-1]

class Mocho:
    def __init__(self, vocab_size=6003, n_embd=512, n_layer=6):
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = [SRULayer(n_embd) for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, idx, c_states=None):
        # idx: (Sequence_Length, Batch_Size)
        x = self.token_emb(idx)
        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)

        # ロジットの出力
        return self.lm_head(x), new_states
