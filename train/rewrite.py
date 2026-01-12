import os
import json
import numpy as np
from tinygrad import Tensor, nn, GlobalCounters, TinyJit
from tinygrad.helpers import getenv
from tokenizers import Tokenizer

# --- 設定 ---
VOCAB_SIZE = 6003
N_EMBD = 512
N_LAYER = 6
BATCH_SIZE = 128
SEQ_LEN = 16
LEARNING_RATE = 5e-4
EPOCHS = 5
DATASET_PATH = "../dataset/train_wikipedia.jsonl"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
SAVE_PATH = "../model/weights/mocho.safetensors" # TinyGradではsafetensors推奨

# --- モデル定義 ---
class SRULayer:
    def __init__(self, n_embd):
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)

    def __call__(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        L, B, D = x.shape
        ufr = self.w_ufr(x)
        u, f, r = ufr.chunk(3, dim=-1)

        f = f.sigmoid()
        r = r.sigmoid()

        # 循環参照を減らすため、ループ内の計算を最小限にする
        curr_c = c
        cs = []
        for t in range(L):
            curr_c = f[t].mul(curr_c).add((1.0 - f[t]).mul(u[t]))
            cs.append(curr_c)

        cs_stack = Tensor.stack(*cs)
        # tanhと残差接続を一括で行う
        hs = r * cs_stack.tanh() + (1.0 - r) * x

        return self.ln(hs.gelu()), curr_c

class Mocho:
    def __init__(self, vocab_size, n_embd, n_layer):
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = [SRULayer(n_embd) for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, idx: Tensor, c_states: list[Tensor] = None):
        # idx: (L, B)
        x = self.token_emb(idx)
        
        if c_states is None:
            c_states = [Tensor.zeros(idx.shape[1], N_EMBD) for _ in self.layers]
        
        new_states = []
        for i, layer in enumerate(self.layers):
            x, c_out = layer(x, c_states[i])
            new_states.append(c_out)
            
        return self.lm_head(x), new_states

# --- トレーニング用ヘルパー ---
def get_parameters(model):
    params = nn.state.get_parameters(model)
    return params

# --- データセット (シンプルなジェネレータ) ---
def wikipedia_dataset_iter(path, tokenizer, seq_len, batch_size):
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("[PAD]")
    input_id = tokenizer.token_to_id("[INPUT]")
    output_id = tokenizer.token_to_id("[OUTPUT]")

    def encode_line(line):
        data = json.loads(line)
        ctx = tokenizer.encode(data.get('left_context') or "").ids
        inp = tokenizer.encode(data.get('input') or "").ids
        out = tokenizer.encode(data.get('output') or "").ids
        ids = [bos_id] + ctx + [input_id] + inp + [output_id] + out + [eos_id]
        
        if len(ids) < 5: return None
        try:
            out_tkn_pos = ids.index(output_id)
        except ValueError: return None
        
        x_ids = ids[:-1]
        y_ids = ids[1:]
        mask = [1.0 if i >= out_tkn_pos else 0.0 for i in range(len(y_ids))]
        
        if len(x_ids) > seq_len:
            return x_ids[:seq_len], y_ids[:seq_len], mask[:seq_len]
        else:
            plen = seq_len - len(x_ids)
            return x_ids + [pad_id]*plen, y_ids + [pad_id]*plen, mask + [0.0]*plen

    batch_x, batch_y, batch_m = [], [], []
    while True:
        with open(path, 'r', encoding='utf-8') as f:
            c = 0
            for line in f:
                res = encode_line(line)
                if res:
                    batch_x.append(res[0]); batch_y.append(res[1]); batch_m.append(res[2])
                    if len(batch_x) == batch_size:
                        # (B, L) -> (L, B) に転置してTensor化
                        yield Tensor(batch_x).T, Tensor(batch_y).T, Tensor(batch_m).T
                        batch_x, batch_y, batch_m = [], [], []
                        if c%10000 === 0: print(f"tokenized {c} lines")
                        c += 1

# --- 学習プロセス ---
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER)
optimizer = nn.optim.Adam(get_parameters(model), lr=LEARNING_RATE)

# 重みの読み込み (存在すれば)
if os.path.exists(SAVE_PATH):
    nn.state.load_state_dict(model, nn.state.safe_load(SAVE_PATH))

# JIT化されたトレーニングステップ
@TinyJit
def train_step(x, y, mask):
    optimizer.zero_grad()

    # 1. 順伝播
    logits, _ = model(x)

    # 2. ログ確率の計算
    log_probs = logits.log_softmax(axis=-1)

    # 3. gather の修正: gather(axis, index) の順序
    y_idx = y.unsqueeze(-1) # (L, B, 1)
    # axis=2 に対して y_idx を適用
    target_probs = log_probs.gather(2, y_idx).squeeze(-1)

    # 4. 損失の計算
    loss = -(target_probs * mask).sum() / (mask.sum() + 1e-8)

    loss.backward()
    optimizer.step()

    return loss.realize()

# --- メインループ ---
data_gen = wikipedia_dataset_iter(DATASET_PATH, tokenizer, SEQ_LEN, BATCH_SIZE)

try:
    with Tensor.train():
        for epoch in range(EPOCHS):
            next_x, next_y, next_m = next(data_gen)
            for step in range(1000): # ステップ数はデータに合わせて調整
                x, y, m = next_x, next_y, next_m
                loss = train_step(x, y, m)

                if step % 10 == 0:
                    print(f"Epoch {epoch} | Step {step} | Loss: {loss.numpy():.4f}")

                if step % 500 == 0:
                    print("saving...")
                    state_dict = nn.state.get_state_dict(model)
                    nn.state.safe_save(state_dict, SAVE_PATH)

               # 学習中に次のデータをバックグラウンドで準備（簡易的なプリフェッチ）
                try:
                    next_x, next_y, next_m = next(data_gen)
                    # realize()を呼ぶことで、JITの外側で転送をスケジュールする
                    next_x.realize(); next_y.realize(); next_m.realize()
                except StopIteration:
                    break

except KeyboardInterrupt:
    print("Saving...")
    nn.state.safe_save(nn.state.get_state_dict(model), SAVE_PATH)
