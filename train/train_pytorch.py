import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from tokenizers import Tokenizer

@torch.jit.script
def sru_compute(x, u, f, r, c_initial):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    L, B, D = x.shape
    c = c_initial
    hs = []

    # このループがTorchScriptによってC++レベルで最適化される
    for t in range(L):
        c = f[t] * c + (1.0 - f[t]) * u[t]
        h = r[t] * torch.tanh(c) + (1.0 - r[t]) * x[t]
        hs.append(h)

    return torch.stack(hs), c

class SRULayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, c=None):
        L, B, D = x.shape
        if c is None:
            c = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # 全タイムステップのゲート計算を一気に行う（ここが並列化ポイント）
        ufr = self.w_ufr(x)
        u, f, r = torch.chunk(ufr, 3, dim=-1)

        f = torch.sigmoid(f)
        r = torch.sigmoid(r)

        # 最適化されたループ関数を呼び出す
        hs, last_c = sru_compute(x, u, f, r, c)

        out = self.ln(F.gelu(hs))
        return out, last_c

class Mocho(nn.Module):
    def __init__(self, vocab_size=6003, n_embd=512, n_layer=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([SRULayer(n_embd) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, c_states=None):
        # idx: (L, B)
        x = self.token_emb(idx)

        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)

        logits = self.lm_head(x) # (L, B, V)
        return logits, new_states

DEVICE = torch.device("cuda")
VOCAB_SIZE = 6003
N_EMBD = 512
N_LAYER = 6
BATCH_SIZE = 180
SEQ_LEN = 256
LEARNING_RATE = 1e-4
EPOCHS = 5
DATASET_PATH = "../dataset/train_wikipedia.jsonl"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
SAVE_PATH = "../model/weights/mocho.pth"

# --- データセット定義 ---
class WikipediaDataset(IterableDataset):
    def __init__(self, path, tokenizer, seq_len):
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.input_id = tokenizer.token_to_id("[INPUT]")
        self.output_id = tokenizer.token_to_id("[OUTPUT]")

    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                ids = self.safe_encode(data)
                if len(ids) < 5: continue

                try:
                    out_tkn_pos = ids.index(self.output_id)
                except ValueError: continue

                x_ids = ids[:-1]
                y_ids = ids[1:]
                mask = [1.0 if i >= out_tkn_pos else 0.0 for i in range(len(y_ids))]

                # パディング/切り詰め
                if len(x_ids) > self.seq_len:
                    x_ids, y_ids, mask = x_ids[:self.seq_len], y_ids[:self.seq_len], mask[:self.seq_len]
                else:
                    plen = self.seq_len - len(x_ids)
                    x_ids += [self.pad_id] * plen
                    y_ids += [self.pad_id] * plen
                    mask += [0.0] * plen

                yield torch.tensor(x_ids), torch.tensor(y_ids), torch.tensor(mask)

    def safe_encode(self, data):
        ctx = self.tokenizer.encode(data.get('left_context') or "").ids
        inp = self.tokenizer.encode(data.get('input') or "").ids
        out = self.tokenizer.encode(data.get('output') or "").ids
        return [self.bos_id] + ctx + [self.input_id] + inp + [self.output_id] + out + [self.eos_id]

# --- 準備 ---
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
dataset = WikipediaDataset(DATASET_PATH, tokenizer, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)

model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER).to(DEVICE)
# model = torch.compile(model)

if os.path.exists(SAVE_PATH):
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizerは定義した後にロード
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss(reduction='none')

# --- 学習ループ ---
model.train()
try:
    for epoch in range(EPOCHS):
        for step, (x, y, m) in enumerate(loader):
            # PyTorchは通常 (Batch, Seq) なので (Seq, Batch) に入れ替え
            x, y, m = x.t().to(DEVICE), y.t().to(DEVICE), m.t().to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(x) # (L, B, V)

            # ロス計算
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            # マスク適用
            masked_loss = (loss * m.reshape(-1)).sum() / (m.sum() + 1e-8)

            masked_loss.backward()
            # 勾配爆発を防ぐ
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {masked_loss.item():.4f}")

            if step % 500 == 0:
                torch.save({
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                }, SAVE_PATH)
except KeyboardInterrupt:
    print("Saving...")
    torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }, SAVE_PATH)
