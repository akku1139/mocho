import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import numpy as np
import os
from datetime import datetime

# --- 設定 ---
DEVICE = torch.device("cuda")
VOCAB_SIZE = 6003
N_EMBD = 512
N_LAYER = 6
BATCH_SIZE = 256
SEQ_LEN = 256
LEARNING_RATE = 1e-4
EPOCHS = 5
BIN_PATH = "../dataset/train_data.bin"
IDX_PATH = "../dataset/train_indices.bin"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
SAVE_PATH = "../model/weights/mocho.pth"

# --- SRU計算部 ---
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

# --- Dataset ---
class BinaryDataset(Dataset):
    def __init__(self, bin_path, idx_path, seq_len, tokenizer_path):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.indices = np.fromfile(idx_path, dtype=np.uint32).reshape(-1, 2)
        self.seq_len = seq_len
        tkn = Tokenizer.from_file(tokenizer_path)
        self.pad_id = tkn.token_to_id("[PAD]")
        self.output_id = tkn.token_to_id("[OUTPUT]")

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        start, length = self.indices[i]
        ids = self.data[start : start + length].astype(np.int64)
        x_ids, y_ids = ids[:-1], ids[1:]
        mask = np.zeros(len(y_ids), dtype=np.float32)
        out_pos = np.where(x_ids == self.output_id)[0]
        if len(out_pos) > 0: mask[out_pos[0]:] = 1.0

        if len(x_ids) > self.seq_len:
            x_ids, y_ids, mask = x_ids[:self.seq_len], y_ids[:self.seq_len], mask[:self.seq_len]
        else:
            diff = self.seq_len - len(x_ids)
            x_ids = np.pad(x_ids, (0, diff), constant_values=self.pad_id)
            y_ids = np.pad(y_ids, (0, diff), constant_values=self.pad_id)
            mask = np.pad(mask, (0, diff), constant_values=0.0)
        return torch.from_numpy(x_ids), torch.from_numpy(y_ids), torch.from_numpy(mask)

def main():
    dataset = BinaryDataset(BIN_PATH, IDX_PATH, SEQ_LEN, TOKENIZER_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # --- チェックポイントのロード ---
    if os.path.exists(SAVE_PATH):
        print(f"Loading weights from {SAVE_PATH}...")
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found. Starting from scratch.")

    print(f"[{datetime.now().strftime("%H:%M:%S")}] compiling model...")
    model = torch.compile(model)
    print(f"[{datetime.now().strftime("%H:%M:%S")}] model compile done.")

    model.train()
    print("Starting training...")
    try:
        for epoch in range(EPOCHS):
            for step, (x, y, m) in enumerate(loader):
                # x, y, m: (B, L) -> (L, B)
                x, y, m = x.t().to(DEVICE), y.t().to(DEVICE), m.t().to(DEVICE)
                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    logits, _ = model(x)
                    
                    # ロス計算の効率化
                    flat_m = m.reshape(-1).bool()
                    active_logits = logits.reshape(-1, VOCAB_SIZE)[flat_m]
                    active_labels = y.reshape(-1)[flat_m]
                    
                    if active_labels.numel() > 0:
                        loss = F.cross_entropy(active_logits, active_labels)
                    else:
                        continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if step % 10 == 0:
                    print(f"[{datetime.now().strftime("%H:%M:%S")}] Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

                if step % 500 == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, SAVE_PATH)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVE_PATH)

if __name__ == "__main__":
    main()
