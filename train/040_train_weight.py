import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import numpy as np
import os
from datetime import datetime
from mocho import Mocho

# --- 設定 ---
DEVICE = torch.device("cuda")
VOCAB_SIZE = 6003
N_EMBD = 512
N_LAYER = 6
BATCH_SIZE = 400
SEQ_LEN = 256
LEARNING_RATE = 1e-4
EPOCHS = 5
BIN_PATH = "../dataset/train_data.bin"
IDX_PATH = "../dataset/train_indices.bin"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
SAVE_PATH = "../model/weights/mocho.pth"

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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    print(f"[{datetime.now().strftime("%H:%M:%S")}] compiling model...")
    model = torch.compile(model, options={
        "triton.cudagraphs": False, # 以前警告が出たCUDAGraphsをオフにする
        "epilogue_fusion": True     # 安全な最適化だけをオンにする
    })
    print(f"[{datetime.now().strftime("%H:%M:%S")}] model compile done.")

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
