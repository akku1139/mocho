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
from safetensors.torch import save_file, load_file

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

# 保存パスの分離
MODEL_SAVE_PATH = "../model/weights/mocho.safetensors"
OPT_SAVE_PATH = "../model/weights/optimizer_state.pth"

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

def save_checkpoint(model, optimizer, model_path, opt_path):
    """
    torch.compileされたモデルからプレフィックスを除去して保存
    """
    # compileされている場合は _orig_mod を取得
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    # 1. モデルの重みをsafetensorsで保存 (プレフィックスなし)
    save_file(raw_model.state_dict(), model_path)
    
    # 2. オプティマイザの状態をpthで保存
    torch.save(optimizer.state_dict(), opt_path)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Checkpoint saved.")

def main():
    dataset = BinaryDataset(BIN_PATH, IDX_PATH, SEQ_LEN, TOKENIZER_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # --- チェックポイントのロード (コンパイル前に行う) ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading weights from {MODEL_SAVE_PATH}...")
        state_dict = load_file(MODEL_SAVE_PATH)
        # compile前のmodelにロードするのでプレフィックス問題は起きない
        model.load_state_dict(state_dict)
        
        if os.path.exists(OPT_SAVE_PATH):
            print(f"Loading optimizer state from {OPT_SAVE_PATH}...")
            opt_state = torch.load(OPT_SAVE_PATH, map_location=DEVICE)
            optimizer.load_state_dict(opt_state)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found. Starting from scratch.")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] compiling model...")
    # ロード後にコンパイルすることで、保存された重みが適用された状態でコンパイルされる
    model = torch.compile(model, options={
        "triton.cudagraphs": False,
        "epilogue_fusion": True
    })
    print(f"[{datetime.now().strftime('%H:%M:%S')}] model compile done.")

    model.train()
    print("Starting training...")
    try:
        for epoch in range(EPOCHS):
            for step, (x, y, m) in enumerate(loader):
                x, y, m = x.t().to(DEVICE), y.t().to(DEVICE), m.t().to(DEVICE)
                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    logits, _ = model(x)
                    
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
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

                if step > 0 and step % 500 == 0:
                    save_checkpoint(model, optimizer, MODEL_SAVE_PATH, OPT_SAVE_PATH)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        save_checkpoint(model, optimizer, MODEL_SAVE_PATH, OPT_SAVE_PATH)

if __name__ == "__main__":
    main()
