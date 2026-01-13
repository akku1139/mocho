import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
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
#N_LAYER = 1
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
#MODEL_SAVE_PATH = "../model/weights/mocho_1layer.safetensors"
#OPT_SAVE_PATH = "../model/weights/optimizer_1layer_state.pth"

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
        # メモリマップから切り出し (int64への変換はここで行う)
        ids = self.data[start : start + length].astype(np.int64)

        # モデルの入力(x)とターゲット(y)
        x_ids = ids[:-1]
        y_ids = ids[1:]

        # 指定の長さを超える場合はカットするだけにする
        if len(x_ids) > self.seq_len:
            x_ids = x_ids[:self.seq_len]
            y_ids = y_ids[:self.seq_len]

        # ここでは mask も padding も行わず、生の長さのまま返す
        return torch.from_numpy(x_ids), torch.from_numpy(y_ids)

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

def collate_fn(batch):
    # batch = [(x1, y1), (x2, y2), ...]
    xs, ys = zip(*batch)
    # 一括でパディングしてテンソル化（これが非常に速い）
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0) # 0はPAD ID
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0)

    # モデルの期待する SEQ_LEN に満たない場合に備えてさらにパディング
    # (SEQ_LEN固定なら、ここでサイズ調整)
    return xs_padded, ys_padded

def main():
    dataset = BinaryDataset(BIN_PATH, IDX_PATH, SEQ_LEN, TOKENIZER_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    output_token_id = dataset.output_id
    pad_token_id = dataset.pad_id

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

    '''
    print(f"[{datetime.now().strftime('%H:%M:%S')}] compiling model...")
    # ロード後にコンパイルすることで、保存された重みが適用された状態でコンパイルされる
    model = torch.compile(model, options={
        "triton.cudagraphs": False,
        "epilogue_fusion": True
    })
    print(f"[{datetime.now().strftime('%H:%M:%S')}] model compile done.")
    '''

    model.train()
    print("Starting training...")
    try:
        for epoch in range(EPOCHS):
            for step, (x, y) in enumerate(loader):
                x, y = x.t().to(DEVICE), y.t().to(DEVICE)

                # 1. GPU上で [OUTPUT] 以降を1にするマスクを作成
                # (x == output_token_id) はその瞬間だけTrueになるテンソル
                # .cumsum(dim=0) でそれ以降をすべて1以上（True）にする
                m = (x == output_token_id).cumsum(dim=0) > 0

                # 2. [PAD] トークンはロス計算から除外する (AND演算)
                m = m & (y != pad_token_id)

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
