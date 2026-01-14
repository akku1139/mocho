import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tokenizers import Tokenizer
import numpy as np
import os
from datetime import datetime
from mocho import Mocho
from safetensors.torch import save_file, load_file

# --- 設定 ---
DEVICE = torch.device("cuda")
VOCAB_SIZE = 6003
BATCH_SIZE = 110
SEQ_LEN = 256
LEARNING_RATE = 5e-4
EPOCHS = 5
BIN_PATH = "../dataset/train_data.bin"
IDX_PATH = "../dataset/train_indices.bin"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"

# 保存パスの分離
MODEL_SAVE_PATH = "../model/weights/v1.3/mocho.safetensors"
OPT_SAVE_PATH = "../model/weights/v1.3/optimizer_state.pth"
LOG_FILE_PATH = "../model/weights/v1.3/train.log"

log_f = open(LOG_FILE_PATH, "a", encoding="utf-8", buffering=1)

def logger(text):
    msg = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
    log_f.write(msg + "\n")
    print(msg)

# --- Dataset ---
class BinaryDataset(Dataset):
    def __init__(self, bin_path, idx_path, seq_len, tokenizer_path):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.indices = np.fromfile(idx_path, dtype=np.uint32).reshape(-1, 2)
        self.seq_len = seq_len
        tkn = Tokenizer.from_file(tokenizer_path)
        self.pad_id = tkn.token_to_id("[PAD]")
        self.output_id = tkn.token_to_id("[OUTPUT]")
        self.eos_id = tkn.token_to_id("</s>")

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        start, length = self.indices[i]
        # uint16のまま取り出す（astypeしない）
        ids = self.data[start : start + length]

        # モデルの入力(x)とターゲット(y)
        x_ids = ids[:-1]
        y_ids = ids[1:]

        # 指定の長さを超える場合はカットするだけにする
        if len(x_ids) > self.seq_len:
            x_ids = x_ids[:self.seq_len]
            y_ids = y_ids[:self.seq_len]

        # ここでは mask も padding も行わず、生の長さのまま返す
        return torch.from_numpy(x_ids), torch.from_numpy(y_ids)

def save_checkpoint(model, optimizer, scheduler, model_path, opt_path):
    """
    torch.compileされたモデルからプレフィックスを除去して保存
    """
    # compileされている場合は _orig_mod を取得
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # 1. モデルの重みをsafetensorsで保存 (プレフィックスなし)
    sd = raw_model.state_dict()
    # 共有されている重みの片方を削除 (lm_head.weight は token_emb.weight と同じなので消して良い)
    if "lm_head.weight" in sd:
        logger("save_checkpoint: removing lm_head.weight")
        del sd["lm_head.weight"]
    save_file(sd, model_path)

    # --- オプティマイザとスケジューラの保存 ---
    checkpoint_states = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint_states, opt_path)
    logger(f"Checkpoint saved (Model, Optimizer, and Scheduler).")

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
    eos_token_id = dataset.eos_id

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    model = Mocho(VOCAB_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # スケジューラの定義
    # 50ステップを1単位とするので、patienceの考え方に注意
    # patience=0 : 50ステップ経過した時点で前回（50ステップ前）より改善してなければ即ダウン
    # patience=2 : 50ステップ×(2+1) = 150ステップ改善がなければダウン
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    # --- チェックポイントのロード (コンパイル前に行う) ---
    if os.path.exists(MODEL_SAVE_PATH):
        logger(f"Loading weights from {MODEL_SAVE_PATH}...")
        state_dict = load_file(MODEL_SAVE_PATH)
        if "lm_head.weight" not in state_dict and "token_emb.weight" in state_dict:
            # 重みをコピーする（Weight Tying なので参照をコピーするだけでOK）
            state_dict["lm_head.weight"] = state_dict["token_emb.weight"]
        # compile前のmodelにロードするのでプレフィックス問題は起きない
        model.load_state_dict(state_dict)

        if os.path.exists(OPT_SAVE_PATH):
            logger(f"Loading optimizer and scheduler state...")
            checkpoint_states = torch.load(OPT_SAVE_PATH, map_location=DEVICE)
            optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
            optimizer.load_state_dict(opt_state)
            scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        logger("Checkpoint loaded successfully.")
    else:
        logger("No checkpoint found. Starting from scratch.")

    '''
    logger(f"compiling model...")
    # ロード後にコンパイルすることで、保存された重みが適用された状態でコンパイルされる
    model = torch.compile(model, options={
        "triton.cudagraphs": False,
        "epilogue_fusion": True
    })
    logger(f"model compile done.")
    '''

    model.train()
    logger("Starting training...")
    try:
        for epoch in range(EPOCHS):
            for step, (x, y) in enumerate(loader):
                x, y = x.t().to(DEVICE, dtype=torch.int64, non_blocking=True), y.t().to(DEVICE, dtype=torch.int64, non_blocking=True)

                # 1. [OUTPUT]の位置以降をTrueにする (L, B)
                m_output = (x == output_token_id).cumsum(dim=0) > 0

                # 2. </s> (EOS) 以降を学習除外する
                # y側にEOSが出現した「次」からをTrueにする
                is_eos_y = (y == eos_token_id)
                after_eos = is_eos_y.cumsum(dim=0) > 0
                # EOS自体は学習したい（モデルに終わるタイミングを教えるため）ので、
                # 「EOS以降」から「EOSそのもの」を引いた範囲をマスク対象にする
                mask_exclude = after_eos & (~is_eos_y)

                # 3. 最終的なマスク: [OUTPUT]以降 かつ EOS以降ではない かつ PADではない
                m = m_output & (~mask_exclude) & (y != pad_token_id)

                # 4. ラベルの作成 (無視する場所は pad_token_id/ignore_index に置換)
                target_y = torch.where(m, y, pad_token_id)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    logits, _ = model(x)
                    loss = criterion(logits.view(-1, VOCAB_SIZE), target_y.view(-1))

                if torch.isnan(loss):
                    logger(f"Epoch {epoch} | Step {step} | Loss is NaN")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if step % 10 == 0:
                    logger(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                    #print(f"Sample target_y: {target_y[:, 0]}")

                if step > 0 and step % 100 == 0:
                    save_checkpoint(model, optimizer, scheduler, MODEL_SAVE_PATH, OPT_SAVE_PATH)

                if step > 0 and step % 50 == 0:
                    scheduler.step(loss.item())
                    current_lr = optimizer.param_groups[0]['lr']
                    logger(f"--- [LR Update Check] Epoch {epoch} | Step {step} | LR: {current_lr:.8f} ---")

                if step % 500 == 0:
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger("\nInterrupted. Saving checkpoint...")
        save_checkpoint(model, optimizer, scheduler, MODEL_SAVE_PATH, OPT_SAVE_PATH)

if __name__ == "__main__":
    main()
