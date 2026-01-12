import json
import os
import numpy as np
from tinygrad import Tensor, nn, TinyJit, dtypes
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_state_dict, safe_save, safe_load, load_state_dict
from tokenizers import Tokenizer
from mocho import Mocho

# --- 設定 ---
VOCAB_SIZE = 6003
N_EMBD = 512
N_LAYER = 6
BATCH_SIZE = 32
SEQ_LEN = 256
LEARNING_RATE = 5e-4
EPOCHS = 5
DATASET_PATH = "../dataset/train_wikipedia.jsonl"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
SAVE_PATH = "../model/weights/mocho.safetensors"

# --- トークナイザーの準備 ---
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
BOS_ID = tokenizer.token_to_id("<s>")
EOS_ID = tokenizer.token_to_id("</s>")
PAD_ID = tokenizer.token_to_id("[PAD]")
INPUT_TKN_ID = tokenizer.token_to_id("[INPUT]")
OUTPUT_TKN_ID = tokenizer.token_to_id("[OUTPUT]")

def safe_encode_sequence(data, tokenizer):
    """IDレベルで安全にシーケンスを結合する"""
    ctx_ids = tokenizer.encode(data.get('left_context') or "").ids
    inp_ids = tokenizer.encode(data.get('input') or "").ids
    out_ids = tokenizer.encode(data.get('output') or "").ids

    # <s> left_context [INPUT] input [OUTPUT] output </s>
    full_ids = [BOS_ID] + ctx_ids + [INPUT_TKN_ID] + inp_ids + [OUTPUT_TKN_ID] + out_ids + [EOS_ID]
    return full_ids

def data_generator(path, tokenizer, batch_size, seq_len):
    """データセットからバッチを生成するジェネレータ"""
    x_batch, y_batch, m_batch = [], [], []

    while True: # ループして学習を続ける場合
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                ids = safe_encode_sequence(data, tokenizer)
                if len(ids) < 5: continue # 短すぎるデータを除外

                # OUTPUTトークンの位置を特定してマスクを作成
                try:
                    out_tkn_pos = ids.index(OUTPUT_TKN_ID)
                except ValueError:
                    continue

                # 入力(x)とターゲット(y: xを1つずらしたもの)
                x_ids = ids[:-1]
                y_ids = ids[1:]

                # マスク作成: 正解(漢字部分)以降のみ1.0
                mask = [1.0 if i >= out_tkn_pos else 0.0 for i in range(len(y_ids))]

                # 長さ調整（パディング / 切り詰め）
                if len(x_ids) > seq_len:
                    x_ids, y_ids, mask = x_ids[:seq_len], y_ids[:seq_len], mask[:seq_len]
                else:
                    pad_len = seq_len - len(x_ids)
                    x_ids += [PAD_ID] * pad_len
                    y_ids += [PAD_ID] * pad_len
                    mask += [0.0] * pad_len

                x_batch.append(x_ids)
                y_batch.append(y_ids)
                m_batch.append(mask)

                if len(x_batch) == batch_size:
                    yield (Tensor(x_batch, dtype=dtypes.int32).transpose(0, 1),
                           Tensor(y_batch, dtype=dtypes.int32).transpose(0, 1),
                           Tensor(m_batch, dtype=dtypes.float32).transpose(0, 1))
                    x_batch, y_batch, m_batch = [], [], []

# --- モデルと最適化 ---
model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER)

# 既存の重みがあればロード
if os.path.exists(SAVE_PATH):
    print(f"既存の重みを {SAVE_PATH} からロードしています...")
    load_state_dict(model, safe_load(SAVE_PATH))

optimizer = Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)

@TinyJit
def train_step(x, y, mask):
    optimizer.zero_grad()

    # 順伝播
    logits, _ = model(x) # (L, B, V)

    # ロス計算
    logits_flat = logits.reshape(-1, VOCAB_SIZE)
    targets_flat = y.reshape(-1)

    loss = logits_flat.sparse_categorical_crossentropy(targets_flat)

    # マスクの適用（漢字部分以外とパディングを無視）
    masked_loss = (loss * mask.reshape(-1)).sum() / (mask.sum() + 1e-8)

    masked_loss.backward()
    optimizer.step()
    return masked_loss.realize()

# --- 学習メインループ ---
print(f"学習を開始します。")
gen = data_generator(DATASET_PATH, tokenizer, BATCH_SIZE, SEQ_LEN)

try:
    for epoch in range(EPOCHS):
        with Tensor.train():
            for step in range(1000):
                x, y, m = next(gen)
                loss_val = train_step(x, y, m)

                if step % 10 == 0:
                    print(f"Epoch {epoch} | Step {step} | Loss: {loss_val.numpy():.4f}")

                if step % 500 == 0:
                    state_dict = get_state_dict(model)
                    safe_save(state_dict, SAVE_PATH)
                    print(f"モデルを {SAVE_PATH} に保存しました。")

except KeyboardInterrupt:
    print("学習を中断します。現在の重みを保存します...")
    safe_save(get_state_dict(model), SAVE_PATH)

print("学習完了。")
