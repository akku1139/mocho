import json
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import os

# 設定
DATASET_PATH = "../dataset/train_wikipedia.jsonl"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
OUTPUT_BIN_PATH = "../dataset/train_data.bin"
OUTPUT_IDX_PATH = "../dataset/train_indices.bin"

def preprocess():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    input_id = tokenizer.token_to_id("[INPUT]")
    output_id = tokenizer.token_to_id("[OUTPUT]")

    all_ids = []
    indices = []
    current_pos = 0

    print("Tokenizing and indexing...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 各セクションをエンコード
            ctx = tokenizer.encode(data.get('left_context') or "").ids
            inp = tokenizer.encode(data.get('input') or "").ids
            out = tokenizer.encode(data.get('output') or "").ids
            
            # 1つの完全なシーケンスを構築
            ids = [bos_id] + ctx + [input_id] + inp + [output_id] + out + [eos_id]
            
            # インデックス（開始位置と長さ）を記録
            indices.append([current_pos, len(ids)])
            all_ids.extend(ids)
            current_pos += len(ids)

    print(f"Saving to {OUTPUT_BIN_PATH}...")
    # トークン列を保存
    np.array(all_ids, dtype=np.uint16).tofile(OUTPUT_BIN_PATH)
    # インデックス（開始位置、長さ）を保存
    np.array(indices, dtype=np.uint32).tofile(OUTPUT_IDX_PATH)
    print("Done!")

if __name__ == "__main__":
    preprocess()
