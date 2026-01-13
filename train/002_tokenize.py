import json
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import os

# 設定
DATASET_PATH = "../dataset/train_wikipedia.jsonl"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
OUTPUT_BIN_PATH = "../dataset/train_data.bin"

def preprocess():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    input_id = tokenizer.token_to_id("[INPUT]")
    output_id = tokenizer.token_to_id("[OUTPUT]")

    all_ids = []

    print("Tokenizing...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            
            # 各セクションをエンコード
            ctx = tokenizer.encode(data.get('left_context') or "").ids
            inp = tokenizer.encode(data.get('input') or "").ids
            out = tokenizer.encode(data.get('output') or "").ids
            
            # 1つのシーケンスとして結合
            # あとで y_ids.index(output_id) を探せるようにそのまま並べる
            ids = [bos_id] + ctx + [input_id] + inp + [output_id] + out + [eos_id]
            
            # 各サンプルの区切りがわかるように、
            # [シーケンスの長さ(uint16), token1, token2, ...] の形式で保存するか、
            # 固定長でパディングして保存する。ここでは「可変長のまま連結」し、
            # 別途インデックス（開始位置）を保存する方法をとります。
            all_ids.extend(ids)

    print(f"Saving to {OUTPUT_BIN_PATH}...")
    # 65535以上の語彙がなければ uint16 でメモリ節約
    np_ids = np.array(all_ids, dtype=np.uint16)
    np_ids.tofile(OUTPUT_BIN_PATH)
    print("Done!")

if __name__ == "__main__":
    preprocess()
