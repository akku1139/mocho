import json
import numpy as np
from tokenizers import Tokenizer, decoders
from tqdm import tqdm

# 設定
DATASET_PATH = "../dataset/train_wikipedia.jsonl"
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
OUTPUT_BIN_PATH = "../dataset/train_data.bin"
OUTPUT_IDX_PATH = "../dataset/train_indices.bin"
BATCH_SIZE = 2000  # まとめて処理する行数

from tokenizers import pre_tokenizers

def preprocess():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # --- ここを追加 ---
    # バイトレベルの前処理を設定
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # デコーダーもバイトレベルにしないと、デコード時に文字化けします
    tokenizer.decoder = decoders.ByteLevel()
    # ------------------

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    input_id = tokenizer.token_to_id("[INPUT]")
    output_id = tokenizer.token_to_id("[OUTPUT]")

    indices = []
    current_pos = 0

    print(f"Tokenizing with batch size {BATCH_SIZE}...")
    
    # 書き出し用ファイルを開く
    with open(DATASET_PATH, 'r', encoding='utf-8') as f, \
         open(OUTPUT_BIN_PATH, 'wb') as f_bin:
        
        batch_lines = []
        for line in tqdm(f):
            batch_lines.append(line)
            
            if len(batch_lines) >= BATCH_SIZE:
                # バッチ処理
                current_pos = process_batch(batch_lines, tokenizer, bos_id, eos_id, input_id, output_id, indices, current_pos, f_bin)
                batch_lines = []
        
        # 残りのデータを処理
        if batch_lines:
            process_batch(batch_lines, tokenizer, bos_id, eos_id, input_id, output_id, indices, current_pos, f_bin)

    print(f"Saving indices to {OUTPUT_IDX_PATH}...")
    np.array(indices, dtype=np.uint32).tofile(OUTPUT_IDX_PATH)
    print("Done!")

def process_batch(lines, tokenizer, bos_id, eos_id, input_id, output_id, indices, current_pos, f_bin):
    contexts = []
    inputs = []
    outputs = []
    
    for line in lines:
        try:
            data = json.loads(line)
            contexts.append(data.get('left_context') or "")
            inputs.append(data.get('input') or "")
            outputs.append(data.get('output') or "")
        except json.JSONDecodeError:
            continue

    # バッチ単位でエンコード（ここがRustで並列実行される）
    enc_ctx = tokenizer.encode_batch(contexts)
    enc_inp = tokenizer.encode_batch(inputs)
    enc_out = tokenizer.encode_batch(outputs)

    all_batch_ids = []
    for c, i, o in zip(enc_ctx, enc_inp, enc_out):
        # 1つのシーケンスを構築
        ids = [bos_id] + c.ids + [input_id] + i.ids + [output_id] + o.ids + [eos_id]
        
        indices.append([current_pos, len(ids)])
        all_batch_ids.extend(ids)
        current_pos += len(ids)
    
    # バッチごとにバイナリを書き出し（メモリ節約）
    np.array(all_batch_ids, dtype=np.uint16).tofile(f_bin)
    return current_pos

if __name__ == "__main__":
    preprocess()
