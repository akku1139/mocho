import numpy as np
from tokenizers import Tokenizer

TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
BIN_PATH = "../dataset/train_data.bin"
IDX_PATH = "../dataset/train_indices.bin"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
indices = np.fromfile(IDX_PATH, dtype=np.uint32).reshape(-1, 2)
data = np.fromfile(BIN_PATH, dtype=np.uint16)

for i in range(5):
    start, length = indices[i]
    sample_ids = data[start:start+length]
    # トークンIDをテキストに戻して表示
    decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
    print(f"--- Sample {i} ---")
    print(f"IDs: {sample_ids}")
    print(f"Decoded: {decoded}\n")
