import numpy as np
from tokenizers import Tokenizer, pre_tokenizers, decoders

TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
BIN_PATH = "../dataset/train_data.bin"
IDX_PATH = "../dataset/train_indices.bin"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

indices = np.fromfile(IDX_PATH, dtype=np.uint32).reshape(-1, 2)
data = np.fromfile(BIN_PATH, dtype=np.uint16)

num_samples = 5
random_indices = np.random.choice(len(indices), num_samples, replace=False)

for i in random_indices:
    start, length = indices[i]
    sample_ids = data[start:start+length]

    # 2. デコードの実行
    # ids をリスト(int)に変換して渡すとより安定します
    decoded = tokenizer.decode(sample_ids.tolist(), skip_special_tokens=False)

    print(f"--- Sample {i} ---")
    print(f"Decoded: {decoded}\n")
