from tokenizers import Tokenizer, pre_tokenizers, decoders, processors

TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"

# 1. 既存のファイルを読み込む（語彙やマージルールは保持される）
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# 2. Pre-tokenizer と Decoder を ByteLevel に設定
# これによりデコード時に「ãģ」が「あ」に結合されるようになります
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# 3. Post-Processor の設定 (任意)
# 特殊トークンを含めた処理を json に覚え込ませたい場合はここで行います
# 今回はシンプルにデコーダーの修正を優先します

# 4. 上書き（または別名で）保存
tokenizer.save(TOKENIZER_PATH)

print(f"Successfully updated and saved {TOKENIZER_PATH}")
