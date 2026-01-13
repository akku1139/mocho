from tokenizers import pre_tokenizers

def preprocess():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # --- ここを追加 ---
    # バイトレベルの前処理を設定
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # デコーダーもバイトレベルにしないと、デコード時に文字化けします
    from tokenizers import decoders
    tokenizer.decoder = decoders.ByteLevel()
    # ------------------

    bos_id = tokenizer.token_to_id("<s>")
    # ...以下、元の処理
