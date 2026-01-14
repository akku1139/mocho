import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, pre_tokenizers, decoders
from safetensors.torch import load_file
import os
import time
from mocho import Mocho

# --- 設定 ---
#DEVICE = torch.device("cuda")
DEVICE = torch.device("cpu")
VOCAB_SIZE = 6003
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
SAVE_PATH = "../model/weights/v1.3/mocho.safetensors"

def generate_fast(model, tokenizer, left_context, input_text, max_new_tokens=100, temperature=0.8):
    model.eval()

    # 1. プロンプト構成
    ids = [tokenizer.token_to_id("<s>")] + \
          tokenizer.encode(left_context).ids + \
          [tokenizer.token_to_id("[INPUT]")] + \
          tokenizer.encode(input_text).ids + \
          [tokenizer.token_to_id("[OUTPUT]")]

    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(1)

    generated = []
    with torch.no_grad():
        # Prefill: プロンプトを一気に読み、その時点の隠れ状態 states を得る
        logits, states = model(x)

        # 最初の予測トークン
        next_token_logits = logits[-1, 0, :] / temperature
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1).item()

        eos_id = tokenizer.token_to_id("</s>")

        # 生成ループ: ここでは入力を更新せず、1トークンとstatesのみで回す
        for _ in range(max_new_tokens):
            if next_token == eos_id:
                break
            generated.append(next_token)

            # 1トークンだけをテンソル化 [1, 1] (L=1, B=1)
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)

            # モデルを直接呼び出す（forwardメソッドが実行される）
            logits, states = model(token_tensor, states)

            next_token_logits = logits[-1, 0, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1).item()

    return tokenizer.decode(generated)

def test(model, tokenizer, ctx, inp):
    print(f"\nContext: {ctx}\nInput: {inp}\nOutput: ", end="", flush=True)
    start_time = time.perf_counter()
    result = generate_fast(model, tokenizer, ctx, inp, 100, 0.2)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(result)
    print(f"time: {elapsed_time:.6f} sec")

if __name__ == "__main__":
    # Tokenizer設定
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    model = Mocho(VOCAB_SIZE).to(DEVICE)
    state_dict = load_file(SAVE_PATH, device=str(DEVICE))
    if "lm_head.weight" not in state_dict and "token_emb.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["token_emb.weight"]
    model.load_state_dict(state_dict)
    # it makes everything slow
    #model = torch.compile(model)
    print(f"Loaded: {SAVE_PATH}")

    test(model, tokenizer, "初期化した", "ケッカハザンネンナモノデアッタ")
    test(model, tokenizer, "日本の歴史において、", "エドジダイニツイテオシエテ")
    test(model, tokenizer, "", "テストニュウリョクデス")
