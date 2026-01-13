import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, pre_tokenizers, decoders
from safetensors.torch import load_file
import os
from mocho import Mocho

# --- 設定 ---
DEVICE = torch.device("cuda")
VOCAB_SIZE = 6003
N_EMBD = 512
N_LAYER = 6
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
SAVE_PATH = "../model/weights/mocho.safetensors"

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

if __name__ == "__main__":
    # Tokenizer設定
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER).to(DEVICE)
    if os.path.exists(SAVE_PATH):
        state_dict = load_file(SAVE_PATH, device=str(DEVICE))
        model.load_state_dict(state_dict)
        print(f"Loaded: {SAVE_PATH}")

    # テスト
    ctx = "日本の歴史において、"
    inp = "エドジダイニツイテオシエテ"
    print(f"\nContext: {ctx}\nInput: {inp}\nOutput: ", end="", flush=True)

    result = generate_fast(model, tokenizer, ctx, inp)
    print(result)
