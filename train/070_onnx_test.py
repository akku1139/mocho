import numpy as np
import onnxruntime as ort
import time
from tokenizers import Tokenizer, pre_tokenizers, decoders

# --- 設定 ---
VOCAB_SIZE = 6003
N_EMBD = 512
N_LAYER = 6
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
ONNX_PATH = "../model/weights/v1.2/mocho.onnx" # 先ほど出力したパス

class MochoONNXInference:
    def __init__(self, model_path):
        # CPUで実行。GPUを使う場合は providers=['CUDAExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
    def forward(self, idx, states):
        # idx: (L, B) の numpy 配列
        # states: (Layers, B, D) の numpy 配列
        inputs = {
            "idx": idx.astype(np.int64),
            "c_states": states.astype(np.float32)
        }
        logits, new_states = self.session.run(None, inputs)
        return logits, new_states

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)

def generate_fast_onnx(model, tokenizer, left_context, input_text, max_new_tokens=100, temperature=0.8):
    # 1. プロンプト構成
    ids = [tokenizer.token_to_id("<s>")] + \
          tokenizer.encode(left_context).ids + \
          [tokenizer.token_to_id("[INPUT]")] + \
          tokenizer.encode(input_text).ids + \
          [tokenizer.token_to_id("[OUTPUT]")]

    # (L, B) 形式に整形
    x = np.array(ids, dtype=np.int64).reshape(-1, 1)
    
    # 初期の states (Layers, B, D) を作成
    states = np.zeros((N_LAYER, 1, N_EMBD), dtype=np.float32)

    generated = []
    
    # Prefill: プロンプトを一気に処理
    #logits, states = model.forward(x, states)
    for token_id in ids:
        token_in = np.array([[token_id]], dtype=np.int64)
        logits, states = model.forward(token_in, states)

    # 最初のトークンを選択
    next_token_logits = logits[-1, 0, :] / temperature
    probs = softmax(next_token_logits)
    next_token = np.random.choice(len(probs), p=probs)

    eos_id = tokenizer.token_to_id("</s>")

    # 生成ループ
    for _ in range(max_new_tokens):
        #print(states)
        if next_token == eos_id:
            break
        generated.append(int(next_token))

        # 次の入力トークン (L=1, B=1)
        token_tensor = np.array([[next_token]], dtype=np.int64)

        # 推論
        logits, states = model.forward(token_tensor, states)

        next_token_logits = logits[0, 0, :] / temperature
        probs = softmax(next_token_logits)
        next_token = np.random.choice(len(probs), p=probs)

    return tokenizer.decode(generated)

def test(model, tokenizer, ctx, inp):
    print(f"\nContext: {ctx}\nInput: {inp}\nOutput: ", end="", flush=True)
    start_time = time.perf_counter()
    result = generate_fast_onnx(model, tokenizer, ctx, inp, 100, 0.2)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(result)
    print(f"time: {elapsed_time:.6f} sec")

if __name__ == "__main__":
    # Tokenizer設定
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # ONNXモデルのロード
    model = MochoONNXInference(ONNX_PATH)
    print(f"Loaded ONNX Model: {ONNX_PATH}")

    test(model, tokenizer, "初期化した", "ケッカハザンネンナモノデアッタ")
    test(model, tokenizer, "日本の歴史において、", "エドジダイニツイテオシエテ")
    test(model, tokenizer, "", "テストニュウリョクデス")
