import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, pre_tokenizers, decoders
import os

# --- 設定 ---
DEVICE = torch.device("cuda")
VOCAB_SIZE = 6003
N_EMBD = 512
N_LAYER = 6
TOKENIZER_PATH = "../model/tokenizer/tokenizer.json"
SAVE_PATH = "../model/weights/mocho.pth"

# --- 推論用：1ステップだけ進めるSRU ---
@torch.jit.script
def sru_step(x_t, u_t, f_t, r_t, c_prev):
    # x_t, u_t, f_t, r_t: (1, B, D)
    # c_prev: (B, D)
    
    # 状態更新
    c_t = f_t[0] * c_prev + (1.0 - f_t[0]) * u_t[0]
    # 出力計算
    h_t = r_t[0] * torch.tanh(c_t) + (1.0 - r_t[0]) * x_t[0]
    
    return h_t.unsqueeze(0), c_t  # h_t: (1, B, D), c_t: (B, D)

class SRULayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)

    def forward_step(self, x_t, c_prev):
        # x_t: (1, B, D) 1タイムステップ分
        ufr = self.w_ufr(x_t)
        u, f, r = torch.chunk(ufr, 3, dim=-1)
        f, r = torch.sigmoid(f), torch.sigmoid(r)
        
        h_t, c_t = sru_step(x_t, u, f, r, c_prev)
        return self.ln(F.gelu(h_t)), c_t

class Mocho(nn.Module):
    def __init__(self, vocab_size=6003, n_embd=512, n_layer=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([SRULayer(n_embd) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward_generate(self, token_id, states):
        # token_id: (1, 1)
        # states: List of c_t
        x = self.token_emb(token_id)
        
        new_states = []
        for i, layer in enumerate(self.layers):
            x, c_out = layer.forward_step(x, states[i])
            new_states.append(c_out)
        
        logits = self.lm_head(x)
        return logits, new_states

# --- 高速推論関数 ---
def generate_fast(model, tokenizer, left_context, input_text, max_new_tokens=100, temperature=0.7):
    model.eval()
    
    # 1. プロンプトのエンコード
    ids = [tokenizer.token_to_id("<s>")] + \
          tokenizer.encode(left_context).ids + \
          [tokenizer.token_to_id("[INPUT]")] + \
          tokenizer.encode(input_text).ids + \
          [tokenizer.token_to_id("[OUTPUT]")]
    
    # 2. Prefill（予習）フェーズ
    # 最初の入力は一括で処理して、最新の内部状態(c)を得る
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(1) # (L, 1)
    with torch.no_grad():
        # Mochoの通常のforwardを使って最後の状態を取得
        # ※Mochoクラスに states を返す実装がある前提
        logits, states = model(x) 
        
        # 3. Generation（生成）フェーズ
        # 最後のトークンの予測からスタート
        next_token_logits = logits[-1, 0, :] / temperature
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1).item()
        
        generated = []
        eos_id = tokenizer.token_to_id("</s>")

        for _ in range(max_new_tokens):
            if next_token == eos_id:
                break
            generated.append(next_token)
            
            # --- ここがポイント ---
            # 入力全体を再計算せず、1つ前のトークンと内部状態だけを使う
            token_tensor = torch.tensor([[next_token]], device=DEVICE)
            logits, states = model.forward_generate(token_tensor, states)
            
            next_token_logits = logits[-1, 0, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1).item()

    return tokenizer.decode(generated)

# --- メイン ---
if __name__ == "__main__":
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER).to(DEVICE)
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

    print(generate_fast(model, tokenizer, "徳川家康は", "江戸幕府を開いた人物ですが、"))
