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

# --- SRU計算部 ---
@torch.jit.script
def sru_compute(x, u, f, r, c_initial):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    L, B, D = x.shape
    c = c_initial
    cs = []
    for t in range(L):
        c = f[t] * c + (1.0 - f[t]) * u[t]
        cs.append(c)
    c_stack = torch.stack(cs)
    hs = r * torch.tanh(c_stack) + (1.0 - r) * x
    return hs, c

@torch.jit.script
def sru_step(x_t, u_t, f_t, r_t, c_prev):
    # 1ステップ推論用
    c_t = f_t[0] * c_prev + (1.0 - f_t[0]) * u_t[0]
    h_t = r_t[0] * torch.tanh(c_t) + (1.0 - r_t[0]) * x_t[0]
    return h_t.unsqueeze(0), c_t

class SRULayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.w_ufr = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, c=None):
        if c is None:
            c = torch.zeros(x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
        ufr = self.w_ufr(x)
        u, f, r = torch.chunk(ufr, 3, dim=-1)
        f, r = torch.sigmoid(f), torch.sigmoid(r)
        hs, last_c = sru_compute(x, u, f, r, c)
        return self.ln(F.gelu(hs)), last_c

    def forward_step(self, x_t, c_prev):
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

    # 学習・Prefill（全読み）用
    def forward(self, idx, c_states=None):
        x = self.token_emb(idx)
        new_states = []
        for i, layer in enumerate(self.layers):
            c_in = c_states[i] if c_states is not None else None
            x, c_out = layer(x, c_in)
            new_states.append(c_out)
        logits = self.lm_head(x)
        return logits, new_states

    # 1トークン生成用
    def forward_generate(self, token_id, states):
        x = self.token_emb(token_id)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, c_out = layer.forward_step(x, states[i])
            new_states.append(c_out)
        logits = self.lm_head(x)
        return logits, new_states

# --- 推論ロジック ---
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
            
            # 状態を引き継ぎながら1ステップだけ進める
            token_tensor = torch.tensor([[next_token]], device=DEVICE)
            logits, states = model.forward_generate(token_tensor, states)
            
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
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded: {SAVE_PATH}")

    # テスト
    ctx = "日本の歴史において、"
    inp = "えどじだいについておしえて"
    print(f"\nContext: {ctx}\nInput: {inp}\nOutput: ", end="", flush=True)
    
    result = generate_fast(model, tokenizer, ctx, inp)
    print(result)
