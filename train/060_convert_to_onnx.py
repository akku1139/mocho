import torch
import os
from mocho import Mocho
from safetensors.torch import load_file
import onnx

def export_onnx():
    # パスの設定
    SAVE_PATH = "../model/weights/v1/mocho.safetensors"
    ONNX_PATH = "../model/weights/v1/mocho.onnx"
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    VOCAB_SIZE, N_EMBD, N_LAYER = 6003, 512, 6
    model = Mocho(VOCAB_SIZE, N_EMBD, N_LAYER)
    
    # 重みのロード
    state_dict = load_file(SAVE_PATH, device="cpu")
    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["token_emb.weight"]
    model.load_state_dict(state_dict)
    model.eval() # 変換前に必ず eval モードにする

    # ONNX用にインターフェースを完全にフラット化するラッパー
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, idx, s0, s1, s2, s3, s4, s5):
            # リストとしてモデルに渡す
            logits, new_states = self.model(idx, [s0, s1, s2, s3, s4, s5])
            # 戻り値もフラットなタプルにする
            return logits, *new_states

    wrapped_model = OnnxWrapper(model)

    # ダミー入力 (L=1, B=1) と 6層分の初期State
    dummy_idx = torch.zeros((1, 1), dtype=torch.long)
    dummy_states = [torch.zeros(1, N_EMBD) for _ in range(N_LAYER)]

    # 入出力名の定義
    input_names = ["idx"] + [f"state_in_{i}" for i in range(N_LAYER)]
    output_names = ["logits"] + [f"state_out_{i}" for i in range(N_LAYER)]

    # --- 修正ポイント: dynamic_shapes を直接定義 ---
    # idx の 0番目の次元（シーケンス長）を可変にする
    # PyTorch 2.5+ では dynamic_axes の代わりにこちらが推奨
    from torch.export import Dim
    seq_len = Dim("seq_len", min=1, max=2048)
    dynamic_shapes = {
        "idx": {0: seq_len},
        # state系は形状固定 (B=1, D=512) なので指定不要
    }

    print(f"Exporting to {ONNX_PATH}...")
    
    # opset 17 を指定しつつ、最新の方式でエクスポート
    torch.onnx.export(
        wrapped_model,
        (dummy_idx, *dummy_states),
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "idx": {0: "seq_len"},
            "logits": {0: "seq_len"}
        },
    )
    print("Success: ONNX model saved.")

    model = onnx.load(ONNX_PATH)
    onnx.save(model, ONNX_PATH, save_as_external_data=False)

    # 不要になった .data ファイルを削除
    data_file = ONNX_PATH + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

if __name__ == "__main__":
    export_onnx()
