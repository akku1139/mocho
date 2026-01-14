import torch
import torch.nn as nn
from safetensors.torch import load_file
from mocho_for_onnx_export import MochoONNX
import onnx
import os

SAVE_PATH = "../model/weights/v1/mocho.safetensors"

def export_to_onnx():
    B, L, D, layers = 1, 1, 512, 6
    model = MochoONNX(n_embd=D, n_layer=layers).eval()
    state_dict = load_file(SAVE_PATH, device="cpu")

    # Weight Tying を手動で適用
    state_dict["lm_head.weight"] = state_dict["token_emb.weight"]

    model.load_state_dict(state_dict)

    dummy_idx = torch.randint(0, 6003, (1, 1), dtype=torch.int64)
    dummy_states = torch.zeros(layers, 1, 512, dtype=torch.float32)

    save_path = "../model/weights/v1/mocho.onnx"

    torch.onnx.export(
        model,
        (dummy_idx, dummy_states),
        save_path,
        opset_version=18,
        do_constant_folding=True,
        input_names=["idx", "c_states"],
        output_names=["logits", "new_states"],
        dynamic_axes={
            "idx": {0: "length", 1: "batch"},
            "c_states": {1: "batch"},
            "logits": {0: "length", 1: "batch"},
            "new_states": {1: "batch"}
        },
        dynamo=False # 依然としてこちらが安定します
    )
    print(f"Successfully exported to {save_path}")

    model = onnx.load(save_path)
    onnx.save(model, save_path, save_as_external_data=False)

    # 不要になった .data ファイルを削除
    data_file = save_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

export_to_onnx()
