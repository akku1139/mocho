import torch
from safetensors.torch import save_file
import os

# 元のファイルのパス
OLD_PTH_PATH = "../model/weights/v1/mocho.pth"

# 出力先のパス
NEW_SAFE_PATH = "../model/weights/v1/mocho.safetensors"
NEW_OPT_PATH = "../model/weights/v1/optimizer_state.pth"

def convert():
    if not os.path.exists(OLD_PTH_PATH):
        print(f"Error: {OLD_PTH_PATH} が見つかりません。")
        return

    print(f"Loading {OLD_PTH_PATH}...")
    # map_location="cpu" で読み込むことでGPUメモリを節約
    checkpoint = torch.load(OLD_PTH_PATH, map_location="cpu")
    
    # 1. モデルの重みを取り出し、名前をクリーンアップ
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # torch.compile 由来の _orig_mod. プレフィックスを除去
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
        if k != new_key:
            print(f"Fixed key: {k} -> {new_key}")

    # 重みの保存
    save_file(new_state_dict, NEW_SAFE_PATH)
    print(f"Successfully saved weights to: {NEW_SAFE_PATH}")

    # 2. オプティマイザの状態を取り出して保存
    if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        torch.save(opt_state, NEW_OPT_PATH)
        print(f"Successfully saved optimizer state to: {NEW_OPT_PATH}")
    else:
        print("Warning: optimizer_state_dict が見つかりませんでした。重みのみ保存しました。")

if __name__ == "__main__":
    # 出力ディレクトリの存在確認
    os.makedirs(os.path.dirname(NEW_SAFE_PATH), exist_ok=True)
    convert()
