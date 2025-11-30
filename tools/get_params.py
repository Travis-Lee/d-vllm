#!/usr/bin/env python3
import os
import argparse
from safetensors.torch import safe_open

def main():
    parser = argparse.ArgumentParser(description="Inspect safetensors file")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model.safetensors file"
    )
    args = parser.parse_args()

    # 展开 ~ 符号
    model_path = os.path.expanduser(args.model)

    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    print(f"🔍 Loading: {model_path}\n")

    # 遍历参数
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"{key}: {tuple(tensor.shape)}")

if __name__ == "__main__":
    main()

