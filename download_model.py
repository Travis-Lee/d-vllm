#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import os

# 设置模型保存路径
local_dir = os.path.expanduser("~/huggingface/Qwen3-0.6B")

# 下载模型
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f"Qwen3 model downloaded to: {local_dir}")

