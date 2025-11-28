#!/bin/bash

# 脚本所在目录
my_dir=$(cd $(dirname "$0"); pwd)
cur_dir=$(pwd)
cd "$my_dir" || exit 1

# 设置 PYTHONPATH，让 Python 能找到 dvllm 模块
export PYTHONPATH="$my_dir/..:$PYTHONPATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 判断是否传入脚本名参数
if [ "$#" -ne 0 ]; then
    python "$@"
else
    python example.py  
fi 

# 返回原目录
cd "$cur_dir" || exit 1

