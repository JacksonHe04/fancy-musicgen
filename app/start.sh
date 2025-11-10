#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate music

# 设置 Python 路径环境变量
export PYTHON_PATH=$(which python)

# 启动服务器
echo "Starting server with Python: $PYTHON_PATH"
PORT=3001 PYTHON_PATH=$PYTHON_PATH node server.js

