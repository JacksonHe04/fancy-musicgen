import os
import sys
import torch
from huggingface_hub import snapshot_download

# 添加项目路径到Python路径
sys.path.append('/root/autodl-tmp/musicgen/audiocraft')

# 设置下载目录
DOWNLOAD_DIR = '/root/autodl-tmp/musicgen/local/musicgen-small'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print(f"开始下载musicgen-small模型到: {DOWNLOAD_DIR}")

# 使用snapshot_download下载模型
snapshot_download(
    repo_id="facebook/musicgen-small",
    local_dir=DOWNLOAD_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.bin", "*.json", "config*.json", "tokenizer*", "*.pth"]
)

print("模型下载完成!")