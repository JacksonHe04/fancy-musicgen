#!/bin/bash
# MusicGen LoRA微调快速启动脚本

set -e

echo "=========================================="
echo "MusicGen LoRA Fine-tuning Quick Start"
echo "=========================================="

# 设置路径
BASE_DIR="/root/autodl-tmp/musicgen"
FINETUNE_DIR="${BASE_DIR}/finetune"
LORA_DIR="${FINETUNE_DIR}/lora"
DATA_DIR="${FINETUNE_DIR}/data"
OUTPUT_DIR="${FINETUNE_DIR}/output"
MODEL_PATH="${BASE_DIR}/local/musicgen-small"

# 创建目录
echo "1. Creating directories..."
mkdir -p ${DATA_DIR}/{train,valid}
mkdir -p ${OUTPUT_DIR}/{checkpoints,logs,generated,evaluation}

# 检查模型路径
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model path not found: ${MODEL_PATH}"
    exit 1
fi

# 检查是否有数据
if [ ! -f "${DATA_DIR}/metadata.json" ]; then
    echo "2. Creating example metadata file..."
    cd ${LORA_DIR}
    python prepare_data.py \
        --create_example \
        --output_dir ${DATA_DIR} \
        --num_samples 10
    echo "   Example metadata created. Please add your audio files to ${DATA_DIR}/train/"
    echo "   Then update metadata.json with your actual data."
    exit 0
fi

# 安装依赖
echo "3. Installing dependencies..."
cd ${LORA_DIR}
pip install -q -r requirements.txt

# 开始训练
echo "4. Starting training..."
python train_lora.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --metadata_file metadata.json \
    --output_dir ${OUTPUT_DIR} \
    --lora_rank 12 \
    --lora_alpha 24 \
    --learning_rate 3e-4 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 5 \
    --warmup_steps 100 \
    --save_steps 100 \
    --logging_steps 10 \
    --fp16 \
    --max_duration 30.0 \
    --target_duration 30.0

echo "=========================================="
echo "Training completed!"
echo "Check outputs in: ${OUTPUT_DIR}"
echo "=========================================="

