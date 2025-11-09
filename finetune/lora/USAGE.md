# MusicGen LoRA微调使用指南

## 快速开始

### 1. 安装依赖

```bash
cd /root/autodl-tmp/musicgen/finetune/lora
bash install.sh
```

### 2. 准备数据

#### 方法1: 使用自动脚本

```bash
python prepare_data.py \
    --input_dir /path/to/your/audio/files \
    --output_dir ../data \
    --metadata_file metadata.json \
    --default_text "hip hop 808 beat"
```

#### 方法2: 手动创建

1. 将音频文件放在 `../data/train/` 目录
2. 创建 `../data/metadata.json` 文件（参考DATA_PREPARATION.md）

### 3. 测试数据集

```bash
python test_dataset.py
```

### 4. 测试模型

```bash
python test_model.py
```

### 5. 开始训练

```bash
python train_lora.py \
    --model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --data_dir ../data \
    --metadata_file metadata.json \
    --output_dir ../output \
    --lora_rank 12 \
    --lora_alpha 24 \
    --batch_size 2 \
    --num_epochs 5 \
    --fp16
```

### 6. 推理

```bash
python inference_lora.py \
    --base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --lora_model_path ../output/best_model \
    --prompts "hip hop 808 beat with trap drums" \
    --output_dir ../output/generated
```

### 7. 评测

```bash
python evaluate.py \
    --base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --lora_model_path ../output/best_model \
    --output_dir ../output/evaluation
```

## 文件说明

### 核心文件

- `dataset.py`: 数据集加载模块
- `train_lora.py`: 训练脚本
- `inference_lora.py`: 推理脚本
- `evaluate.py`: 评测脚本
- `prepare_data.py`: 数据预处理脚本

### 测试文件

- `test_dataset.py`: 测试数据集加载
- `test_model.py`: 测试模型前向传播

### 配置文件

- `config.yaml`: 训练配置文件
- `requirements.txt`: 依赖文件

### 文档

- `README.md`: 主要文档
- `DATA_PREPARATION.md`: 数据准备指南
- `USAGE.md`: 本文件

### 脚本

- `install.sh`: 安装依赖脚本
- `quick_start.sh`: 快速启动脚本

## 常见问题

### 1. 显存不足

减少batch size或增加gradient accumulation:

```bash
python train_lora.py --batch_size 1 --gradient_accumulation_steps 8
```

### 2. 训练速度慢

- 使用 `--fp16` 启用混合精度
- 减少 `--num_epochs`
- 减少 `--max_duration`

### 3. 数据集加载错误

- 检查metadata.json格式
- 检查音频文件路径
- 运行 `test_dataset.py` 诊断问题

### 4. 模型训练错误

- 运行 `test_model.py` 检查模型
- 检查输入数据格式
- 查看错误日志

## 下一步

1. 准备你的数据集
2. 运行测试脚本验证环境
3. 开始训练
4. 评估结果
5. 调整参数优化效果

## 参考

- [README.md](README.md): 详细文档
- [DATA_PREPARATION.md](../DATA_PREPARATION.md): 数据准备指南
- [MusicGen文档](https://huggingface.co/docs/transformers/model_doc/musicgen)

