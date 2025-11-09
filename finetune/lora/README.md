# MusicGen LoRA微调指南

本目录包含MusicGen模型LoRA微调的完整实现，用于生成嘻哈808 Beat风格的音乐。

## 目录结构

```
finetune/lora/
├── dataset.py           # 数据集加载模块
├── train_lora.py        # 训练脚本
├── inference_lora.py    # 推理脚本
├── evaluate.py          # 评测脚本
├── prepare_data.py      # 数据预处理脚本
├── config.yaml          # 配置文件
├── requirements.txt     # 依赖文件
└── README.md           # 本文件
```

## 环境准备

### 1. 安装依赖

```bash
cd /root/autodl-tmp/musicgen/finetune/lora
pip install -r requirements.txt
```

### 2. 验证CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 数据准备

### 1. 准备音频文件

将你的嘻哈808 Beat音频文件放在 `finetune/data/train/` 目录下。

### 2. 创建元数据文件

运行数据预处理脚本：

```bash
# 从文件夹创建数据集
python prepare_data.py \
    --input_dir /path/to/your/audio/files \
    --output_dir ./finetune/data \
    --metadata_file metadata.json \
    --default_text "hip hop 808 beat"

# 或者创建示例元数据文件（用于测试）
python prepare_data.py \
    --create_example \
    --output_dir ./finetune/data \
    --num_samples 10
```

### 3. 元数据格式

`metadata.json` 文件格式：

```json
[
  {
    "audio": "train/audio1.wav",
    "text": "hip hop 808 beat with trap drums"
  },
  {
    "audio": "train/audio2.wav",
    "text": "808 bass and hip hop drum pattern"
  }
]
```

## 训练

### 基本训练命令

```bash
python train_lora.py \
    --model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --data_dir ./finetune/data \
    --metadata_file metadata.json \
    --output_dir ./finetune/output \
    --lora_rank 12 \
    --lora_alpha 24 \
    --learning_rate 3e-4 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 5 \
    --fp16
```

### 训练参数说明

- `--model_path`: 基础模型路径
- `--data_dir`: 数据目录
- `--metadata_file`: 元数据文件名
- `--output_dir`: 输出目录
- `--lora_rank`: LoRA rank（默认12）
- `--lora_alpha`: LoRA alpha（默认24）
- `--learning_rate`: 学习率（默认3e-4）
- `--batch_size`: 批次大小（默认2，根据GPU显存调整）
- `--gradient_accumulation_steps`: 梯度累积步数（默认4）
- `--num_epochs`: 训练轮数（默认5）
- `--fp16`: 使用混合精度训练

### 训练监控

训练过程中会输出：
- 训练损失
- 验证损失
- 学习率
- 训练进度

模型checkpoint会保存在 `output_dir/checkpoints/` 目录下。

## 推理

### 基本推理命令

```bash
python inference_lora.py \
    --base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --lora_model_path ./finetune/output/best_model \
    --prompts "hip hop 808 beat with trap drums" \
    --output_dir ./finetune/output/generated \
    --duration 10
```

### 批量推理

```bash
python inference_lora.py \
    --base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --lora_model_path ./finetune/output/best_model \
    --prompts "hip hop 808 beat" "808 bass and trap drums" "trap beat with 808" \
    --output_dir ./finetune/output/generated \
    --duration 10
```

## 评测

### 基本评测

```bash
python evaluate.py \
    --base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --lora_model_path ./finetune/output/best_model \
    --output_dir ./finetune/output/evaluation \
    --duration 10
```

### 对比评测（基础模型 vs 微调模型）

```bash
python evaluate.py \
    --base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --lora_model_path ./finetune/output/best_model \
    --output_dir ./finetune/output/evaluation \
    --compare \
    --duration 10
```

## 数据集要求

### 音频格式
- 格式: WAV, MP3, FLAC等
- 采样率: 32kHz（会自动重采样）
- 时长: 建议15-30秒
- 通道: 单声道或立体声

### 文本描述
- 应包含"hip hop", "808", "beat", "trap"等关键词
- 描述应准确反映音频内容
- 建议使用英文描述

### 数据集规模
- 最小: 50-100个样本
- 推荐: 200-500个样本
- 理想: 500-1000个样本

## 常见问题

### 1. 显存不足

减少 `--batch_size` 或增加 `--gradient_accumulation_steps`：

```bash
python train_lora.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8
```

### 2. 训练速度慢

- 使用 `--fp16` 启用混合精度训练
- 减少 `--num_epochs`
- 减少 `--max_duration`

### 3. 生成质量不佳

- 增加训练数据量
- 调整 `--lora_rank` 和 `--lora_alpha`
- 增加训练轮数
- 检查文本描述质量

## 参考资源

- [MusicGen文档](https://huggingface.co/docs/transformers/model_doc/musicgen)
- [PEFT文档](https://huggingface.co/docs/peft)
- [LoRA论文](https://arxiv.org/abs/2106.09685)

## 许可证

本代码遵循MIT许可证。

