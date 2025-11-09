<!-- ea572f5c-2e63-495f-91eb-9676dc65efd9 59d4fdf9-fe5b-4a07-b5c3-ecec72c77002 -->
# MusicGen模型微调方案

## 方案概述

提供两种微调方案：**LoRA微调**（推荐，快速）和**AudioCraft原生微调**（完整但耗时）。基于RTX 4090 (24GB) GPU，优先使用LoRA以缩短训练时间。

## 方案一：LoRA微调（推荐）

### 优势

- 训练速度快（仅训练少量参数）
- 显存占用低
- 可快速迭代
- 兼容transformers格式模型

### 实现步骤

1. **环境准备**

   - 安装peft、transformers、accelerate等依赖
   - 配置训练环境

2. **数据准备**

   - 创建数据集加载脚本
   - 支持音频文件+文本描述的数据格式
   - 实现数据预处理和tokenization

3. **模型准备**

   - 加载本地musicgen-small模型（transformers格式）
   - 配置LoRA适配器（rank、alpha等参数）
   - 冻结基础模型，仅训练LoRA层

4. **训练脚本**

   - 实现训练循环
   - 支持梯度检查点、混合精度训练
   - 配置学习率调度器
   - 实现checkpoint保存

5. **推理脚本**

   - 加载微调后的LoRA权重
   - 集成到现有生成流程

### 预期训练时间

- 小数据集（<100样本）：10-30分钟
- 中等数据集（100-1000样本）：30分钟-2小时
- 大数据集（>1000样本）：2-6小时

## 方案二：AudioCraft原生微调

### 优势

- 完整的训练流程
- 支持更多自定义配置
- 与AudioCraft生态完全兼容

### 实现步骤

1. **环境配置**

   - 配置dora实验管理器
   - 设置AUDIOCRAFT_TEAM环境变量
   - 配置输出目录

2. **数据准备**

   - 创建manifest文件（JSONL格式）
   - 配置数据集yaml文件
   - 支持MusicDataset格式（音频+元数据）

3. **训练配置**

   - 创建微调配置文件
   - 设置continue_from指向本地模型
   - 配置优化器和学习率
   - 设置batch size和训练步数

4. **训练执行**

   - 使用dora run命令启动训练
   - 监控训练进度
   - 保存checkpoint

### 预期训练时间

- 小数据集：2-4小时
- 中等数据集：4-12小时
- 大数据集：12-24小时+

## 文件结构

```
/root/autodl-tmp/musicgen/
├── audiocraft/                 # AudioCraft仓库
├── local/musicgen-small/       # 本地模型
├── finetune/                   # 微调相关文件（新建）
│   ├── lora/                   # LoRA微调方案
│   │   ├── train_lora.py       # LoRA训练脚本
│   │   ├── dataset.py          # 数据集加载
│   │   ├── inference_lora.py   # LoRA推理脚本
│   │   ├── evaluate.py         # 评测脚本
│   │   ├── prepare_data.py     # 数据预处理脚本
│   │   ├── requirements.txt    # 依赖文件
│   │   └── config.yaml         # 训练配置
│   ├── data/                   # 训练数据
│   │   ├── train/              # 训练集音频
│   │   │   ├── audio1.wav
│   │   │   └── audio2.wav
│   │   ├── valid/              # 验证集音频
│   │   └── metadata.json       # 音频描述元数据
│   │       # 格式: [{"audio": "train/audio1.wav", "text": "hip hop 808 beat with trap drums"}]
│   └── output/                 # 训练输出
│       ├── checkpoints/        # 模型checkpoint
│       └── logs/               # 训练日志
└── generate_music_local.py     # 现有推理脚本
```

## 实施步骤（详细）

### 步骤1：环境准备

- 安装依赖：peft, transformers, accelerate, datasets, soundfile, librosa
- 验证CUDA可用性
- 创建目录结构

### 步骤2：数据准备

- 收集嘻哈808 Beat音频样本（50-500个）
- 为每个音频创建文本描述
- 创建metadata.json文件
- 数据预处理（重采样到32kHz，标准化格式）
- 划分train/val集（80/20）

### 步骤3：实现数据集加载

- 实现MusicGenDataset类
- 支持音频加载和预处理
- 文本tokenization
- 数据增强（可选）

### 步骤4：实现LoRA训练

- 加载musicgen-small模型
- 配置LoRA（target_modules选择decoder layers）
- 实现训练循环
- 实现验证循环
- Checkpoint保存和加载

### 步骤5：实现推理脚本

- 加载基础模型+LoRA权重
- 文本到音频生成
- 保存生成的音频

### 步骤6：实现评测

- 生成测试样本
- 对比微调前后效果
- 评估生成质量

## 关键技术参数

- **LoRA配置**：
  - rank: 8-16（推荐12）
  - alpha: 16-32（推荐24）
  - target_modules: decoder的attention层
  - dropout: 0.1

- **训练配置**：
  - 学习率: 1e-4到5e-4（推荐3e-4）
  - Batch size: 2-4（根据显存调整）
  - Gradient accumulation: 4-8（等效batch size 8-32）
  - Epochs: 5-10（根据数据集大小）
  - Warmup steps: 100-500
  - Max length: 1500 tokens（约30秒音频）

- **优化设置**：
  - 混合精度: fp16
  - 梯度检查点: 启用（节省显存）
  - 优化器: AdamW
  - 学习率调度: cosine with warmup

## 预期效果

- **训练时间**：
  - 50样本：30分钟-1小时
  - 200样本：1-2小时
  - 500样本：2-4小时

- **显存占用**：
  - 基础模型：~8GB
  - 训练时：~12-16GB（batch size 2-4）

- **生成效果**：
  - 微调后应能生成更符合嘻哈808 Beat风格的音乐
  - 对相关文本提示的响应更准确
  - 保持原模型的通用生成能力