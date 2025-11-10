# Fancy MusicGen

一个基于MusicGen模型的音乐生成与微调整合框架，提供完整的训练、推理和Web界面交互功能。

## 项目概述

Fancy MusicGen 是一个综合性音乐生成平台，它整合了 Meta 的 Audiocraft 库，并扩展了 LoRA 微调功能，同时提供了直观的 Web 界面。该项目旨在简化音乐生成模型的训练、微调、推理和交互过程，使用户能够轻松生成自定义风格的音乐。

## 主要功能

- **MusicGen 模型集成**：基于 Meta 的 Audiocraft 库，支持音乐生成功能
- **LoRA 微调框架**：参数高效的模型微调，支持自定义数据集训练
- **完整的工作流**：数据准备、训练、推理、评测一体化流程
- **Web 界面**：基于 React 的用户友好界面，支持模型加载、音乐生成和音频播放
- **批量处理**：支持批量音乐生成和评测
- **多种输出格式**：支持 WAV 音频格式，带响度标准化和压缩

## 项目结构

```
fancy-musicgen/
├── app/                  # Web 应用界面（React + Express）
├── audiocraft/           # Meta 的 Audiocraft 库
├── docs/                 # 项目文档
├── finetune/             # LoRA 微调相关代码和资源
│   ├── data/             # 训练数据目录
│   ├── lora/             # LoRA 微调核心代码
│   └── output/           # 训练输出目录
├── scripts/              # 辅助脚本
├── generate_music_local.py  # 本地音乐生成脚本
├── install_deps.sh       # 依赖安装脚本
├── INSTALL_GUIDE.md      # 安装指南
├── requirements.txt      # Python 依赖
└── README.md             # 项目说明文档（当前文件）
```

## 系统要求

- Python 3.8+
- CUDA 支持（推荐用于模型训练和推理加速）
- Node.js 16+（用于 Web 界面）
- 至少 8GB GPU 显存（推荐 16GB+）

## 安装指南

### 1. 环境准备

```bash
# 创建并激活 conda 环境
conda create -n music python=3.9
conda activate music
```

### 2. 安装依赖

由于 `av==10.0.0` 包与新版 Cython 存在兼容性问题，我们提供了专用的安装脚本：

```bash
# 运行安装脚本
bash install_deps.sh
```

该脚本会：
- 安装兼容的 Cython 版本
- 正确安装 `av==10.0.0` 包
- 安装所有其他必要的依赖

### 3. 安装 Web 界面依赖

```bash
cd app
npm install
```

## 使用指南

### 1. 使用预训练模型生成音乐

可以使用根目录下的脚本直接生成音乐：

```bash
python generate_music_from_text.py
```

修改脚本中的提示词和参数以自定义生成内容：

```python
prompts = [
    "欢快的钢琴音乐，充满活力和希望",
    "平静的爵士背景音乐，适合咖啡馆",
    "Trap嘻哈音乐，充满能量和激情"
]

output_files = generate_music_from_text(
    prompts=prompts,
    duration=10,  # 生成10秒音乐
    output_dir='./output_music'
)
```

### 2. 微调模型

#### 数据准备

1. 将音频文件放在 `finetune/data/train/` 目录下
2. 创建元数据文件：

```bash
cd finetune/lora
python prepare_data.py \
    --input_dir ../data/train \
    --output_dir ../data \
    --metadata_file metadata.json \
    --default_text "hip hop 808 beat"
```

#### 模型训练

```bash
python train_lora.py \
    --model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --data_dir ../data \
    --metadata_file metadata.json \
    --output_dir ../output \
    --lora_rank 12 \
    --lora_alpha 24 \
    --learning_rate 3e-4 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 5 \
    --fp16
```

#### 模型推理

```bash
python inference_lora.py \
    --base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --lora_model_path ../output/best_model \
    --prompts "hip hop 808 beat with trap drums" \
    --output_dir ../output/generated \
    --duration 10
```

### 3. 使用 Web 界面

#### 启动服务

```bash
cd app

# 使用 conda 环境启动完整服务（推荐）
npm run dev:full:conda

# 或者手动启动
# 终端1: 启动后端
npm run server:conda
# 终端2: 启动前端
npm run dev
```

服务启动后，可以通过以下地址访问：
- 前端界面: http://localhost:5173
- 后端 API: http://localhost:3001

#### 使用界面

1. 选择微调方案（从输出目录中自动检测）
2. 选择要使用的模型（MusicGen-Small 或微调模型）
3. 设置生成参数（音乐风格、标签等）
4. 点击"加载模型"按钮
5. 点击"生成音乐"按钮
6. 使用音频播放器播放和下载生成的音乐

## 评测功能

对微调后的模型进行评估：

```bash
cd finetune/lora
python evaluate.py \
    --base_model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --lora_model_path ../output/best_model \
    --output_dir ../output/evaluation \
    --compare \
    --duration 10
```

## 常见问题

### 显存不足

减少批处理大小或增加梯度累积步数：

```bash
python train_lora.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8
```

### 训练速度慢

- 使用 `--fp16` 启用混合精度训练
- 减少训练轮数或音频最大时长

### 模型加载错误

- 检查模型路径是否正确
- 确保 CUDA 可用（`torch.cuda.is_available()` 返回 True）

## 技术细节

### 模型架构

- 基础模型：MusicGen-Small（来自 Meta 的 Audiocraft）
- 微调方法：LoRA（Low-Rank Adaptation）
- 采样率：32kHz
- 音频格式：WAV

### API 端点

Web 界面提供以下 API 端点：

- `GET /api/schemes` - 获取可用的微调方案
- `POST /api/load-models` - 加载选定的模型
- `POST /api/generate` - 生成音乐
- `GET /api/health` - 健康检查

## 文档资源

- **安装指南**: [INSTALL_GUIDE.md](INSTALL_GUIDE.md)
- **微调实施总结**: [finetune/IMPLEMENTATION_SUMMARY.md](finetune/IMPLEMENTATION_SUMMARY.md)
- **LoRA微调指南**: [finetune/lora/README.md](finetune/lora/README.md)
- **数据准备指南**: [finetune/DATA_PREPARATION.md](finetune/DATA_PREPARATION.md)
- **项目文档**: [docs/](docs/)

## 许可证

本项目基于 MIT 许可证。

## 参考资源

- [MusicGen 文档](https://huggingface.co/docs/transformers/model_doc/musicgen)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [Audiocraft 库](https://github.com/facebookresearch/audiocraft)