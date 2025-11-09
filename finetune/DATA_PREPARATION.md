# 数据准备指南

本指南说明如何为MusicGen LoRA微调准备数据集。

## 数据集要求

### 音频文件

- **格式**: WAV, MP3, FLAC, M4A, OGG
- **采样率**: 建议44.1kHz或更高（会自动重采样到32kHz）
- **时长**: 建议15-30秒（可以更长，但训练时会裁剪）
- **通道**: 单声道或立体声均可
- **质量**: 建议使用高质量音频，避免噪音和失真

### 文本描述

- **语言**: 建议使用英文
- **格式**: 简洁的描述性文本
- **内容**: 应包含音乐风格、节奏、乐器等关键信息

### 数据集规模

- **最小数据集**: 50-100个样本（快速测试）
- **推荐数据集**: 200-500个样本（较好的效果）
- **理想数据集**: 500-1000个样本（最佳效果）

## 数据准备步骤

### 步骤1: 收集音频文件

将你的嘻哈808 Beat音频文件放在一个文件夹中，例如：

```
/path/to/your/audio/files/
├── beat1.wav
├── beat2.wav
├── beat3.wav
└── ...
```

### 步骤2: 创建元数据文件

#### 方法1: 使用自动脚本（推荐）

```bash
cd /root/autodl-tmp/musicgen/finetune/lora
python prepare_data.py \
    --input_dir /path/to/your/audio/files \
    --output_dir ./finetune/data \
    --metadata_file metadata.json \
    --default_text "hip hop 808 beat" \
    --train_ratio 0.8
```

#### 方法2: 手动创建

创建 `finetune/data/metadata.json` 文件：

```json
[
  {
    "audio": "train/beat1.wav",
    "text": "hip hop 808 beat with trap drums"
  },
  {
    "audio": "train/beat2.wav",
    "text": "808 bass and hip hop drum pattern"
  },
  {
    "audio": "train/beat3.wav",
    "text": "trap beat with heavy 808 kick"
  }
]
```

### 步骤3: 组织文件结构

```
finetune/data/
├── metadata.json
├── train/
│   ├── beat1.wav
│   ├── beat2.wav
│   └── ...
└── valid/
    ├── beat1.wav
    ├── beat2.wav
    └── ...
```

## 文本描述建议

### 针对嘻哈808 Beat的描述模板

- "hip hop 808 beat with trap drums"
- "808 bass and hip hop drum pattern"
- "trap beat with heavy 808 kick"
- "hip hop instrumental with 808"
- "808 drum pattern with trap elements"
- "hip hop beat with 808 bass"
- "trap music with 808 drums"
- "808 kick and hip hop snare"
- "hip hop 808 beat"
- "trap beat 808 bass"

### 描述技巧

1. **包含关键词**: 确保包含"hip hop", "808", "beat", "trap"等关键词
2. **描述节奏**: 可以描述节奏类型，如"fast", "slow", "energetic"
3. **描述乐器**: 可以描述主要乐器，如"drums", "bass", "synth"
4. **保持简洁**: 描述应该简洁明了，不要太长

## 数据来源建议

### 免费资源

1. **Freesound.org**
   - 搜索"808", "hip hop beat", "trap beat"
   - 注意许可证要求

2. **YouTube Audio Library**
   - 筛选嘻哈/说唱类别
   - 注意使用条款

3. **Hugging Face Datasets**
   - 搜索"music", "hip hop"相关数据集
   - 注意数据许可

### 付费资源

1. **Splice**
   - 高质量音乐样本库
   - 提供大量808 beats

2. **Loopmasters**
   - 专业音乐样本库
   - 多种风格选择

### 自制数据

1. **使用音乐制作软件**
   - FL Studio
   - Ableton Live
   - Logic Pro
   - 创建808 beats

2. **从现有音乐提取**
   - 使用音频编辑软件
   - 提取器乐片段
   - 注意版权问题

## 数据预处理

### 自动预处理

使用 `prepare_data.py` 脚本会自动：
- 重采样到32kHz
- 统一音频格式
- 创建train/valid分割
- 生成metadata.json

### 手动预处理

如果需要手动预处理：

```python
import librosa
import soundfile as sf

# 加载音频
audio, sr = librosa.load('input.wav', sr=32000)

# 保存处理后的音频
sf.write('output.wav', audio, 32000)
```

## 数据验证

### 检查数据

运行测试脚本验证数据：

```bash
cd /root/autodl-tmp/musicgen/finetune/lora
python test_dataset.py
```

### 常见问题

1. **音频文件不存在**
   - 检查文件路径是否正确
   - 确保音频文件在正确的位置

2. **音频格式不支持**
   - 转换为WAV格式
   - 使用librosa或ffmpeg转换

3. **文本描述为空**
   - 确保metadata.json中每个样本都有text字段
   - 检查JSON格式是否正确

4. **数据集为空**
   - 检查metadata.json是否正确
   - 确保音频文件路径正确

## 下一步

数据准备完成后，可以开始训练：

```bash
cd /root/autodl-tmp/musicgen/finetune/lora
python train_lora.py \
    --model_path /root/autodl-tmp/musicgen/local/musicgen-small \
    --data_dir ./finetune/data \
    --metadata_file metadata.json \
    --output_dir ./finetune/output
```

## 参考资源

- [MusicGen文档](https://huggingface.co/docs/transformers/model_doc/musicgen)
- [音频处理指南](https://librosa.org/doc/latest/index.html)
- [数据集准备最佳实践](https://huggingface.co/docs/datasets/preparing_dataset)

