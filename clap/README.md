# CLAP音乐评估工具

这个工具用于计算三个模型（base_model、best_model、final_model）生成的音乐与文本提示之间的CLAP（Contrastive Language-Audio Pre-training）相似度分数。

## 文件说明

- `calculate_clap.py` - 主要的CLAP计算脚本
- `requirements.txt` - Python依赖包列表
- `run_clap.sh` - 自动化运行脚本（使用audiocraft_env conda环境）
- `README.md` - 本说明文档

## 使用方法

### 方法1：使用自动化脚本（推荐）

```bash
cd /Users/jackson/Codes/fancy-musicgen/clap
./run_clap.sh
```

脚本会自动：
1. 检查名为"audiocraft_env"的conda环境是否存在
2. 激活audiocraft_env环境
3. 安装所需的Python依赖包
4. 运行CLAP计算
5. 显示结果摘要

**注意**: 脚本需要预先存在名为"audiocraft_env"的conda环境。如果不存在，请先创建：
```bash
conda create -n audiocraft_env python=3.9 -y
```

### 方法2：手动运行

```bash
# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate audiocraft_env

# 安装依赖
pip install -r requirements.txt

# 运行计算
python3 calculate_clap.py
```

## 自定义参数

您可以通过命令行参数自定义输入和输出路径：

```bash
python3 calculate_clap.py \
    --comparison_file /path/to/comparison_results.jsonl \
    --audio_base_dir /path/to/audio/files \
    --output_file /path/to/output.json
```

## 输出结果

脚本会生成一个JSON文件，包含：

1. **摘要信息**：每个模型的平均分数、标准差、最小值和最大值
2. **详细结果**：每个音频-文本对的CLAP分数

示例输出：
```json
{
  "summary": {
    "total_samples": 50,
    "base_model": {
      "average_score": 0.3245,
      "std_score": 0.0567,
      "min_score": 0.2103,
      "max_score": 0.4521
    },
    "best_model": {
      "average_score": 0.3892,
      "std_score": 0.0623,
      "min_score": 0.2541,
      "max_score": 0.5234
    },
    "final_model": {
      "average_score": 0.4128,
      "std_score": 0.0589,
      "min_score": 0.2987,
      "max_score": 0.5456
    }
  },
  "detailed_results": [
    {
      "text_prompt": "detroit hiphop beat, aggressive dark threatening",
      "duration": 15,
      "base_model_score": 0.3245,
      "best_model_score": 0.3892,
      "final_model_score": 0.4128,
      "base_model_file": "base_model/20251111_000244_0_detroit_hiphop_beat__aggressive_dark_threatening.wav",
      "best_model_file": "best_model/20251111_000300_0_detroit_hiphop_beat__aggressive_dark_threatening.wav",
      "final_model_file": "final_model/20251111_000316_0_detroit_hiphop_beat__aggressive_dark_threatening.wav"
    },
    ...
  ]
}
```

## 注意事项

1. 确保有足够的GPU内存（如果使用GPU）或CPU资源
2. 首次运行时会下载CLAP模型，可能需要较长时间
3. 音频文件应为WAV格式
4. 确保所有音频文件路径都正确存在

## 技术细节

- 使用LAION CLAP模型（laion/clap-htsat-fused）
- 音频重采样到48kHz
- 计算音频和文本嵌入之间的余弦相似度
- 支持批量处理多个音频文件