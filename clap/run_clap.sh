#!/bin/bash

# CLAP计算运行脚本
# 用于计算三个模型生成音乐与文本提示的CLAP分数

echo "===== CLAP音乐评估脚本 ====="

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 检查是否存在audiocraft_env环境
if ! conda env list | grep -q "audiocraft_env"; then
    echo "错误: 未找到audiocraft_env环境，请先创建该环境"
    echo "可以使用以下命令创建:"
    echo "conda create -n audiocraft_env python=3.9 -y"
    exit 1
fi

# 激活conda环境
echo "激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate audiocraft_env

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 设置环境变量
export TORCH_HOME="$(pwd)/torch_cache"
mkdir -p $TORCH_HOME

# 运行CLAP计算
echo "开始计算CLAP分数..."
python3 calculate_clap.py

# 检查结果
if [ -f "clap_results.json" ]; then
    echo "计算完成！结果已保存到 clap_results.json"
    echo ""
    echo "结果摘要:"
    python3 -c "
import json
with open('clap_results.json', 'r') as f:
    data = json.load(f)
    summary = data['summary']
    print(f\"Base模型平均分数: {summary['base_model']['average_score']:.4f} ± {summary['base_model']['std_score']:.4f}\")
    print(f\"Best模型平均分数: {summary['best_model']['average_score']:.4f} ± {summary['best_model']['std_score']:.4f}\")
    print(f\"Final模型平均分数: {summary['final_model']['average_score']:.4f} ± {summary['final_model']['std_score']:.4f}\")
"
else
    echo "错误: 未找到结果文件"
    exit 1
fi

echo ""
echo "===== 计算完成 ====="