#!/usr/bin/env python3
"""
创建主观评测结果JSON文件的脚本

该脚本会为5个评测者(hjc, jm, sry, lzj, gr)创建评测结果文件，
每个文件包含50个评测条目，对应evaluation_prompts.json中的提示词，
但selected_model字段为空，等待评测者填写。
"""

import json
import os
from datetime import datetime

# 评测者列表
evaluators = ["hjc", "jm", "sry", "lzj", "gr"]

# 输入和输出路径
eval_prompts_path = "/Users/jackson/Codes/fancy-musicgen/finetune/data/eval/evaluation_prompts.json"
output_dir = "/Users/jackson/Codes/fancy-musicgen/subjective/eval-results"

def create_evaluation_files():
    """为每个评测者创建评测结果JSON文件"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取评测提示词
    with open(eval_prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # 为每个评测者创建文件
    for evaluator in evaluators:
        # 创建评测结果数据结构
        evaluation_data = {
            "evaluator": evaluator,
            "evaluation_date": current_date,
            "evaluations": []
        }
        
        # 为每个提示词创建评测条目
        for prompt in prompts:
            evaluation_entry = {
                "text_prompt": prompt,
                "selected_model": ""  # 初始为空，等待评测者填写
            }
            evaluation_data["evaluations"].append(evaluation_entry)
        
        # 写入JSON文件
        output_file = os.path.join(output_dir, f"{evaluator}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        print(f"已创建评测文件: {output_file}")
        print(f"包含 {len(evaluation_data['evaluations'])} 个评测条目")

if __name__ == "__main__":
    create_evaluation_files()
    print("\n所有评测文件已创建完成！")