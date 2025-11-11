#!/usr/bin/env python3
"""
详细评测报告生成脚本

该脚本会生成一个详细的评测报告，包括：
1. 每个评测者的偏好分析
2. 模型对比分析
3. 评测者一致性分析
4. 生成Markdown格式的报告
"""

import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# 评测结果文件路径
eval_results_dir = "/Users/jackson/Codes/fancy-musicgen/subjective/eval-results"
analysis_output_dir = "/Users/jackson/Codes/fancy-musicgen/subjective/analysis"

# 确保输出目录存在
os.makedirs(analysis_output_dir, exist_ok=True)

def load_evaluation_data():
    """加载所有评测者的评测结果数据"""
    all_data = []
    evaluators = []
    
    # 获取所有评测结果文件
    for filename in os.listdir(eval_results_dir):
        if filename.endswith('.json') and filename != 'example.json':
            evaluator = filename.split('.')[0]
            evaluators.append(evaluator)
            
            # 读取评测结果文件
            file_path = os.path.join(eval_results_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取评测数据
            for evaluation in data["evaluations"]:
                all_data.append({
                    'evaluator': evaluator,
                    'text_prompt': evaluation["text_prompt"],
                    'selected_model': evaluation["selected_model"]
                })
    
    return pd.DataFrame(all_data), evaluators

def analyze_evaluator_preferences(df):
    """分析每个评测者的偏好"""
    evaluator_analysis = {}
    
    for evaluator in df['evaluator'].unique():
        evaluator_df = df[df['evaluator'] == evaluator]
        counts = evaluator_df['selected_model'].value_counts()
        percentages = evaluator_df['selected_model'].value_counts(normalize=True) * 100
        
        # 确定评测者的首选模型
        preferred_model = counts.idxmax()
        preference_strength = percentages.max()
        
        evaluator_analysis[evaluator] = {
            'counts': counts.to_dict(),
            'percentages': {k: round(v, 2) for k, v in percentages.to_dict().items()},
            'preferred_model': preferred_model,
            'preference_strength': round(preference_strength, 2)
        }
    
    return evaluator_analysis

def analyze_model_consistency(df):
    """分析模型之间的一致性"""
    model_consistency = {}
    
    # 计算每个提示词上模型选择的分布
    prompt_analysis = df.groupby('text_prompt')['selected_model'].value_counts().unstack(fill_value=0)
    
    # 计算每个提示词上的主导模型
    dominant_models = prompt_analysis.idxmax(axis=1)
    dominant_percentages = prompt_analysis.max(axis=1) / prompt_analysis.sum(axis=1) * 100
    
    # 计算一致性指标
    consistency_score = dominant_percentages.mean()
    
    model_consistency = {
        'consistency_score': round(consistency_score, 2),
        'prompt_count': len(prompt_analysis),
        'fully_consistent_prompts': len(dominant_percentages[dominant_percentages == 100]),
        'highly_consistent_prompts': len(dominant_percentages[dominant_percentages >= 80]),
        'moderately_consistent_prompts': len(dominant_percentages[(dominant_percentages >= 60) & (dominant_percentages < 80)]),
        'low_consistency_prompts': len(dominant_percentages[dominant_percentages < 60])
    }
    
    return model_consistency, prompt_analysis

def generate_markdown_report(df, evaluator_analysis, model_consistency, model_scores):
    """生成Markdown格式的报告"""
    
    # 加载分析结果
    with open(os.path.join(analysis_output_dir, 'analysis_results.json'), 'r', encoding='utf-8') as f:
        analysis_results = json.load(f)
    
    # 开始构建报告
    report = []
    report.append("# 音乐生成模型主观评测报告\n")
    
    # 1. 评测概述
    report.append("## 1. 评测概述\n")
    report.append(f"- **评测者数量**: {len(df['evaluator'].unique())} 人")
    report.append(f"- **评测样本数量**: {len(df)} 条")
    report.append(f"- **评测模型**: base (预训练模型), best (验证损失最小模型), final (训练最终模型)\n")
    
    # 2. 总体偏好统计
    report.append("## 2. 总体偏好统计\n")
    report.append("| 模型 | 选择次数 | 选择比例 | 评分 |")
    report.append("|------|---------|----------|------|")
    
    for model in ['base', 'best', 'final']:
        count = analysis_results['preference_statistics']['counts'].get(model, 0)
        percentage = analysis_results['preference_statistics']['percentages'].get(model, 0)
        score = model_scores.get(model, 0)
        report.append(f"| {model} | {count} | {percentage}% | {score} |")
    
    report.append("")
    
    # 3. 显著性检验结果
    report.append("## 3. 显著性检验结果\n")
    report.append(f"- **检验方法**: 卡方检验")
    report.append(f"- **卡方值**: {analysis_results['significance_test']['chi2']}")
    report.append(f"- **P值**: {analysis_results['significance_test']['p_value']}")
    report.append(f"- **显著性**: {'是' if analysis_results['significance_test']['significant'] else '否'}")
    report.append(f"- **结论**: {'评测者对模型的偏好存在显著差异' if analysis_results['significance_test']['significant'] else '评测者对模型的偏好没有显著差异'}\n")
    
    # 4. 各评测者偏好分析
    report.append("## 4. 各评测者偏好分析\n")
    
    for evaluator, data in evaluator_analysis.items():
        report.append(f"### {evaluator.upper()}\n")
        report.append(f"- **首选模型**: {data['preferred_model']}")
        report.append(f"- **偏好强度**: {data['preference_strength']}%")
        report.append("- **模型选择分布**:")
        
        for model in ['base', 'best', 'final']:
            count = data['counts'].get(model, 0)
            percentage = data['percentages'].get(model, 0)
            report.append(f"  - {model}: {count}次 ({percentage}%)")
        
        report.append("")
    
    # 5. 模型一致性分析
    report.append("## 5. 模型一致性分析\n")
    report.append(f"- **一致性评分**: {model_consistency['consistency_score']}%")
    report.append(f"- **完全一致的提示词数量**: {model_consistency['fully_consistent_prompts']}")
    report.append(f"- **高度一致的提示词数量**: {model_consistency['highly_consistent_prompts']}")
    report.append(f"- **中等一致的提示词数量**: {model_consistency['moderately_consistent_prompts']}")
    report.append(f"- **低一致性的提示词数量**: {model_consistency['low_consistency_prompts']}\n")
    
    # 6. 结论与建议
    report.append("## 6. 结论与建议\n")
    
    # 确定最佳模型
    best_model = max(model_scores.items(), key=lambda x: x[1])[0]
    report.append(f"### 最佳模型\n")
    report.append(f"根据评测结果，**{best_model}** 模型表现最佳，评分为 {model_scores[best_model]} 分。\n")
    
    # 模型特点分析
    report.append("### 模型特点分析\n")
    
    if best_model == 'base':
        report.append("- **base模型** (预训练模型): 在评测中表现最佳，说明预训练模型已经具备了良好的音乐生成能力。")
        report.append("- **best模型** (验证损失最小模型): 虽然在验证集上表现最佳，但在主观评测中表现一般。")
        report.append("- **final模型** (训练最终模型): 表现介于base和best之间，可能存在过拟合问题。")
    elif best_model == 'best':
        report.append("- **best模型** (验证损失最小模型): 在评测中表现最佳，验证了验证损失作为模型选择指标的有效性。")
        report.append("- **base模型** (预训练模型): 作为基线模型，表现良好但不如best模型。")
        report.append("- **final模型** (训练最终模型): 可能存在过拟合问题，表现不如best模型。")
    else:  # final
        report.append("- **final模型** (训练最终模型): 在评测中表现最佳，说明完整的训练过程有助于提升模型性能。")
        report.append("- **base模型** (预训练模型): 作为基线模型，表现良好但不如final模型。")
        report.append("- **best模型** (验证损失最小模型): 虽然在验证集上表现最佳，但在主观评测中表现一般。")
    
    report.append("")
    
    # 建议
    report.append("### 建议\n")
    report.append("1. **模型选择**: 建议在实际应用中使用表现最佳的模型。")
    report.append("2. **进一步分析**: 可以进一步分析不同提示词上模型的表现差异，以找出模型的优缺点。")
    report.append("3. **用户研究**: 可以进行更大规模的用户研究，以验证当前评测结果的可靠性。")
    report.append("4. **模型改进**: 根据评测结果，可以针对性地改进模型在某些方面的表现。\n")
    
    # 7. 附录
    report.append("## 7. 附录\n")
    report.append("### 评测方法\n")
    report.append("1. 评测者对每个提示词生成的音乐进行盲听比较。")
    report.append("2. 评测者从三个模型中选择自己认为最佳的模型。")
    report.append("3. 评测结果使用统计方法进行分析。\n")
    
    # 将报告写入文件
    report_path = os.path.join(analysis_output_dir, 'evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return report_path

def main():
    """主函数"""
    print("生成详细评测报告...")
    
    # 加载数据
    df, evaluators = load_evaluation_data()
    
    # 分析评测者偏好
    evaluator_analysis = analyze_evaluator_preferences(df)
    
    # 分析模型一致性
    model_consistency, prompt_analysis = analyze_model_consistency(df)
    
    # 加载模型评分
    with open(os.path.join(analysis_output_dir, 'analysis_results.json'), 'r', encoding='utf-8') as f:
        analysis_results = json.load(f)
    model_scores = analysis_results['model_scores']
    
    # 生成Markdown报告
    report_path = generate_markdown_report(df, evaluator_analysis, model_consistency, model_scores)
    
    print(f"详细评测报告已生成: {report_path}")
    return report_path

if __name__ == "__main__":
    main()