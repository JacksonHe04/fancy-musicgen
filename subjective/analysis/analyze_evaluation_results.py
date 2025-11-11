#!/usr/bin/env python3
"""
主观评测结果分析脚本

该脚本会对评测结果数据进行统计分析，包括：
1. 偏好统计 - 计算每个模型被选择的次数和比例
2. 显著性检验 - 使用卡方检验评估模型间差异的显著性
3. 归一化处理 - 将选择次数转换为百分制分数
4. 可视化结果 - 生成图表展示分析结果
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from collections import Counter, defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def preference_statistics(df):
    """计算偏好统计"""
    print("=" * 50)
    print("偏好统计分析")
    print("=" * 50)
    
    # 总体偏好统计
    total_counts = df['selected_model'].value_counts()
    total_percentages = df['selected_model'].value_counts(normalize=True) * 100
    
    print("总体偏好统计:")
    for model in ['base', 'best', 'final']:
        count = total_counts.get(model, 0)
        percentage = total_percentages.get(model, 0)
        print(f"{model}: {count}次 ({percentage:.2f}%)")
    
    # 每个评测者的偏好统计
    print("\n各评测者偏好统计:")
    evaluator_stats = {}
    for evaluator in df['evaluator'].unique():
        evaluator_df = df[df['evaluator'] == evaluator]
        counts = evaluator_df['selected_model'].value_counts()
        percentages = evaluator_df['selected_model'].value_counts(normalize=True) * 100
        
        evaluator_stats[evaluator] = {
            'counts': counts.to_dict(),
            'percentages': percentages.to_dict()
        }
        
        print(f"\n{evaluator}:")
        for model in ['base', 'best', 'final']:
            count = counts.get(model, 0)
            percentage = percentages.get(model, 0)
            print(f"  {model}: {count}次 ({percentage:.2f}%)")
    
    return total_counts, total_percentages, evaluator_stats

def significance_test(df):
    """进行显著性检验"""
    print("\n" + "=" * 50)
    print("显著性检验 (卡方检验)")
    print("=" * 50)
    
    # 创建列联表
    contingency_table = pd.crosstab(df['evaluator'], df['selected_model'])
    
    print("列联表:")
    print(contingency_table)
    
    # 进行卡方检验
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\n卡方检验结果:")
    print(f"卡方值: {chi2:.4f}")
    print(f"自由度: {dof}")
    print(f"P值: {p_value:.4f}")
    
    # 解释结果
    alpha = 0.05
    if p_value < alpha:
        print(f"\n由于P值({p_value:.4f}) < {alpha}，拒绝原假设，认为评测者对模型的偏好存在显著差异。")
    else:
        print(f"\n由于P值({p_value:.4f}) >= {alpha}，不能拒绝原假设，认为评测者对模型的偏好没有显著差异。")
    
    return chi2, p_value, dof, expected

def calculate_scores(df):
    """计算每个模型的百分制分数"""
    print("\n" + "=" * 50)
    print("模型评分计算")
    print("=" * 50)
    
    # 计算每个模型的总选择次数
    model_counts = df['selected_model'].value_counts()
    total_evaluations = len(df)
    
    # 计算每个模型的选择比例
    model_percentages = (model_counts / total_evaluations) * 100
    
    # 使用更合理的评分方法
    # 方法：将选择比例直接映射到百分制，但设置基准分和调整系数
    # 基准分：50分（表示随机选择水平，三个模型平均约为33.3%）
    # 调整系数：1.5（用于放大差异，使评分更有区分度）
    
    base_score = 50  # 基准分
    adjustment_factor = 1.5  # 调整系数
    
    model_scores = {}
    for model, percentage in model_percentages.items():
        # 计算得分：基准分 + (选择比例 - 随机期望) * 调整系数
        random_expectation = 100 / len(model_counts)  # 随机选择期望值
        score = base_score + (percentage - random_expectation) * adjustment_factor
        
        # 确保分数在0-100范围内
        score = max(0, min(100, score))
        model_scores[model] = round(score, 2)
    
    # 添加评分方法说明
    print("评分方法说明:")
    print(f"- 基准分：{base_score}分（表示随机选择水平）")
    print(f"- 随机选择期望：{100/len(model_counts):.1f}%（三个模型平均选择比例）")
    print(f"- 调整系数：{adjustment_factor}（用于放大差异）")
    print("- 计算公式：得分 = 基准分 + (选择比例 - 随机期望) × 调整系数")
    print("- 分数范围：0-100分")
    
    print("\n模型评分结果:")
    for model in ['base', 'best', 'final']:
        if model in model_percentages:
            percentage = model_percentages[model]
            score = model_scores.get(model, 0)
            print(f"{model}: 选择率 {percentage:.2f}%, 评分 {score}")
    
    return model_percentages, model_scores

def visualize_results(total_counts, total_percentages, model_scores):
    """可视化分析结果"""
    print("\n生成可视化图表...")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('主观评测结果分析', fontsize=16)
    
    # 1. 模型选择次数柱状图
    ax1 = axes[0, 0]
    models = ['base', 'best', 'final']
    counts = [total_counts.get(model, 0) for model in models]
    bars = ax1.bar(models, counts, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_title('模型选择次数')
    ax1.set_ylabel('选择次数')
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{count}', ha='center', va='bottom')
    
    # 2. 模型选择比例饼图
    ax2 = axes[0, 1]
    percentages = [total_percentages.get(model, 0) for model in models]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax2.pie(percentages, labels=models, colors=colors, autopct='%1.1f%%',
                                       startangle=90)
    ax2.set_title('模型选择比例')
    
    # 3. 模型评分柱状图
    ax3 = axes[1, 0]
    scores = [model_scores.get(model, 0) for model in models]
    bars = ax3.bar(models, scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax3.set_title('模型评分')
    ax3.set_ylabel('评分')
    ax3.set_ylim(0, 100)
    
    # 在柱状图上添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{score}', ha='center', va='bottom')
    
    # 4. 模型对比雷达图
    ax4 = axes[1, 1]
    # 归一化数据到0-1范围用于雷达图
    max_count = max(counts)
    normalized_counts = [count/max_count for count in counts]
    max_score = max(scores)
    normalized_scores = [score/max_score for score in scores]
    
    # 雷达图设置
    angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    normalized_counts += normalized_counts[:1]
    normalized_scores += normalized_scores[:1]
    
    ax4.plot(angles, normalized_counts, 'o-', linewidth=2, label='选择率')
    ax4.fill(angles, normalized_counts, alpha=0.25)
    ax4.plot(angles, normalized_scores, 'o-', linewidth=2, label='评分')
    ax4.fill(angles, normalized_scores, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(models)
    ax4.set_ylim(0, 1)
    ax4.set_title('模型对比雷达图')
    ax4.legend()
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 保存图表
    output_path = os.path.join(analysis_output_dir, 'evaluation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    return output_path

def save_analysis_results(total_counts, total_percentages, model_scores, chi2, p_value):
    """保存分析结果到JSON文件"""
    results = {
        "preference_statistics": {
            "counts": total_counts.to_dict(),
            "percentages": {k: round(v, 2) for k, v in total_percentages.to_dict().items()}
        },
        "model_scores": model_scores,
        "significance_test": {
            "chi2": round(chi2, 4),
            "p_value": round(p_value, 4),
            "significant": bool(p_value < 0.05)
        }
    }
    
    output_path = os.path.join(analysis_output_dir, 'analysis_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n分析结果已保存至: {output_path}")
    return output_path

def main():
    """主函数"""
    print("开始主观评测结果分析...")
    
    # 加载数据
    df, evaluators = load_evaluation_data()
    print(f"已加载 {len(evaluators)} 位评测者的 {len(df)} 条评测数据")
    
    # 偏好统计
    total_counts, total_percentages, evaluator_stats = preference_statistics(df)
    
    # 显著性检验
    chi2, p_value, dof, expected = significance_test(df)
    
    # 计算模型评分
    model_percentages, model_scores = calculate_scores(df)
    
    # 可视化结果
    chart_path = visualize_results(total_counts, total_percentages, model_scores)
    
    # 保存分析结果
    results_path = save_analysis_results(total_counts, total_percentages, model_scores, chi2, p_value)
    
    print("\n分析完成！")
    return df, total_counts, total_percentages, model_scores, chi2, p_value

if __name__ == "__main__":
    main()