"""
批量对比评测脚本
处理 evaluation_prompts.json 中的所有prompts，生成对比结果
"""
import os
import sys
import json
import argparse
from pathlib import Path

# 添加当前目录到路径，以便导入 evaluate 模块
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import compare_models, compare_multi_models


def main():
    parser = argparse.ArgumentParser(description="批量对比评测 MusicGen LoRA 模型")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/musicgen/local/musicgen-small",
        help="基础模型路径"
    )
    parser.add_argument("--lora_model_path", type=str, help="LoRA模型路径")
    parser.add_argument("--best_model_path", type=str, help="best模型路径")
    parser.add_argument("--final_model_path", type=str, help="final模型路径")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="/root/autodl-tmp/musicgen/finetune/data/eval/evaluation_prompts.json",
        help="测试提示词文件路径（JSON格式，字符串数组）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/musicgen/finetune/output/evaluation",
        help="输出目录"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="生成时长（秒）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # 设置HuggingFace镜像（如果需要）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 读取测试提示词
    print(f"Loading prompts from: {args.prompts_file}")
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    # 处理不同的数据格式
    if isinstance(prompts_data, list):
        test_prompts = prompts_data
    elif isinstance(prompts_data, dict):
        test_prompts = prompts_data.get('prompts', [])
    else:
        raise ValueError(f"Unsupported prompts file format: {type(prompts_data)}")
    
    print(f"Loaded {len(test_prompts)} prompts")
    
    # 确保输出目录存在
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 执行对比评测
    print("\n" + "=" * 70)
    print("Starting Batch Comparison Evaluation")
    print("=" * 70)
    print(f"Base Model: {args.base_model_path}")
    if args.lora_model_path:
        print(f"LoRA Model: {args.lora_model_path}")
    if args.best_model_path:
        print(f"Best Model: {args.best_model_path}")
    if args.final_model_path:
        print(f"Final Model: {args.final_model_path}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Duration: {args.duration}s")
    print(f"Device: {args.device}")
    print(f"Total Prompts: {len(test_prompts)}")
    print("=" * 70 + "\n")
    
    model_map = {"base": args.base_model_path}
    if args.lora_model_path:
        model_map["lora"] = args.lora_model_path
    if args.best_model_path:
        model_map["best"] = args.best_model_path
    if args.final_model_path:
        model_map["final"] = args.final_model_path

    compare_multi_models(
        base_model_path=args.base_model_path,
        model_map=model_map,
        test_prompts=test_prompts,
        output_dir=args.output_dir,
        device=args.device,
        duration=args.duration,
    )
    
    print("\n" + "=" * 70)
    print("Batch Evaluation Completed!")
    print("=" * 70)
    print(f"Results saved to: {output_path / 'comparison_results.jsonl'}")
    print(f"Per-model audio dirs saved under: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

