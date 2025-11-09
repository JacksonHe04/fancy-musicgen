"""
MusicGen LoRA微调评测脚本
"""
import os
import argparse
from pathlib import Path
import torch
import json
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel
import numpy as np
from inference_lora import load_lora_model, generate_music, save_audio


def evaluate_model(
    base_model_path: str,
    lora_model_path: str,
    test_prompts: list,
    output_dir: str,
    device: str = "cuda",
    duration: int = 10,
):
    """评测模型"""
    print("=" * 50)
    print("Evaluating MusicGen LoRA Model")
    print("=" * 50)
    
    # 加载模型
    processor, model = load_lora_model(base_model_path, lora_model_path, device)
    
    # 测试提示词
    if test_prompts is None:
        test_prompts = [
            "hip hop 808 beat with trap drums",
            "808 bass and hip hop drum pattern",
            "trap beat with heavy 808 kick",
            "hip hop instrumental with 808",
            "808 drum pattern with trap elements",
        ]
    
    print(f"\n测试提示词数量: {len(test_prompts)}")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. {prompt}")
    
    # 生成音频
    print(f"\n开始生成音频（每个提示词生成1个样本）...")
    audio_values, sampling_rate = generate_music(
        processor,
        model,
        test_prompts,
        duration=duration,
        device=device,
    )
    
    # 保存音频
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_files = save_audio(audio_values, sampling_rate, str(output_path), test_prompts)
    
    # 保存评测结果
    results = {
        "model_path": lora_model_path,
        "test_prompts": test_prompts,
        "output_files": output_files,
        "sampling_rate": int(sampling_rate),
        "duration": duration,
    }
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评测结果已保存到: {results_file}")
    print(f"生成的音频文件:")
    for file in output_files:
        print(f"  - {file}")
    
    return results


def compare_models(
    base_model_path: str,
    lora_model_path: str,
    test_prompts: list,
    output_dir: str,
    device: str = "cuda",
    duration: int = 10,
):
    """对比基础模型和微调后的模型"""
    print("=" * 50)
    print("Comparing Base Model vs Fine-tuned Model")
    print("=" * 50)
    
    # 加载基础模型
    print("\n1. Loading base model...")
    base_processor = AutoProcessor.from_pretrained(base_model_path)
    base_model = MusicgenForConditionalGeneration.from_pretrained(base_model_path)
    base_model.to(device)
    base_model.eval()
    
    # 加载LoRA模型
    print("2. Loading LoRA model...")
    lora_processor, lora_model = load_lora_model(base_model_path, lora_model_path, device)
    
    # 测试提示词
    if test_prompts is None:
        test_prompts = [
            "hip hop 808 beat with trap drums",
            "808 bass and hip hop drum pattern",
        ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成对比音频
    print(f"\n3. Generating audio with base model...")
    base_output_dir = output_path / "base_model"
    base_audio, base_sr = generate_music(
        base_processor,
        base_model,
        test_prompts,
        duration=duration,
        device=device,
    )
    base_files = save_audio(base_audio, base_sr, str(base_output_dir), test_prompts)
    
    print(f"4. Generating audio with LoRA model...")
    lora_output_dir = output_path / "lora_model"
    lora_audio, lora_sr = generate_music(
        lora_processor,
        lora_model,
        test_prompts,
        duration=duration,
        device=device,
    )
    lora_files = save_audio(lora_audio, lora_sr, str(lora_output_dir), test_prompts)
    
    # 保存对比结果
    comparison = {
        "test_prompts": test_prompts,
        "base_model_files": base_files,
        "lora_model_files": lora_files,
        "duration": duration,
    }
    
    comparison_file = output_path / "comparison_results.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n对比结果已保存到: {comparison_file}")
    print(f"\n基础模型输出:")
    for file in base_files:
        print(f"  - {file}")
    print(f"\nLoRA模型输出:")
    for file in lora_files:
        print(f"  - {file}")


def main():
    parser = argparse.ArgumentParser(description="MusicGen LoRA评测")
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/musicgen/local/musicgen-small", help="基础模型路径")
    parser.add_argument("--lora_model_path", type=str, required=True, help="LoRA模型路径")
    parser.add_argument("--test_prompts", type=str, nargs="+", help="测试提示词")
    parser.add_argument("--test_prompts_file", type=str, help="测试提示词文件（JSON格式）")
    parser.add_argument("--output_dir", type=str, default="./finetune/output/evaluation", help="输出目录")
    parser.add_argument("--duration", type=int, default=10, help="生成时长（秒）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--compare", action="store_true", help="对比基础模型和微调模型")
    
    args = parser.parse_args()
    
    # 设置HuggingFace镜像（如果需要）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # 加载测试提示词
    test_prompts = args.test_prompts
    if args.test_prompts_file:
        with open(args.test_prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            test_prompts = data.get('prompts', [])
    
    # 执行评测
    if args.compare:
        compare_models(
            args.base_model_path,
            args.lora_model_path,
            test_prompts,
            args.output_dir,
            args.device,
            args.duration,
        )
    else:
        evaluate_model(
            args.base_model_path,
            args.lora_model_path,
            test_prompts,
            args.output_dir,
            args.device,
            args.duration,
        )


if __name__ == "__main__":
    main()

