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
    base_output_dir = output_path / "base_model"
    lora_output_dir = output_path / "lora_model"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    lora_output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSONL文件路径
    comparison_file = output_path / "comparison_results.jsonl"
    
    print(f"\n3. Processing {len(test_prompts)} prompts...")
    
    # 打开JSONL文件进行写入
    with open(comparison_file, 'w', encoding='utf-8') as f:
        # 逐个处理每个prompt
        for idx, prompt in enumerate(test_prompts, 1):
            print(f"\n[{idx}/{len(test_prompts)}] Processing: {prompt}")
            
            # 使用单个prompt列表进行生成
            prompt_list = [prompt]
            
            # 生成基础模型音频
            print(f"  Generating audio with base model...")
            base_audio, base_sr = generate_music(
                base_processor,
                base_model,
                prompt_list,
                duration=duration,
                device=device,
            )
            base_files = save_audio(base_audio, base_sr, str(base_output_dir), prompt_list)
            base_file = base_files[0] if base_files else None
            # 转换为相对路径（相对于输出目录）
            if base_file:
                base_file = str(Path(base_file).relative_to(output_path))
            
            # 生成LoRA模型音频
            print(f"  Generating audio with LoRA model...")
            lora_audio, lora_sr = generate_music(
                lora_processor,
                lora_model,
                prompt_list,
                duration=duration,
                device=device,
            )
            lora_files = save_audio(lora_audio, lora_sr, str(lora_output_dir), prompt_list)
            lora_file = lora_files[0] if lora_files else None
            # 转换为相对路径（相对于输出目录）
            if lora_file:
                lora_file = str(Path(lora_file).relative_to(output_path))
            
            # 创建记录
            record = {
                "text_prompt": prompt,
                "base_model_file": base_file,
                "lora_model_file": lora_file,
                "duration": duration,
            }
            
            # 写入JSONL文件（每行一个JSON对象）
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()  # 确保立即写入
            
            print(f"  ✓ Completed: base={base_file}, lora={lora_file}")
    
    print(f"\n对比结果已保存到: {comparison_file}")
    print(f"共处理 {len(test_prompts)} 个prompt")


def compare_multi_models(
    base_model_path: str,
    model_map: dict,
    test_prompts: list,
    output_dir: str,
    device: str = "cuda",
    duration: int = 10,
):
    """对比多个模型（可选包含 base/best/final 等）。
    model_map: { model_key(str)-> model_path(str) }, 如 {"base": base_model_path, "best": "/path/best", "final": "/path/final"}
    """
    print("=" * 50)
    print("Comparing Multiple Models")
    print("=" * 50)

    # 预加载各模型与输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processors = {}
    models = {}
    subdirs = {}

    for key, path in model_map.items():
        if key == "base":
            print(f"Loading base model: {path}")
            processors[key] = AutoProcessor.from_pretrained(base_model_path)
            models[key] = MusicgenForConditionalGeneration.from_pretrained(base_model_path).to(device).eval()
        else:
            print(f"Loading LoRA model '{key}': {path}")
            processors[key], models[key] = load_lora_model(base_model_path, path, device)
        sub = output_path / f"{key}_model"
        sub.mkdir(parents=True, exist_ok=True)
        subdirs[key] = sub

    if test_prompts is None:
        test_prompts = ["hip hop 808 beat with trap drums"]

    comparison_file = output_path / "comparison_results.jsonl"
    print(f"\nProcessing {len(test_prompts)} prompts across {list(model_map.keys())}...")

    with open(comparison_file, 'w', encoding='utf-8') as f:
        for idx, prompt in enumerate(test_prompts, 1):
            print(f"\n[{idx}/{len(test_prompts)}] {prompt}")
            prompt_list = [prompt]

            record = {
                "text_prompt": prompt,
                "duration": duration,
            }

            for key in model_map.keys():
                audio, sr = generate_music(
                    processors[key],
                    models[key],
                    prompt_list,
                    duration=duration,
                    device=device,
                )
                files = save_audio(audio, sr, str(subdirs[key]), prompt_list)
                rel = None
                if files:
                    rel = str(Path(files[0]).relative_to(output_path))
                record[f"{key}_model_file"] = rel

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\n对比结果已保存到: {comparison_file}")
    print(f"共处理 {len(test_prompts)} 个prompt")


def main():
    parser = argparse.ArgumentParser(description="MusicGen LoRA评测")
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/musicgen/local/musicgen-small", help="基础模型路径")
    parser.add_argument("--lora_model_path", type=str, help="LoRA模型路径（单模型或与base对比）")
    parser.add_argument("--best_model_path", type=str, help="可选：best模型路径")
    parser.add_argument("--final_model_path", type=str, help="可选：final模型路径")
    parser.add_argument("--test_prompts", type=str, nargs="+", help="测试提示词")
    parser.add_argument("--test_prompts_file", type=str, help="测试提示词文件（JSON格式）")
    parser.add_argument("--output_dir", type=str, default="./finetune/output/evaluation", help="输出目录")
    parser.add_argument("--duration", type=int, default=10, help="生成时长（秒）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--compare", action="store_true", help="对比基础模型和微调模型（双模型）")
    parser.add_argument("--compare_multi", action="store_true", help="对比多模型（base/best/final 任意组合）")
    
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
            # 如果data是列表，直接使用；如果是字典，尝试获取prompts键
            if isinstance(data, list):
                test_prompts = data
            elif isinstance(data, dict):
                test_prompts = data.get('prompts', [])
            else:
                test_prompts = []
    
    # 执行评测
    if args.compare_multi:
        model_map = {}
        # base 总是可选
        model_map["base"] = args.base_model_path
        if args.lora_model_path:
            model_map["lora"] = args.lora_model_path
        if args.best_model_path:
            model_map["best"] = args.best_model_path
        if args.final_model_path:
            model_map["final"] = args.final_model_path
        compare_multi_models(
            args.base_model_path,
            model_map,
            test_prompts,
            args.output_dir,
            args.device,
            args.duration,
        )
    elif args.compare:
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

