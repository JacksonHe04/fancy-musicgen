"""
MusicGen LoRA推理脚本
"""
import os
import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel
import scipy.io.wavfile as wavfile
from datetime import datetime


def load_lora_model(base_model_path: str, lora_model_path: str, device: str = "cuda"):
    """加载基础模型和LoRA权重"""
    print(f"Loading base model from {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path)
    model = MusicgenForConditionalGeneration.from_pretrained(base_model_path)
    
    print(f"Loading LoRA weights from {lora_model_path}")
    model = PeftModel.from_pretrained(model, lora_model_path)
    
    model.to(device)
    model.eval()
    
    return processor, model


def generate_music(
    processor,
    model,
    prompts: list,
    duration: int = 10,
    guidance_scale: float = 3.0,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
    device: str = "cuda",
):
    """生成音乐"""
    # 处理输入
    inputs = processor(
        text=prompts,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # 生成音频
    print(f"Generating music for prompts: {prompts}")
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=int(duration * 50),  # 大约50 tokens/秒
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p if top_p > 0 else None,
            do_sample=True,
        )
    
    # 获取采样率
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    return audio_values, sampling_rate


def save_audio(audio_values: torch.Tensor, sampling_rate: int, output_dir: str, prompts: list):
    """保存音频文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = []
    
    # audio_values shape: (batch_size, num_channels, sequence_length)
    for idx, audio in enumerate(audio_values):
        # 转换为numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio
        
        # 如果是立体声，取第一个通道
        if audio_np.ndim > 1:
            audio_np = audio_np[0]
        
        # 创建文件名（清理特殊字符）
        prompt_text = prompts[idx]
        # 替换可能影响文件名的字符
        safe_text = prompt_text.replace(' ', '_').replace('/', '_').replace('\\', '_')
        safe_text = safe_text.replace(',', '_').replace('.', '_').replace(':', '_')
        safe_text = safe_text.replace(';', '_').replace('!', '_').replace('?', '_')
        safe_text = safe_text.replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_')
        safe_text = safe_text.replace('{', '_').replace('}', '_').replace('|', '_')
        safe_text = safe_text[:50]  # 限制长度
        filename = f"{timestamp}_{idx}_{safe_text}.wav"
        filepath = output_path / filename
        
        # 保存音频
        wavfile.write(str(filepath), sampling_rate, audio_np.astype(np.float32))
        output_files.append(str(filepath))
        print(f"Saved audio to: {filepath}")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(description="MusicGen LoRA推理")
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/musicgen/local/musicgen-small", help="基础模型路径")
    parser.add_argument("--lora_model_path", type=str, required=True, help="LoRA模型路径")
    parser.add_argument("--prompts", type=str, nargs="+", default=["hip hop 808 beat with trap drums"], help="文本提示")
    parser.add_argument("--output_dir", type=str, default="./finetune/output/generated", help="输出目录")
    parser.add_argument("--duration", type=int, default=10, help="生成时长（秒）")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="引导尺度")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度")
    parser.add_argument("--top_k", type=int, default=250, help="Top-k采样")
    parser.add_argument("--top_p", type=float, default=0.0, help="Top-p采样")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    # 设置HuggingFace镜像（如果需要）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # 加载模型
    processor, model = load_lora_model(args.base_model_path, args.lora_model_path, args.device)
    
    # 生成音乐
    audio_values, sampling_rate = generate_music(
        processor,
        model,
        args.prompts,
        duration=args.duration,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )
    
    # 保存音频
    output_files = save_audio(audio_values, sampling_rate, args.output_dir, args.prompts)
    
    print(f"\n生成完成！共生成 {len(output_files)} 个音频文件")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

