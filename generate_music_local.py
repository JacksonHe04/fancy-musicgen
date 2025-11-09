import os
import sys
import torch
import torchaudio
import numpy as np
from datetime import datetime

# 设置HuggingFace镜像源（如果需要）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目路径到Python路径
sys.path.append('/root/autodl-tmp/musicgen/audiocraft')

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_music_from_text(prompts, duration=8, output_dir='./output', model_path='/root/autodl-tmp/musicgen/local/musicgen-small'):
    """
    使用本地MusicGen模型根据文本提示生成音乐
    
    Args:
        prompts (list): 文本提示列表
        duration (int): 生成音乐的时长（秒）
        output_dir (str): 输出目录
        model_path (str): 本地模型路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载本地预训练模型
    print(f"加载本地模型: {model_path}")
    model = MusicGen.get_pretrained(model_path)
    # MusicGen对象不支持to()方法，移除这行
    
    # 设置生成参数
    model.set_generation_params(
        duration=duration,
        temperature=1.0,  # 控制生成的随机性，较低的值使输出更确定
        top_k=250,        # 仅考虑概率最高的k个token
        top_p=0.0,        # 不使用nucleus sampling
        cfg_coef=3.0      # 分类器自由引导系数，控制提示的影响强度
    )
    
    # 生成音乐
    print(f"生成音乐中，基于提示: {prompts}")
    with torch.no_grad():
        wav = model.generate(prompts)
    
    # 保存生成的音乐
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = []
    
    for idx, one_wav in enumerate(wav):
        prompt_text = prompts[idx].replace(' ', '_')[:20]  # 为文件名创建简短提示
        # audio_write会自动添加.wav扩展名，所以这里不需要包含扩展名
        output_file_stem = os.path.join(output_dir, f"{timestamp}_{idx}_{prompt_text}")
        
        # 保存音频，使用响度标准化
        output_file = audio_write(
            output_file_stem,
            one_wav.cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True
        )
        
        print(f"已保存音乐到: {output_file}")
        output_files.append(str(output_file))
    
    return output_files

def main():
    # 示例文本提示
    prompts = [
        # "Cheerful piano music, full of vitality and hope",
        # "Calm jazz background music, suitable for cafes",
        "Trap hip hop music, full of energy and excitement"
    ]
    
    # 生成音乐
    output_files = generate_music_from_text(
        prompts=prompts,
        duration=10,  # 生成10秒音乐
        output_dir='./output_music',
        model_path='/root/autodl-tmp/musicgen/local/musicgen-small'
    )
    
    print(f"\n音乐生成完成! 共生成 {len(output_files)} 个音频文件。")

if __name__ == "__main__":
    main()