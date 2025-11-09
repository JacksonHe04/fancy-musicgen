"""
数据预处理脚本，用于准备训练数据
"""
import os
import json
import argparse
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil


def resample_audio(input_path: str, output_path: str, target_sr: int = 32000):
    """重采样音频到目标采样率"""
    try:
        audio, sr = librosa.load(input_path, sr=None)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def create_metadata_from_folder(
    input_dir: str,
    output_dir: str,
    metadata_file: str,
    target_sr: int = 32000,
    train_ratio: float = 0.8,
    default_text: str = "hip hop 808 beat"
):
    """
    从文件夹创建数据集和元数据
    
    Args:
        input_dir: 输入音频文件夹
        output_dir: 输出目录
        metadata_file: 元数据JSON文件路径
        target_sr: 目标采样率
        train_ratio: 训练集比例
        default_text: 默认文本描述
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的音频格式
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    # 查找所有音频文件
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(input_path.rglob(f"*{ext}")))
        audio_files.extend(list(input_path.rglob(f"*{ext.upper()}")))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # 随机打乱
    import random
    random.shuffle(audio_files)
    
    # 划分训练集和验证集
    split_idx = int(len(audio_files) * train_ratio)
    train_files = audio_files[:split_idx]
    valid_files = audio_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}")
    
    # 处理文件并创建元数据
    metadata = []
    
    # 处理训练集
    for idx, audio_file in enumerate(tqdm(train_files, desc="Processing training set")):
        output_file = train_dir / f"{idx:04d}.wav"
        if resample_audio(str(audio_file), str(output_file), target_sr):
            metadata.append({
                "audio": f"train/{output_file.name}",
                "text": default_text
            })
    
    # 处理验证集
    for idx, audio_file in enumerate(tqdm(valid_files, desc="Processing validation set")):
        output_file = valid_dir / f"{idx:04d}.wav"
        if resample_audio(str(audio_file), str(output_file), target_sr):
            metadata.append({
                "audio": f"valid/{output_file.name}",
                "text": default_text
            })
    
    # 保存元数据
    metadata_path = output_path / metadata_file
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved to {metadata_path}")
    print(f"Total samples: {len(metadata)}")


def create_example_metadata(output_dir: str, num_samples: int = 10):
    """创建示例元数据文件（用于测试）"""
    output_path = Path(output_dir)
    metadata_path = output_path / "metadata.json"
    
    # 创建示例数据目录
    train_dir = output_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    hip_hop_texts = [
        "hip hop 808 beat with trap drums",
        "808 bass and hip hop drum pattern",
        "trap beat with heavy 808 kick",
        "hip hop instrumental with 808",
        "808 drum pattern with trap elements",
        "hip hop beat with 808 bass",
        "trap music with 808 drums",
        "808 kick and hip hop snare",
        "hip hop 808 beat",
        "trap beat 808 bass"
    ]
    
    for i in range(num_samples):
        text = hip_hop_texts[i % len(hip_hop_texts)]
        metadata.append({
            "audio": f"train/sample_{i:04d}.wav",
            "text": text
        })
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Example metadata created at {metadata_path}")
    print(f"Note: You need to add actual audio files to {train_dir}")


def main():
    parser = argparse.ArgumentParser(description="准备MusicGen训练数据")
    parser.add_argument("--input_dir", type=str, help="输入音频文件夹")
    parser.add_argument("--output_dir", type=str, default="./finetune/data", help="输出目录")
    parser.add_argument("--metadata_file", type=str, default="metadata.json", help="元数据文件名")
    parser.add_argument("--target_sr", type=int, default=32000, help="目标采样率")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--default_text", type=str, default="hip hop 808 beat", help="默认文本描述")
    parser.add_argument("--create_example", action="store_true", help="创建示例元数据文件")
    parser.add_argument("--num_samples", type=int, default=10, help="示例样本数量")
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_metadata(args.output_dir, args.num_samples)
    elif args.input_dir:
        create_metadata_from_folder(
            args.input_dir,
            args.output_dir,
            args.metadata_file,
            args.target_sr,
            args.train_ratio,
            args.default_text
        )
    else:
        print("Please provide --input_dir or use --create_example")
        parser.print_help()


if __name__ == "__main__":
    main()

