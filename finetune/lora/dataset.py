"""
数据集加载模块，用于MusicGen LoRA微调
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import numpy as np
from transformers import AutoProcessor


class MusicGenDataset(Dataset):
    """MusicGen数据集类，用于加载音频和文本描述"""
    
    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        processor: AutoProcessor,
        sample_rate: int = 32000,
        max_duration: float = 30.0,
        target_duration: float = 30.0,
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 音频文件目录
            metadata_path: 元数据JSON文件路径
            processor: MusicGen处理器
            sample_rate: 目标采样率
            max_duration: 最大音频时长（秒）
            target_duration: 目标音频时长（秒）
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.target_duration = target_duration
        
        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 验证数据
        self.samples = []
        for item in self.metadata:
            audio_path = self.data_dir / item['audio']
            if audio_path.exists():
                self.samples.append({
                    'audio': str(audio_path),
                    'text': item['text']
                })
            else:
                print(f"Warning: Audio file not found: {audio_path}")
        
        print(f"Loaded {len(self.samples)} samples from {metadata_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个数据样本"""
        sample = self.samples[idx]
        
        # 加载音频
        try:
            audio, sr = librosa.load(sample['audio'], sr=self.sample_rate, duration=self.max_duration)
        except Exception as e:
            print(f"Error loading audio {sample['audio']}: {e}")
            # 返回静音音频作为fallback
            audio = np.zeros(int(self.sample_rate * self.target_duration))
            sr = self.sample_rate
        
        # 确保音频长度一致
        target_length = int(self.sample_rate * self.target_duration)
        if len(audio) < target_length:
            # 如果音频太短，进行填充
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            # 如果音频太长，进行裁剪
            audio = audio[:target_length]
        
        # 处理文本和音频
        # MusicGen processor需要音频作为numpy数组
        inputs = self.processor(
            text=[sample['text']],
            audio=[audio],  # 作为列表传递
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        
        # 移除batch维度（因为只有一个样本）
        # 但保留batch维度用于训练
        result = {}
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                # 保持batch维度
                result[key] = inputs[key]
            else:
                result[key] = inputs[key]
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """批处理函数 - MusicGen processor已经处理了batch，这里只需要简单合并"""
    # processor返回的tensor已经包含了batch维度
    # 我们只需要concatenate或者stack
    
    batch_dict = {}
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch]
        
        if isinstance(values[0], torch.Tensor):
            # 如果tensor已经有batch维度，需要stack
            # processor返回的input_ids等通常是2D: [batch_size, seq_len]
            # audio_values可能是3D: [batch_size, channels, seq_len]
            try:
                # 尝试直接stack
                if values[0].dim() >= 1:
                    batch_dict[key] = torch.cat(values, dim=0)
                else:
                    batch_dict[key] = torch.stack(values)
            except Exception as e:
                # 如果cat失败，尝试stack
                try:
                    batch_dict[key] = torch.stack(values)
                except Exception as e2:
                    # 如果都失败，保持列表
                    batch_dict[key] = values
        else:
            # 对于非tensor值，保持列表
            batch_dict[key] = values
    
    return batch_dict

