#!/usr/bin/env python3
"""
CLAP (Contrastive Language-Audio Pretraining) 计算脚本
用于评估三个模型生成音乐与文本提示的匹配度
"""

import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm

try:
    import laion_clap
    from laion_clap import CLAP_Module
except ImportError as e:
    print(f"导入laion_clap库失败: {e}")
    print("请尝试手动安装: pip install laion_clap")
    exit(1)

class CLAPEvaluator:
    """CLAP评估器类"""
    
    def __init__(self, model_name: str = "laion/clap-htsat-fused"):
        """
        初始化CLAP模型
        
        Args:
            model_name: CLAP模型名称
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载CLAP模型
        self.model = CLAP_Module(enable_fusion=True, device=self.device)
        self.model.load_ckpt()
        
    def get_audio_embedding(self, audio_path: str) -> torch.Tensor:
        """
        获取音频的CLAP嵌入
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频嵌入向量
        """
        try:
            # 加载音频
            audio, sr = torchaudio.load(audio_path)
            
            # 确保音频是单声道
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # 重采样到48kHz（CLAP模型的期望采样率）
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000).to(self.device)
                audio = resampler(audio.to(self.device))
            else:
                audio = audio.to(self.device)
            
            # 获取音频嵌入
            with torch.no_grad():
                audio_embed = self.model.get_audio_embedding_from_data(x=audio, use_tensor=True)
                
            return audio_embed
        except Exception as e:
            print(f"处理音频文件 {audio_path} 时出错: {e}")
            return None
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        获取文本的CLAP嵌入
        
        Args:
            text: 文本提示
            
        Returns:
            文本嵌入向量
        """
        try:
            with torch.no_grad():
                text_embed = self.model.get_text_embedding([text], use_tensor=True)
                
            return text_embed
        except Exception as e:
            print(f"处理文本 '{text}' 时出错: {e}")
            return None
    
    def calculate_similarity(self, audio_embed: torch.Tensor, text_embed: torch.Tensor) -> float:
        """
        计算音频和文本嵌入之间的余弦相似度
        
        Args:
            audio_embed: 音频嵌入向量
            text_embed: 文本嵌入向量
            
        Returns:
            余弦相似度分数
        """
        if audio_embed is None or text_embed is None:
            return 0.0
            
        # 确保向量在CPU上用于计算
        audio_embed = audio_embed.cpu()
        text_embed = text_embed.cpu()
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(audio_embed, text_embed, dim=1)
        
        return similarity.item()
    
    def evaluate_audio_text_pair(self, audio_path: str, text_prompt: str) -> float:
        """
        评估单个音频-文本对的CLAP分数
        
        Args:
            audio_path: 音频文件路径
            text_prompt: 文本提示
            
        Returns:
            CLAP相似度分数
        """
        # 获取嵌入
        audio_embed = self.get_audio_embedding(audio_path)
        text_embed = self.get_text_embedding(text_prompt)
        
        # 计算相似度
        similarity = self.calculate_similarity(audio_embed, text_embed)
        
        return similarity


def load_comparison_data(jsonl_path: str) -> List[Dict]:
    """
    加载比较结果数据
    
    Args:
        jsonl_path: JSONL文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算音频-文本对的CLAP分数")
    parser.add_argument("--comparison_file", type=str, 
                        default="/Users/jackson/Codes/fancy-musicgen/finetune/output_epoch3/evaluation_full/comparison_results.jsonl",
                        help="比较结果JSONL文件路径")
    parser.add_argument("--audio_base_dir", type=str,
                        default="/Users/jackson/Codes/fancy-musicgen/finetune/output_epoch3/evaluation_full",
                        help="音频文件基础目录")
    parser.add_argument("--output_file", type=str,
                        default="/Users/jackson/Codes/fancy-musicgen/clap/clap_results.json",
                        help="输出结果文件路径")
    
    args = parser.parse_args()
    
    # 初始化CLAP评估器
    print("初始化CLAP模型...")
    evaluator = CLAPEvaluator()
    
    # 加载比较数据
    print(f"加载比较数据从 {args.comparison_file}...")
    comparison_data = load_comparison_data(args.comparison_file)
    print(f"加载了 {len(comparison_data)} 条记录")
    
    # 准备结果存储
    results = []
    
    # 处理每个数据项
    for item in tqdm(comparison_data, desc="计算CLAP分数"):
        text_prompt = item["text_prompt"]
        
        # 为每个模型计算CLAP分数
        result_item = {
            "text_prompt": text_prompt,
            "duration": item["duration"]
        }
        
        # 处理base_model
        base_model_path = os.path.join(args.audio_base_dir, item["base_model_file"])
        if os.path.exists(base_model_path):
            base_score = evaluator.evaluate_audio_text_pair(base_model_path, text_prompt)
            result_item["base_model_score"] = base_score
            result_item["base_model_file"] = item["base_model_file"]
        else:
            print(f"文件不存在: {base_model_path}")
            result_item["base_model_score"] = 0.0
        
        # 处理best_model
        best_model_path = os.path.join(args.audio_base_dir, item["best_model_file"])
        if os.path.exists(best_model_path):
            best_score = evaluator.evaluate_audio_text_pair(best_model_path, text_prompt)
            result_item["best_model_score"] = best_score
            result_item["best_model_file"] = item["best_model_file"]
        else:
            print(f"文件不存在: {best_model_path}")
            result_item["best_model_score"] = 0.0
        
        # 处理final_model
        final_model_path = os.path.join(args.audio_base_dir, item["final_model_file"])
        if os.path.exists(final_model_path):
            final_score = evaluator.evaluate_audio_text_pair(final_model_path, text_prompt)
            result_item["final_model_score"] = final_score
            result_item["final_model_file"] = item["final_model_file"]
        else:
            print(f"文件不存在: {final_model_path}")
            result_item["final_model_score"] = 0.0
        
        results.append(result_item)
    
    # 计算平均分数
    base_scores = [r["base_model_score"] for r in results if "base_model_score" in r]
    best_scores = [r["best_model_score"] for r in results if "best_model_score" in r]
    final_scores = [r["final_model_score"] for r in results if "final_model_score" in r]
    
    summary = {
        "total_samples": len(results),
        "base_model": {
            "average_score": np.mean(base_scores) if base_scores else 0.0,
            "std_score": np.std(base_scores) if base_scores else 0.0,
            "min_score": np.min(base_scores) if base_scores else 0.0,
            "max_score": np.max(base_scores) if base_scores else 0.0
        },
        "best_model": {
            "average_score": np.mean(best_scores) if best_scores else 0.0,
            "std_score": np.std(best_scores) if best_scores else 0.0,
            "min_score": np.min(best_scores) if best_scores else 0.0,
            "max_score": np.max(best_scores) if best_scores else 0.0
        },
        "final_model": {
            "average_score": np.mean(final_scores) if final_scores else 0.0,
            "std_score": np.std(final_scores) if final_scores else 0.0,
            "min_score": np.min(final_scores) if final_scores else 0.0,
            "max_score": np.max(final_scores) if final_scores else 0.0
        }
    }
    
    # 保存结果
    output_data = {
        "summary": summary,
        "detailed_results": results
    }
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n结果已保存到 {args.output_file}")
    print("\n===== CLAP评估摘要 =====")
    print(f"Base模型平均分数: {summary['base_model']['average_score']:.4f} ± {summary['base_model']['std_score']:.4f}")
    print(f"Best模型平均分数: {summary['best_model']['average_score']:.4f} ± {summary['best_model']['std_score']:.4f}")
    print(f"Final模型平均分数: {summary['final_model']['average_score']:.4f} ± {summary['final_model']['std_score']:.4f}")


if __name__ == "__main__":
    main()