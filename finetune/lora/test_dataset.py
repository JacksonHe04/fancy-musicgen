"""
测试数据集加载脚本
"""
import os
import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from dataset import MusicGenDataset, collate_fn
from torch.utils.data import DataLoader

def test_dataset():
    """测试数据集加载"""
    # 设置路径
    base_dir = Path("/root/autodl-tmp/musicgen")
    model_path = base_dir / "local" / "musicgen-small"
    data_dir = base_dir / "finetune" / "data"
    metadata_file = data_dir / "metadata.json"
    
    # 设置HuggingFace镜像（如果需要）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("1. Loading processor...")
    processor = AutoProcessor.from_pretrained(str(model_path))
    
    print("2. Creating dataset...")
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        print("Please run prepare_data.py first to create metadata.json")
        return
    
    dataset = MusicGenDataset(
        data_dir=str(data_dir),
        metadata_path=str(metadata_file),
        processor=processor,
        sample_rate=32000,
        max_duration=30.0,
        target_duration=30.0,
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("   Warning: Dataset is empty!")
        return
    
    print("3. Testing dataset item...")
    try:
        sample = dataset[0]
        print(f"   Sample keys: {sample.keys()}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
    except Exception as e:
        print(f"   Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("4. Testing dataloader...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # 使用0避免多进程问题
        )
        
        print("5. Loading batch...")
        batch = next(iter(dataloader))
        print(f"   Batch keys: {batch.keys()}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
        
        print("\n✅ Dataset test passed!")
        
    except Exception as e:
        print(f"   Error loading batch: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()

