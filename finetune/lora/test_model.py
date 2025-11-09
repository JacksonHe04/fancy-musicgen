"""
测试MusicGen模型的前向传播
"""
import os
import torch
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration

def test_model_forward():
    """测试模型前向传播"""
    # 设置路径
    model_path = "/root/autodl-tmp/musicgen/local/musicgen-small"
    
    # 设置HuggingFace镜像（如果需要）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("1. Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = MusicgenForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   Using device: {device}")
    
    # 创建测试数据
    print("2. Creating test data...")
    text = ["hip hop 808 beat with trap drums"]
    
    # 创建随机音频（用于测试）
    sample_rate = 32000
    duration = 10
    audio = np.random.randn(sample_rate * duration).astype(np.float32)
    
    # 处理输入
    print("3. Processing inputs...")
    inputs = processor(
        text=text,
        audio=[audio],
        sampling_rate=sample_rate,
        padding=True,
        return_tensors="pt",
    )
    
    print("4. Input keys:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   {key}: {type(value)}")
    
    # 移动到设备
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # 测试前向传播
    print("5. Testing forward pass...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("6. Output keys:")
        for key in outputs.keys():
            if hasattr(outputs[key], 'shape'):
                print(f"   {key}: shape={outputs[key].shape}")
            else:
                print(f"   {key}: {type(outputs[key])}")
        
        # 检查是否有loss
        if hasattr(outputs, 'loss'):
            print(f"   loss: {outputs.loss}")
        else:
            print("   No loss in outputs (this is normal for inference)")
        
        print("\n✅ Model forward pass test passed!")
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试训练模式
    print("\n7. Testing training mode...")
    model.train()
    try:
        outputs = model(**inputs)
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            print(f"   Training loss: {outputs.loss.item()}")
            print("✅ Training mode test passed!")
        else:
            print("⚠️  Warning: No loss in training mode outputs")
            print("   This might indicate that the model needs labels for training")
    except Exception as e:
        print(f"❌ Error in training mode: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_forward()

