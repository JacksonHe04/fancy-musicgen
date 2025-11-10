"""
MusicGen Model Service
提供模型加载和音乐生成的 Python 服务
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel
import scipy.io.wavfile as wavfile

# 设置HuggingFace镜像（如果需要）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 全局模型缓存
loaded_models = {}
BASE_MODEL_PATH = '/root/autodl-tmp/musicgen/local/musicgen-small'
FINETUNE_BASE = '/root/autodl-tmp/musicgen/finetune'


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_base_model(model_id='musicgen-small'):
    """加载基础模型（musicgen-small）"""
    if model_id in loaded_models:
        print(f"Model {model_id} already loaded")
        return loaded_models[model_id]
    
    print(f"Loading base model: {model_id} from {BASE_MODEL_PATH}")
    device = get_device()
    
    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
        model = MusicgenForConditionalGeneration.from_pretrained(BASE_MODEL_PATH)
        model.to(device)
        model.eval()
        
        loaded_models[model_id] = {
            'processor': processor,
            'model': model,
            'device': device,
            'is_lora': False
        }
        
        print(f"Base model {model_id} loaded successfully")
        return loaded_models[model_id]
    except Exception as e:
        print(f"Error loading base model: {e}")
        raise


def load_lora_model(model_id, scheme, model_type='best'):
    """加载 LoRA 微调模型"""
    cache_key = f"{scheme}-{model_type}"
    if cache_key in loaded_models:
        print(f"Model {cache_key} already loaded")
        return loaded_models[cache_key]
    
    lora_model_path = os.path.join(FINETUNE_BASE, scheme, f"{model_type}_model")
    
    if not os.path.exists(lora_model_path):
        raise FileNotFoundError(f"LoRA model path does not exist: {lora_model_path}")
    
    print(f"Loading LoRA model: {cache_key} from {lora_model_path}")
    device = get_device()
    
    try:
        # 先加载基础模型（如果还没加载）
        if 'musicgen-small' not in loaded_models:
            load_base_model()
        
        base_model = loaded_models['musicgen-small']['model']
        processor = loaded_models['musicgen-small']['processor']
        
        # 加载 LoRA 权重
        # PeftModel.from_pretrained 会创建一个新的模型，不会修改原模型
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model.to(device)
        model.eval()
        
        loaded_models[cache_key] = {
            'processor': processor,
            'model': model,
            'device': device,
            'is_lora': True,
            'base_model': 'musicgen-small'
        }
        
        print(f"LoRA model {cache_key} loaded successfully")
        return loaded_models[cache_key]
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_models(model_ids, scheme):
    """加载指定的模型列表"""
    results = {}
    
    for model_id in model_ids:
        try:
            if model_id == 'musicgen-small':
                load_base_model(model_id)
                results[model_id] = {'success': True, 'message': 'Model loaded successfully'}
            elif model_id.endswith('-best'):
                load_lora_model(model_id, scheme, 'best')
                results[model_id] = {'success': True, 'message': 'Model loaded successfully'}
            elif model_id.endswith('-final'):
                load_lora_model(model_id, scheme, 'final')
                results[model_id] = {'success': True, 'message': 'Model loaded successfully'}
            else:
                results[model_id] = {'success': False, 'message': f'Unknown model type: {model_id}'}
        except Exception as e:
            results[model_id] = {'success': False, 'message': str(e)}
    
    return results


def generate_music(model_id, scheme, prompt, tags, duration=10, output_dir=None):
    """生成音乐"""
    # 确定使用哪个模型
    if model_id == 'musicgen-small':
        cache_key = 'musicgen-small'
    elif model_id.endswith('-best'):
        cache_key = f"{scheme}-best"
    elif model_id.endswith('-final'):
        cache_key = f"{scheme}-final"
    else:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    if cache_key not in loaded_models:
        raise ValueError(f"Model {model_id} is not loaded. Please load it first.")
    
    model_info = loaded_models[cache_key]
    processor = model_info['processor']
    model = model_info['model']
    device = model_info['device']
    
    # 构建完整的提示词
    full_prompt = prompt
    if tags:
        full_prompt = f"{prompt}, {', '.join(tags)}"
    
    print(f"Generating music with model {model_id}")
    print(f"Prompt: {full_prompt}")
    
    # 处理输入
    inputs = processor(
        text=[full_prompt],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # 生成音频
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=int(duration * 50),  # 大约50 tokens/秒
            guidance_scale=3.0,
            temperature=1.0,
            top_k=250,
            top_p=None,
            do_sample=True,
        )
    
    # 获取采样率
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    # 保存音频
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'public', 'generated')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 清理文件名
    safe_prompt = prompt.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_prompt = safe_prompt.replace(',', '_').replace('.', '_').replace(':', '_')
    safe_prompt = safe_prompt.replace(';', '_').replace('!', '_').replace('?', '_')
    safe_prompt = safe_prompt.replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_')
    safe_prompt = safe_prompt.replace('{', '_').replace('}', '_').replace('|', '_')
    safe_prompt = safe_prompt[:50]  # 限制长度
    
    filename = f"{timestamp}_{model_id}_{safe_prompt}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # 转换为numpy并保存
    audio_np = audio_values[0].cpu().numpy()
    
    # 处理音频格式：如果是多通道，取第一个通道；如果是单通道，直接使用
    if audio_np.ndim > 1:
        if audio_np.shape[0] == 1:
            audio_np = audio_np[0]
        else:
            audio_np = audio_np[0]  # 取第一个通道
    
    # 确保音频值在合理范围内 [-1, 1]
    audio_np = np.clip(audio_np, -1.0, 1.0)
    
    # 转换为 int16 格式 (范围: -32768 到 32767)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    # 保存为WAV文件
    wavfile.write(filepath, int(sampling_rate), audio_int16)
    
    print(f"Audio saved to: {filepath}")
    
    return {
        'filename': filename,
        'filepath': filepath,
        'url': f'/generated/{filename}',
        'model_id': model_id,
        'prompt': prompt,
        'tags': tags
    }


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_service.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'load':
        # python model_service.py load '["musicgen-small"]' output
        model_ids = json.loads(sys.argv[2])
        scheme = sys.argv[3] if len(sys.argv) > 3 else ''
        results = load_models(model_ids, scheme)
        print(json.dumps(results))
    
    elif command == 'generate':
        # python model_service.py generate musicgen-small output "test prompt" '["tag1"]' 10
        model_id = sys.argv[2]
        scheme = sys.argv[3]
        prompt = sys.argv[4]
        tags = json.loads(sys.argv[5]) if len(sys.argv) > 5 else []
        duration = int(sys.argv[6]) if len(sys.argv) > 6 else 10
        result = generate_music(model_id, scheme, prompt, tags, duration)
        print(json.dumps(result))

