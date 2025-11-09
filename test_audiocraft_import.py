import sys
import torch
import torchaudio

# 测试基本导入
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("TorchAudio version:", torchaudio.__version__)

# 添加项目路径到Python路径
sys.path.append('/Users/jackson/Codes/musicgen/audiocraft')

try:
    # 尝试导入audiocraft的核心模块
    from audiocraft.models import MusicGen, AudioGen
    print("Successfully imported MusicGen and AudioGen")
except ImportError as e:
    print(f"Import error: {e}")
    print("This might be expected if not all dependencies are properly linked")