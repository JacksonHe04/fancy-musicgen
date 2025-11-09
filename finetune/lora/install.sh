#!/bin/bash
# 安装依赖脚本

echo "Installing MusicGen LoRA fine-tuning dependencies..."

pip install torch>=2.0.0
pip install transformers>=4.31.0
pip install peft>=0.5.0
pip install accelerate>=0.20.0
pip install datasets>=2.14.0
pip install soundfile>=0.12.1
pip install librosa>=0.10.0
pip install numpy>=1.24.0
pip install tqdm>=4.65.0
pip install scipy>=1.10.0
pip install torchaudio>=2.0.0

echo "Dependencies installed!"

# 验证安装
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Installation complete!"

