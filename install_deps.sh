#!/bin/bash
# 安装脚本：解决av包编译问题
# 此脚本解决av==10.0.0在编译时与新版Cython不兼容的问题
# 使用方法: bash install_deps.sh

# 不使用set -e，以便在安装失败时能够尝试备用方案

echo "=========================================="
echo "开始安装依赖..."
echo "=========================================="

# 步骤1: 安装兼容的Cython版本和构建工具
echo ""
echo "步骤1: 安装兼容的Cython版本（<3.0）..."
pip install "Cython<3.0" wheel --quiet
echo "✓ Cython安装完成"

# 步骤2: 单独安装av包（使用--no-build-isolation）
# 这样可以使用当前环境的Cython版本，而不是构建隔离环境中的新版本
echo ""
echo "步骤2: 安装av包（使用--no-build-isolation避免构建隔离问题）..."
pip install --no-build-isolation av==10.0.0
echo "✓ av包安装完成"

# 步骤2.5: 确保libstdc++版本足够新（解决GLIBCXX版本问题）
echo ""
echo "步骤2.5: 更新libstdc++库（如果需要）..."
# 尝试更新libstdc++，如果失败也不影响后续安装
conda install -y -c conda-forge libstdcxx-ng 2>/dev/null || echo "  (libstdc++更新跳过，如果后续有导入错误请手动安装)"
echo "✓ libstdc++检查完成"

# 步骤3: 安装requirements.txt中的其他依赖
# pip会检测到av已经安装并跳过它
echo ""
echo "步骤3: 安装其他依赖..."
echo "（如果镜像源中某些包不可用，将自动尝试官方PyPI源）"

# 首先尝试从当前镜像源安装
pip install -r requirements.txt 2>&1 | tee /tmp/pip_install.log
PIP_EXIT_CODE=${PIPESTATUS[0]}

if [ $PIP_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ 所有依赖安装成功！"
else
    echo ""
    echo "从镜像源安装遇到问题，尝试从官方PyPI安装..."
    # 从官方PyPI安装（会跳过已安装的包）
    pip install -r requirements.txt -i https://pypi.org/simple
    if [ $? -eq 0 ]; then
        echo "✓ 从官方PyPI安装完成"
    else
        echo "⚠ 安装过程中可能有错误，请检查上面的输出"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "依赖安装完成！"
echo "=========================================="

