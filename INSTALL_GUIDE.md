# 依赖安装指南

## 问题说明

在安装 `requirements.txt` 时，`av==10.0.0` 包会因为与新版 Cython 不兼容而编译失败。错误信息显示 Cython 编译时类型不兼容。

## 解决方案

已创建 `install_deps.sh` 脚本来解决这个问题。该脚本会：

1. 安装兼容的 Cython 版本（<3.0）
2. 使用 `--no-build-isolation` 标志单独安装 `av==10.0.0` 包
3. 安装 `requirements.txt` 中的其他依赖
4. 如果镜像源中某些包不可用，会自动尝试从官方 PyPI 安装

## 使用方法

```bash
# 确保在正确的 conda 环境中
conda activate music

# 运行安装脚本
bash install_deps.sh
```

## 脚本工作原理

- **步骤1**: 安装兼容的 Cython 版本（<3.0），避免新版 Cython 的类型检查问题
- **步骤2**: 使用 `--no-build-isolation` 安装 av 包，这样可以使用当前环境的 Cython，而不是构建隔离环境中的新版本
- **步骤3**: 安装其他依赖，pip 会检测到 av 已安装并跳过它

## 注意事项

- 确保系统已安装 FFmpeg 开发库（通常已安装）
- 如果安装过程中遇到网络问题，脚本会自动尝试使用官方 PyPI 源
- 脚本不会修改 `requirements.txt` 文件

