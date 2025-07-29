#!/bin/bash

# macOS 环境配置脚本
# 为 FunASR 转录服务器配置 macOS 开发环境

echo "=== FunASR 转录服务器 macOS 环境配置 ==="

# 检查 Homebrew 是否安装
if ! command -v brew &> /dev/null; then
    echo "错误: 请先安装 Homebrew"
    echo "访问 https://brew.sh/ 获取安装说明"
    exit 1
fi

# 更新 Homebrew
echo "更新 Homebrew..."
brew update

# 安装系统依赖
echo "安装系统依赖..."
brew install python@3.10 ffmpeg portaudio

# 检查 Python 版本
python3 --version

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装 Python 依赖
echo "安装 Python 依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p models uploads temp logs data

# 检查配置文件
if [ ! -f "config.json" ]; then
    echo "创建默认配置文件..."
    python -c "
from src.core.config import Config
config = Config()
config.save_to_file('config.json')
print('配置文件已创建：config.json')
"
fi

echo "=== macOS 环境配置完成 ==="
echo "运行以下命令启动服务器:"
echo "  source venv/bin/activate"
echo "  python run_server.py"