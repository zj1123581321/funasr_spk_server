#!/bin/bash
# macOS 一键环境配置脚本
#
# 完整部署文档（含 prod PM2 / dev 前台、为什么不能 Docker 等）见：
#   docs/部署.md
#
# 本脚本仅做开发环境一次性 setup，生产部署不会调用此脚本。

set -e

echo "=== FunASR 转录服务器 macOS 环境配置 ==="

# 1. 检查 Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ 请先安装 Homebrew：https://brew.sh/"
    exit 1
fi

# 2. 系统依赖（Python 3.11 + ffmpeg）
echo "→ 安装系统依赖..."
brew update
brew install python@3.11 ffmpeg

# 3. 校验 Python
python3.11 --version

# 4. 创建项目 venv
echo "→ 创建 Python 虚拟环境..."
python3.11 -m venv venv

# 5. 装 Python 依赖
echo "→ 安装依赖（约 3-5 分钟）..."
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# 6. 准备运行时目录
echo "→ 创建必要目录..."
mkdir -p models uploads temp logs data

# 7. .env 提示
if [ ! -f ".env" ]; then
    echo "→ 复制 .env.example 为 .env（需手动编辑配置）"
    cp .env.example .env
    echo "  ⚠ 请编辑 .env 中的 FUNASR_WEBHOOK_URL / FUNASR_AUTH_SECRET_KEY"
fi

echo ""
echo "=== ✅ 配置完成 ==="
echo ""
echo "下一步："
echo "  开发模式（前台直跑）:  venv/bin/python run_server.py"
echo "  生产模式（PM2 守护）:  pm2 start ecosystem.config.cjs"
echo ""
echo "FunASR 模型首次推理时会自动从 ModelScope 下载（~1.2GB，存 ~/.cache/modelscope/）"
echo "详细部署文档见 docs/部署.md"
