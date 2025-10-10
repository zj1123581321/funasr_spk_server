# FunASR 转录服务器多平台部署指南

## 环境要求
- Python 3.10+
- FFmpeg
- 8GB+ 内存（推荐）
- Docker（可选）

## 平台特定安装指南

### 🍎 macOS 部署

#### 自动安装（推荐）
```bash
# 运行 macOS 配置脚本
chmod +x setup_mac.sh
./setup_mac.sh
```

#### 手动安装
```bash
# 1. 安装 Homebrew（如果没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装系统依赖
brew install python@3.10 ffmpeg portaudio

# 3. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 4. 安装 Python 依赖
pip install --upgrade pip
pip install -r requirements.txt

# 5. 创建目录和配置
mkdir -p models uploads temp logs data
cp .env.example .env  # 可选：自定义环境变量

# 6. 启动服务器
python run_server.py
```

#### macOS 特定注意事项
- 可能需要允许终端访问麦克风权限
- M1/M2 芯片需要 Rosetta 2 支持某些依赖
- 建议使用虚拟环境避免权限问题

### 🐧 Linux 部署

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev
sudo apt-get install ffmpeg git wget curl gcc g++

# CentOS/RHEL
sudo yum install python3.10 python3-pip
sudo yum install ffmpeg git wget curl gcc gcc-c++

# 创建虚拟环境和安装依赖
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 启动服务
python run_server.py
```

### 🐳 Docker 部署（推荐用于生产环境）

#### 使用 Docker Compose（推荐）
```bash
# 1. 克隆项目
git clone <repository-url>
cd funasr_spk_server

# 2. 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 文件设置自定义配置

# 3. 启动服务
docker-compose up -d

# 4. 查看日志
docker-compose logs -f

# 5. 停止服务
docker-compose down
```

#### 使用 Docker 直接运行
```bash
# 构建镜像
docker build -t funasr-server .

# 运行容器
docker run -d \
  --name funasr_spk_server \
  -p 8765:8765 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.json:/app/config.json \
  funasr-server
```

## 跨平台兼容性优化

### 已实现的优化
1. **路径处理统一**：使用 `pathlib.Path` 处理所有文件路径
2. **平台特定依赖**：`uvloop` 仅在非 Windows 平台安装
3. **环境变量配置**：支持通过环境变量自定义配置
4. **Docker 多阶段构建**：优化镜像大小和安全性

### 性能优化建议

#### macOS 特定优化
```bash
# 设置环境变量优化性能
export OMP_NUM_THREADS=4
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

#### Linux 特定优化
```bash
# 设置环境变量
export OMP_NUM_THREADS=$(nproc)
export MALLOC_MMAP_THRESHOLD_=65536
```

## 配置说明

### 环境变量配置
复制 `.env.example` 为 `.env` 并根据需要修改：

```bash
cp .env.example .env
```

### 配置文件
主要配置文件为 `config.json`，包含：
- 服务器设置（端口、连接数等）
- 模型配置（模型路径、设备等）
- 转录设置（并发数、缓存等）
- 通知配置（企微 webhook 等）

## 故障排除

### 常见问题

#### 1. 模型下载失败
```bash
# 手动下载模型
mkdir -p models
# 从 ModelScope 或 HuggingFace 下载所需模型
```

#### 2. 权限问题（macOS/Linux）
```bash
# 确保目录权限正确
chmod -R 755 models uploads temp logs data
```

#### 3. 端口被占用
```bash
# 检查端口占用
lsof -i :8765  # macOS/Linux
netstat -ano | findstr :8765  # Windows

# 修改端口
export PORT=8766  # 或在 .env 文件中设置
```

#### 4. 内存不足
- 减少 `max_concurrent_tasks` 配置
- 增加系统交换空间
- 使用更小的批处理大小

## 监控和日志

### 日志文件位置
- Docker: `/app/logs/`
- 本地: `./logs/`

### 健康检查
```bash
# Docker 环境
docker-compose ps
docker-compose exec funasr-server python -c "import websockets; print('OK')"

# 本地环境
curl -f http://localhost:8765/health || echo "服务异常"
```

## 安全建议

1. **生产环境**：
   - 修改默认的 `secret_key`
   - 启用认证功能
   - 使用 HTTPS 代理

2. **网络安全**：
   - 限制访问 IP 范围
   - 使用防火墙规则
   - 定期更新依赖

3. **数据安全**：
   - 定期备份数据库
   - 清理临时文件
   - 监控磁盘空间