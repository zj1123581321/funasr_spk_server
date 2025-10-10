# FunASR Speaker Recognition Server

基于FunASR的音视频转录服务器，支持说话人识别和时间戳标注，提供JSON和SRT两种输出格式。

## 功能特性

- ✅ 支持多种音视频格式（wav, mp3, mp4, m4a, flac等）
- ✅ 自动说话人识别和分离
- ✅ 精确到秒/毫秒的时间戳标注
- ✅ 双输出格式：JSON（合并说话人）& SRT（原始分割）
- ✅ WebSocket实时通信
- ✅ 任务队列管理，支持并发处理
- ✅ 智能缓存，相同文件直接返回
- ✅ 原始结果保存，支持格式转换
- ✅ 企微机器人通知
- ✅ JWT认证机制
- ✅ 支持Docker部署

## 架构说明

### 服务端架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebSocket     │───▶│  Task Manager   │───▶│ FunASR Engine   │
│   Handler       │    │                 │    │                 │
│  (接收请求)      │    │  (队列管理)      │    │  (AI转录)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Auth &        │    │   Database      │    │  File Manager   │
│   Validation    │    │   (缓存&原始)    │    │  (上传&清理)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### 核心组件

1. **WebSocket Handler** (`src/api/websocket_handler.py`)
   - 处理客户端连接和消息路由
   - 支持认证和连接管理
   - 实时进度推送

2. **Task Manager** (`src/core/task_manager.py`)
   - 并发任务队列管理
   - 智能文件生命周期管理
   - 缓存策略和重试机制

3. **FunASR Transcriber** (`src/core/funasr_transcriber.py`)
   - 基于FunASR的AI转录引擎
   - 支持双格式输出（JSON/SRT）
   - 说话人识别和时间戳标注

4. **Database Manager** (`src/core/database.py`)
   - 转录结果缓存
   - 原始数据保存
   - 支持格式转换

5. **File Manager** (`src/utils/file_utils.py`)
   - 文件上传和验证
   - 格式转换和清理

## 快速开始

### 环境要求

- Python 3.8+
- FFmpeg
- 4GB+ 内存

### 本地运行

1. 克隆项目
```bash
git clone <repository_url>
cd funasr_spk_server
```

2. 创建虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
复制 `.env.example` 为 `.env` 并编辑：
```bash
cp .env.example .env
# 编辑 .env 文件，配置必需的环境变量
```

**必需配置项：**
- `FUNASR_WEBHOOK_URL`: 企业微信机器人 Webhook 地址（如果启用通知）
- `FUNASR_AUTH_SECRET_KEY`: JWT 认证密钥（如果启用认证，生产环境必须修改）

**可选配置项：**
- 服务器地址和端口
- 日志级别
- 目录路径
- FunASR 设备配置
- 性能参数

详细配置说明见 `.env.example` 文件注释。

5. 启动服务器
```bash
python run_server.py
```

### Docker部署

1. 构建镜像
```bash
docker-compose build
```

2. 启动服务
```bash
docker-compose up -d
```

## 客户端用法

### 测试客户端

基础测试：
```bash
python tests/server/test_server_transcription.py
```

并发测试：
```bash
python tests/server/test_concurrent_transcription.py [客户端数量]
```

### WebSocket API

#### 1. 连接到服务器
```javascript
const ws = new WebSocket('ws://localhost:8767');
```

#### 2. 认证（如果启用）
```json
{
  "type": "auth",
  "data": {
    "token": "your-jwt-token"
  }
}
```

#### 3. 上传文件请求
```json
{
  "type": "upload_request",
  "data": {
    "file_name": "test.mp3",
    "file_size": 1024000,
    "file_hash": "md5-hash",
    "force_refresh": false,
    "output_format": "json"  // 或 "srt"
  }
}
```

#### 4. 上传文件数据
```json
{
  "type": "upload_data",
  "data": {
    "task_id": "task-id",
    "file_data": "base64-encoded-data"
  }
}
```

#### 5. 接收转录结果
服务器会发送以下类型的消息：
- `task_progress`: 转录进度更新
- `task_complete`: 转录完成
- `error`: 错误信息

## 输出格式对比

### JSON 格式（推荐用于数据处理）

**特点**：
- 自动合并相同说话人的连续句子
- 提供完整的元数据和统计信息
- 便于程序处理和分析

**适用场景**：
- 会议纪要整理
- 对话分析
- 数据挖掘

```json
{
  "task_id": "uuid",
  "file_name": "meeting.mp3",
  "file_hash": "md5-hash",
  "duration": 120.5,
  "segments": [
    {
      "start_time": 0.88,
      "end_time": 5.195,
      "text": "欢迎大家来体验达摩院推出的语音识别模型。",
      "speaker": "Speaker1"
    }
  ],
  "speakers": ["Speaker1"],
  "processing_time": 1.03
}
```

### SRT 格式（推荐用于字幕制作）

**特点**：
- 保持FunASR原始的句子分割
- 不合并说话人内容，保持原始粒度
- 标准SRT字幕格式，兼容性好

**适用场景**：
- 视频字幕制作
- 直播转录
- 播客字幕

```srt
1
00:00:00,880 --> 00:00:05,195
Speaker1:欢迎大家来体验达摩院推出的语音识别模型。
```

### 请求格式选择

在上传请求中指定 `output_format` 参数：

```json
// JSON格式（默认）
{
  "type": "upload_request",
  "data": {
    "file_name": "test.mp3",
    "output_format": "json"
  }
}

// SRT格式
{
  "type": "upload_request", 
  "data": {
    "file_name": "test.mp3",
    "output_format": "srt"
  }
}
```

### 响应格式

#### JSON格式响应
```json
{
  "type": "task_complete",
  "data": {
    "task_id": "uuid",
    "result": {
      "task_id": "uuid",
      "file_name": "test.mp3",
      "segments": [...],
      "speakers": [...],
      // ... 完整的转录结果
    }
  }
}
```

#### SRT格式响应
```json
{
  "type": "task_complete", 
  "data": {
    "task_id": "uuid",
    "result": {
      "format": "srt",
      "content": "1\n00:00:00,880 --> 00:00:05,195\nSpeaker1:欢迎大家...\n\n",
      "file_name": "test.mp3",
      "file_hash": "md5-hash"
    }
  }
}
```

## 配置说明

### 配置文件说明

本项目采用**双配置文件**架构：

1. **`.env`** - 环境变量配置（敏感信息和环境相关）
   - 企业微信 Webhook URL
   - JWT 认证密钥
   - 服务器地址和端口
   - 目录路径
   - 日志级别
   - FunASR 设备配置
   - 性能参数覆盖

2. **`config.json`** - 业务逻辑配置（无敏感信息）
   - FunASR 模型配置
   - 转录任务配置
   - 缓存策略
   - WebSocket 连接参数

### 配置优先级

**环境变量 (.env) > config.json > 代码默认值**

### 快速配置步骤

1. **复制环境变量模板**
```bash
cp .env.example .env
```

2. **编辑 `.env` 文件，配置必需项**
```env
# 如果启用通知功能，必须配置 Webhook URL
FUNASR_NOTIFICATION_ENABLED=true
FUNASR_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your-key

# 如果启用认证功能，必须修改密钥（生产环境）
FUNASR_AUTH_ENABLED=false
FUNASR_AUTH_SECRET_KEY=your-secret-key-change-this-in-production
```

3. **根据需要调整性能参数**
```env
# 并发任务数（建议为 CPU 核心数的一半）
FUNASR_MAX_CONCURRENT_TASKS=4

# 设备配置（Mac GPU 加速）
FUNASR_DEVICE=auto
FUNASR_DEVICE_PRIORITY=mps,cpu
```

### 主要配置项

#### 环境变量配置 (.env)

| 配置项 | 说明 | 默认值 | 必需 |
|--------|------|--------|------|
| `FUNASR_SERVER_HOST` | 服务器监听地址 | 0.0.0.0 | 否 |
| `FUNASR_SERVER_PORT` | 服务器监听端口 | 8767 | 否 |
| `FUNASR_WEBHOOK_URL` | 企业微信 Webhook URL | - | 条件必需* |
| `FUNASR_AUTH_SECRET_KEY` | JWT 认证密钥 | - | 条件必需** |
| `FUNASR_NOTIFICATION_ENABLED` | 是否启用通知 | true | 否 |
| `FUNASR_AUTH_ENABLED` | 是否启用认证 | false | 否 |
| `FUNASR_LOG_LEVEL` | 日志级别 | INFO | 否 |
| `FUNASR_DEVICE` | 计算设备 | auto | 否 |
| `FUNASR_DEVICE_PRIORITY` | 设备优先级 | mps,cpu | 否 |
| `FUNASR_MAX_CONCURRENT_TASKS` | 最大并发任务数 | 2 | 否 |

\* 仅当 `FUNASR_NOTIFICATION_ENABLED=true` 时必需
\** 仅当 `FUNASR_AUTH_ENABLED=true` 时必需，且生产环境必须修改默认值

完整配置项说明请参考 `.env.example` 文件。

#### 业务配置 (config.json)

- **server**: WebSocket 服务器业务配置（连接数、文件大小限制、支持的格式等）
- **funasr**: FunASR 模型配置（模型名称、版本等）
- **transcription**: 转录任务管理配置（超时时间、重试次数、缓存策略等）
- **database**: 数据库配置（缓存过期时间等）
- **notification**: 通知配置（重试次数、超时时间等）
- **auth**: 认证配置（算法、令牌过期时间等）
- **logging**: 日志配置（格式、轮转、保留期等）

**注意**: `config.json` 中的大部分配置都可以通过环境变量覆盖，详见文件中的 `_comment` 注释。

## 缓存机制

### 智能缓存策略

1. **文件哈希识别**：基于MD5哈希避免重复转录
2. **原始结果保存**：保存FunASR原始输出，支持格式转换
3. **格式灵活切换**：同一音频可输出不同格式而无需重新转录
4. **过期清理**：自动清理过期缓存数据

### 缓存优势

- **性能提升**：相同文件秒级返回结果
- **成本节约**：避免重复计算消耗
- **格式支持**：缓存后可随时切换输出格式

## 并发处理

### 任务队列管理

- **多工作线程**：支持配置并发任务数量
- **智能调度**：队列化处理，避免资源竞争
- **文件管理**：同一文件多任务时智能文件生命周期管理
- **错误重试**：自动重试机制，提高成功率

### 并发测试

使用并发测试工具验证服务器性能：

```bash
# 默认4个并发客户端
python tests/server/test_concurrent_transcription.py

# 自定义并发数
python tests/server/test_concurrent_transcription.py 8
```

## 注意事项

### 系统要求

1. **内存要求**：建议16GB以上内存以获得最佳性能
2. **存储空间**：首次运行会自动下载模型文件（约2GB）
3. **网络要求**：模型下载需要稳定的网络连接
4. **FFmpeg依赖**：必须安装FFmpeg用于音频处理

### 生产环境

1. **认证配置**：生产环境请修改JWT密钥
2. **并发调优**：根据CPU核心数调整并发任务数
3. **缓存管理**：定期清理过期缓存，控制数据库大小
4. **监控日志**：关注错误日志和性能指标

## 故障排除

### 常见问题

1. **模型下载失败**
   ```bash
   # 检查网络连接
   ping huggingface.co
   # 手动下载模型到models目录
   ```

2. **转录速度慢**
   ```json
   // 增加并发数
   "max_concurrent_tasks": 8,
   // 调整批处理大小
   "batch_size_s": 200
   ```

3. **内存不足错误**
   ```json
   // 减少并发数
   "max_concurrent_tasks": 2,
   // 减小批处理大小
   "batch_size_s": 100
   ```

4. **WebSocket连接失败**
   ```bash
   # 检查防火墙设置
   netstat -an | findstr 8767
   # 检查端口占用
   ```

5. **文件格式不支持**
   ```json
   // 添加支持的文件扩展名
   "allowed_extensions": [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
   ```

### 调试模式

开启详细日志：
```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

## 性能优化

### 硬件优化

- **CPU**：多核心CPU提升并发处理能力
- **内存**：大内存支持更多并发任务
- **存储**：SSD提升文件I/O性能

### 软件优化

- **并发配置**：根据硬件调整并发任务数
- **缓存策略**：合理配置缓存过期时间
- **文件清理**：及时清理临时文件

## 开发扩展

### 自定义格式支持

1. 在 `funasr_transcriber.py` 中添加新的格式处理方法
2. 在 `database.py` 中添加格式转换逻辑
3. 更新API文档和客户端示例

### 第三方集成

- **Web前端**：提供HTTP REST API封装
- **移动端SDK**：基于WebSocket的移动端集成
- **云服务**：Docker容器化部署

## License

MIT License