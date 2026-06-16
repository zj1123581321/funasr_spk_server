# FunASR音频转文本服务器兼容性开发文档

> **适用范围**: 本文档描述 **FunASR 生产引擎路径** 的协议和实现细节,用于开发兼容客户端。
> 项目当前是**多引擎架构**, 除 FunASR 外, Qwen3 引擎已通过多 worker pool 接入生产(`src/core/qwen3_pool_transcriber.py`),
> 协议层通过 `upload_request.engine` 字段路由 — 详见 `docs/开发/Server-Client 交互协议.md` §2.1 + `CLAUDE.md` ASR 引擎章节。
> 引擎选择对客户端透明,本文档描述的字段/流程对两种引擎都适用。

## 项目概述

本文档为您分析了现有FunASR音频转文本服务器项目的完整技术架构和实现细节，以便您开发完全兼容的新服务器。该项目是一个基于FunASR的音视频转录服务器，支持说话人识别、时间戳标注，提供JSON和SRT两种输出格式。

## 1. 系统架构

### 1.1 核心架构模式

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

### 1.2 核心组件

1. **主服务器** (`src/main.py`)：管理整体服务器生命周期
2. **WebSocket处理器** (`src/api/websocket_handler.py`)：处理客户端连接和消息路由
3. **任务管理器** (`src/core/task_manager.py`)：并发任务队列管理
4. **FunASR转录器** (`src/core/funasr_transcriber.py`)：AI转录引擎
5. **数据库管理器** (`src/core/database.py`)：转录结果缓存
6. **文件管理器** (`src/utils/file_utils.py`)：文件上传和验证

## 2. 通信协议规范

### 2.1 WebSocket协议

**服务器地址**: `ws://host:port` (默认: `ws://localhost:8767`)

**连接配置**:
```javascript
{
  max_size: 5000 * 1024 * 1024,  // 最大消息5GB
  max_queue: 100,                 // 最大连接数
  ping_interval: 60,              // 心跳间隔60秒
  ping_timeout: 120,              // 心跳超时120秒
  read_limit: 1MB,                // 读缓冲区
  write_limit: 1MB,               // 写缓冲区
  compression: null               // 禁用压缩
}
```

### 2.2 消息格式

**基础消息结构**:
```json
{
  "type": "消息类型",
  "data": {
    // 消息数据
  },
  "timestamp": "ISO时间戳"
}
```

### 2.3 消息类型详解

#### 2.3.1 连接阶段

**服务器欢迎消息**:
```json
{
  "type": "connected",
  "data": {
    "connection_id": "连接ID",
    "message": "连接成功",
    "server_time": "2024-01-01T12:00:00Z"
  }
}
```

**客户端心跳**:
```json
{
  "type": "ping",
  "data": {}
}
```

**服务器心跳响应**:
```json
{
  "type": "pong",
  "data": {
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

#### 2.3.2 认证阶段 (可选)

**认证要求通知**:
```json
{
  "type": "auth_required",
  "data": {
    "message": "请提供认证令牌"
  }
}
```

**客户端认证**:
```json
{
  "type": "auth",
  "data": {
    "token": "JWT-TOKEN"
  }
}
```

**认证成功**:
```json
{
  "type": "auth_success",
  "data": {
    "message": "认证成功",
    "user": {
      "username": "用户名"
    }
  }
}
```

#### 2.3.3 文件上传阶段

**上传请求** (支持单文件和分片两种模式):
```json
{
  "type": "upload_request",
  "data": {
    "file_name": "test.mp3",
    "file_size": 1024000,
    "file_hash": "md5-hash",
    "force_refresh": false,
    "output_format": "json",  // "json" 或 "srt"
    "upload_mode": "single"   // "single" 或 "chunked"
  }
}
```

**分片上传请求**:
```json
{
  "type": "upload_request",
  "data": {
    "file_name": "large_file.mp3",
    "file_size": 50000000,
    "file_hash": "md5-hash",
    "upload_mode": "chunked",
    "chunk_size": 1048576,    // 1MB分片
    "total_chunks": 48,
    "output_format": "json"
  }
}
```

**上传就绪响应**:
```json
{
  "type": "upload_ready",
  "data": {
    "task_id": "任务ID",
    "message": "准备接收文件数据"
  }
}
```

#### 2.3.4 文件数据传输

**单文件上传数据**:
```json
{
  "type": "upload_data",
  "data": {
    "task_id": "任务ID",
    "file_data": "base64编码的文件数据"
  }
}
```

**分片数据上传**:
```json
{
  "type": "upload_chunk",
  "data": {
    "task_id": "任务ID",
    "chunk_index": 0,
    "chunk_data": "base64编码的分片数据",
    "chunk_hash": "分片MD5哈希"
  }
}
```

**分片接收确认**:
```json
{
  "type": "chunk_received",
  "data": {
    "task_id": "任务ID",
    "chunk_index": 0,
    "progress": 10.5,
    "status": "received"
  }
}
```

**上传完成确认**:
```json
{
  "type": "upload_complete",
  "data": {
    "task_id": "任务ID",
    "message": "文件上传成功，开始处理"
  }
}
```

#### 2.3.5 任务处理阶段

**排队通知** (队列满时):
```json
{
  "type": "task_queued",
  "data": {
    "task_id": "任务ID",
    "queue_position": 3,
    "estimated_wait_minutes": 6,
    "message": "任务排队中"
  }
}
```

**处理进度**:
```json
{
  "type": "task_progress",
  "data": {
    "task_id": "任务ID",
    "progress": 45.5,
    "status": "processing",
    "message": "转录进度: 45.5%",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

#### 2.3.6 结果返回阶段

**JSON格式完成**:
```json
{
  "type": "task_complete",
  "data": {
    "task_id": "任务ID",
    "result": {
      "task_id": "任务ID",
      "file_name": "test.mp3",
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
      "created_at": "2024-01-01T12:00:00Z",
      "processing_time": 1.03,
      "error": null
    },
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

**SRT格式完成**:
```json
{
  "type": "task_complete",
  "data": {
    "task_id": "任务ID",
    "result": {
      "format": "srt",
      "content": "1\\n00:00:00,880 --> 00:00:05,195\\nSpeaker1:欢迎大家...",
      "file_name": "test.mp3",
      "file_hash": "md5-hash"
    },
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

#### 2.3.7 错误处理

**错误响应**:
```json
{
  "type": "error",
  "data": {
    "error": "错误类型",
    "message": "错误描述",
    "details": {
      // 可选的详细信息
    }
  }
}
```

**常见错误类型**:
- `auth_failed`: 认证失败
- `invalid_json`: 无效JSON格式
- `invalid_file_type`: 不支持的文件类型
- `file_too_large`: 文件过大
- `hash_mismatch`: 文件哈希不匹配
- `upload_error`: 上传错误
- `task_not_found`: 任务不存在
- `queue_full`: 任务队列已满

#### 2.3.8 任务控制

**查询任务状态**:
```json
{
  "type": "task_status",
  "data": {
    "task_id": "任务ID"
  }
}
```

**取消任务**:
```json
{
  "type": "cancel_task",
  "data": {
    "task_id": "任务ID"
  }
}
```

**任务取消确认**:
```json
{
  "type": "task_cancelled",
  "data": {
    "task_id": "任务ID",
    "message": "任务已取消"
  }
}
```

## 3. 数据模型规范

### 3.1 任务状态枚举
```python
class TaskStatus(str, Enum):
    PENDING = "pending"        # 等待中
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消
```

### 3.2 转录片段模型
```python
{
  "start_time": float,    # 开始时间(秒)
  "end_time": float,      # 结束时间(秒)
  "text": str,            # 转录文本
  "speaker": str          # 说话人标识
}
```

### 3.3 转录结果模型
```python
{
  "task_id": str,                    # 任务ID
  "file_name": str,                  # 文件名
  "file_hash": str,                  # 文件哈希
  "duration": float,                 # 音频时长(秒)
  "segments": [TranscriptionSegment], # 转录片段列表
  "speakers": [str],                 # 说话人列表
  "created_at": str,                 # 创建时间(ISO)
  "processing_time": float,          # 处理时长(秒)
  "error": str | null                # 错误信息
}
```

## 4. 配置规范

### 4.1 服务器配置
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8767,
    "max_connections": 100,
    "max_file_size_mb": 5000,
    "allowed_extensions": [".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg", ".opus"],
    "temp_dir": "./temp",
    "upload_dir": "./uploads"
  }
}
```

### 4.2 转录配置
```json
{
  "transcription": {
    "max_concurrent_tasks": 3,
    "concurrency_mode": "pool",
    "task_timeout_minutes": 30,
    "retry_times": 2,
    "cache_enabled": true,
    "delete_after_transcription": true,
    "transcription_speed_ratio": 10,
    "max_queue_size": 100,
    "queue_status_enabled": true
  }
}
```

### 4.3 认证配置
```json
{
  "auth": {
    "enabled": false,
    "secret_key": "your-secret-key",
    "algorithm": "HS256",
    "access_token_expire_minutes": 1440
  }
}
```

## 5. 核心功能实现要求

### 5.1 文件上传机制

#### 单文件上传 (<5MB)
1. 客户端发送`upload_request`
2. 服务器检查缓存，如有缓存直接返回结果
3. 服务器返回`upload_ready`
4. 客户端发送`upload_data`(Base64编码)
5. 服务器验证文件大小和哈希
6. 服务器返回`upload_complete`

#### 分片上传 (≥5MB)
1. 客户端发送分片`upload_request` (`upload_mode: "chunked"`)
2. 服务器创建上传会话，返回`upload_ready`
3. 客户端发送多个`upload_chunk`消息
4. 服务器确认每个分片：`chunk_received`
5. 所有分片上传完成后验证完整性
6. 服务器返回`upload_complete`

#### 关键特性
- **文件哈希验证**: MD5哈希验证文件完整性
- **分片去重**: 支持分片重传，避免重复上传
- **缓存机制**: 基于文件哈希的智能缓存
- **格式转换**: 同一文件支持JSON/SRT格式切换

### 5.2 并发处理机制

#### 任务队列设计
```python
class TaskManager:
    def __init__(self):
        self.task_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_tasks = 0
        self.max_concurrent_tasks = 3  # 可配置
```

#### 排队状态通知
- 当并发数超过限制时，新任务进入排队
- 实时计算排队位置和预估等待时间
- 发送`task_queued`消息通知客户端

#### 任务调度策略
- FIFO队列处理
- 动态超时算法避免任务卡住
- 智能重试机制处理临时错误
- 资源清理保证系统稳定性

### 5.3 缓存系统

#### SQLite数据库设计
```sql
CREATE TABLE transcription_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    result TEXT NOT NULL,          -- JSON格式结果
    raw_result TEXT,               -- 原始FunASR结果
    duration REAL,
    processing_time REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 缓存策略
- **智能缓存**: 基于MD5哈希自动缓存
- **格式灵活**: 支持JSON/SRT格式互转
- **过期清理**: 定期清理超过30天的缓存
- **访问更新**: 更新最后访问时间

### 5.4 错误处理机制

#### 错误分类
```python
# 不可重试错误
non_retryable_errors = [
    "音频时长过短",
    "文件不存在",
    "不支持的文件格式",
    "文件太大",
    "认证失败"
]

# 模型相关错误(可重试)
model_errors = [
    "VAD algorithm",
    "index .* out of bounds",
    "list index out of range"
]
```

#### 重试机制
- 智能重试: 根据错误类型决定是否重试
- 重试次数限制: 最多重试2次
- 模型重置: 遇到模型错误时重置模型状态
- 详细错误分类: 为客户端提供精确错误类型

## 6. 性能优化要点

### 6.1 内存管理
- 分片上传减少内存占用
- 及时清理临时文件
- 流式处理避免大文件内存溢出

### 6.2 并发优化
- 异步I/O处理
- 连接池管理
- 任务队列限制防止过载

### 6.3 网络优化
- 禁用WebSocket压缩减少CPU负载
- 增加缓冲区大小支持大文件传输
- 心跳机制保持长连接稳定

## 7. 系统监控与通知

### 7.1 企业微信通知
```python
# 任务完成/失败通知
await send_wework_notification(task, "completed")
await send_wework_notification(task, "failed")

# 服务器启动/停止通知
await send_custom_notification("🚀 服务器已启动", details)
await send_custom_notification("🛑 服务器已停止", details)
```

### 7.2 系统统计
```python
# 任务统计
{
    "total_tasks": int,
    "pending_tasks": int,
    "processing_tasks": int,
    "completed_tasks": int,
    "failed_tasks": int,
    "queue_size": int
}

# 缓存统计
{
    "total_count": int,
    "today_count": int,
    "cache_size_mb": float
}
```

## 8. 兼容性检查清单

### 8.1 协议兼容性
- [ ] WebSocket连接参数一致
- [ ] 消息格式完全匹配
- [ ] 错误代码对应正确
- [ ] 认证流程兼容

### 8.2 功能兼容性
- [ ] 支持单文件和分片上传
- [ ] JSON/SRT格式输出
- [ ] 说话人识别和合并
- [ ] 缓存机制工作正常
- [ ] 排队状态通知

### 8.3 性能兼容性
- [ ] 并发处理能力
- [ ] 文件大小限制
- [ ] 响应时间要求
- [ ] 内存使用控制

### 8.4 API兼容性
- [ ] 所有消息类型支持
- [ ] 数据模型结构一致
- [ ] 时间戳格式统一
- [ ] 编码格式正确

## 9. 开发建议

### 9.1 技术栈选择
- **异步框架**: 推荐使用asyncio/uvloop
- **WebSocket库**: websockets或类似库
- **数据库**: SQLite用于缓存(可扩展到PostgreSQL)
- **消息序列化**: JSON
- **文件处理**: 支持FFmpeg音频转换

### 9.2 关键实现点
1. **严格遵循消息格式**: 确保与现有客户端完全兼容
2. **实现完整的错误处理**: 包括所有错误类型和重试机制
3. **性能优化**: 特别是大文件处理和并发管理
4. **测试覆盖**: 使用提供的测试客户端验证兼容性

### 9.3 测试验证
项目包含完整的测试套件:
- `tests/server/test_server_transcription.py`: 基础功能测试
- `tests/server/test_concurrent_transcription.py`: 并发能力测试

使用这些测试确保新服务器与现有客户端完全兼容。

## 10. 总结

本文档详细分析了FunASR音频转文本服务器的完整技术规范。新开发的服务器必须严格遵循：

1. **协议兼容**: WebSocket消息格式和交互流程
2. **功能完整**: 支持所有现有功能特性
3. **性能对等**: 保持相同的并发处理能力
4. **错误处理**: 实现完整的错误分类和重试机制

按照本文档开发的新服务器将与现有客户端完全兼容，无需修改客户端代码即可实现无缝迁移。