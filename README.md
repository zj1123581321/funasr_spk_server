# FunASR Speaker Recognition Server

基于FunASR的音视频转录服务器，支持说话人识别和时间戳标注。

## 功能特性

- ✅ 支持多种音视频格式（wav, mp3, mp4, m4a, flac等）
- ✅ 自动说话人识别和分离
- ✅ 精确到秒的时间戳标注
- ✅ WebSocket实时通信
- ✅ 任务队列管理，支持并发处理
- ✅ 结果缓存，相同文件直接返回
- ✅ 企微机器人通知
- ✅ JWT认证机制
- ✅ 支持Docker部署

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

4. 配置服务器
编辑 `config.json` 文件，设置相关参数：
- 企微webhook地址
- 认证密钥
- 并发数等

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

## 使用方法

### 客户端测试

运行测试客户端：
```bash
python client_test.py
```

批量测试：
```bash
python client_test.py batch
```

### WebSocket API

1. 连接到服务器
```
ws://localhost:8765
```

2. 认证（如果启用）
```json
{
  "type": "auth",
  "data": {
    "token": "your-jwt-token"
  }
}
```

3. 上传文件请求
```json
{
  "type": "upload_request",
  "data": {
    "file_name": "test.mp3",
    "file_size": 1024000,
    "file_hash": "md5-hash",
    "force_refresh": false
  }
}
```

4. 上传文件数据
```json
{
  "type": "upload_data",
  "data": {
    "task_id": "task-id",
    "file_data": "base64-encoded-data"
  }
}
```

5. 查询任务状态
```json
{
  "type": "task_status",
  "data": {
    "task_id": "task-id"
  }
}
```

## 输出格式

```json
{
  "task_id": "uuid",
  "file_name": "test.mp3",
  "file_hash": "md5-hash",
  "duration": 120.5,
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.2,
      "text": "你好，欢迎使用FunASR",
      "speaker": "Speaker1"
    }
  ],
  "speakers": ["Speaker1", "Speaker2"],
  "processing_time": 15.3
}
```

## 配置说明

查看 `config.json` 了解所有配置选项：

- `server`: 服务器配置
- `funasr`: 模型配置
- `transcription`: 转录任务配置
- `database`: 缓存数据库配置
- `notification`: 企微通知配置
- `auth`: 认证配置
- `logging`: 日志配置

## 注意事项

1. 首次运行会自动下载模型文件（约2GB）
2. 建议使用16GB以上内存以获得最佳性能
3. 默认用户名/密码：admin/admin123
4. 生产环境请修改认证密钥

## 故障排除

1. 模型下载失败
   - 检查网络连接
   - 手动下载模型到models目录

2. 转录速度慢
   - 增加并发数
   - 使用更高性能的CPU

3. 内存不足
   - 减少并发数
   - 调整batch_size参数

## License

MIT License