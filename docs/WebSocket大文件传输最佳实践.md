# WebSocket 大文件传输最佳实践

## 概述

在使用 WebSocket 进行大文件传输时，不当的实现会导致连接超时、内存溢出、传输失败等问题。本文档基于 CapsWriter-Offline 项目的实践经验，总结了服务端和客户端的最佳实践方案。

## 核心问题分析

### 常见问题

1. **WebSocket keepalive 超时**
   ```
   received 1011 (internal error) keepalive ping timeout
   ```

2. **内存溢出**
   - 大文件一次性加载到内存
   - Base64 编码导致内存占用翻倍

3. **传输中断**
   - 网络波动导致连接断开
   - 长时间传输无法恢复

4. **用户体验差**
   - 无进度反馈
   - 传输失败无重试机制

### 问题根因

- **单消息过大**：WebSocket 在发送大消息时会阻塞连接
- **传输时间过长**：超出 keepalive 超时时间
- **缺乏流控制**：没有分片、重试、进度反馈机制

## 客户端最佳实践

### 1. 连接配置优化

```python
import asyncio
import websockets
import hashlib
import base64
from pathlib import Path

class WebSocketClient:
    def __init__(self, server_url="ws://localhost:8767"):
        self.server_url = server_url
        self.websocket = None
        
    async def connect(self):
        """建立 WebSocket 连接"""
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                # 关键配置：适当延长 keepalive 参数
                ping_interval=60,           # 60秒发送一次心跳
                ping_timeout=120,           # 心跳响应超时120秒  
                close_timeout=60,           # 关闭连接超时60秒
                max_size=10 * 1024 * 1024,  # 单消息最大10MB
                # 增加读写缓冲区
                read_limit=2**20,           # 1MB读缓冲
                write_limit=2**20           # 1MB写缓冲
            )
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
```

### 2. 分片上传策略

```python
async def upload_file_chunked(self, file_path: Path, chunk_size=1024*1024):
    """分片上传大文件"""
    
    # 1. 计算文件信息
    file_size = file_path.stat().st_size
    file_hash = self._calculate_file_hash(file_path)
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    
    print(f"文件大小: {file_size/1024/1024:.2f}MB")
    print(f"分片大小: {chunk_size/1024/1024:.2f}MB") 
    print(f"总分片数: {total_chunks}")
    
    # 2. 发送上传请求
    upload_request = {
        "type": "upload_request",
        "data": {
            "file_name": file_path.name,
            "file_size": file_size,
            "file_hash": file_hash,
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "upload_mode": "chunked"  # 标识分片上传
        }
    }
    
    await self._send_message(upload_request)
    response = await self._receive_message()
    
    if response["type"] != "upload_ready":
        raise Exception(f"上传准备失败: {response}")
    
    task_id = response["data"]["task_id"]
    
    # 3. 分片读取和上传
    with open(file_path, 'rb') as f:
        for chunk_index in range(total_chunks):
            # 读取分片数据
            chunk_data = f.read(chunk_size)
            chunk_hash = hashlib.md5(chunk_data).hexdigest()
            
            # 发送分片
            chunk_message = {
                "type": "upload_chunk",
                "data": {
                    "task_id": task_id,
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_data),
                    "chunk_hash": chunk_hash,
                    "chunk_data": base64.b64encode(chunk_data).decode(),
                    "is_last": chunk_index == total_chunks - 1
                }
            }
            
            await self._send_message(chunk_message)
            
            # 等待分片确认
            chunk_response = await self._receive_message(timeout=30)
            if chunk_response["type"] != "chunk_received":
                raise Exception(f"分片 {chunk_index} 上传失败")
            
            # 显示进度
            progress = (chunk_index + 1) / total_chunks * 100
            print(f"上传进度: {progress:.1f}% ({chunk_index + 1}/{total_chunks})")
    
    print("文件上传完成")
    return task_id

def _calculate_file_hash(self, file_path: Path, chunk_size=8192):
    """高效计算文件哈希"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块计算哈希，避免大文件内存占用
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
```

### 3. 连接保活和重试

```python
async def _send_message_with_retry(self, message, max_retry=3):
    """带重试的消息发送"""
    for attempt in range(max_retry):
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except websockets.exceptions.ConnectionClosed:
            print(f"连接断开，尝试重连... (第{attempt+1}次)")
            if await self._reconnect():
                continue
            else:
                return False
        except Exception as e:
            print(f"发送失败: {e}")
            if attempt < max_retry - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
            else:
                return False
    return False

async def _reconnect(self):
    """重新连接"""
    try:
        if self.websocket:
            await self.websocket.close()
        return await self.connect()
    except Exception:
        return False

async def _keep_alive(self):
    """保持连接活跃"""
    while True:
        try:
            # 定期发送心跳
            await asyncio.sleep(30)  # 30秒间隔
            if self.websocket and not self.websocket.closed:
                pong = await self.websocket.ping()
                await asyncio.wait_for(pong, timeout=10)
        except Exception as e:
            print(f"心跳失败: {e}")
            break
```

### 4. 流式传输（音频文件专用）

```python
async def upload_audio_stream(self, audio_path: Path):
    """音频文件流式上传（参考 CapsWriter-Offline 实现）"""
    
    # 使用 FFmpeg 转换音频格式
    import subprocess
    ffmpeg_cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-f", "f32le",      # 32位浮点格式
        "-ac", "1",         # 单声道
        "-ar", "16000",     # 16kHz采样率
        "-"
    ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data = process.stdout.read()
    
    # 按时间分段传输（每60秒为一段）
    task_id = str(uuid.uuid4())
    segment_duration = 60  # 60秒
    samples_per_segment = 16000 * segment_duration * 4  # 4字节/样本
    
    offset = 0
    segment_index = 0
    
    while offset < len(audio_data):
        # 计算当前分段
        segment_end = min(offset + samples_per_segment, len(audio_data))
        segment_data = audio_data[offset:segment_end]
        is_final = segment_end >= len(audio_data)
        
        # 发送分段
        message = {
            "type": "audio_segment",
            "data": {
                "task_id": task_id,
                "segment_index": segment_index,
                "segment_duration": len(segment_data) / 4 / 16000,  # 实际时长
                "is_final": is_final,
                "audio_data": base64.b64encode(segment_data).decode()
            }
        }
        
        await self._send_message(message)
        print(f"发送音频分段 {segment_index}, 时长: {len(segment_data)/4/16000:.2f}s")
        
        offset = segment_end
        segment_index += 1
    
    return task_id
```

## 服务端最佳实践

### 1. 连接管理

```python
import asyncio
import websockets
from typing import Dict, Set
import tempfile
import os

class WebSocketServer:
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.upload_sessions: Dict[str, dict] = {}  # 上传会话管理
        
    async def handle_connection(self, websocket, path):
        """处理客户端连接"""
        conn_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.connections[conn_id] = websocket
        
        try:
            print(f"客户端连接: {conn_id}")
            
            # 发送欢迎消息
            await self._send_message(websocket, "connected", {
                "connection_id": conn_id,
                "message": "连接成功"
            })
            
            # 处理消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, conn_id, data)
                except Exception as e:
                    await self._send_error(websocket, "message_error", str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"客户端断开: {conn_id}")
        finally:
            self._cleanup_connection(conn_id)
    
    def start_server(self, host="0.0.0.0", port=8767):
        """启动服务器"""
        return websockets.serve(
            self.handle_connection,
            host, port,
            # 服务端 keepalive 配置
            ping_interval=60,
            ping_timeout=120,
            max_size=10 * 1024 * 1024,  # 10MB
            # 增加连接限制
            max_queue=100,
            compression=None  # 禁用压缩以减少CPU负载
        )
```

### 2. 分片接收处理

```python
async def _handle_chunked_upload(self, websocket, conn_id: str, data: dict):
    """处理分片上传"""
    msg_type = data["type"]
    
    if msg_type == "upload_request":
        # 初始化上传会话
        task_id = str(uuid.uuid4())
        upload_mode = data["data"].get("upload_mode", "single")
        
        if upload_mode == "chunked":
            # 创建临时文件存储分片
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            
            session = {
                "task_id": task_id,
                "file_name": data["data"]["file_name"],
                "file_size": data["data"]["file_size"],
                "file_hash": data["data"]["file_hash"],
                "chunk_size": data["data"]["chunk_size"],
                "total_chunks": data["data"]["total_chunks"],
                "received_chunks": 0,
                "temp_file_path": temp_file.name,
                "temp_file": temp_file,
                "chunks_received": set()  # 记录已收到的分片
            }
            
            self.upload_sessions[task_id] = session
            temp_file.close()
            
            await self._send_message(websocket, "upload_ready", {
                "task_id": task_id,
                "message": "准备接收分片数据"
            })
        
    elif msg_type == "upload_chunk":
        # 处理分片数据
        await self._handle_chunk_data(websocket, data["data"])

async def _handle_chunk_data(self, websocket, data: dict):
    """处理单个分片数据"""
    task_id = data["task_id"]
    chunk_index = data["chunk_index"]
    
    if task_id not in self.upload_sessions:
        await self._send_error(websocket, "session_not_found", "上传会话不存在")
        return
    
    session = self.upload_sessions[task_id]
    
    # 检查分片是否重复
    if chunk_index in session["chunks_received"]:
        await self._send_message(websocket, "chunk_received", {
            "task_id": task_id,
            "chunk_index": chunk_index,
            "status": "duplicate"
        })
        return
    
    try:
        # 解码和验证分片数据
        chunk_data = base64.b64decode(data["chunk_data"])
        chunk_hash = hashlib.md5(chunk_data).hexdigest()
        
        if chunk_hash != data["chunk_hash"]:
            await self._send_error(websocket, "chunk_hash_mismatch", 
                                 f"分片 {chunk_index} 哈希校验失败")
            return
        
        # 写入分片数据到临时文件
        with open(session["temp_file_path"], "r+b") as f:
            f.seek(chunk_index * session["chunk_size"])
            f.write(chunk_data)
        
        # 更新会话状态
        session["chunks_received"].add(chunk_index)
        session["received_chunks"] += 1
        
        # 发送分片接收确认
        await self._send_message(websocket, "chunk_received", {
            "task_id": task_id,
            "chunk_index": chunk_index,
            "progress": session["received_chunks"] / session["total_chunks"] * 100
        })
        
        # 检查是否所有分片都已接收
        if session["received_chunks"] == session["total_chunks"]:
            await self._finalize_upload(websocket, task_id)
            
    except Exception as e:
        await self._send_error(websocket, "chunk_process_error", str(e))

async def _finalize_upload(self, websocket, task_id: str):
    """完成文件上传"""
    session = self.upload_sessions[task_id]
    
    try:
        # 验证完整文件哈希
        file_hash = self._calculate_file_hash(session["temp_file_path"])
        if file_hash != session["file_hash"]:
            await self._send_error(websocket, "file_hash_mismatch", "文件完整性校验失败")
            return
        
        # 移动文件到最终位置
        final_path = os.path.join("uploads", f"{task_id}_{session['file_name']}")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        os.rename(session["temp_file_path"], final_path)
        
        # 发送上传完成消息
        await self._send_message(websocket, "upload_complete", {
            "task_id": task_id,
            "message": "文件上传成功",
            "file_path": final_path
        })
        
        # 开始处理任务（转录等）
        asyncio.create_task(self._process_file(websocket, task_id, final_path))
        
    except Exception as e:
        await self._send_error(websocket, "finalize_error", str(e))
    finally:
        # 清理会话
        if task_id in self.upload_sessions:
            del self.upload_sessions[task_id]
```

### 3. 内存和资源管理

```python
import psutil
import gc

class ResourceManager:
    def __init__(self, max_memory_usage=0.8):
        self.max_memory_usage = max_memory_usage  # 最大内存使用率
        
    def check_memory_usage(self):
        """检查内存使用情况"""
        memory = psutil.virtual_memory()
        return memory.percent / 100.0
    
    async def _process_with_memory_control(self, file_path: str):
        """带内存控制的文件处理"""
        try:
            # 检查内存使用
            if self.check_memory_usage() > self.max_memory_usage:
                print("内存使用率过高，等待释放...")
                await asyncio.sleep(5)
                gc.collect()  # 强制垃圾回收
            
            # 处理文件
            result = await self._actual_process_file(file_path)
            return result
            
        finally:
            # 清理临时资源
            gc.collect()

async def _cleanup_temp_files(self):
    """定期清理临时文件"""
    temp_dir = tempfile.gettempdir()
    current_time = time.time()
    
    for filename in os.listdir(temp_dir):
        if filename.startswith("upload_"):
            file_path = os.path.join(temp_dir, filename)
            # 删除超过1小时的临时文件
            if current_time - os.path.getctime(file_path) > 3600:
                try:
                    os.remove(file_path)
                    print(f"清理临时文件: {filename}")
                except Exception:
                    pass
```

### 4. 并发控制

```python
import asyncio
from asyncio import Semaphore

class ConcurrencyController:
    def __init__(self, max_concurrent_uploads=5, max_concurrent_processing=3):
        self.upload_semaphore = Semaphore(max_concurrent_uploads)
        self.processing_semaphore = Semaphore(max_concurrent_processing)
        self.active_tasks = set()
    
    async def handle_upload(self, websocket, upload_data):
        """并发控制的上传处理"""
        async with self.upload_semaphore:
            task_id = await self._process_upload(websocket, upload_data)
            return task_id
    
    async def handle_processing(self, websocket, task_id, file_path):
        """并发控制的文件处理"""
        async with self.processing_semaphore:
            try:
                self.active_tasks.add(task_id)
                result = await self._process_file(websocket, task_id, file_path)
                return result
            finally:
                self.active_tasks.discard(task_id)
```

## 完整示例

### 客户端完整示例

```python
# client_example.py
import asyncio
import json
import base64
import hashlib
from pathlib import Path
import websockets

class LargeFileClient:
    def __init__(self, server_url="ws://localhost:8767"):
        self.server_url = server_url
        self.websocket = None
    
    async def connect(self):
        self.websocket = await websockets.connect(
            self.server_url,
            ping_interval=60,
            ping_timeout=120,
            max_size=10 * 1024 * 1024
        )
    
    async def upload_file(self, file_path: Path):
        """上传大文件"""
        try:
            await self.connect()
            
            # 分片上传
            if file_path.stat().st_size > 5 * 1024 * 1024:  # >5MB使用分片
                return await self.upload_file_chunked(file_path)
            else:
                return await self.upload_file_single(file_path)
                
        finally:
            if self.websocket:
                await self.websocket.close()
    
    # ... (其他方法的实现)

# 使用示例
async def main():
    client = LargeFileClient()
    file_path = Path("large_audio.mp3")
    
    task_id = await client.upload_file(file_path)
    print(f"上传完成，任务ID: {task_id}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 服务端完整示例

```python
# server_example.py
import asyncio
import websockets
import json

class LargeFileServer:
    def __init__(self):
        self.connections = {}
        self.upload_sessions = {}
    
    async def handle_connection(self, websocket, path):
        # ... (连接处理实现)
        pass
    
    def start(self, host="0.0.0.0", port=8767):
        return websockets.serve(
            self.handle_connection,
            host, port,
            ping_interval=60,
            ping_timeout=120,
            max_size=10 * 1024 * 1024
        )

# 启动示例
async def main():
    server = LargeFileServer()
    async with server.start():
        print("服务器启动在 ws://localhost:8767")
        await asyncio.Future()  # 持续运行

if __name__ == "__main__":
    asyncio.run(main())
```

## 性能优化建议

### 1. 传输优化
- **分片大小**：建议 1-5MB，平衡传输效率和内存占用
- **并发上传**：同时上传多个分片（需要服务端支持）
- **压缩传输**：对文本数据启用 gzip 压缩

### 2. 内存优化
- **流式处理**：避免大文件完全加载到内存
- **及时释放**：处理完分片后立即释放内存
- **内存监控**：实时监控内存使用率

### 3. 网络优化
- **重试机制**：指数退避重试策略
- **心跳优化**：根据网络质量调整心跳间隔
- **缓冲区调优**：增大 TCP 缓冲区大小

### 4. 错误处理
- **分片校验**：每个分片进行哈希校验
- **断点续传**：支持上传中断后继续传输
- **超时处理**：合理设置各种超时时间

## 监控和调试

### 1. 关键指标
- 连接数量和存活时间
- 分片传输成功率
- 内存使用情况
- 传输速度和延迟

### 2. 日志记录
```python
import logging

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('websocket_transfer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 在关键点添加日志
logger.info(f"开始上传文件: {file_name}, 大小: {file_size}")
logger.debug(f"发送分片 {chunk_index}/{total_chunks}")
logger.warning(f"重试连接: 第 {retry_count} 次")
logger.error(f"上传失败: {error_message}")
```

### 3. 性能分析
```python
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
    
    def report(self, message=""):
        elapsed = time.time() - self.start_time
        current_memory = psutil.virtual_memory().used
        memory_diff = current_memory - self.start_memory
        
        print(f"[性能] {message}")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  内存变化: {memory_diff/1024/1024:.2f}MB")
```

## 总结

WebSocket 大文件传输的关键在于：

1. **合理分片**：避免单个消息过大导致的阻塞
2. **连接保活**：正确配置 keepalive 参数
3. **资源管理**：控制内存使用和并发数量
4. **错误处理**：完善的重试和恢复机制
5. **性能监控**：实时监控传输状态和系统资源

通过遵循这些最佳实践，可以实现稳定、高效的 WebSocket 大文件传输系统。