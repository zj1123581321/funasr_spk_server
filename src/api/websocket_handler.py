"""
WebSocket处理器
"""
import json
import asyncio
from typing import Dict, Set, Optional
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from loguru import logger
from src.models.schemas import WebSocketMessage, FileUploadRequest, TaskStatusResponse, ErrorResponse
from src.core.config import config
from src.utils.auth import verify_token
import base64
import hashlib
import tempfile
import os
import uuid


class WebSocketHandler:
    """WebSocket连接处理器"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.task_connections: Dict[str, Set[str]] = {}  # task_id -> connection_ids
        self.upload_sessions: Dict[str, dict] = {}  # 分片上传会话管理
        
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """处理WebSocket连接"""
        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}_{datetime.now().timestamp()}"
        
        try:
            # 认证
            if config.auth.enabled:
                auth_success = await self._authenticate(websocket)
                if not auth_success:
                    await self._send_error(websocket, "auth_failed", "认证失败")
                    return
            
            # 注册连接
            self.connections[connection_id] = websocket
            logger.info(f"WebSocket连接建立: {connection_id}")
            
            # 发送欢迎消息
            await self._send_message(websocket, "connected", {
                "connection_id": connection_id,
                "message": "连接成功",
                "server_time": datetime.now().isoformat()
            })
            
            # 处理消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, connection_id, data)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "invalid_json", "无效的JSON格式")
                except Exception as e:
                    logger.error(f"处理消息失败: {e}")
                    await self._send_error(websocket, "message_error", str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket连接关闭: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket处理错误: {e}")
        finally:
            # 清理连接
            self._cleanup_connection(connection_id)
    
    async def _authenticate(self, websocket: WebSocketServerProtocol) -> bool:
        """认证WebSocket连接"""
        try:
            # 等待认证消息
            await self._send_message(websocket, "auth_required", {
                "message": "请提供认证令牌"
            })
            
            # 设置超时
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            data = json.loads(auth_message)
            
            if data.get("type") != "auth":
                return False
            
            token = data.get("token")
            if not token:
                return False
            
            # 验证令牌
            from src.utils.auth import verify_token
            user_info = verify_token(token)
            if not user_info:
                return False
            
            await self._send_message(websocket, "auth_success", {
                "message": "认证成功",
                "user": user_info
            })
            return True
            
        except (asyncio.TimeoutError, json.JSONDecodeError):
            return False
        except Exception as e:
            logger.error(f"认证失败: {e}")
            return False
    
    async def _handle_message(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理接收到的消息"""
        msg_type = data.get("type")
        msg_data = data.get("data", {})
        
        if msg_type == "ping":
            # 心跳响应
            await self._send_message(websocket, "pong", {
                "timestamp": datetime.now().isoformat()
            })
            
        elif msg_type == "upload_request":
            # 文件上传请求
            await self._handle_upload_request(websocket, connection_id, msg_data)
            
        elif msg_type == "upload_data":
            # 文件数据上传（单文件模式）
            await self._handle_upload_data(websocket, connection_id, msg_data)
            
        elif msg_type == "upload_chunk":
            # 分片文件上传
            await self._handle_chunk_upload(websocket, connection_id, msg_data)
            
        elif msg_type == "task_status":
            # 查询任务状态
            task_id = msg_data.get("task_id")
            if task_id:
                await self._send_task_status(websocket, task_id)
            else:
                await self._send_error(websocket, "missing_task_id", "缺少task_id参数")
                
        elif msg_type == "cancel_task":
            # 取消任务
            task_id = msg_data.get("task_id")
            if task_id:
                await self._handle_cancel_task(websocket, task_id)
            else:
                await self._send_error(websocket, "missing_task_id", "缺少task_id参数")
                
        else:
            await self._send_error(websocket, "unknown_message_type", f"未知的消息类型: {msg_type}")
    
    async def _handle_upload_request(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理文件上传请求"""
        try:
            # 提取输出格式（默认为json）
            output_format = data.get("output_format", "json")
            if output_format not in ["json", "srt"]:
                await self._send_error(websocket, "invalid_format", 
                                     "不支持的输出格式，支持: json, srt")
                return
            
            # 验证请求数据
            request = FileUploadRequest(**data)
            
            # 检查文件大小
            from src.utils.file_utils import validate_file_size
            if not validate_file_size(request.file_size):
                await self._send_error(websocket, "file_too_large", 
                                     f"文件太大，最大支持{config.server.max_file_size_mb}MB")
                return
            
            # 检查文件类型
            from src.utils.file_utils import is_allowed_file
            if not is_allowed_file(request.file_name):
                await self._send_error(websocket, "invalid_file_type", 
                                     f"不支持的文件类型，支持: {', '.join(config.server.allowed_extensions)}")
                return
            
            # 检查是否为分片上传模式
            upload_mode = data.get("upload_mode", "single")
            
            if upload_mode == "chunked":
                # 处理分片上传请求
                await self._handle_chunked_upload_request(websocket, connection_id, data)
                return
            
            # 创建任务（单文件模式）
            from src.core.task_manager import task_manager
            task = await task_manager.create_task(request)
            
            # 关联任务和连接
            if task.task_id not in self.task_connections:
                self.task_connections[task.task_id] = set()
            self.task_connections[task.task_id].add(connection_id)
            
            # 检查缓存（如果不强制刷新）
            if not request.force_refresh:
                from src.core.database import db_manager
                cached_result = await db_manager.get_cached_result(request.file_hash, output_format)
                if cached_result:
                    logger.info(f"使用缓存结果（upload_request阶段）: {task.task_id}")
                    
                    # 根据输出格式准备结果
                    if output_format == "srt":
                        if isinstance(cached_result, dict) and cached_result.get("format") == "srt":
                            result_data = cached_result
                        else:
                            # 如果缓存中没有SRT格式，跳过缓存
                            logger.info("缓存中没有SRT格式，继续处理")
                            result_data = None
                    else:
                        # JSON格式
                        result_data = cached_result.dict() if cached_result else None
                    
                    if result_data:
                        # 直接返回缓存结果
                        await self._send_message(websocket, "task_complete", {
                            "task_id": task.task_id,
                            "result": result_data
                        })
                        return
            
            # 发送响应
            await self._send_message(websocket, "upload_ready", {
                "task_id": task.task_id,
                "message": "准备接收文件数据"
            })
            
        except Exception as e:
            logger.error(f"处理上传请求失败: {e}")
            await self._send_error(websocket, "upload_error", str(e))
    
    async def _handle_upload_data(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理文件数据上传"""
        try:
            task_id = data.get("task_id")
            if not task_id:
                await self._send_error(websocket, "missing_task_id", "缺少task_id参数")
                return
            
            # 获取任务
            from src.core.task_manager import task_manager
            task = task_manager.get_task(task_id)
            if not task:
                await self._send_error(websocket, "task_not_found", "任务不存在")
                return
            
            # 处理文件数据
            file_data_base64 = data.get("file_data")
            if not file_data_base64:
                await self._send_error(websocket, "missing_file_data", "缺少文件数据")
                return
            
            # 解码base64数据
            file_data = base64.b64decode(file_data_base64)
            
            # 验证文件大小
            if len(file_data) != task.file_size:
                await self._send_error(websocket, "size_mismatch", 
                                     f"文件大小不匹配: 期望{task.file_size}, 实际{len(file_data)}")
                return
            
            # 验证文件哈希
            actual_hash = hashlib.md5(file_data).hexdigest()
            if actual_hash != task.file_hash:
                await self._send_error(websocket, "hash_mismatch", "文件哈希不匹配")
                return
            
            # 保存文件
            from src.utils.file_utils import save_uploaded_file
            file_path, _ = await save_uploaded_file(file_data, task.file_name)
            
            # 提交任务到队列，获取排队信息
            queue_info = await task_manager.submit_task(task_id, file_path)
            
            # 发送响应，包含排队信息
            response_data = {
                "task_id": task_id,
                "message": "文件上传成功，开始处理"
            }
            
            # 如果启用了排队状态通知且任务在排队
            if config.transcription.queue_status_enabled and queue_info:
                if queue_info.get("queued", False):
                    response_data.update({
                        "queue_position": queue_info["position"],
                        "estimated_wait_minutes": queue_info["estimated_wait"],
                        "message": f"文件上传成功，排队位置: {queue_info['position']}"
                    })
                    # 发送排队状态
                    await self._send_message(websocket, "task_queued", response_data)
                    return
            
            await self._send_message(websocket, "upload_complete", response_data)
            
        except Exception as e:
            logger.error(f"处理文件数据失败: {e}")
            await self._send_error(websocket, "upload_data_error", str(e))
    
    async def _send_task_status(self, websocket: WebSocketServerProtocol, task_id: str):
        """发送任务状态"""
        try:
            from src.core.task_manager import task_manager
            task = task_manager.get_task(task_id)
            
            if not task:
                await self._send_error(websocket, "task_not_found", "任务不存在")
                return
            
            response = TaskStatusResponse(
                task_id=task.task_id,
                status=task.status,
                progress=task.progress,
                result=task.result,
                error=task.error
            )
            
            await self._send_message(websocket, "task_status", response.dict())
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            await self._send_error(websocket, "status_error", str(e))
    
    async def _handle_cancel_task(self, websocket: WebSocketServerProtocol, task_id: str):
        """处理取消任务请求"""
        try:
            from src.core.task_manager import task_manager
            success = await task_manager.cancel_task(task_id)
            
            if success:
                await self._send_message(websocket, "task_cancelled", {
                    "task_id": task_id,
                    "message": "任务已取消"
                })
            else:
                await self._send_error(websocket, "cancel_failed", "任务取消失败")
                
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            await self._send_error(websocket, "cancel_error", str(e))
    
    async def notify_task_progress(self, task_id: str, progress: float, status: str, message: str = None):
        """通知任务进度"""
        connection_ids = self.task_connections.get(task_id, set())
        
        # 对于失败状态，提供更详细的错误分类
        error_type = None
        if status == "failed" and message:
            if "VAD" in message:
                error_type = "vad_error"
            elif "索引" in message or "index" in message:
                error_type = "index_error"
            elif "音频" in message:
                error_type = "audio_error"
            elif "模型" in message:
                error_type = "model_error"
            else:
                error_type = "unknown_error"
        
        data = {
            "task_id": task_id,
            "progress": progress,
            "status": status,
            "message": message,
            "error_type": error_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # 向所有相关连接发送进度更新
        for conn_id in list(connection_ids):
            if conn_id in self.connections:
                websocket = self.connections[conn_id]
                try:
                    await self._send_message(websocket, "task_progress", data)
                except Exception as e:
                    logger.error(f"发送进度通知失败: {e}")
                    self._cleanup_connection(conn_id)
    
    async def notify_task_complete(self, task_id: str, result: dict):
        """通知任务完成"""
        connection_ids = self.task_connections.get(task_id, set())
        
        data = {
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # 向所有相关连接发送完成通知
        for conn_id in list(connection_ids):
            if conn_id in self.connections:
                websocket = self.connections[conn_id]
                try:
                    await self._send_message(websocket, "task_complete", data)
                except Exception as e:
                    logger.error(f"发送完成通知失败: {e}")
                    self._cleanup_connection(conn_id)
        
        # 清理任务连接关系
        if task_id in self.task_connections:
            del self.task_connections[task_id]
    
    async def _send_message(self, websocket: WebSocketServerProtocol, msg_type: str, data: dict):
        """发送消息"""
        message = WebSocketMessage(type=msg_type, data=data)
        await websocket.send(message.json())
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_type: str, message: str):
        """发送错误消息"""
        error = ErrorResponse(error=error_type, message=message)
        await self._send_message(websocket, "error", error.dict())
    
    def _cleanup_connection(self, connection_id: str):
        """清理连接"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        # 从任务连接中移除
        for task_id, conn_ids in list(self.task_connections.items()):
            if connection_id in conn_ids:
                conn_ids.remove(connection_id)
                if not conn_ids:
                    del self.task_connections[task_id]
        
        logger.debug(f"连接已清理: {connection_id}")
    
    async def _handle_chunked_upload_request(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理分片上传请求"""
        try:
            # 生成任务ID
            task_id = str(uuid.uuid4())
            
            # 创建临时文件存储分片
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            
            # 创建上传会话
            session = {
                "task_id": task_id,
                "file_name": data["file_name"],
                "file_size": data["file_size"],
                "file_hash": data["file_hash"],
                "chunk_size": data.get("chunk_size", 1024 * 1024),  # 默认1MB
                "total_chunks": data["total_chunks"],
                "received_chunks": 0,
                "temp_file_path": temp_file.name,
                "temp_file": temp_file,
                "chunks_received": set(),  # 记录已收到的分片索引
                "output_format": data.get("output_format", "json"),
                "force_refresh": data.get("force_refresh", False),
                "connection_id": connection_id
            }
            
            self.upload_sessions[task_id] = session
            temp_file.close()
            
            # 关联任务和连接
            if task_id not in self.task_connections:
                self.task_connections[task_id] = set()
            self.task_connections[task_id].add(connection_id)
            
            logger.info(f"创建分片上传会话: {task_id}, 文件: {session['file_name']}, "
                       f"总大小: {session['file_size']/1024/1024:.2f}MB, "
                       f"分片数: {session['total_chunks']}")
            
            await self._send_message(websocket, "upload_ready", {
                "task_id": task_id,
                "message": "准备接收分片数据",
                "chunk_size": session["chunk_size"],
                "total_chunks": session["total_chunks"]
            })
            
        except Exception as e:
            logger.error(f"处理分片上传请求失败: {e}")
            await self._send_error(websocket, "chunked_upload_error", str(e))
    
    async def _handle_chunk_upload(self, websocket: WebSocketServerProtocol, connection_id: str, data: dict):
        """处理分片数据上传"""
        try:
            task_id = data.get("task_id")
            chunk_index = data.get("chunk_index")
            
            if not task_id or chunk_index is None:
                await self._send_error(websocket, "missing_chunk_data", "缺少task_id或chunk_index")
                return
            
            if task_id not in self.upload_sessions:
                await self._send_error(websocket, "session_not_found", "上传会话不存在")
                return
            
            session = self.upload_sessions[task_id]
            
            # 检查分片是否重复
            if chunk_index in session["chunks_received"]:
                await self._send_message(websocket, "chunk_received", {
                    "task_id": task_id,
                    "chunk_index": chunk_index,
                    "status": "duplicate",
                    "progress": session["received_chunks"] / session["total_chunks"] * 100
                })
                return
            
            # 解码和验证分片数据
            chunk_data_base64 = data.get("chunk_data")
            if not chunk_data_base64:
                await self._send_error(websocket, "missing_chunk_data", "缺少分片数据")
                return
            
            chunk_data = base64.b64decode(chunk_data_base64)
            chunk_hash = hashlib.md5(chunk_data).hexdigest()
            
            # 验证分片哈希
            expected_hash = data.get("chunk_hash")
            if expected_hash and chunk_hash != expected_hash:
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
            
            progress = session["received_chunks"] / session["total_chunks"] * 100
            
            logger.debug(f"接收分片 {chunk_index}/{session['total_chunks']}, "
                        f"进度: {progress:.1f}%")
            
            # 发送分片接收确认
            await self._send_message(websocket, "chunk_received", {
                "task_id": task_id,
                "chunk_index": chunk_index,
                "progress": progress,
                "status": "received"
            })
            
            # 检查是否所有分片都已接收
            if session["received_chunks"] == session["total_chunks"]:
                await self._finalize_chunked_upload(websocket, task_id)
                
        except Exception as e:
            logger.error(f"处理分片上传失败: {e}")
            await self._send_error(websocket, "chunk_upload_error", str(e))
    
    async def _finalize_chunked_upload(self, websocket: WebSocketServerProtocol, task_id: str):
        """完成分片上传"""
        try:
            session = self.upload_sessions[task_id]
            
            logger.info(f"完成分片上传: {task_id}, 验证文件完整性...")
            
            # 验证完整文件哈希
            file_hash = self._calculate_file_hash(session["temp_file_path"])
            if file_hash != session["file_hash"]:
                await self._send_error(websocket, "file_hash_mismatch", "文件完整性校验失败")
                return
            
            # 检查缓存（如果不强制刷新）
            if not session["force_refresh"]:
                from src.core.database import db_manager
                cached_result = await db_manager.get_cached_result(
                    session["file_hash"], session["output_format"]
                )
                if cached_result:
                    logger.info(f"使用缓存结果（分片上传阶段）: {task_id}")
                    
                    # 根据输出格式准备结果
                    if session["output_format"] == "srt":
                        if isinstance(cached_result, dict) and cached_result.get("format") == "srt":
                            result_data = cached_result
                        else:
                            result_data = None
                    else:
                        result_data = cached_result.dict() if cached_result else None
                    
                    if result_data:
                        # 清理临时文件和会话
                        await self._cleanup_upload_session(task_id)
                        
                        # 直接返回缓存结果
                        await self._send_message(websocket, "task_complete", {
                            "task_id": task_id,
                            "result": result_data
                        })
                        return
            
            # 移动文件到最终位置
            from src.utils.file_utils import save_uploaded_file
            
            # 读取临时文件内容
            with open(session["temp_file_path"], "rb") as f:
                file_data = f.read()
            
            file_path, _ = await save_uploaded_file(file_data, session["file_name"])
            
            # 创建任务请求对象
            from src.models.schemas import FileUploadRequest
            request = FileUploadRequest(
                file_name=session["file_name"],
                file_size=session["file_size"],
                file_hash=session["file_hash"],
                force_refresh=session["force_refresh"],
                output_format=session["output_format"]
            )
            
            # 创建任务
            from src.core.task_manager import task_manager
            task = await task_manager.create_task(request, task_id=task_id)
            
            # 提交任务到队列
            queue_info = await task_manager.submit_task(task_id, file_path)
            
            # 清理上传会话
            await self._cleanup_upload_session(task_id)
            
            # 发送上传完成消息
            response_data = {
                "task_id": task_id,
                "message": "分片文件上传成功，开始处理"
            }
            
            # 如果启用了排队状态通知且任务在排队
            if config.transcription.queue_status_enabled and queue_info:
                if queue_info.get("queued", False):
                    response_data.update({
                        "queue_position": queue_info["position"],
                        "estimated_wait_minutes": queue_info["estimated_wait"],
                        "message": f"分片文件上传成功，排队位置: {queue_info['position']}"
                    })
                    await self._send_message(websocket, "task_queued", response_data)
                    return
            
            await self._send_message(websocket, "upload_complete", response_data)
            
            logger.info(f"分片上传完成: {task_id}, 文件: {session['file_name']}")
            
        except Exception as e:
            logger.error(f"完成分片上传失败: {e}")
            await self._send_error(websocket, "finalize_error", str(e))
    
    def _calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """高效计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # 分块计算哈希，避免大文件内存占用
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def _cleanup_upload_session(self, task_id: str):
        """清理上传会话"""
        if task_id in self.upload_sessions:
            session = self.upload_sessions[task_id]
            
            # 删除临时文件
            temp_file_path = session.get("temp_file_path")
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"清理临时文件: {temp_file_path}")
                except Exception as e:
                    logger.error(f"清理临时文件失败: {e}")
            
            # 删除会话
            del self.upload_sessions[task_id]
            logger.debug(f"清理上传会话: {task_id}")


# 全局WebSocket处理器实例
ws_handler = WebSocketHandler()