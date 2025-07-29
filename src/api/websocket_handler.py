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


class WebSocketHandler:
    """WebSocket连接处理器"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.task_connections: Dict[str, Set[str]] = {}  # task_id -> connection_ids
        
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
            # 文件数据上传
            await self._handle_upload_data(websocket, connection_id, msg_data)
            
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
            
            # 创建任务
            from src.core.task_manager import task_manager
            task = await task_manager.create_task(request)
            
            # 关联任务和连接
            if task.task_id not in self.task_connections:
                self.task_connections[task.task_id] = set()
            self.task_connections[task.task_id].add(connection_id)
            
            # 检查缓存（如果不强制刷新）
            if not request.force_refresh:
                from src.core.database import db_manager
                cached_result = await db_manager.get_cached_result(request.file_hash)
                if cached_result:
                    logger.info(f"使用缓存结果（upload_request阶段）: {task.task_id}")
                    # 直接返回缓存结果
                    await self._send_message(websocket, "task_complete", {
                        "task_id": task.task_id,
                        "result": cached_result
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
            
            # 提交任务到队列
            await task_manager.submit_task(task_id, file_path)
            
            # 发送响应
            await self._send_message(websocket, "upload_complete", {
                "task_id": task_id,
                "message": "文件上传成功，开始处理"
            })
            
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
        
        data = {
            "task_id": task_id,
            "progress": progress,
            "status": status,
            "message": message,
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


# 全局WebSocket处理器实例
ws_handler = WebSocketHandler()