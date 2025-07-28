"""
文件上传处理模块
"""
import os
import asyncio
import hashlib
from typing import Optional
from loguru import logger
import websockets
from src.core.config import config
from src.utils.file_utils import save_uploaded_file, calculate_file_hash
from src.models.schemas import FileUploadRequest, TranscriptionTask, TaskStatus


class FileUploadHandler:
    """文件上传处理器"""
    
    def __init__(self):
        self.upload_sessions = {}  # task_id -> upload_data
    
    async def handle_file_upload(
        self,
        websocket: websockets.WebSocketServerProtocol,
        task_id: str,
        file_data: bytes,
        chunk_index: int,
        total_chunks: int,
        file_info: FileUploadRequest
    ) -> Optional[str]:
        """处理文件上传"""
        try:
            # 初始化上传会话
            if task_id not in self.upload_sessions:
                self.upload_sessions[task_id] = {
                    "chunks": {},
                    "file_info": file_info,
                    "total_chunks": total_chunks,
                    "received_chunks": 0
                }
            
            session = self.upload_sessions[task_id]
            
            # 存储数据块
            session["chunks"][chunk_index] = file_data
            session["received_chunks"] += 1
            
            # 计算进度
            progress = (session["received_chunks"] / total_chunks) * 100
            
            # 发送进度更新
            await self._send_upload_progress(websocket, task_id, progress)
            
            # 检查是否所有块都已接收
            if session["received_chunks"] == total_chunks:
                # 合并所有数据块
                file_data = b""
                for i in range(total_chunks):
                    if i not in session["chunks"]:
                        raise Exception(f"缺少数据块 {i}")
                    file_data += session["chunks"][i]
                
                # 验证文件大小
                if len(file_data) != file_info.file_size:
                    raise Exception(f"文件大小不匹配: 期望{file_info.file_size}, 实际{len(file_data)}")
                
                # 验证文件哈希
                actual_hash = hashlib.md5(file_data).hexdigest()
                if actual_hash != file_info.file_hash:
                    raise Exception(f"文件哈希不匹配")
                
                # 保存文件
                file_path, _ = await save_uploaded_file(file_data, file_info.file_name)
                
                # 清理上传会话
                del self.upload_sessions[task_id]
                
                logger.info(f"文件上传完成: {file_info.file_name} -> {file_path}")
                return file_path
            
            return None
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            # 清理上传会话
            if task_id in self.upload_sessions:
                del self.upload_sessions[task_id]
            raise
    
    async def handle_direct_upload(
        self,
        websocket: websockets.WebSocketServerProtocol,
        task_id: str,
        message_data: dict
    ) -> Optional[str]:
        """处理直接上传（单次传输）"""
        try:
            # 获取文件数据
            file_data_base64 = message_data.get("file_data")
            if not file_data_base64:
                raise Exception("缺少文件数据")
            
            # 解码base64数据
            import base64
            file_data = base64.b64decode(file_data_base64)
            
            # 获取文件信息
            file_info = FileUploadRequest(
                file_name=message_data.get("file_name"),
                file_size=len(file_data),
                file_hash=hashlib.md5(file_data).hexdigest(),
                force_refresh=message_data.get("force_refresh", False)
            )
            
            # 验证文件大小
            from src.utils.file_utils import validate_file_size
            if not validate_file_size(file_info.file_size):
                raise Exception(f"文件太大，最大支持{config.server.max_file_size_mb}MB")
            
            # 保存文件
            file_path, file_hash = await save_uploaded_file(file_data, file_info.file_name)
            
            logger.info(f"直接上传完成: {file_info.file_name} -> {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"直接上传失败: {e}")
            raise
    
    async def _send_upload_progress(
        self,
        websocket: websockets.WebSocketServerProtocol,
        task_id: str,
        progress: float
    ):
        """发送上传进度"""
        try:
            from src.api.websocket_handler import ws_handler
            await ws_handler.notify_task_progress(
                task_id=task_id,
                progress=progress,
                status="uploading",
                message=f"上传进度: {progress:.1f}%"
            )
        except Exception as e:
            logger.error(f"发送上传进度失败: {e}")
    
    def cleanup_session(self, task_id: str):
        """清理上传会话"""
        if task_id in self.upload_sessions:
            del self.upload_sessions[task_id]
            logger.debug(f"清理上传会话: {task_id}")


# 全局文件上传处理器实例
file_upload_handler = FileUploadHandler()