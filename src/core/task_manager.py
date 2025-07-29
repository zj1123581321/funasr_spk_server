"""
任务队列管理模块
"""
import asyncio
import uuid
from typing import Dict, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from src.core.config import config
from src.core.database import db_manager
from src.models.schemas import TranscriptionTask, TaskStatus, FileUploadRequest, TranscriptionResult


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, TranscriptionTask] = {}
        self.task_queue = asyncio.Queue()
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=config.transcription.max_concurrent_tasks)
        self.is_running = False
    
    async def start(self):
        """启动任务管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动工作线程
        for i in range(config.transcription.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        
        logger.info(f"任务管理器已启动，{config.transcription.max_concurrent_tasks}个工作线程")
    
    async def stop(self):
        """停止任务管理器"""
        self.is_running = False
        
        # 取消所有工作线程
        for worker in self.workers:
            worker.cancel()
        
        # 等待所有工作线程结束
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("任务管理器已停止")
    
    async def create_task(self, request: FileUploadRequest) -> TranscriptionTask:
        """创建新任务"""
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建任务对象
        task = TranscriptionTask(
            task_id=task_id,
            file_name=request.file_name,
            file_path="",  # 文件路径在上传完成后设置
            file_size=request.file_size,
            file_hash=request.file_hash,
            force_refresh=request.force_refresh,
            output_format=request.output_format
        )
        
        # 保存任务
        self.tasks[task_id] = task
        
        logger.info(f"创建任务: {task_id} - {request.file_name}")
        
        return task
    
    def get_task(self, task_id: str) -> Optional[TranscriptionTask]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    async def submit_task(self, task_id: str, file_path: str):
        """提交任务到队列"""
        task = self.get_task(task_id)
        if not task:
            raise Exception(f"任务不存在: {task_id}")
        
        # 更新文件路径
        task.file_path = file_path
        
        # 检查缓存
        if not task.force_refresh:
            cached_result = await db_manager.get_cached_result(task.file_hash, task.output_format)
            if cached_result:
                logger.info(f"使用缓存结果: {task_id}")
                task.status = TaskStatus.COMPLETED
                
                if task.output_format == "srt":
                    # SRT格式缓存结果
                    if isinstance(cached_result, dict) and cached_result.get("format") == "srt":
                        task.srt_content = cached_result["content"]
                        # 创建简化的结果对象
                        from src.models.schemas import TranscriptionResult
                        task.result = TranscriptionResult(
                            task_id=task_id,
                            file_name=task.file_name,
                            file_hash=task.file_hash,
                            duration=cached_result.get("duration", 0),
                            segments=[],
                            speakers=[],
                            processing_time=0,
                            error=None
                        )
                else:
                    # JSON格式缓存结果
                    task.result = cached_result
                
                task.completed_at = datetime.now()
                
                # 通知完成
                await self._notify_task_complete(task)
                return
        
        # 将任务加入队列
        await self.task_queue.put(task_id)
        logger.info(f"任务已加入队列: {task_id}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.get_task(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.error = "用户取消"
        task.completed_at = datetime.now()
        
        # 删除文件（如果配置了且没有其他任务使用）
        if config.transcription.delete_after_transcription and task.file_path:
            has_pending_tasks = any(
                t.file_hash == task.file_hash and t.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]
                for t in self.tasks.values()
                if t.task_id != task.task_id
            )
            
            if not has_pending_tasks:
                from src.utils.file_utils import delete_file
                await delete_file(task.file_path)
                logger.debug(f"任务取消，文件已删除: {task.file_path}")
        
        logger.info(f"任务已取消: {task_id}")
        return True
    
    async def _worker(self, worker_id: int):
        """工作线程"""
        logger.info(f"工作线程 {worker_id} 已启动")
        
        while self.is_running:
            try:
                # 获取任务（超时1秒）
                try:
                    task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # 处理任务
                await self._process_task(task_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"工作线程 {worker_id} 错误: {e}")
        
        logger.info(f"工作线程 {worker_id} 已停止")
    
    async def _process_task(self, task_id: str):
        """处理任务"""
        task = self.get_task(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return
        
        if task.status == TaskStatus.CANCELLED:
            logger.info(f"任务已取消，跳过处理: {task_id}")
            return
        
        try:
            # 更新任务状态
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            
            # 通知开始处理
            await self._notify_task_progress(task, 0, "开始处理")
            
            # 定义进度回调
            async def progress_callback(progress: float):
                await self._notify_task_progress(task, progress, f"转录进度: {progress:.1f}%")
            
            # 执行转录
            from src.core.funasr_transcriber import transcriber
            result = await transcriber.transcribe(
                audio_path=task.file_path,
                task_id=task_id,
                progress_callback=progress_callback,
                output_format=task.output_format
            )
            
            # 更新任务结果
            task.status = TaskStatus.COMPLETED
            if task.output_format == "srt":
                # SRT格式结果
                task.srt_content = result["content"]
                
                # 创建转录结果对象用于缓存
                from src.models.schemas import TranscriptionResult
                transcription_result = TranscriptionResult(
                    task_id=task_id,
                    file_name=result["file_name"],
                    file_hash=result["file_hash"],
                    duration=result["duration"],
                    segments=[],  # SRT格式不存储片段信息
                    speakers=[],  # SRT格式不存储说话人列表
                    processing_time=result["processing_time"],
                    error=None
                )
                task.result = transcription_result
                
                # 保存到缓存（包含原始结果）
                await db_manager.save_result(transcription_result, result["raw_result"])
            else:
                # JSON格式结果
                transcription_result, raw_result = result
                task.result = transcription_result
                
                # 保存到缓存（包含原始结果）
                await db_manager.save_result(transcription_result, raw_result)
            
            task.completed_at = datetime.now()
            task.progress = 100
            
            # 通知完成
            await self._notify_task_complete(task)
            
            # 删除文件（如果配置了），但要检查是否还有其他任务使用这个文件
            if config.transcription.delete_after_transcription:
                # 检查是否还有其他任务使用相同的文件哈希
                has_pending_tasks = any(
                    t.file_hash == task.file_hash and t.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]
                    for t in self.tasks.values()
                    if t.task_id != task.task_id
                )
                
                if not has_pending_tasks:
                    from src.utils.file_utils import delete_file
                    await delete_file(task.file_path)
                    logger.debug(f"文件已删除: {task.file_path}")
                else:
                    logger.debug(f"保留文件，还有其他任务使用: {task.file_path}")
            
            logger.info(f"任务完成: {task_id}")
            
        except Exception as e:
            logger.error(f"任务处理失败 {task_id}: {e}")
            
            # 更新任务状态
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            # 重试
            if task.retry_count < config.transcription.retry_times:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"任务重试 {task_id}: 第{task.retry_count}次")
                await self.task_queue.put(task_id)
            else:
                # 通知失败
                await self._notify_task_failed(task)
                
                # 删除文件（如果配置了且没有其他任务使用）
                if config.transcription.delete_after_transcription and task.file_path:
                    has_pending_tasks = any(
                        t.file_hash == task.file_hash and t.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]
                        for t in self.tasks.values()
                        if t.task_id != task.task_id
                    )
                    
                    if not has_pending_tasks:
                        from src.utils.file_utils import delete_file
                        await delete_file(task.file_path)
                        logger.debug(f"任务失败，文件已删除: {task.file_path}")
                
                # 发送企微通知
                await self._send_wework_notification(task, "failed")
    
    async def _notify_task_progress(self, task: TranscriptionTask, progress: float, message: str):
        """通知任务进度"""
        task.progress = progress
        
        try:
            from src.api.websocket_handler import ws_handler
            await ws_handler.notify_task_progress(
                task_id=task.task_id,
                progress=progress,
                status=task.status.value,
                message=message
            )
        except Exception as e:
            logger.error(f"通知任务进度失败: {e}")
    
    async def _notify_task_complete(self, task: TranscriptionTask):
        """通知任务完成"""
        try:
            from src.api.websocket_handler import ws_handler
            
            # 根据输出格式准备结果
            if task.output_format == "srt":
                result_data = {
                    "format": "srt",
                    "content": task.srt_content,
                    "file_name": task.file_name,
                    "file_hash": task.file_hash
                }
            else:
                result_data = task.result.dict() if task.result else None
            
            await ws_handler.notify_task_complete(
                task_id=task.task_id,
                result=result_data
            )
            
            # 发送企微通知
            await self._send_wework_notification(task, "completed")
            
        except Exception as e:
            logger.error(f"通知任务完成失败: {e}")
    
    async def _notify_task_failed(self, task: TranscriptionTask):
        """通知任务失败"""
        try:
            from src.api.websocket_handler import ws_handler
            await ws_handler.notify_task_progress(
                task_id=task.task_id,
                progress=task.progress,
                status=task.status.value,
                message=f"任务失败: {task.error}"
            )
        except Exception as e:
            logger.error(f"通知任务失败失败: {e}")
    
    def _convert_json_to_srt(self, result: TranscriptionResult) -> str:
        """将JSON格式的转录结果转换为SRT格式"""
        srt_lines = []
        
        for idx, segment in enumerate(result.segments, 1):
            # 转换时间格式
            start_time = self._seconds_to_srt_time(segment.start_time)
            end_time = self._seconds_to_srt_time(segment.end_time)
            
            # SRT格式：序号 -> 时间 -> 文本
            srt_lines.append(f"{idx}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(f"{segment.speaker}:{segment.text}")
            srt_lines.append("")  # 空行分隔
        
        return "\n".join(srt_lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """将秒数转换为SRT时间格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    
    async def _send_wework_notification(self, task: TranscriptionTask, event_type: str):
        """发送企微通知"""
        if not config.notification.enabled or not config.notification.webhook_url:
            return
        
        try:
            from src.utils.notification import send_wework_notification
            await send_wework_notification(task, event_type)
        except Exception as e:
            logger.error(f"发送企微通知失败: {e}")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = {
            "total_tasks": len(self.tasks),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "processing_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PROCESSING),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            "queue_size": self.task_queue.qsize()
        }
        return stats


# 全局任务管理器实例
task_manager = TaskManager()