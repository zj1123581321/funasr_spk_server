"""
Qwen3PoolTranscriber — 主进程 wrapper, 通过 multi-process worker pool 调度任务

PR3: Qwen3 引擎走 multi-process worker pool, 跟 FunASR 同一套架构.
本类替代主进程直接持有 Qwen3DiarizeTranscriber 的旧路径.

为什么:
- libllama.cpp 单 context 设计上不支持并发, 同进程多协程同时调会串台
- 改为每个 worker subprocess 独立加载 Qwen3DiarizeTranscriber, libllama context 自然 per-worker 隔离
- 主进程只负责派发任务给 pool, 不再 import vendor 引擎

接口与 Qwen3DiarizeTranscriber.transcribe 鸭子兼容(同形):
    async def transcribe(audio_path, task_id, progress_callback, output_format)
        JSON: (TranscriptionResult, raw_result_dict)
        SRT:  {format, content, file_name, file_hash, duration, processing_time, raw_result}
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from loguru import logger

from src.core.file_based_process_pool import FileBasedProcessPool


class Qwen3PoolTranscriber:
    """Qwen3 池化转录器 — 把 transcribe 调用转成 worker subprocess 任务派发"""

    def __init__(
        self,
        pool_size: int,
        pool: Optional[FileBasedProcessPool] = None,
    ):
        """
        Args:
            pool_size: worker 进程数(PoC v5 sweet spot N=3).
            pool: 注入式 pool(测试用). 不传则按 qwen3_worker_process.py 作为 entry 新建.
        """
        if pool is not None:
            self._pool = pool
        else:
            # task_dir 必须与 FunASR 池物理隔离 — 否则同机器 PM2 daemon FunASR 和
            # 临时跑的 Qwen3 测试 / 进程会抢 worker_X_*.task 文件, 导致结果串台.
            self._pool = FileBasedProcessPool(
                pool_size=pool_size,
                worker_entry_script="src/core/qwen3_worker_process.py",
                task_dir="./temp/tasks_qwen3",
            )

    async def initialize(self):
        """提前初始化 pool(启动所有 worker subprocess 并加载模型)"""
        await self._pool.initialize()

    async def transcribe(
        self,
        audio_path: str,
        task_id: str,
        progress_callback: Optional[Callable] = None,
        output_format: str = "json",
    ) -> Any:
        """通过 pool 派发任务给 worker subprocess.

        跨进程 progress 难精细传递, 简化为开始 0% / 完成 100% 两次回调.
        worker 内部的进度日志走 worker 自己的 log 文件.
        """
        await self._notify_progress(progress_callback, 0, task_id)

        try:
            result = await self._pool.generate_with_pool(
                audio_path=audio_path,
                extra_task_fields={"output_format": output_format},
            )
        except Exception:
            # pool 错误透传, 由上层 task_manager 处理重试逻辑
            raise

        await self._notify_progress(progress_callback, 100, task_id)
        return result

    @staticmethod
    async def _notify_progress(callback: Optional[Callable], pct: int, task_id: str):
        if callback is None:
            return
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(pct)
            else:
                callback(pct)
        except Exception as e:
            logger.warning(f"[{task_id}] progress_callback 异常(忽略): {e}")


# ==================== 全局单例 ====================

_qwen3_pool_singleton: Optional[Qwen3PoolTranscriber] = None


def get_qwen3_pool_transcriber() -> Qwen3PoolTranscriber:
    """获取 Qwen3 池化转录器单例 — 从 config.transcription.qwen3_pool_size 读池大小"""
    global _qwen3_pool_singleton
    if _qwen3_pool_singleton is not None:
        return _qwen3_pool_singleton

    from src.core.config import config

    pool_size = config.transcription.qwen3_pool_size
    _qwen3_pool_singleton = Qwen3PoolTranscriber(pool_size=pool_size)
    return _qwen3_pool_singleton


def reset_qwen3_pool_singleton():
    """重置单例(仅测试用)"""
    global _qwen3_pool_singleton
    _qwen3_pool_singleton = None
