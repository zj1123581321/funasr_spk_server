"""
基于文件系统的进程池 - 完全独立的多进程方案
不使用共享内存，不使用Manager，通过文件系统通信
"""
import asyncio
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger

from src.core.config import config as global_config


class FileBasedProcessPool:
    """
    基于文件系统的进程池管理器

    通过文件系统实现进程间通信，完全避免共享状态问题
    每个工作进程完全独立运行
    """

    def __init__(self, config_path: str = "config.json", pool_size: Optional[int] = None):
        """
        初始化进程池

        Args:
            config_path: 配置文件路径（已废弃，保留用于兼容性）
            pool_size: 进程池大小
        """
        self.pool_size = pool_size or global_config.transcription.max_concurrent_tasks

        # 任务目录
        self.task_dir = Path("./temp/tasks")
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # 进程管理
        self.worker_processes: List[Optional[subprocess.Popen]] = []
        self.is_initialized = False
        self.next_worker_id = 0  # 轮询分配任务

        # 协调重启
        self._management_lock = asyncio.Lock()

        # 巡检任务
        self._health_task: Optional[asyncio.Task] = None
        self._health_check_interval = 30  # seconds

        logger.info(f"初始化文件系统进程池，池大小: {self.pool_size}")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _log_worker_states(self, context: str) -> None:
        """输出当前已知的 worker 进程状态"""
        states = []
        for idx in range(self.pool_size):
            if idx >= len(self.worker_processes):
                states.append(f"{idx}[未创建]")
                continue
            process = self.worker_processes[idx]
            if process is None:
                states.append(f"{idx}[缺失]")
                continue
            pid = process.pid
            exit_code = process.poll()
            status = "运行中" if exit_code is None else f"已退出({exit_code})"
            states.append(f"{idx}[pid={pid},{status}]")
        if states:
            logger.info(f"{context} | 工作进程状态: {', '.join(states)}")

    def _ensure_capacity(self, worker_id: int) -> None:
        """确保 worker 列表扩容到指定索引"""
        needed = worker_id + 1
        while len(self.worker_processes) < needed:
            self.worker_processes.append(None)

    def _launch_worker_process(self, worker_id: int) -> subprocess.Popen:
        """在当前事件循环内同步启动进程"""
        cmd = [
            sys.executable,
            "src/core/worker_process.py",
            "--worker-id",
            str(worker_id),
            "--task-dir",
            str(self.task_dir),
        ]

        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                text=True,
                encoding="utf-8",
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        return process

    async def _wait_for_worker_ready(
        self,
        worker_id: int,
        process: subprocess.Popen,
        timeout: int = 300,
    ) -> None:
        """等待 worker 写入 ready 文件"""
        ready_file = self.task_dir / f"worker_{worker_id}.ready"
        start = time.time()
        loop = asyncio.get_event_loop()

        while time.time() - start < timeout:
            if ready_file.exists():
                return

            # 如果进程已退出则立即失败
            exit_code = await loop.run_in_executor(None, process.poll)
            if exit_code is not None:
                raise RuntimeError(f"工作进程 {worker_id} 在就绪前退出 (exit={exit_code})")

            await asyncio.sleep(0.5)

        raise TimeoutError(f"工作进程 {worker_id} 在 {timeout} 秒内未写入就绪标记")

    async def _spawn_worker(self, worker_id: int) -> None:
        """创建并注册新的工作进程"""
        async with self._management_lock:
            self._ensure_capacity(worker_id)

            # 终止旧进程
            old_process = self.worker_processes[worker_id]
            if old_process is not None and old_process.poll() is None:
                loop = asyncio.get_event_loop()
                try:
                    await loop.run_in_executor(None, old_process.wait, 5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"工作进程 {worker_id} 在 {os.getpid()} 中未及时退出，发送 terminate")
                    old_process.terminate()
                    try:
                        await loop.run_in_executor(None, old_process.wait, 5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"工作进程 {worker_id} 强制 kill")
                        old_process.kill()
                except Exception as wait_error:
                    logger.warning(f"等待工作进程 {worker_id} 退出时出现异常: {wait_error}")
                    if old_process.poll() is None:
                        old_process.terminate()
                        try:
                            await loop.run_in_executor(None, old_process.wait, 5)
                        except Exception:
                            old_process.kill()

            # 清理旧文件
            ready_file = self.task_dir / f"worker_{worker_id}.ready"
            stop_file = self.task_dir / f"worker_{worker_id}.stop"
            for artifact in (ready_file, stop_file):
                try:
                    artifact.unlink()
                except FileNotFoundError:
                    pass
                except Exception as cleanup_error:
                    logger.debug(f"清理 {artifact.name} 失败: {cleanup_error}")

            # 清理残留任务文件，避免重复执行
            task_pattern = f"worker_{worker_id}_*.task"
            for stale_task in self.task_dir.glob(task_pattern):
                try:
                    stale_task.unlink()
                except Exception as cleanup_error:
                    logger.debug(f"删除残留任务文件 {stale_task.name} 失败: {cleanup_error}")

            process = self._launch_worker_process(worker_id)
            self.worker_processes[worker_id] = process

        try:
            await self._wait_for_worker_ready(worker_id, process)
        except Exception as spawn_error:
            logger.error(f"工作进程 {worker_id} 启动失败: {spawn_error}")
            if process.poll() is None:
                process.terminate()
                try:
                    await asyncio.get_event_loop().run_in_executor(None, process.wait, 5)
                except Exception:
                    process.kill()
            async with self._management_lock:
                self.worker_processes[worker_id] = None
            raise

        logger.info(f"工作进程 {worker_id} 已启动 (PID: {process.pid})")

    # ------------------------------------------------------------------
    # 巡检任务
    # ------------------------------------------------------------------
    def _start_health_monitor(self) -> None:
        """启动后台巡检任务"""
        if self._health_task and not self._health_task.done():
            return
        loop = asyncio.get_event_loop()
        self._health_task = loop.create_task(self._health_check_loop())
        logger.debug("工作进程健康巡检任务已启动")

    def _stop_health_monitor(self) -> None:
        """停止后台巡检任务"""
        if not self._health_task:
            return

        task = self._health_task
        self._health_task = None

        def _finalizer(t: asyncio.Task) -> None:
            try:
                t.result()
            except asyncio.CancelledError:
                pass
            except Exception as err:
                logger.debug(f"巡检任务结束时捕获的异常: {err}")

        task.add_done_callback(_finalizer)
        task.cancel()
        logger.debug("工作进程健康巡检任务已请求停止")

    async def _health_check_loop(self) -> None:
        """周期性校验工作进程数量"""
        try:
            while self.is_initialized:
                await asyncio.sleep(self._health_check_interval)
                if not self.is_initialized:
                    break
                try:
                    await self._ensure_workers_alive()
                except asyncio.CancelledError:
                    raise
                except Exception as monitor_error:
                    logger.error(f"巡检任务执行失败: {monitor_error}")
        except asyncio.CancelledError:
            pass
        finally:
            logger.debug("工作进程健康巡检任务结束")

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------
    async def initialize(self):
        """初始化进程池 - 启动独立的工作进程"""
        if self.is_initialized:
            return

        logger.info(f"启动 {self.pool_size} 个独立工作进程...")

        # 清理旧目录
        self._cleanup_task_dir()

        try:
            spawn_tasks = [
                asyncio.create_task(self._spawn_worker(i)) for i in range(self.pool_size)
            ]
            await asyncio.gather(*spawn_tasks)

            self.is_initialized = True
            self._log_worker_states("初始化完成")
            self._start_health_monitor()

        except Exception as e:
            logger.error(f"进程池初始化失败: {e}")
            await self.cleanup()
            raise

    def _cleanup_task_dir(self):
        """清理任务目录"""
        try:
            for entry in self.task_dir.iterdir():
                try:
                    if entry.is_file() or entry.is_symlink():
                        entry.unlink()
                    elif entry.is_dir():
                        shutil.rmtree(entry)
                except Exception as inner_error:
                    logger.debug(f"删除 {entry} 失败: {inner_error}")
        except Exception as e:
            logger.warning(f"清理任务目录时出错: {e}")

    async def _ensure_workers_alive(self):
        """检查并补齐所有工作进程"""
        self._log_worker_states("巡检前")
        for worker_id in range(self.pool_size):
            self._ensure_capacity(worker_id)
            process = self.worker_processes[worker_id]
            needs_spawn = process is None or process.poll() is not None
            if needs_spawn:
                logger.warning(f"检测到工作进程 {worker_id} 不可用，尝试重启")
                await self._spawn_worker(worker_id)
        self._log_worker_states("巡检后")

    # ------------------------------------------------------------------
    # 任务调度
    # ------------------------------------------------------------------
    async def generate_with_pool(
        self,
        audio_path: str,
        batch_size_s: int = 300,
        hotword: str = "",
        use_pickle: bool = True,
    ) -> Any:
        """
        使用进程池进行推理
        """
        if not self.is_initialized:
            await self.initialize()

        await self._ensure_workers_alive()

        task_id = str(uuid.uuid4())

        # 选择 worker
        worker_id = self.next_worker_id
        self.next_worker_id = (self.next_worker_id + 1) % self.pool_size

        self._ensure_capacity(worker_id)
        process = self.worker_processes[worker_id]
        if process is None or process.poll() is not None:
            logger.warning(f"工作进程 {worker_id} 不可用，正在重启")
            await self._spawn_worker(worker_id)
            self._log_worker_states(f"重新选择工作进程 {worker_id} 后状态")

        # 准备任务文件
        task_file = self.task_dir / f"worker_{worker_id}_{task_id}.task"
        result_file = (
            self.task_dir / f"worker_{worker_id}_{task_id}.pkl"
            if use_pickle
            else self.task_dir / f"worker_{worker_id}_{task_id}.result"
        )

        source_audio_path = Path(audio_path)
        if not source_audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        local_audio_ext = source_audio_path.suffix or ".wav"
        local_audio_path = self.task_dir / f"{task_id}{local_audio_ext}"

        try:
            shutil.copy2(source_audio_path, local_audio_path)
        except Exception as copy_error:
            raise RuntimeError(f"复制音频文件失败: {copy_error}") from copy_error

        task_data = {
            "task_id": task_id,
            "audio_path": str(local_audio_path),
            "source_audio_path": str(source_audio_path),
            "batch_size_s": batch_size_s,
            "hotword": hotword,
            "use_pickle": use_pickle,
        }

        try:
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False)
        except Exception as write_error:
            try:
                local_audio_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise write_error

        logger.debug(f"任务 {task_id} 分配给工作进程 {worker_id}")

        max_wait_time = self._calculate_timeout(audio_path)
        start_time = time.time()
        logger.info(f"任务 {task_id} 设置超时时间: {max_wait_time/60:.1f} 分钟")

        while time.time() - start_time < max_wait_time:
            if result_file.exists():
                try:
                    file_size = result_file.stat().st_size
                    logger.debug(f"结果文件大小: {file_size / 1024:.2f} KB")

                    # 读取结果
                    if result_file.suffix == ".pkl":
                        with open(result_file, "rb") as f:
                            result_data = pickle.load(f)
                    else:
                        with open(result_file, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        result_data = json.loads(content)

                    try:
                        result_file.unlink()
                    except Exception:
                        pass

                    await self._spawn_worker(worker_id)

                    if result_data["success"]:
                        logger.debug(f"任务 {task_id} 处理成功 (工作进程 {worker_id})")
                        return result_data["result"]

                    error_msg = result_data.get("error", "未知错误")
                    logger.error(f"任务 {task_id} 处理失败: {error_msg}")
                    if "traceback" in result_data:
                        logger.debug(f"错误堆栈: {result_data['traceback']}")
                    raise Exception(f"处理失败: {error_msg}")

                except Exception as e:
                    try:
                        result_file.unlink()
                    except Exception:
                        pass
                    await self._spawn_worker(worker_id)
                    raise e

            # 检查 worker 是否意外退出
            process = self.worker_processes[worker_id]
            if process is None or process.poll() is not None:
                logger.warning(f"工作进程 {worker_id} 在任务执行中退出，重新分配")
                await self._spawn_worker(worker_id)
                # 重新提交任务
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, ensure_ascii=False)
                start_time = time.time()  # 重置超时

            await asyncio.sleep(0.1)

        logger.error(f"任务 {task_id} 处理超时")
        if task_file.exists():
            task_file.unlink()
        raise TimeoutError(f"任务处理超时（{max_wait_time}秒）")

    # ------------------------------------------------------------------
    # 其它工具
    # ------------------------------------------------------------------
    def _calculate_timeout(self, audio_path: str) -> int:
        """根据音频时长动态计算超时时间"""
        try:
            import librosa

            duration = librosa.get_duration(filename=audio_path)

            base_timeout = 300  # 5 分钟
            duration_factor = 0.3  # 每秒音频需要 0.3 秒处理
            calculated_timeout = int(base_timeout + duration * duration_factor)

            min_timeout = 600
            max_timeout = 3600

            timeout = max(min_timeout, min(calculated_timeout, max_timeout))

            logger.info(
                f"音频时长: {duration:.1f}s ({duration/60:.1f}分钟), 计算超时: {timeout}s ({timeout/60:.1f}分钟)"
            )
            return timeout

        except Exception as e:
            logger.warning(f"无法获取音频时长，使用默认超时: {e}")
            return 1200  # 默认20分钟

    def _is_worker_alive(self, worker_id: int) -> bool:
        """检查工作进程是否存活"""
        if worker_id < len(self.worker_processes):
            process = self.worker_processes[worker_id]
            return process is not None and process.poll() is None
        return False

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    async def cleanup(self):
        """清理进程池资源"""
        logger.info("清理进程池资源...")

        if self.is_initialized:
            self.is_initialized = False
        self._stop_health_monitor()

        for i in range(self.pool_size):
            stop_file = self.task_dir / f"worker_{i}.stop"
            try:
                stop_file.touch()
            except Exception:
                pass

        for i, process in enumerate(self.worker_processes):
            if process is None:
                continue
            if process.poll() is None:
                logger.info(f"等待工作进程 {i} 退出...")
                try:
                    await asyncio.get_event_loop().run_in_executor(None, process.wait, 5)
                except Exception:
                    logger.warning(f"强制终止工作进程 {i}")
                    process.terminate()
                    try:
                        await asyncio.get_event_loop().run_in_executor(None, process.wait, 2)
                    except Exception:
                        process.kill()

        self._cleanup_task_dir()

        self.worker_processes.clear()
        self._log_worker_states("清理完成")
        logger.info("进程池资源已清理")

    def __del__(self):
        """析构函数 - 确保资源清理"""
        if hasattr(self, "is_initialized") and self.is_initialized:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())


# 全局进程池实例
file_based_pool = None


def get_file_based_pool(
    config_path: str = "config.json", pool_size: Optional[int] = None
) -> FileBasedProcessPool:
    """
    获取全局进程池实例
    """
    global file_based_pool
    if file_based_pool is None:
        file_based_pool = FileBasedProcessPool(config_path, pool_size)
    return file_based_pool
