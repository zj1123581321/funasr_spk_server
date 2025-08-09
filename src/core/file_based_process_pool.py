"""
基于文件系统的进程池 - 完全独立的多进程方案
不使用共享内存，不使用Manager，通过文件系统通信
"""
import os
import sys
import json
import pickle
import time
import asyncio
import subprocess
import uuid
from typing import Optional, Any, List
from pathlib import Path
from loguru import logger


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
            config_path: 配置文件路径
            pool_size: 进程池大小
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.pool_size = pool_size or self.config["transcription"]["max_concurrent_tasks"]
        
        # 任务目录
        self.task_dir = Path("./temp/tasks")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        
        # 进程管理
        self.workers = []
        self.worker_processes = []
        self.is_initialized = False
        self.next_worker_id = 0  # 轮询分配任务
        
        logger.info(f"初始化文件系统进程池，池大小: {self.pool_size}")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    async def initialize(self):
        """初始化进程池 - 启动独立的工作进程"""
        if self.is_initialized:
            return
        
        logger.info(f"启动 {self.pool_size} 个独立工作进程...")
        
        # 清理旧的任务和结果文件
        self._cleanup_task_dir()
        
        try:
            # 启动工作进程
            for i in range(self.pool_size):
                # 构建命令
                cmd = [
                    sys.executable,  # Python解释器
                    "src/core/worker_process.py",
                    "--worker-id", str(i),
                    "--config", self.config_path,
                    "--task-dir", str(self.task_dir)
                ]
                
                # 启动进程（使用CREATE_NO_WINDOW避免弹出控制台窗口）
                if sys.platform == "win32":
                    # Windows下避免弹出新窗口
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                    
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        startupinfo=startupinfo,
                        text=True,
                        encoding='utf-8'
                    )
                else:
                    # Unix/Linux
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                
                self.worker_processes.append(process)
                logger.info(f"启动工作进程 {i} (PID: {process.pid})")
            
            # 等待所有工作进程就绪
            logger.info("等待工作进程初始化...")
            await self._wait_for_workers_ready()
            
            self.is_initialized = True
            logger.info(f"进程池初始化完成，共 {len(self.worker_processes)} 个工作进程")
            
        except Exception as e:
            logger.error(f"进程池初始化失败: {e}")
            self.cleanup()
            raise
    
    async def _wait_for_workers_ready(self, timeout: int = 60):
        """等待所有工作进程就绪"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ready_count = 0
            for i in range(self.pool_size):
                ready_file = self.task_dir / f"worker_{i}.ready"
                if ready_file.exists():
                    ready_count += 1
            
            if ready_count == self.pool_size:
                logger.info(f"所有 {self.pool_size} 个工作进程已就绪")
                return
            
            await asyncio.sleep(0.5)
        
        raise TimeoutError(f"工作进程初始化超时（{timeout}秒）")
    
    def _cleanup_task_dir(self):
        """清理任务目录"""
        try:
            # 删除所有任务和结果文件
            for file in self.task_dir.glob("*.task"):
                file.unlink()
            for file in self.task_dir.glob("*.result"):
                file.unlink()
            for file in self.task_dir.glob("*.ready"):
                file.unlink()
            for file in self.task_dir.glob("*.stop"):
                file.unlink()
        except Exception as e:
            logger.warning(f"清理任务目录时出错: {e}")
    
    async def generate_with_pool(self, audio_path: str, batch_size_s: int = 300, hotword: str = '', use_pickle: bool = True) -> Any:
        """
        使用进程池进行推理
        
        Args:
            audio_path: 音频文件路径
            batch_size_s: 批处理大小
            hotword: 热词
            
        Returns:
            推理结果
        """
        if not self.is_initialized:
            await self.initialize()
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 选择工作进程（轮询）
        worker_id = self.next_worker_id
        self.next_worker_id = (self.next_worker_id + 1) % self.pool_size
        
        # 创建任务文件
        task_file = self.task_dir / f"worker_{worker_id}_{task_id}.task"
        
        # 根据是否使用pickle决定结果文件扩展名
        if use_pickle:
            result_file = self.task_dir / f"worker_{worker_id}_{task_id}.pkl"
        else:
            result_file = self.task_dir / f"worker_{worker_id}_{task_id}.result"
        
        task_data = {
            'task_id': task_id,
            'audio_path': audio_path,
            'batch_size_s': batch_size_s,
            'hotword': hotword,
            'use_pickle': use_pickle
        }
        
        # 写入任务文件
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False)
        
        logger.debug(f"任务 {task_id} 分配给工作进程 {worker_id}")
        
        # 异步等待结果
        max_wait_time = 600  # 最长等待10分钟
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # 检查结果文件
            if result_file.exists():
                try:
                    # 获取文件大小
                    file_size = result_file.stat().st_size
                    logger.debug(f"结果文件大小: {file_size / 1024:.2f} KB")
                    
                    # 多次尝试读取文件（文件可能还在写入中）
                    max_read_attempts = 3
                    for attempt in range(max_read_attempts):
                        try:
                            # 根据文件类型选择读取方式
                            if result_file.suffix == '.pkl':
                                # 使用pickle读取
                                with open(result_file, 'rb') as f:
                                    result_data = pickle.load(f)
                            else:
                                # 使用JSON读取
                                with open(result_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                # 尝试解析JSON
                                result_data = json.loads(content)
                            
                            # 删除结果文件
                            try:
                                result_file.unlink()
                            except:
                                pass  # 忽略删除失败
                            
                            # 检查结果
                            if result_data['success']:
                                logger.debug(f"任务 {task_id} 处理成功 (工作进程 {worker_id})")
                                return result_data['result']
                            else:
                                error_msg = result_data.get('error', '未知错误')
                                logger.error(f"任务 {task_id} 处理失败: {error_msg}")
                                if 'traceback' in result_data:
                                    logger.debug(f"错误堆栈: {result_data['traceback']}")
                                raise Exception(f"处理失败: {error_msg}")
                                
                        except (json.JSONDecodeError, pickle.UnpicklingError, EOFError) as e:
                            if attempt < max_read_attempts - 1:
                                logger.warning(f"文件解析失败（尝试 {attempt + 1}/{max_read_attempts}）: {e}")
                                await asyncio.sleep(0.5)  # 等待文件写入完成
                            else:
                                logger.error(f"解析结果文件失败: {e}")
                                logger.error(f"文件大小: {file_size} bytes, 类型: {result_file.suffix}")
                                # 尝试读取文件内容进行调试
                                try:
                                    with open(result_file, 'rb') as f:
                                        raw_content = f.read(1000)  # 读取前1000字节
                                    logger.debug(f"文件开头内容: {raw_content[:200]}")
                                except:
                                    pass
                                # 删除损坏的文件
                                try:
                                    result_file.unlink()
                                except:
                                    pass
                                raise Exception(f"文件解析失败: {e}")
                        
                except Exception as e:
                    logger.error(f"处理结果文件时出错: {e}")
                    # 尝试删除文件
                    try:
                        result_file.unlink()
                    except:
                        pass
                    raise
            
            # 检查工作进程是否还活着
            if not self._is_worker_alive(worker_id):
                logger.error(f"工作进程 {worker_id} 已退出")
                # 尝试重启工作进程
                await self._restart_worker(worker_id)
                # 重新提交任务
                with open(task_file, 'w', encoding='utf-8') as f:
                    json.dump(task_data, f, ensure_ascii=False)
            
            # 短暂等待
            await asyncio.sleep(0.1)
        
        # 超时
        logger.error(f"任务 {task_id} 处理超时")
        # 尝试删除任务文件
        if task_file.exists():
            task_file.unlink()
        raise TimeoutError(f"任务处理超时（{max_wait_time}秒）")
    
    def _is_worker_alive(self, worker_id: int) -> bool:
        """检查工作进程是否存活"""
        if worker_id < len(self.worker_processes):
            process = self.worker_processes[worker_id]
            return process.poll() is None  # None表示进程还在运行
        return False
    
    async def _restart_worker(self, worker_id: int):
        """重启工作进程"""
        logger.info(f"重启工作进程 {worker_id}...")
        
        # 终止旧进程
        if worker_id < len(self.worker_processes):
            old_process = self.worker_processes[worker_id]
            if old_process.poll() is None:
                old_process.terminate()
                old_process.wait(timeout=5)
        
        # 启动新进程
        cmd = [
            sys.executable,
            "src/core/worker_process.py",
            "--worker-id", str(worker_id),
            "--config", self.config_path,
            "--task-dir", str(self.task_dir)
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
                encoding='utf-8'
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        
        self.worker_processes[worker_id] = process
        logger.info(f"工作进程 {worker_id} 已重启 (PID: {process.pid})")
        
        # 等待进程就绪
        for _ in range(60):
            ready_file = self.task_dir / f"worker_{worker_id}.ready"
            if ready_file.exists():
                break
            await asyncio.sleep(0.5)
    
    def cleanup(self):
        """清理进程池资源"""
        logger.info("清理进程池资源...")
        
        # 发送停止信号给所有工作进程
        for i in range(self.pool_size):
            stop_file = self.task_dir / f"worker_{i}.stop"
            try:
                stop_file.touch()
            except:
                pass
        
        # 等待进程退出
        for i, process in enumerate(self.worker_processes):
            if process.poll() is None:
                logger.info(f"等待工作进程 {i} 退出...")
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"强制终止工作进程 {i}")
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except:
                        process.kill()
        
        # 清理任务目录
        self._cleanup_task_dir()
        
        self.worker_processes.clear()
        self.is_initialized = False
        logger.info("进程池资源已清理")
    
    def __del__(self):
        """析构函数 - 确保资源清理"""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            self.cleanup()


# 全局进程池实例
file_based_pool = None

def get_file_based_pool(config_path: str = "config.json", pool_size: Optional[int] = None) -> FileBasedProcessPool:
    """
    获取全局进程池实例
    
    Args:
        config_path: 配置文件路径
        pool_size: 池大小
        
    Returns:
        FileBasedProcessPool实例
    """
    global file_based_pool
    if file_based_pool is None:
        file_based_pool = FileBasedProcessPool(config_path, pool_size)
    return file_based_pool