"""
FunASR进程池模型 - 使用多进程实现真正的并发
"""
import os
import json
import asyncio
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from typing import Optional, Dict, Any
from pathlib import Path
import time
import signal
import sys
from loguru import logger


def worker_process(worker_id: int, config: dict, task_queue: Queue, result_queue: Queue, ready_event):
    """
    工作进程函数 - 每个进程独立加载模型
    
    Args:
        worker_id: 工作进程ID
        config: 配置字典
        task_queue: 任务队列
        result_queue: 结果队列
        ready_event: 就绪事件
    """
    # 在子进程中导入FunASR，避免主进程的共享问题
    from funasr import AutoModel
    
    # 设置日志
    logger.remove()  # 移除默认handler
    logger.add(sys.stdout, level="INFO")
    logger.info(f"工作进程 {worker_id} 启动中...")
    
    try:
        # 加载模型
        funasr_config = config["funasr"]
        cache_dir = funasr_config["model_dir"]
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        model = AutoModel(
            model=funasr_config["model"],
            model_revision=funasr_config["model_revision"],
            vad_model=funasr_config["vad_model"],
            vad_model_revision=funasr_config["vad_model_revision"],
            punc_model=funasr_config["punc_model"],
            punc_model_revision=funasr_config["punc_model_revision"],
            spk_model=funasr_config["spk_model"],
            spk_model_revision=funasr_config["spk_model_revision"],
            cache_dir=cache_dir,
            ncpu=funasr_config.get("ncpu", 8),
            device=funasr_config["device"],
            disable_update=funasr_config.get("disable_update", True),
            disable_pbar=funasr_config.get("disable_pbar", True)
        )
        
        logger.info(f"工作进程 {worker_id} 模型加载完成")
        ready_event.set()  # 通知主进程该工作进程已就绪
        
        # 处理任务
        while True:
            try:
                # 获取任务（阻塞等待）
                task = task_queue.get(timeout=1)
                
                if task is None:  # 退出信号
                    logger.info(f"工作进程 {worker_id} 收到退出信号")
                    break
                
                task_id = task['task_id']
                audio_path = task['audio_path']
                batch_size_s = task.get('batch_size_s', 300)
                hotword = task.get('hotword', '')
                
                logger.info(f"工作进程 {worker_id} 处理任务 {task_id}")
                
                try:
                    # 执行模型推理
                    result = model.generate(
                        input=audio_path,
                        batch_size_s=batch_size_s,
                        hotword=hotword
                    )
                    
                    # 返回结果
                    result_queue.put({
                        'task_id': task_id,
                        'success': True,
                        'result': result,
                        'worker_id': worker_id
                    })
                    
                    logger.info(f"工作进程 {worker_id} 完成任务 {task_id}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"工作进程 {worker_id} 处理任务 {task_id} 失败: {error_msg}")
                    
                    result_queue.put({
                        'task_id': task_id,
                        'success': False,
                        'error': error_msg,
                        'worker_id': worker_id
                    })
                    
            except mp.TimeoutError:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"工作进程 {worker_id} 任务循环异常: {e}")
                
    except Exception as e:
        logger.error(f"工作进程 {worker_id} 初始化失败: {e}")
        ready_event.set()  # 即使失败也要设置事件，避免主进程永远等待
    finally:
        logger.info(f"工作进程 {worker_id} 退出")


class ModelProcessPool:
    """
    进程池模型管理器 - 使用多进程实现真正的并发
    
    每个进程独立加载模型，完全隔离，避免共享状态问题
    """
    
    def __init__(self, config_path: str = "config.json", pool_size: Optional[int] = None):
        """
        初始化进程池
        
        Args:
            config_path: 配置文件路径
            pool_size: 进程池大小
        """
        self.config = self._load_config(config_path)
        self.pool_size = pool_size or self.config["transcription"]["max_concurrent_tasks"]
        
        # 进程管理
        self.workers = []
        self.manager = None
        self.task_queue = None
        self.result_queue = None
        self.ready_events = []
        self.is_initialized = False
        self.pending_tasks = {}  # 存储待处理任务
        
        logger.info(f"初始化进程池，池大小: {self.pool_size}")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    async def initialize(self):
        """初始化进程池 - 启动工作进程"""
        if self.is_initialized:
            return
        
        logger.info(f"启动 {self.pool_size} 个工作进程...")
        
        try:
            # 创建Manager
            self.manager = Manager()
            self.task_queue = self.manager.Queue()
            self.result_queue = self.manager.Queue()
            
            # 创建就绪事件
            for i in range(self.pool_size):
                ready_event = self.manager.Event()
                self.ready_events.append(ready_event)
            
            # 启动工作进程
            for i in range(self.pool_size):
                worker = Process(
                    target=worker_process,
                    args=(i, self.config, self.task_queue, self.result_queue, self.ready_events[i])
                )
                worker.start()
                self.workers.append(worker)
                logger.info(f"启动工作进程 {i} (PID: {worker.pid})")
            
            # 等待所有工作进程就绪
            logger.info("等待所有工作进程加载模型...")
            for i, event in enumerate(self.ready_events):
                if not event.wait(timeout=60):  # 60秒超时
                    raise Exception(f"工作进程 {i} 初始化超时")
            
            self.is_initialized = True
            logger.info(f"进程池初始化完成，共 {len(self.workers)} 个工作进程")
            
        except Exception as e:
            logger.error(f"进程池初始化失败: {e}")
            self.cleanup()
            raise
    
    async def generate_with_pool(self, audio_path: str, batch_size_s: int = 300, hotword: str = '') -> Any:
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
        task_id = f"{time.time()}_{os.path.basename(audio_path)}"
        
        # 创建任务
        task = {
            'task_id': task_id,
            'audio_path': audio_path,
            'batch_size_s': batch_size_s,
            'hotword': hotword
        }
        
        # 提交任务到队列
        self.task_queue.put(task)
        logger.debug(f"任务 {task_id} 已提交到进程池")
        
        # 异步等待结果
        loop = asyncio.get_event_loop()
        
        async def wait_for_result():
            while True:
                # 非阻塞地检查结果队列
                try:
                    # 在线程中运行阻塞操作
                    result = await loop.run_in_executor(
                        None,
                        self.result_queue.get,
                        True,  # block
                        0.1    # timeout
                    )
                    
                    if result['task_id'] == task_id:
                        if result['success']:
                            logger.debug(f"任务 {task_id} 处理成功 (工作进程 {result['worker_id']})")
                            return result['result']
                        else:
                            raise Exception(f"处理失败: {result['error']}")
                    else:
                        # 不是我们的结果，放回队列
                        self.result_queue.put(result)
                        
                except:
                    # 队列为空或超时，继续等待
                    await asyncio.sleep(0.1)
        
        try:
            result = await wait_for_result()
            return result
        except Exception as e:
            logger.error(f"进程池处理失败: {e}")
            raise
    
    def cleanup(self):
        """清理进程池资源"""
        logger.info("清理进程池资源...")
        
        # 发送退出信号
        for _ in range(len(self.workers)):
            self.task_queue.put(None)
        
        # 等待工作进程退出
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5)
                if worker.is_alive():
                    logger.warning(f"强制终止工作进程 {worker.pid}")
                    worker.terminate()
                    worker.join()
        
        self.workers.clear()
        self.is_initialized = False
        logger.info("进程池资源已清理")
    
    def __del__(self):
        """析构函数 - 确保资源清理"""
        if self.is_initialized:
            self.cleanup()


# 全局进程池实例
model_process_pool = None

def get_model_process_pool(config_path: str = "config.json", pool_size: Optional[int] = None) -> ModelProcessPool:
    """
    获取全局进程池实例
    
    Args:
        config_path: 配置文件路径
        pool_size: 池大小
        
    Returns:
        ModelProcessPool实例
    """
    global model_process_pool
    if model_process_pool is None:
        model_process_pool = ModelProcessPool(config_path, pool_size)
    return model_process_pool