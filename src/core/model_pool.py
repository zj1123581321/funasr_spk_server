"""
FunASR模型池 - 管理多个模型实例以支持真正的并发处理
"""
import asyncio
import json
import threading
from typing import Optional, List
from pathlib import Path
from loguru import logger
from funasr import AutoModel


class ModelPool:
    """
    模型池管理器 - 创建和管理多个FunASR模型实例
    
    通过维护多个独立的模型实例，避免并发访问冲突，
    实现真正的并发处理能力。
    """
    
    def __init__(self, config_path: str = "config.json", pool_size: Optional[int] = None):
        """
        初始化模型池
        
        Args:
            config_path: 配置文件路径
            pool_size: 模型池大小，默认为配置中的max_concurrent_tasks
        """
        self.config = self._load_config(config_path)
        self.pool_size = pool_size or self.config["transcription"]["max_concurrent_tasks"]
        self.cache_dir = self.config["funasr"]["model_dir"]
        
        # 模型池和信号量
        self.models = []
        self.semaphore = asyncio.Semaphore(self.pool_size)
        self.available_models = []
        self.lock = threading.Lock()
        self.is_initialized = False
        
        logger.info(f"初始化模型池，池大小: {self.pool_size}")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            # 返回默认配置
            return {
                "funasr": {
                    "model": "paraformer-zh",
                    "model_revision": "v2.0.4",
                    "vad_model": "fsmn-vad",
                    "vad_model_revision": "v2.0.4",
                    "punc_model": "ct-punc-c",
                    "punc_model_revision": "v2.0.4",
                    "spk_model": "cam++",
                    "spk_model_revision": "v2.0.2",
                    "model_dir": "./models",
                    "batch_size_s": 300,
                    "device": "cpu",
                    "disable_update": True,
                    "disable_pbar": True,
                    "ncpu": 8
                },
                "transcription": {
                    "max_concurrent_tasks": 4
                }
            }
    
    async def initialize(self):
        """初始化模型池 - 创建多个模型实例"""
        if self.is_initialized:
            return
        
        logger.info(f"开始初始化模型池，创建 {self.pool_size} 个模型实例...")
        
        # 确保缓存目录存在
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        funasr_config = self.config["funasr"]
        
        # 创建多个模型实例
        for i in range(self.pool_size):
            try:
                logger.info(f"创建模型实例 {i + 1}/{self.pool_size}...")
                
                model = AutoModel(
                    model=funasr_config["model"],
                    model_revision=funasr_config["model_revision"],
                    vad_model=funasr_config["vad_model"],
                    vad_model_revision=funasr_config["vad_model_revision"],
                    punc_model=funasr_config["punc_model"],
                    punc_model_revision=funasr_config["punc_model_revision"],
                    spk_model=funasr_config["spk_model"],
                    spk_model_revision=funasr_config["spk_model_revision"],
                    cache_dir=self.cache_dir,
                    ncpu=funasr_config.get("ncpu", 8),
                    device=funasr_config["device"],
                    disable_update=funasr_config.get("disable_update", True),
                    disable_pbar=funasr_config.get("disable_pbar", True)
                )
                
                self.models.append(model)
                self.available_models.append(model)
                
                logger.info(f"模型实例 {i + 1} 创建成功")
                
            except Exception as e:
                logger.error(f"创建模型实例 {i + 1} 失败: {e}")
                # 如果创建失败，清理已创建的模型
                self.models.clear()
                self.available_models.clear()
                raise Exception(f"模型池初始化失败: {e}")
        
        self.is_initialized = True
        logger.info(f"模型池初始化完成，共 {len(self.models)} 个模型实例")
    
    async def acquire_model(self):
        """
        获取一个可用的模型实例
        
        Returns:
            可用的模型实例
        """
        if not self.is_initialized:
            await self.initialize()
        
        # 等待获取信号量
        await self.semaphore.acquire()
        
        # 获取可用模型
        with self.lock:
            if not self.available_models:
                # 理论上不应该发生，因为有信号量控制
                logger.error("没有可用的模型实例")
                self.semaphore.release()
                raise Exception("模型池耗尽")
            
            model = self.available_models.pop()
            logger.debug(f"获取模型实例，剩余可用: {len(self.available_models)}")
            return model
    
    def release_model(self, model):
        """
        释放模型实例回池
        
        Args:
            model: 要释放的模型实例
        """
        with self.lock:
            if model in self.models and model not in self.available_models:
                self.available_models.append(model)
                logger.debug(f"释放模型实例，当前可用: {len(self.available_models)}")
        
        # 释放信号量
        self.semaphore.release()
    
    async def generate_with_pool(self, audio_path: str, batch_size_s: int = 300, hotword: str = ''):
        """
        使用模型池进行推理
        
        Args:
            audio_path: 音频文件路径
            batch_size_s: 批处理大小
            hotword: 热词
            
        Returns:
            推理结果
        """
        model = None
        try:
            # 获取模型
            model = await self.acquire_model()
            
            # 在线程池中执行推理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: model.generate(
                    input=audio_path,
                    batch_size_s=batch_size_s,
                    hotword=hotword
                )
            )
            
            return result
            
        finally:
            # 确保模型被释放
            if model:
                self.release_model(model)
    
    def cleanup(self):
        """清理模型池资源"""
        logger.info("清理模型池资源...")
        self.models.clear()
        self.available_models.clear()
        self.is_initialized = False
        logger.info("模型池资源已清理")


# 全局模型池实例（可选使用）
model_pool = None

def get_model_pool(config_path: str = "config.json", pool_size: Optional[int] = None) -> ModelPool:
    """
    获取全局模型池实例
    
    Args:
        config_path: 配置文件路径
        pool_size: 池大小
        
    Returns:
        ModelPool实例
    """
    global model_pool
    if model_pool is None:
        model_pool = ModelPool(config_path, pool_size)
    return model_pool