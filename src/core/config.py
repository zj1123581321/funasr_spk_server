"""
配置管理模块
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from loguru import logger


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 100
    max_file_size_mb: int = 5000
    allowed_extensions: list[str] = [".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg", ".opus"]
    temp_dir: str = "./temp"
    upload_dir: str = "./uploads"
    
    model_config = {"protected_namespaces": ()}


class FunASRConfig(BaseModel):
    """FunASR模型配置"""
    model: str = "paraformer-zh"
    model_revision: str = "v2.0.4"
    vad_model: str = "fsmn-vad"
    vad_model_revision: str = "v2.0.4"
    punc_model: str = "ct-punc-c"
    punc_model_revision: str = "v2.0.4"
    spk_model: str = "cam++"
    spk_model_revision: str = "v2.0.2"
    model_dir: str = "./models"
    batch_size_s: int = 300
    ncpu: int = 8  # 添加 ncpu 属性
    device: str = "cpu"
    disable_update: bool = True
    disable_pbar: bool = True
    
    model_config = {"protected_namespaces": ()}


class TranscriptionConfig(BaseModel):
    """转录配置"""
    max_concurrent_tasks: int = 4
    task_timeout_minutes: int = 30
    retry_times: int = 2
    cache_enabled: bool = True
    delete_after_transcription: bool = True
    
    model_config = {"protected_namespaces": ()}


class DatabaseConfig(BaseModel):
    """数据库配置"""
    path: str = "./data/transcription_cache.db"
    max_cache_days: int = 30
    
    model_config = {"protected_namespaces": ()}


class NotificationConfig(BaseModel):
    """通知配置"""
    enabled: bool = True
    webhook_url: str = ""
    retry_times: int = 3
    timeout_seconds: int = 10
    
    model_config = {"protected_namespaces": ()}


class AuthConfig(BaseModel):
    """认证配置"""
    enabled: bool = True
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    model_config = {"protected_namespaces": ()}


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    rotation: str = "100 MB"
    retention: str = "7 days"
    log_dir: str = "./logs"
    
    model_config = {"protected_namespaces": ()}


class Config(BaseModel):
    """总配置"""
    server: ServerConfig = ServerConfig()
    funasr: FunASRConfig = FunASRConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    database: DatabaseConfig = DatabaseConfig()
    notification: NotificationConfig = NotificationConfig()
    auth: AuthConfig = AuthConfig()
    logging: LoggingConfig = LoggingConfig()
    
    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> "Config":
        """从文件加载配置"""
        if not os.path.exists(config_path):
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            config = cls()
            config.save_to_file(config_path)
            return config
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.warning("使用默认配置")
            return cls()
    
    def save_to_file(self, config_path: str = "config.json"):
        """保存配置到文件"""
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到 {config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            self.server.temp_dir,
            self.server.upload_dir,
            self.funasr.model_dir,
            Path(self.database.path).parent,
            self.logging.log_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"确保目录存在: {directory}")


# 全局配置实例
config = Config.load_from_file()
config.setup_directories()