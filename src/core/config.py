"""
配置管理模块 - 支持环境变量覆盖和配置验证
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from loguru import logger
from dotenv import load_dotenv


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8767
    max_connections: int = 100
    max_file_size_mb: int = 5000
    allowed_extensions: list[str] = [".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".webm"]
    temp_dir: str = "./temp"
    upload_dir: str = "./uploads"
    connection_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 60

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
    batch_size_s: int = 500
    ncpu: int = 16
    device: str = "auto"
    device_priority: List[str] = ["mps", "cpu"]
    fallback_on_error: bool = True
    disable_update: bool = True
    disable_pbar: bool = True

    model_config = {"protected_namespaces": ()}


class TranscriptionConfig(BaseModel):
    """转录配置"""
    max_concurrent_tasks: int = 2
    max_queue_size: int = 50
    concurrency_mode: str = "pool"
    task_timeout_minutes: int = 30
    retry_times: int = 2
    cache_enabled: bool = True
    delete_after_transcription: bool = True
    transcription_speed_ratio: int = 10
    queue_status_enabled: bool = True

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
    enabled: bool = False
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    model_config = {"protected_namespaces": ()}


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    rotation: str = "100 MB"
    retention: str = "7 days"
    log_dir: str = "./logs"

    model_config = {"protected_namespaces": ()}


class Config(BaseModel):
    """总配置 - 支持环境变量覆盖"""
    server: ServerConfig = ServerConfig()
    funasr: FunASRConfig = FunASRConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    database: DatabaseConfig = DatabaseConfig()
    notification: NotificationConfig = NotificationConfig()
    auth: AuthConfig = AuthConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> "Config":
        """
        从文件加载配置，并支持环境变量覆盖
        优先级: 环境变量 > config.json > 默认值
        """
        # 加载 .env 文件
        load_dotenv()

        # 从 config.json 加载基础配置
        config_data = cls._load_json_config(config_path)

        # 应用环境变量覆盖
        config_data = cls._apply_env_overrides(config_data)

        # 创建配置实例
        config = cls(**config_data)

        # 验证配置
        cls._validate_config(config)

        # 注意: 配置打印延迟到 setup_logger() 之后进行，以保证日志格式统一
        # 不在这里调用 cls._print_config(config)

        return config

    @classmethod
    def _load_json_config(cls, config_path: str) -> Dict[str, Any]:
        """从 JSON 文件加载配置"""
        if not os.path.exists(config_path):
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 过滤掉注释字段
            return cls._filter_comments(data)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.warning("使用默认配置")
            return {}

    @classmethod
    def _filter_comments(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """递归过滤掉所有以 _comment 开头的键"""
        filtered = {}
        for key, value in data.items():
            if not key.startswith("_comment"):
                if isinstance(value, dict):
                    filtered[key] = cls._filter_comments(value)
                else:
                    filtered[key] = value
        return filtered

    @classmethod
    def _apply_env_overrides(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""

        # 初始化嵌套字典
        if "server" not in config_data:
            config_data["server"] = {}
        if "funasr" not in config_data:
            config_data["funasr"] = {}
        if "transcription" not in config_data:
            config_data["transcription"] = {}
        if "database" not in config_data:
            config_data["database"] = {}
        if "notification" not in config_data:
            config_data["notification"] = {}
        if "auth" not in config_data:
            config_data["auth"] = {}
        if "logging" not in config_data:
            config_data["logging"] = {}

        # ==================== 服务器配置 ====================
        cls._override_if_set(config_data["server"], "host", "FUNASR_SERVER_HOST")
        cls._override_if_set(config_data["server"], "port", "FUNASR_SERVER_PORT", int)
        cls._override_if_set(config_data["server"], "max_connections", "FUNASR_SERVER_MAX_CONNECTIONS", int)
        cls._override_if_set(config_data["server"], "max_file_size_mb", "FUNASR_SERVER_MAX_FILE_SIZE_MB", int)
        cls._override_if_set(config_data["server"], "temp_dir", "FUNASR_TEMP_DIR")
        cls._override_if_set(config_data["server"], "upload_dir", "FUNASR_UPLOAD_DIR")

        # ==================== FunASR 配置 ====================
        cls._override_if_set(config_data["funasr"], "model_dir", "FUNASR_MODEL_DIR")
        cls._override_if_set(config_data["funasr"], "device", "FUNASR_DEVICE")
        cls._override_if_set(config_data["funasr"], "ncpu", "FUNASR_NCPU", int)
        cls._override_if_set(config_data["funasr"], "batch_size_s", "FUNASR_BATCH_SIZE_S", int)

        # 设备优先级 (逗号分隔)
        device_priority = os.getenv("FUNASR_DEVICE_PRIORITY")
        if device_priority:
            config_data["funasr"]["device_priority"] = [d.strip() for d in device_priority.split(",")]

        # ==================== 转录配置 ====================
        cls._override_if_set(config_data["transcription"], "max_concurrent_tasks", "FUNASR_MAX_CONCURRENT_TASKS", int)
        cls._override_if_set(config_data["transcription"], "task_timeout_minutes", "FUNASR_TASK_TIMEOUT_MINUTES", int)
        cls._override_if_set(config_data["transcription"], "transcription_speed_ratio", "FUNASR_TRANSCRIPTION_SPEED_RATIO", int)

        # ==================== 数据库配置 ====================
        # 数据库路径由 FUNASR_DATA_DIR 构建
        data_dir = os.getenv("FUNASR_DATA_DIR")
        if data_dir:
            config_data["database"]["path"] = os.path.join(data_dir, "transcription_cache.db")

        # ==================== 通知配置 ====================
        cls._override_if_set(config_data["notification"], "enabled", "FUNASR_NOTIFICATION_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["notification"], "webhook_url", "FUNASR_WEBHOOK_URL")

        # ==================== 认证配置 ====================
        cls._override_if_set(config_data["auth"], "enabled", "FUNASR_AUTH_ENABLED", cls._parse_bool)
        cls._override_if_set(config_data["auth"], "secret_key", "FUNASR_AUTH_SECRET_KEY")

        # ==================== 日志配置 ====================
        cls._override_if_set(config_data["logging"], "level", "FUNASR_LOG_LEVEL")
        cls._override_if_set(config_data["logging"], "log_dir", "FUNASR_LOG_DIR")

        return config_data

    @staticmethod
    def _override_if_set(config_dict: Dict[str, Any], key: str, env_var: str, converter=None):
        """如果环境变量存在，则覆盖配置值"""
        value = os.getenv(env_var)
        if value is not None:
            try:
                config_dict[key] = converter(value) if converter else value
                # 配置加载阶段不输出日志,避免影响日志系统初始化
            except Exception as e:
                # 错误仍需要输出
                logger.error(f"环境变量 {env_var} 转换失败: {e}")

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """解析布尔值环境变量"""
        return value.lower() in ("true", "1", "yes", "on")

    @classmethod
    def _validate_config(cls, config: "Config"):
        """验证配置的完整性和正确性"""
        # 注意: 此方法在 setup_logger() 之前调用，所以不输出 info 级别日志
        # 只输出错误和警告（如果有的话）
        is_worker = os.getenv('FUNASR_WORKER_MODE') == '1'

        errors = []
        warnings = []

        # 验证通知配置
        if config.notification.enabled:
            if not config.notification.webhook_url:
                errors.append("通知功能已启用 (FUNASR_NOTIFICATION_ENABLED=true)，但未设置 FUNASR_WEBHOOK_URL")

        # 验证认证配置
        if config.auth.enabled:
            if config.auth.secret_key == "your-secret-key-change-this-in-production":
                warnings.append("认证功能已启用，但仍在使用默认密钥，生产环境请务必修改 FUNASR_AUTH_SECRET_KEY！")

        # 验证目录配置
        directories = {
            "模型目录": config.funasr.model_dir,
            "上传目录": config.server.upload_dir,
            "临时目录": config.server.temp_dir,
            "日志目录": config.logging.log_dir,
            "数据目录": str(Path(config.database.path).parent),
        }

        for name, path in directories.items():
            if not path:
                errors.append(f"{name} 未配置")

        # 验证端口范围
        if not (1 <= config.server.port <= 65535):
            errors.append(f"服务器端口 {config.server.port} 超出有效范围 (1-65535)")

        # 验证并发任务数
        if config.transcription.max_concurrent_tasks < 1:
            errors.append(f"最大并发任务数必须 >= 1，当前值: {config.transcription.max_concurrent_tasks}")

        # 打印警告
        for warning in warnings:
            if not is_worker:
                logger.warning(f"⚠️  {warning}")

        # 打印错误并退出
        if errors:
            if not is_worker:
                logger.error("配置验证失败，发现以下错误:")
                for error in errors:
                    logger.error(f"❌ {error}")
                logger.error("=" * 60)
                logger.error("请检查 .env 文件和 config.json 配置，修正后重新启动")
            sys.exit(1)

        # 验证通过的日志延迟到 print_config() 中输出，以保证格式统一

    def print_config(self):
        """打印配置信息（敏感信息脱敏）- 应在 setup_logger() 之后调用"""
        # Worker 模式下跳过配置打印
        is_worker = os.getenv('FUNASR_WORKER_MODE') == '1'
        if is_worker:
            return

        config = self

        # 首先输出配置验证结果
        logger.info("=" * 60)
        logger.info("✅ 配置验证通过")
        logger.info("=" * 60)
        logger.info("当前配置:")
        logger.info("-" * 60)

        # 服务器配置
        logger.info(f"[服务器]")
        logger.info(f"  地址: {config.server.host}:{config.server.port}")
        logger.info(f"  最大连接数: {config.server.max_connections}")
        logger.info(f"  最大文件大小: {config.server.max_file_size_mb} MB")
        logger.info(f"  上传目录: {config.server.upload_dir}")
        logger.info(f"  临时目录: {config.server.temp_dir}")

        # FunASR 配置
        logger.info(f"[FunASR]")
        logger.info(f"  模型: {config.funasr.model} ({config.funasr.model_revision})")
        logger.info(f"  模型目录: {config.funasr.model_dir}")
        logger.info(f"  设备: {config.funasr.device}")
        logger.info(f"  设备优先级: {config.funasr.device_priority}")
        logger.info(f"  CPU 线程数: {config.funasr.ncpu}")
        logger.info(f"  批处理大小: {config.funasr.batch_size_s}s")

        # 转录配置
        logger.info(f"[转录]")
        logger.info(f"  最大并发任务: {config.transcription.max_concurrent_tasks}")
        logger.info(f"  任务超时: {config.transcription.task_timeout_minutes} 分钟")
        logger.info(f"  缓存启用: {config.transcription.cache_enabled}")

        # 通知配置
        logger.info(f"[通知]")
        logger.info(f"  启用状态: {config.notification.enabled}")
        if config.notification.enabled and config.notification.webhook_url:
            # 脱敏显示 webhook URL (仅显示前 30 字符)
            masked_url = config.notification.webhook_url[:30] + "..." if len(config.notification.webhook_url) > 30 else config.notification.webhook_url
            logger.info(f"  Webhook URL: {masked_url}")

        # 认证配置
        logger.info(f"[认证]")
        logger.info(f"  启用状态: {config.auth.enabled}")
        if config.auth.enabled:
            logger.info(f"  密钥: ***hidden***")
            logger.info(f"  算法: {config.auth.algorithm}")

        # 日志配置
        logger.info(f"[日志]")
        logger.info(f"  级别: {config.logging.level}")
        logger.info(f"  目录: {config.logging.log_dir}")

        logger.info("=" * 60)

    def setup_directories(self):
        """创建必要的目录"""
        is_worker = os.getenv('FUNASR_WORKER_MODE') == '1'

        directories = [
            self.server.temp_dir,
            self.server.upload_dir,
            self.funasr.model_dir,
            Path(self.database.path).parent,
            self.logging.log_dir
        ]

        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                # 目录创建成功时不输出日志,避免影响日志系统初始化
            except Exception as e:
                logger.error(f"创建目录失败 {directory}: {e}")
                sys.exit(1)


# 全局配置实例
config = Config.load_from_file()
config.setup_directories()
