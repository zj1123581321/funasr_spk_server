"""
日志管理模块
"""
import sys
from pathlib import Path
from loguru import logger
from src.core.config import config


def setup_logger():
    """设置日志系统"""
    # 移除默认的处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format=config.logging.format,
        level=config.logging.level,
        colorize=True
    )
    
    # 添加文件输出
    log_file = Path(config.logging.log_dir) / "funasr_server.log"
    logger.add(
        log_file,
        format=config.logging.format,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        encoding="utf-8"
    )
    
    # 添加错误日志文件
    error_log_file = Path(config.logging.log_dir) / "error.log"
    logger.add(
        error_log_file,
        format=config.logging.format,
        level="ERROR",
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        encoding="utf-8"
    )
    
    logger.info("日志系统初始化完成")


# 初始化日志
setup_logger()