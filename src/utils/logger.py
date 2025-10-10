"""
日志管理模块
"""
import sys
import os
from pathlib import Path
from loguru import logger


def setup_logger():
    """设置日志系统 - 必须在 config 加载后调用"""
    # 延迟导入 config,避免循环依赖
    from src.core.config import config

    # 移除默认的处理器
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stdout,
        format=config.logging.format,
        level=config.logging.level,
        colorize=True
    )

    # 确保日志目录存在
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 添加文件输出
    log_file = log_dir / "funasr_server.log"
    logger.add(
        log_file,
        format=config.logging.format,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        encoding="utf-8"
    )

    # 添加错误日志文件
    error_log_file = log_dir / "error.log"
    logger.add(
        error_log_file,
        format=config.logging.format,
        level="ERROR",
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        encoding="utf-8"
    )

    logger.info("日志系统初始化完成")
    logger.info(f"日志级别: {config.logging.level}")
    logger.info(f"日志目录: {config.logging.log_dir}")