"""
平台相关工具模块
"""
import os
import sys
import platform
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def get_platform_info() -> Dict[str, Any]:
    """获取平台信息"""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "is_docker": is_running_in_docker(),
        "is_macos": sys.platform == "darwin",
        "is_linux": sys.platform.startswith("linux"),
        "is_windows": sys.platform == "win32",
    }


def is_running_in_docker() -> bool:
    """检测是否在 Docker 容器中运行"""
    # 检查 /.dockerenv 文件
    if Path("/.dockerenv").exists():
        return True
    
    # 检查 /proc/1/cgroup 文件
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read()
    except (FileNotFoundError, PermissionError):
        pass
    
    # 检查环境变量
    return os.getenv("DOCKER_CONTAINER") == "true"


def get_optimal_worker_count() -> int:
    """根据平台获取最优工作线程数"""
    cpu_count = os.cpu_count() or 4
    platform_info = get_platform_info()
    
    if platform_info["is_docker"]:
        # Docker 环境中可能有 CPU 限制
        return min(cpu_count, 4)
    elif platform_info["is_macos"]:
        # macOS 可能有内存压力管理
        return min(cpu_count, 6)
    else:
        # Linux 环境通常可以使用更多线程
        return min(cpu_count, 8)


def setup_platform_specific_env():
    """设置平台特定的环境变量"""
    platform_info = get_platform_info()
    
    if platform_info["is_macos"]:
        # macOS 特定优化
        if "M1" in platform_info["machine"] or "M2" in platform_info["machine"]:
            # Apple Silicon 优化
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        
        # 设置最大线程数
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        
    elif platform_info["is_linux"]:
        # Linux 特定优化
        os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
        os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "65536")
        
    elif platform_info["is_windows"]:
        # Windows 特定优化
        os.environ.setdefault("OMP_NUM_THREADS", "4")
    
    # Docker 环境优化
    if platform_info["is_docker"]:
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.environ.setdefault("PYTHONPATH", "/app")


def get_platform_specific_config() -> Dict[str, Any]:
    """获取平台特定的配置调整"""
    platform_info = get_platform_info()
    config_overrides = {}
    
    if platform_info["is_macos"]:
        # macOS 可能需要减少并发数以避免内存压力
        config_overrides.update({
            "transcription": {
                "max_concurrent_tasks": min(get_optimal_worker_count(), 4)
            }
        })
        
    elif platform_info["is_docker"]:
        # Docker 环境配置
        config_overrides.update({
            "server": {
                "host": "0.0.0.0"  # Docker 需要绑定到所有接口
            }
        })
    
    return config_overrides


def check_system_requirements() -> bool:
    """检查系统要求"""
    platform_info = get_platform_info()
    issues = []
    
    # 检查 Python 版本
    python_version = tuple(map(int, platform_info["python_version"].split(".")))
    if python_version < (3, 10):
        issues.append(f"Python 版本过低: {platform_info['python_version']}，需要 3.10+")
    
    # 检查 FFmpeg
    try:
        import ffmpeg
        ffmpeg.probe("dummy", v=None)  # 这会失败但会检查 ffmpeg 是否可用
    except ffmpeg.Error:
        pass  # 正常，因为文件不存在
    except FileNotFoundError:
        issues.append("FFmpeg 未安装或不在 PATH 中")
    
    # 检查可用内存（如果可能）
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            issues.append(f"内存不足: {memory_gb:.1f}GB，推荐 8GB+")
    except ImportError:
        logger.warning("无法检查内存，psutil 未安装")
    
    # 记录检查结果
    if issues:
        logger.warning("系统要求检查发现以下问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("系统要求检查通过")
        return True


def log_platform_info():
    """记录平台信息"""
    platform_info = get_platform_info()
    
    logger.info("平台信息:")
    logger.info(f"  系统: {platform_info['system']} {platform_info['machine']}")
    logger.info(f"  Python: {platform_info['python_version']}")
    logger.info(f"  Docker: {'是' if platform_info['is_docker'] else '否'}")
    logger.info(f"  优化工作线程数: {get_optimal_worker_count()}")


# 在模块加载时自动设置环境变量
setup_platform_specific_env()