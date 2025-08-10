"""
设备信息获取模块
"""
import socket
import platform
from typing import Dict, Optional
from loguru import logger


def get_local_ip() -> str:
    """获取本机IP地址"""
    try:
        # 创建一个UDP socket连接到外部地址(不实际发送数据)
        # 这种方法可以获取到实际使用的网卡IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 连接到一个外部地址（这里使用DNS服务器）
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            return ip
    except Exception as e:
        logger.warning(f"获取本机IP失败: {e}")
        try:
            # 备用方案：获取主机名对应的IP
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception as e2:
            logger.error(f"获取IP地址失败: {e2}")
            return "127.0.0.1"


def get_hostname() -> str:
    """获取主机名"""
    try:
        return socket.gethostname()
    except Exception as e:
        logger.error(f"获取主机名失败: {e}")
        return "unknown"


def get_device_info() -> Dict[str, str]:
    """获取设备信息"""
    return {
        "hostname": get_hostname(),
        "ip": get_local_ip(),
        "platform": platform.system(),
        "platform_version": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine()
    }


def get_device_identifier() -> str:
    """获取设备标识符（用于通知）"""
    hostname = get_hostname()
    ip = get_local_ip()
    return f"{hostname} ({ip})"


# 缓存设备信息，避免重复获取
_cached_device_info: Optional[Dict[str, str]] = None
_cached_device_identifier: Optional[str] = None


def get_cached_device_info() -> Dict[str, str]:
    """获取缓存的设备信息"""
    global _cached_device_info
    if _cached_device_info is None:
        _cached_device_info = get_device_info()
        logger.info(f"设备信息: {_cached_device_info}")
    return _cached_device_info


def get_cached_device_identifier() -> str:
    """获取缓存的设备标识符"""
    global _cached_device_identifier
    if _cached_device_identifier is None:
        _cached_device_identifier = get_device_identifier()
        logger.info(f"设备标识: {_cached_device_identifier}")
    return _cached_device_identifier