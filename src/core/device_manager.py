"""
设备管理器 - 统一管理加速设备的检测、选择和配置

支持的设备类型：
- MPS (Metal Performance Shaders) - Apple Silicon GPU
- CPU - 通用 CPU 模式

未来可扩展：CUDA, ROCm, XPU 等
"""
import os
from typing import List, Dict, Optional
from loguru import logger
import torch


class DeviceManager:
    """统一设备管理器 - 支持多种加速后端"""

    # 当前支持的设备类型（按优先级排序）
    SUPPORTED_DEVICES = ["mps", "cpu"]

    # 默认设备选择优先级
    DEFAULT_PRIORITY = ["mps", "cpu"]

    @staticmethod
    def detect_available_devices() -> List[str]:
        """
        检测当前环境可用的设备

        Returns:
            可用设备列表，如 ["mps", "cpu"] 或 ["cpu"]
        """
        available = []

        # 检测 MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            available.append("mps")
            logger.debug("检测到 MPS 设备可用")

        # CPU 始终可用
        available.append("cpu")

        logger.info(f"可用设备: {', '.join(available)}")
        return available

    @staticmethod
    def select_device(config: dict) -> str:
        """
        根据配置选择最优设备

        Args:
            config: 配置字典，包含 funasr.device 和可选的 funasr.device_priority

        Returns:
            选定的设备名称 ("mps", "cpu" 等)
        """
        funasr_config = config.get("funasr", {})
        device_config = funasr_config.get("device", "auto")
        fallback_on_error = funasr_config.get("fallback_on_error", True)

        # 获取可用设备列表
        available_devices = DeviceManager.detect_available_devices()

        # 情况 1: auto 模式 - 自动选择最优设备
        if device_config == "auto":
            priority = funasr_config.get("device_priority", DeviceManager.DEFAULT_PRIORITY)

            for device in priority:
                if device in available_devices:
                    logger.info(f"自动选择设备: {device} (优先级: {priority})")
                    return device

            # 理论上不会到这里，因为 CPU 始终可用
            logger.warning("未找到可用设备，回退到 CPU")
            return "cpu"

        # 情况 2: 指定设备模式
        else:
            # 检查设备是否被支持
            if device_config not in DeviceManager.SUPPORTED_DEVICES:
                logger.warning(
                    f"不支持的设备类型: {device_config}，支持的设备: {DeviceManager.SUPPORTED_DEVICES}"
                )
                if fallback_on_error:
                    logger.warning("回退到 CPU 模式")
                    return "cpu"
                else:
                    raise ValueError(f"不支持的设备类型: {device_config}")

            # 检查设备是否可用
            if device_config not in available_devices:
                logger.warning(f"指定的设备 {device_config} 不可用，可用设备: {available_devices}")

                if fallback_on_error:
                    logger.warning("回退到 CPU 模式")
                    return "cpu"
                else:
                    raise RuntimeError(f"设备 {device_config} 不可用")

            logger.info(f"使用指定设备: {device_config}")
            return device_config

    @staticmethod
    def apply_patches(device: str) -> None:
        """
        为指定设备应用必要的补丁

        Args:
            device: 设备名称 ("mps", "cpu" 等)
        """
        if device == "mps":
            # 动态导入 MPS 补丁模块
            try:
                from src.core.patches.mps_patch import apply_mps_patch
                apply_mps_patch()
                logger.success("✅ MPS 设备补丁已应用")
            except Exception as e:
                logger.error(f"应用 MPS 补丁失败: {e}")
                raise

        elif device == "cpu":
            # CPU 模式不需要补丁
            logger.debug("CPU 模式，无需应用补丁")

        else:
            logger.warning(f"未知设备类型: {device}，跳过补丁应用")

    @staticmethod
    def get_device_info(device: str) -> Dict[str, any]:
        """
        获取设备详细信息（用于日志和监控）

        Args:
            device: 设备名称

        Returns:
            设备信息字典
        """
        info = {
            "device": device,
            "device_name": DeviceManager._get_device_display_name(device),
            "torch_version": torch.__version__,
        }

        if device == "mps":
            info.update({
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
            })

        return info

    @staticmethod
    def _get_device_display_name(device: str) -> str:
        """获取设备的显示名称"""
        display_names = {
            "mps": "Apple Silicon GPU (MPS)",
            "cpu": "CPU",
            # 未来扩展：
            # "cuda": "NVIDIA GPU (CUDA)",
            # "rocm": "AMD GPU (ROCm)",
            # "xpu": "Intel GPU (XPU)",
        }
        return display_names.get(device, device.upper())

    @staticmethod
    def log_device_selection(device: str, config: dict) -> None:
        """
        记录设备选择信息（用于调试和监控）

        Args:
            device: 选定的设备
            config: 配置信息
        """
        info = DeviceManager.get_device_info(device)

        logger.info("="*60)
        logger.info("设备配置信息")
        logger.info("="*60)
        logger.info(f"选定设备: {info['device_name']}")
        logger.info(f"配置模式: {config.get('funasr', {}).get('device', 'auto')}")
        logger.info(f"PyTorch 版本: {info['torch_version']}")

        if device == "mps":
            logger.info(f"MPS 可用: {info['mps_available']}")
            logger.info(f"MPS 已构建: {info['mps_built']}")

        logger.info("="*60)
