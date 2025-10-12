#!/usr/bin/env python3
"""
MPS 内存和张量诊断脚本
用于追踪长音频处理时的内存使用和张量维度变化
"""

import torch
import sys
import os
from pathlib import Path
import traceback
import psutil
from loguru import logger

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.device_manager import DeviceManager
from src.core.config import config

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG"
)


class MPSMemoryTracer:
    """MPS 内存追踪器"""

    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_events = []

    def log_memory(self, stage: str):
        """记录当前内存使用"""
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024  # MB

        if torch.backends.mps.is_available():
            # MPS 与系统内存共享，我们主要关注进程内存
            self.current_memory = rss_mb
            self.peak_memory = max(self.peak_memory, rss_mb)

        event = {
            'stage': stage,
            'memory_mb': rss_mb,
            'peak_mb': self.peak_memory
        }
        self.memory_events.append(event)
        logger.debug(f"[内存] {stage}: {rss_mb:.1f} MB (峰值: {self.peak_memory:.1f} MB)")

    def print_summary(self):
        """打印内存使用总结"""
        logger.info("\n" + "=" * 80)
        logger.info("内存使用总结")
        logger.info("=" * 80)
        for event in self.memory_events:
            logger.info(f"{event['stage']:30s} | {event['memory_mb']:8.1f} MB | 峰值: {event['peak_mb']:8.1f} MB")
        logger.info("=" * 80)


class TensorDimensionTracer:
    """张量维度追踪器 - 通过 Hook 监控张量操作"""

    def __init__(self):
        self.tensor_events = []
        self.error_tensors = []

    def log_tensor(self, stage: str, tensor, name: str = "tensor"):
        """记录张量信息"""
        if tensor is None:
            event = {
                'stage': stage,
                'name': name,
                'is_none': True,
                'shape': None,
                'device': None,
                'dtype': None
            }
            logger.warning(f"[张量] {stage} - {name}: None!")
        elif isinstance(tensor, torch.Tensor):
            event = {
                'stage': stage,
                'name': name,
                'is_none': False,
                'shape': tuple(tensor.shape),
                'device': str(tensor.device),
                'dtype': str(tensor.dtype),
                'numel': tensor.numel()
            }
            logger.debug(
                f"[张量] {stage} - {name}: "
                f"shape={tensor.shape}, device={tensor.device}, "
                f"dtype={tensor.dtype}, numel={tensor.numel()}"
            )

            # 检查空张量
            if tensor.numel() == 0 or len(tensor.shape) == 0:
                logger.error(f"[张量异常] {stage} - {name}: 空张量或无维度! shape={tensor.shape}")
                self.error_tensors.append(event)
        else:
            event = {
                'stage': stage,
                'name': name,
                'type': type(tensor).__name__
            }
            logger.debug(f"[张量] {stage} - {name}: type={type(tensor).__name__}")

        self.tensor_events.append(event)

    def print_summary(self):
        """打印张量追踪总结"""
        logger.info("\n" + "=" * 80)
        logger.info("张量维度追踪总结")
        logger.info("=" * 80)
        logger.info(f"总共追踪了 {len(self.tensor_events)} 个张量操作")

        if self.error_tensors:
            logger.error(f"\n发现 {len(self.error_tensors)} 个异常张量:")
            for event in self.error_tensors:
                logger.error(f"  - {event['stage']} | {event['name']} | shape={event['shape']}")
        else:
            logger.success("未发现异常张量")
        logger.info("=" * 80)


def patch_funasr_with_tracing(tensor_tracer: TensorDimensionTracer):
    """
    为 FunASR 添加张量追踪
    通过 monkey patch 在关键位置插入追踪代码
    """
    try:
        from funasr import AutoModel

        # 保存原始的 generate 方法
        original_generate = AutoModel.generate

        def traced_generate(self, input, **kwargs):
            """带追踪的 generate 方法"""
            tensor_tracer.log_tensor("generate_input", input, "input")

            try:
                result = original_generate(self, input, **kwargs)
                tensor_tracer.log_tensor("generate_output", result, "result")
                return result
            except Exception as e:
                logger.error(f"generate 方法出错: {e}")
                tensor_tracer.log_tensor("generate_error", input, "input_on_error")
                raise

        # 应用 patch
        AutoModel.generate = traced_generate
        logger.success("已为 FunASR 添加张量追踪")

    except Exception as e:
        logger.warning(f"无法为 FunASR 添加张量追踪: {e}")


def test_audio_with_tracing(audio_path: str, config_obj):
    """测试音频处理并追踪内存和张量"""

    memory_tracer = MPSMemoryTracer()
    tensor_tracer = TensorDimensionTracer()

    logger.info("=" * 80)
    logger.info("MPS 内存和张量诊断测试")
    logger.info("=" * 80)
    logger.info(f"音频文件: {audio_path}")

    try:
        # 1. 检测音频时长
        import librosa
        memory_tracer.log_memory("开始")

        # 使用 path 参数（新版本）或 filename 参数（旧版本）
        try:
            duration = librosa.get_duration(path=audio_path)
        except TypeError:
            # 兼容旧版本 librosa
            duration = librosa.get_duration(filename=audio_path)
        logger.info(f"音频时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        memory_tracer.log_memory("音频加载后")

        # 2. 设置设备
        # 将 Config 对象转换为字典（DeviceManager 需要字典格式）
        config_dict = {
            "funasr": {
                "device": config_obj.funasr.device,
                "device_priority": config_obj.funasr.device_priority,
                "fallback_on_error": config_obj.funasr.fallback_on_error,
            }
        }

        device = DeviceManager.select_device(config_dict)
        DeviceManager.apply_patches(device)
        DeviceManager.log_device_selection(device, config_dict)
        memory_tracer.log_memory("设备初始化后")

        # 3. 为 FunASR 添加追踪
        patch_funasr_with_tracing(tensor_tracer)

        # 4. 初始化模型
        logger.info("初始化 FunASR 模型...")
        from funasr import AutoModel

        model = AutoModel(
            model=config_obj.funasr.model,
            model_revision=config_obj.funasr.model_revision,
            device=device,
            disable_pbar=True,
            disable_log=False,
        )
        memory_tracer.log_memory("模型加载后")

        # 5. 执行转录
        logger.info("开始转录...")
        logger.info(f"batch_size_s: {config_obj.funasr.batch_size_s}")

        try:
            result = model.generate(
                input=audio_path,
                batch_size_s=config_obj.funasr.batch_size_s,
                hotword="",
            )
            memory_tracer.log_memory("转录完成后")

            # 检查结果
            tensor_tracer.log_tensor("final_result", result, "transcription_result")

            if result:
                logger.success(f"转录成功! 结果类型: {type(result)}")
                if isinstance(result, list) and len(result) > 0:
                    logger.info(f"结果数量: {len(result)}")
                    # 检查第一个结果
                    first_result = result[0]
                    logger.info(f"第一个结果键: {first_result.keys() if isinstance(first_result, dict) else 'N/A'}")
            else:
                logger.error("转录结果为空!")

        except Exception as e:
            logger.error(f"转录过程出错: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            memory_tracer.log_memory("转录出错时")
            raise

    except Exception as e:
        logger.error(f"测试失败: {e}")
        logger.error(traceback.format_exc())

    finally:
        # 打印总结
        memory_tracer.print_summary()
        tensor_tracer.print_summary()


def main():
    """主函数"""

    # 检查 MPS 可用性
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"MPS 可用: {torch.backends.mps.is_available()}")
    logger.info(f"MPS 已构建: {torch.backends.mps.is_built()}")

    if not torch.backends.mps.is_available():
        logger.error("MPS 不可用，无法进行测试")
        return

    # 使用全局配置实例
    logger.info(f"使用配置: batch_size_s = {config.funasr.batch_size_s}")

    # 测试音频文件（用户需要提供长音频路径）
    if len(sys.argv) < 2:
        logger.error("请提供音频文件路径作为参数")
        logger.info("用法: python test_mps_memory_trace.py <audio_file_path>")
        return

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        logger.error(f"音频文件不存在: {audio_path}")
        return

    # 运行测试
    test_audio_with_tracing(audio_path, config)


if __name__ == "__main__":
    main()
