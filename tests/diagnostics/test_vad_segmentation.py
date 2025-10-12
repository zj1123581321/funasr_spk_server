#!/usr/bin/env python3
"""
测试 VAD 分割策略：使用 max_single_segment_time 限制音频段长度
理论：通过 VAD 将长音频分割成短片段，避免 MPS 缓冲区 bug
"""

import os
import sys
from pathlib import Path
import time

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


def test_with_vad_segmentation(audio_path: str, max_segment_time_ms: int):
    """
    测试使用 VAD 分割策略

    Args:
        audio_path: 音频文件路径
        max_segment_time_ms: VAD 最大分割时长（毫秒）
    """

    logger.info("=" * 80)
    logger.info(f"测试 VAD 分割策略: max_single_segment_time = {max_segment_time_ms} ms ({max_segment_time_ms/1000}s)")
    logger.info("=" * 80)

    from src.core.config import config
    from src.core.device_manager import DeviceManager

    # 将 Config 对象转换为字典
    config_dict = {
        "funasr": {
            "device": config.funasr.device,
            "device_priority": config.funasr.device_priority,
            "fallback_on_error": config.funasr.fallback_on_error,
        }
    }

    try:
        # 选择设备
        device = DeviceManager.select_device(config_dict)
        DeviceManager.apply_patches(device)
        DeviceManager.log_device_selection(device, config_dict)

        # 初始化模型（带 VAD）
        logger.info("初始化 FunASR 模型（启用 VAD 分割）...")
        from funasr import AutoModel

        model = AutoModel(
            model=config.funasr.model,
            model_revision=config.funasr.model_revision,
            vad_model=config.funasr.vad_model,
            vad_model_revision=config.funasr.vad_model_revision,
            punc_model=config.funasr.punc_model,
            punc_model_revision=config.funasr.punc_model_revision,
            spk_model=config.funasr.spk_model,
            spk_model_revision=config.funasr.spk_model_revision,
            device=device,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
        )

        logger.info(f"VAD 配置: max_single_segment_time = {max_segment_time_ms} ms")
        logger.info(f"batch_size_s: {config.funasr.batch_size_s}")

        # 执行转录
        logger.info("开始转录...")
        logger.info("理论：VAD 会将长音频分割成短片段，每个片段独立处理")
        logger.info("")

        start_time = time.time()

        result = model.generate(
            input=audio_path,
            batch_size_s=config.funasr.batch_size_s,
            hotword="",
            # VAD 配置
            vad_kwargs={
                "max_single_segment_time": max_segment_time_ms
            }
        )

        elapsed_time = time.time() - start_time

        # 检查结果
        if result and len(result) > 0:
            logger.success(f"✅ 成功! 耗时: {elapsed_time:.1f} 秒")
            logger.info("")

            # 统计结果
            first_result = result[0]
            if isinstance(first_result, dict):
                text = first_result.get("text", "")
                logger.info(f"结果统计:")
                logger.info(f"  - 总结果数: {len(result)}")
                logger.info(f"  - 第一段文本: {text[:80]}...")

                # 统计说话人
                speakers = set()
                for item in result:
                    if isinstance(item, dict) and "spk" in item:
                        speakers.add(item["spk"])

                if speakers:
                    logger.info(f"  - 说话人数: {len(speakers)}")
                    logger.info(f"  - 说话人列表: {sorted(speakers)}")

            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ VAD 分割策略成功！")
            logger.info("=" * 80)
            logger.info("建议配置:")
            logger.info("")
            logger.info("在初始化模型时添加 vad_kwargs:")
            logger.info("```python")
            logger.info("model = AutoModel(")
            logger.info("    model=config.funasr.model,")
            logger.info("    vad_model=config.funasr.vad_model,")
            logger.info("    vad_kwargs={")
            logger.info(f"        'max_single_segment_time': {max_segment_time_ms}")
            logger.info("    },")
            logger.info("    device=device,")
            logger.info(")")
            logger.info("```")
            logger.info("=" * 80)

            return True

        else:
            logger.error("❌ 失败: 结果为空")
            return False

    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        logger.error(f"   错误类型: {type(e).__name__}")

        if "buffer" in str(e).lower():
            logger.error("")
            logger.error("仍然出现缓冲区错误!")
            logger.error("这说明 VAD 分割不能完全解决 MPS bug")
            logger.error("")
            logger.error("可能的原因:")
            logger.error("1. batch_size_s 仍然太大，即使音频被分割")
            logger.error("2. MPS bug 发生在其他地方（不仅仅是音频加载）")
            logger.error("")
            logger.error("建议:")
            logger.error("1. 尝试更小的 max_single_segment_time (例如 15000)")
            logger.error("2. 同时降低 batch_size_s")
            logger.error("3. 或使用 CPU 模式")

        return False


def main():
    """主函数"""

    # 检查参数
    if len(sys.argv) < 2:
        logger.error("请提供音频文件路径作为参数")
        logger.info("用法: python test_vad_segmentation.py <audio_file_path> [max_segment_time_ms]")
        logger.info("示例: python test_vad_segmentation.py audio.mp3 30000")
        return

    audio_path = sys.argv[1]
    max_segment_time_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 30000

    if not os.path.exists(audio_path):
        logger.error(f"音频文件不存在: {audio_path}")
        return

    # 显示音频信息
    try:
        import librosa
        try:
            duration = librosa.get_duration(path=audio_path)
        except TypeError:
            duration = librosa.get_duration(filename=audio_path)

        logger.info(f"音频文件: {Path(audio_path).name}")
        logger.info(f"音频时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")

        # 计算预计分割数
        num_segments = int(duration / (max_segment_time_ms / 1000)) + 1
        logger.info(f"预计 VAD 分割段数: ~{num_segments} 段（每段最多 {max_segment_time_ms/1000}s）")
    except Exception as e:
        logger.warning(f"无法获取音频时长: {e}")

    logger.info("")

    # 测试
    success = test_with_vad_segmentation(audio_path, max_segment_time_ms)

    if not success:
        logger.info("")
        logger.info("=" * 80)
        logger.info("测试其他配置")
        logger.info("=" * 80)

        # 提供备选方案
        logger.info("如果 VAD 分割失败，可以尝试:")
        logger.info("")
        logger.info("1. 更小的分割时长:")
        logger.info(f"   python {sys.argv[0]} {audio_path} 15000")
        logger.info("")
        logger.info("2. 同时降低 batch_size_s:")
        logger.info("   在 .env 中添加: FUNASR_BATCH_SIZE_S=100")
        logger.info("")
        logger.info("3. 使用 CPU 模式（最稳定）:")
        logger.info("   FUNASR_DEVICE=cpu python {sys.argv[0]} {audio_path}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
