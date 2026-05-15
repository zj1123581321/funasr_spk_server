#!/usr/bin/env python3
"""
安全的单次 batch_size 测试
避免内存泄漏，只测试一个 batch_size 值
"""

import os
import sys
from pathlib import Path

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


def main():
    """主函数"""

    # 检查参数
    if len(sys.argv) < 3:
        logger.error("请提供音频文件路径和 batch_size 作为参数")
        logger.info("用法: python test_single_batch_size.py <audio_file_path> <batch_size_s>")
        logger.info("示例: python test_single_batch_size.py audio.mp3 100")
        return

    audio_path = sys.argv[1]
    batch_size = int(sys.argv[2])

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
        logger.info(f"预计 batch 数: {int(duration / batch_size) + 1}")
    except Exception as e:
        logger.warning(f"无法获取音频时长: {e}")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"测试 batch_size_s = {batch_size}")
    logger.info("=" * 80)

    # 临时设置环境变量
    os.environ["FUNASR_BATCH_SIZE_S"] = str(batch_size)

    # 加载配置
    from src.core.config import config
    from src.core.device_manager import DeviceManager

    logger.info(f"配置: batch_size_s = {config.funasr.batch_size_s}")
    logger.info(f"配置: device = {config.funasr.device}")

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

        # 初始化模型
        logger.info("初始化 FunASR 模型...")
        from funasr import AutoModel

        model = AutoModel(
            model=config.funasr.model,
            model_revision=config.funasr.model_revision,
            device=device,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,  # 禁用更新检查
        )

        # 执行转录
        logger.info(f"开始转录（batch_size_s={batch_size}）...")
        logger.info("提示: 如果内存开始飙升，请立即 Ctrl+C 终止")
        logger.info("")

        import time
        start_time = time.time()

        result = model.generate(
            input=audio_path,
            batch_size_s=batch_size,
            hotword="",
        )

        elapsed_time = time.time() - start_time

        # 检查结果
        if result and len(result) > 0:
            logger.success(f"✅ 成功! 耗时: {elapsed_time:.1f} 秒")

            # 统计结果
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, dict):
                    text = first_result.get("text", "")
                    logger.info(f"   结果片段数: {len(result)}")
                    logger.info(f"   第一段文本: {text[:50]}...")

            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ 测试成功！建议配置:")
            logger.info("=" * 80)
            logger.info(f"在 .env 文件中添加:")
            logger.info(f"  FUNASR_BATCH_SIZE_S={batch_size}")
            logger.info("")
            logger.info("或在 config.json 中修改:")
            logger.info(f'  "batch_size_s": {batch_size}')
            logger.info("=" * 80)

        else:
            logger.error("❌ 失败: 结果为空")

    except KeyboardInterrupt:
        logger.warning("\n测试被用户中断")
        logger.warning("如果内存已经飙升，说明此 batch_size 仍然太大")
        logger.info("建议尝试更小的值，或使用 CPU 模式")

    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        logger.error(f"   错误类型: {type(e).__name__}")

        if "buffer" in str(e).lower() or "memory" in str(e).lower():
            logger.error("")
            logger.error("检测到内存/缓冲区错误!")
            logger.error("这是 MPS 在处理长音频时的已知问题")
            logger.error("")
            logger.error("建议:")
            logger.error("1. 尝试更小的 batch_size (例如 50)")
            logger.error("2. 或者对长音频使用 CPU 模式:")
            logger.error("   - 在 .env 中设置: FUNASR_DEVICE=cpu")


if __name__ == "__main__":
    main()
