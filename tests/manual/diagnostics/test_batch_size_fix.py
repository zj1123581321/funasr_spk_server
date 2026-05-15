#!/usr/bin/env python3
"""
快速测试：降低 batch_size_s 是否能解决长音频问题

测试不同的 batch_size_s 值：500 (当前), 200, 100
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


def test_batch_size(audio_path: str, batch_size: int):
    """测试指定的 batch_size"""

    logger.info("=" * 80)
    logger.info(f"测试 batch_size_s = {batch_size}")
    logger.info("=" * 80)

    # 临时设置环境变量
    os.environ["FUNASR_BATCH_SIZE_S"] = str(batch_size)

    # 重新加载配置
    import importlib
    if 'src.core.config' in sys.modules:
        importlib.reload(sys.modules['src.core.config'])

    from src.core.config import config
    from src.core.device_manager import DeviceManager

    logger.info(f"配置已加载: batch_size_s = {config.funasr.batch_size_s}")

    try:
        # 将 Config 对象转换为字典（DeviceManager 需要字典格式）
        config_dict = {
            "funasr": {
                "device": config.funasr.device,
                "device_priority": config.funasr.device_priority,
                "fallback_on_error": config.funasr.fallback_on_error,
            }
        }

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
        )

        # 执行转录
        logger.info(f"开始转录（batch_size_s={batch_size}）...")
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

            return True
        else:
            logger.error("❌ 失败: 结果为空")
            return False

    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        logger.error(f"   错误类型: {type(e).__name__}")
        return False

    finally:
        # 清理模型
        if 'model' in locals():
            del model

        # 清理 FunASR 模块（强制重新初始化）
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('funasr')]
        for module in modules_to_remove:
            del sys.modules[module]


def main():
    """主函数"""

    # 检查参数
    if len(sys.argv) < 2:
        logger.error("请提供音频文件路径作为参数")
        logger.info("用法: python test_batch_size_fix.py <audio_file_path>")
        return

    audio_path = sys.argv[1]
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
    except Exception as e:
        logger.warning(f"无法获取音频时长: {e}")

    logger.info("")

    # 测试不同的 batch_size
    batch_sizes = [500, 200, 100]  # 从大到小测试
    results = {}

    for batch_size in batch_sizes:
        success = test_batch_size(audio_path, batch_size)
        results[batch_size] = success

        logger.info("")

        if success:
            logger.success(f"✅ batch_size_s = {batch_size} 成功！")
            logger.info("")
            logger.info("=" * 80)
            logger.info("建议:")
            logger.info(f"1. 在 .env 文件中添加: FUNASR_BATCH_SIZE_S={batch_size}")
            logger.info("2. 或者在 config.json 中修改: \"batch_size_s\": " + str(batch_size))
            logger.info("=" * 80)
            break
        else:
            logger.warning(f"⚠️  batch_size_s = {batch_size} 失败，尝试更小的值...")
            logger.info("")

    # 打印总结
    logger.info("")
    logger.info("=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    for batch_size, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"  batch_size_s = {batch_size:3d} : {status}")

    if not any(results.values()):
        logger.error("")
        logger.error("所有 batch_size 测试都失败了！")
        logger.error("建议:")
        logger.error("1. 运行深度诊断: python tests/diagnostics/test_mps_memory_trace.py <audio_path>")
        logger.error("2. 或者对长音频使用 CPU 模式")
        logger.error("   - 在 .env 中设置: FUNASR_DEVICE=cpu")


if __name__ == "__main__":
    main()
