"""
MPS 并发问题诊断测试脚本

此脚本用于复现和诊断 MPS 多进程并发时的错误：
"Dimension specified as -1 but tensor has no dimensions"

测试策略：
1. 并发提交多个音频转录任务
2. 监控每个 worker 的运行状态
3. 收集详细的错误信息和堆栈
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.file_based_process_pool import FileBasedProcessPool
from loguru import logger

# 配置详细日志
logger.remove()
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
)
logger.add(
    "logs/mps_concurrent_test_{time}.log",
    level="DEBUG",
    rotation="10 MB"
)


async def test_concurrent_transcription():
    """测试并发转录"""

    # 测试文件
    test_files = [
        "temp/samples/podcast_en.mp3",
        "temp/samples/test.m4a"
    ]

    # 检查文件是否存在
    for f in test_files:
        if not os.path.exists(f):
            logger.error(f"测试文件不存在: {f}")
            return

    logger.info("="*80)
    logger.info("MPS 并发诊断测试开始")
    logger.info("="*80)
    logger.info(f"测试文件: {test_files}")
    logger.info(f"并发数: {len(test_files)}")

    # 创建进程池
    pool = FileBasedProcessPool(pool_size=2)  # 使用 2 个 worker

    try:
        # 初始化进程池
        logger.info("初始化进程池...")
        await pool.initialize()
        logger.success("进程池初始化完成")

        # 定义任务处理函数
        async def process_file(file_path: str, task_num: int):
            """处理单个文件"""
            logger.info(f"[任务{task_num}] 开始处理: {file_path}")
            start_time = time.time()

            try:
                result = await pool.generate_with_pool(
                    audio_path=file_path,
                    batch_size_s=500,
                    hotword='',
                    use_pickle=True
                )

                elapsed = time.time() - start_time
                logger.success(f"[任务{task_num}] 完成! 耗时: {elapsed:.2f}秒")
                logger.info(f"[任务{task_num}] 结果片段数: {len(result)}")
                return {"success": True, "file": file_path, "time": elapsed, "segments": len(result)}

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[任务{task_num}] 失败! 错误: {e}")
                logger.error(f"[任务{task_num}] 错误类型: {type(e).__name__}")
                import traceback
                logger.error(f"[任务{task_num}] 堆栈:\n{traceback.format_exc()}")
                return {"success": False, "file": file_path, "error": str(e), "time": elapsed}

        # 测试 1: 串行执行（基准测试）
        logger.info("\n" + "="*80)
        logger.info("测试 1: 串行执行（基准测试）")
        logger.info("="*80)

        serial_results = []
        for i, file_path in enumerate(test_files):
            result = await process_file(file_path, i+1)
            serial_results.append(result)
            await asyncio.sleep(2)  # 任务间隔

        logger.info("\n串行测试结果:")
        for i, r in enumerate(serial_results):
            logger.info(f"  任务{i+1}: {'成功' if r['success'] else '失败'} - {r.get('time', 0):.2f}秒")

        # 等待一段时间，确保 worker 状态清理
        logger.info("\n等待 5 秒...")
        await asyncio.sleep(5)

        # 测试 2: 并发执行（压力测试）
        logger.info("\n" + "="*80)
        logger.info("测试 2: 并发执行（压力测试）")
        logger.info("="*80)

        # 同时提交所有任务
        tasks = [process_file(f, i+1) for i, f in enumerate(test_files)]
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("\n并发测试结果:")
        for i, r in enumerate(concurrent_results):
            if isinstance(r, Exception):
                logger.error(f"  任务{i+1}: 异常 - {r}")
            elif isinstance(r, dict):
                logger.info(f"  任务{i+1}: {'成功' if r['success'] else '失败'} - {r.get('time', 0):.2f}秒")
            else:
                logger.error(f"  任务{i+1}: 未知结果类型 - {type(r)}")

        # 测试 3: 重复并发测试（稳定性测试）
        logger.info("\n" + "="*80)
        logger.info("测试 3: 重复并发测试（3 轮）")
        logger.info("="*80)

        for round_num in range(3):
            logger.info(f"\n--- 第 {round_num + 1} 轮 ---")
            await asyncio.sleep(3)  # 轮次间隔

            tasks = [process_file(f, i+1) for i, f in enumerate(test_files)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            logger.info(f"第 {round_num + 1} 轮: {success_count}/{len(test_files)} 成功")

    except Exception as e:
        logger.error(f"测试过程异常: {e}")
        import traceback
        logger.error(f"堆栈:\n{traceback.format_exc()}")

    finally:
        # 清理进程池
        logger.info("\n清理进程池...")
        pool.cleanup()
        logger.info("进程池已清理")

    logger.info("\n" + "="*80)
    logger.info("MPS 并发诊断测试结束")
    logger.info("="*80)


if __name__ == "__main__":
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)

    # 运行测试
    asyncio.run(test_concurrent_transcription())
