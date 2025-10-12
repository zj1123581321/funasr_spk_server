"""
MPS 6 Workers 并发压力测试 - 复现问题

使用 6 个真实音频文件 + 6 个 worker 进行压力测试
目标：复现 "Dimension specified as -1 but tensor has no dimensions" 错误
"""

import os
import sys
import asyncio
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.file_based_process_pool import FileBasedProcessPool
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)
logger.add(
    "logs/mps_6workers_test_{time}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
)


async def main():
    """测试 6 workers 并发"""

    # 6 个音频文件
    test_files = [
        "temp/samples/test_60s.wav",            # 1.8M - 最小
        "temp/samples/test_3min.wav",           # 5.5M
        "temp/samples/podcast_en.mp3",          # 13M
        "temp/samples/test.m4a",                # 15M
        "temp/samples/4-person-example.m4a",    # 35M
        "temp/samples/「月月轻松」救助流浪狗，到哪一步该收手？- EP237 - ULSUM RADIO - 小宇宙 - 听播客，上小宇宙.mp3",  # 120M - 最大
    ]

    # 检查文件存在
    for f in test_files:
        if not os.path.exists(f):
            logger.error(f"文件不存在: {f}")
            return

    logger.info("="*100)
    logger.info(f"MPS 6 Workers 并发压力测试 - 目标：复现并发错误")
    logger.info("="*100)
    logger.info(f"测试文件：")
    for i, f in enumerate(test_files, 1):
        size_mb = os.path.getsize(f) / 1024 / 1024
        logger.info(f"  {i}. {os.path.basename(f)} ({size_mb:.1f}MB)")

    # 创建 6 个 worker 的进程池
    pool = FileBasedProcessPool(pool_size=6)

    try:
        logger.info("\n初始化 6 个 MPS worker 进程...")
        await pool.initialize()
        logger.success("✓ 进程池初始化完成")

        # 定义任务处理函数
        async def process_file(file_path: str, task_id: int):
            file_name = os.path.basename(file_path)
            logger.info(f"[任务{task_id}] 开始处理: {file_name}")
            start = time.time()

            try:
                result = await pool.generate_with_pool(
                    audio_path=file_path,
                    batch_size_s=500,
                    use_pickle=True
                )
                elapsed = time.time() - start
                logger.success(f"[任务{task_id}] ✓ 成功 - {file_name} - {elapsed:.1f}秒")
                return {"success": True, "file": file_name, "time": elapsed}

            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[任务{task_id}] ✗ 失败 - {file_name} - {e}")
                return {"success": False, "file": file_name, "error": str(e), "time": elapsed}

        # 测试轮数
        total_rounds = 3
        all_results = []

        for round_num in range(total_rounds):
            logger.info(f"\n{'='*100}")
            logger.info(f"第 {round_num+1}/{total_rounds} 轮测试 - 6 个任务同时提交")
            logger.info(f"{'='*100}")

            # 同时提交所有 6 个任务
            tasks = [process_file(f, i+1) for i, f in enumerate(test_files)]
            round_start = time.time()

            results = await asyncio.gather(*tasks, return_exceptions=True)

            round_elapsed = time.time() - round_start
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            fail_count = len(test_files) - success_count

            logger.info(f"\n第 {round_num+1} 轮结果: {success_count}/{len(test_files)} 成功, {fail_count} 失败, 总耗时: {round_elapsed:.1f}秒")

            # 记录失败详情
            if fail_count > 0:
                logger.error(f"\n⚠️  检测到 {fail_count} 个失败任务:")
                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        logger.error(f"  任务{i+1}: 异常 - {r}")
                    elif isinstance(r, dict) and not r.get('success'):
                        logger.error(f"  任务{i+1} ({r.get('file')}): {r.get('error')}")

            all_results.append({
                'round': round_num + 1,
                'success': success_count,
                'fail': fail_count,
                'time': round_elapsed,
                'results': results
            })

            # 轮次间隔
            if round_num < total_rounds - 1:
                logger.info(f"\n等待 5 秒后开始下一轮...")
                await asyncio.sleep(5)

        # 汇总结果
        logger.info(f"\n{'='*100}")
        logger.info(f"测试总结")
        logger.info(f"{'='*100}")

        total_tasks = total_rounds * len(test_files)
        total_success = sum(r['success'] for r in all_results)
        total_fail = sum(r['fail'] for r in all_results)
        success_rate = (total_success / total_tasks) * 100

        logger.info(f"总任务数: {total_tasks}")
        logger.info(f"成功: {total_success}")
        logger.info(f"失败: {total_fail}")
        logger.info(f"成功率: {success_rate:.1f}%")

        if total_fail > 0:
            logger.error(f"\n⚠️  问题已复现！失败率: {100-success_rate:.1f}%")
        else:
            logger.success(f"\n✓ 所有任务成功！MPS 6 并发稳定")

    except Exception as e:
        logger.error(f"测试过程异常: {e}")
        import traceback
        logger.error(f"堆栈:\n{traceback.format_exc()}")

    finally:
        logger.info("\n清理进程池...")
        pool.cleanup()
        logger.info("测试完成")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    asyncio.run(main())
