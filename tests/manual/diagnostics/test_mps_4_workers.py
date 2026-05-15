"""
MPS 4 Workers 并发压力测试

测试 4 个 worker 同时运行时的稳定性
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

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>")


async def main():
    """测试 4 workers 并发"""

    test_files = [
        # "temp/samples/podcast_en.mp3",
        # "temp/samples/test.m4a"
        # "temp/samples/podcast_en.mp3",  # 重复文件模拟4个任务
        "temp/samples/「月月轻松」救助流浪狗，到哪一步该收手？- EP237 - ULSUM RADIO - 小宇宙 - 听播客，上小宇宙.mp3"
    ]

    logger.info("="*80)
    logger.info(f"测试 4 Workers MPS 并发 - 共 {len(test_files)} 个任务")
    logger.info("="*80)

    # 创建 4 个 worker 的进程池
    pool = FileBasedProcessPool(pool_size=4)

    try:
        await pool.initialize()
        logger.success("进程池初始化完成 (4 workers)")

        # 并发提交所有任务
        async def process_file(file_path: str, task_id: int):
            logger.info(f"[任务{task_id}] 开始: {os.path.basename(file_path)}")
            start = time.time()
            try:
                result = await pool.generate_with_pool(file_path, batch_size_s=500, use_pickle=True)
                elapsed = time.time() - start
                logger.success(f"[任务{task_id}] 成功 - {elapsed:.1f}秒")
                return {"success": True, "time": elapsed}
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[任务{task_id}] 失败 - {e}")
                return {"success": False, "error": str(e), "time": elapsed}

        # 多轮测试
        for round_num in range(5):
            logger.info(f"\n{'='*80}")
            logger.info(f"第 {round_num+1} 轮测试")
            logger.info(f"{'='*80}")

            tasks = [process_file(f, i+1) for i, f in enumerate(test_files)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            logger.info(f"第 {round_num+1} 轮结果: {success}/{len(test_files)} 成功")

            if success < len(test_files):
                logger.warning("检测到失败任务！")
                for i, r in enumerate(results):
                    if isinstance(r, dict) and not r.get('success'):
                        logger.error(f"  任务{i+1} 失败: {r.get('error')}")

            await asyncio.sleep(3)

    finally:
        pool.cleanup()
        logger.info("测试完成")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    asyncio.run(main())
