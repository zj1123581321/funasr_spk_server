"""
测试进程池模式的并发转录
"""
import asyncio
import json
import time
from pathlib import Path
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.model_process_pool import ModelProcessPool
from loguru import logger


async def test_process_pool():
    """测试进程池并发处理"""
    
    # 确保配置正确
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置为进程池模式
    config["transcription"]["concurrency_mode"] = "pool"
    config["transcription"]["max_concurrent_tasks"] = 2
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info("配置已更新为进程池模式")
    
    # 创建进程池
    pool = ModelProcessPool(config_path, pool_size=2)
    
    try:
        # 初始化进程池
        logger.info("初始化进程池...")
        await pool.initialize()
        
        # 查找测试音频文件
        test_dir = Path("tests/test_data")
        audio_files = list(test_dir.glob("*.wav"))
        
        if not audio_files:
            logger.warning(f"未找到测试音频文件，请在 {test_dir} 目录下放置 .wav 文件")
            return
        
        logger.info(f"找到 {len(audio_files)} 个测试音频文件")
        
        # 并发处理音频文件
        tasks = []
        start_time = time.time()
        
        for audio_file in audio_files[:2]:  # 只测试前2个文件
            logger.info(f"提交任务: {audio_file.name}")
            task = pool.generate_with_pool(str(audio_file))
            tasks.append(task)
        
        # 等待所有任务完成
        logger.info("等待所有任务完成...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"任务 {i} 失败: {result}")
            else:
                logger.success(f"任务 {i} 成功完成")
                success_count += 1
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n测试完成:")
        logger.info(f"  总任务数: {len(tasks)}")
        logger.info(f"  成功: {success_count}")
        logger.info(f"  失败: {len(tasks) - success_count}")
        logger.info(f"  总耗时: {elapsed_time:.2f}秒")
        
    finally:
        # 清理资源
        logger.info("清理进程池...")
        pool.cleanup()


async def test_with_transcriber():
    """使用 FunASRTranscriber 测试进程池模式"""
    from src.core.funasr_transcriber import FunASRTranscriber
    
    # 确保配置正确
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置为进程池模式
    config["transcription"]["concurrency_mode"] = "pool"
    config["transcription"]["max_concurrent_tasks"] = 2
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info("使用 FunASRTranscriber 测试进程池模式")
    
    # 创建转录器
    transcriber = FunASRTranscriber(config_path)
    await transcriber.initialize()
    
    # 查找测试音频文件
    test_dir = Path("tests/test_data")
    audio_files = list(test_dir.glob("*.wav"))
    
    if not audio_files:
        logger.warning(f"未找到测试音频文件，请在 {test_dir} 目录下放置 .wav 文件")
        return
    
    # 并发处理
    tasks = []
    for i, audio_file in enumerate(audio_files[:2]):
        task_id = f"test_{i}"
        logger.info(f"创建任务 {task_id}: {audio_file.name}")
        task = transcriber.transcribe(
            str(audio_file),
            task_id,
            enable_speaker=True,
            output_format="json"
        )
        tasks.append(task)
    
    # 等待所有任务完成
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed_time = time.time() - start_time
    
    # 分析结果
    success_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"任务 {i} 失败: {result}")
        else:
            logger.success(f"任务 {i} 成功")
            success_count += 1
    
    logger.info(f"\n测试结果:")
    logger.info(f"  成功: {success_count}/{len(tasks)}")
    logger.info(f"  耗时: {elapsed_time:.2f}秒")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("开始测试进程池模式")
    logger.info("=" * 50)
    
    # 测试进程池基础功能
    asyncio.run(test_process_pool())
    
    logger.info("\n" + "=" * 50)
    logger.info("测试 FunASRTranscriber 进程池模式")
    logger.info("=" * 50)
    
    # 测试完整转录器
    asyncio.run(test_with_transcriber())