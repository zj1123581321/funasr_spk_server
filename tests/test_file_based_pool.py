"""
测试文件系统进程池的并发转录
"""
import asyncio
import json
import time
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.file_based_process_pool import FileBasedProcessPool
from loguru import logger


async def test_single_task():
    """测试单个任务处理"""
    logger.info("=" * 50)
    logger.info("测试单个任务处理")
    logger.info("=" * 50)
    
    # 配置文件路径
    config_path = "config.json"
    
    # 确保配置正确
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置为进程池模式
    config["transcription"]["concurrency_mode"] = "pool"
    config["transcription"]["max_concurrent_tasks"] = 2
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info("配置已更新为文件系统进程池模式")
    
    # 创建进程池
    pool = FileBasedProcessPool(config_path, pool_size=2)
    
    try:
        # 初始化进程池
        logger.info("初始化进程池...")
        await pool.initialize()
        
        # 查找测试音频文件
        test_files = [
            "tests/test_data/test.wav",
            "tests/test_data/test.mp3",
            "tests/test_data/test.m4a"
        ]
        
        audio_file = None
        for test_file in test_files:
            if Path(test_file).exists():
                audio_file = test_file
                break
        
        if not audio_file:
            # 创建一个简单的测试文件
            test_dir = Path("tests/test_data")
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # 查找上传目录中的文件
            upload_dir = Path("uploads")
            if upload_dir.exists():
                audio_files = (list(upload_dir.glob("*.m4a")) + 
                              list(upload_dir.glob("*.mp3")) + 
                              list(upload_dir.glob("*.wav")) +
                              list(upload_dir.glob("*.mp4")))
                if audio_files:
                    audio_file = str(audio_files[0])
                    logger.info(f"使用上传目录中的文件: {audio_file}")
        
        if not audio_file:
            logger.warning("未找到测试音频文件")
            logger.info("请在 tests/test_data 或 uploads 目录下放置音频文件")
            return
        
        logger.info(f"测试文件: {audio_file}")
        
        # 处理单个任务
        start_time = time.time()
        logger.info("提交任务到进程池...")
        
        result = await pool.generate_with_pool(audio_file)
        
        elapsed_time = time.time() - start_time
        
        if result:
            logger.success(f"任务成功完成，耗时: {elapsed_time:.2f}秒")
            if isinstance(result, list) and len(result) > 0:
                logger.info(f"结果类型: {type(result[0])}")
                if isinstance(result[0], dict):
                    logger.info(f"结果键: {list(result[0].keys())}")
        else:
            logger.error("任务失败")
        
    finally:
        # 清理资源
        logger.info("清理进程池...")
        pool.cleanup()


async def test_concurrent_tasks():
    """测试并发任务处理"""
    logger.info("=" * 50)
    logger.info("测试并发任务处理")
    logger.info("=" * 50)
    
    # 配置文件路径
    config_path = "config.json"
    
    # 确保配置正确
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置为进程池模式
    config["transcription"]["concurrency_mode"] = "pool"
    config["transcription"]["max_concurrent_tasks"] = 2
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 创建进程池
    pool = FileBasedProcessPool(config_path, pool_size=2)
    
    try:
        # 初始化进程池
        logger.info("初始化进程池...")
        await pool.initialize()
        
        # 查找测试音频文件
        upload_dir = Path("uploads")
        audio_files = []
        
        if upload_dir.exists():
            audio_files = (list(upload_dir.glob("*.m4a")) + 
                          list(upload_dir.glob("*.mp3")) + 
                          list(upload_dir.glob("*.wav")) +
                          list(upload_dir.glob("*.mp4")))
            audio_files = audio_files[:4]  # 最多测试4个文件
        
        if not audio_files:
            logger.warning("未找到测试音频文件")
            return
        
        logger.info(f"找到 {len(audio_files)} 个测试文件")
        
        # 并发处理任务
        tasks = []
        start_time = time.time()
        
        for i, audio_file in enumerate(audio_files):
            logger.info(f"提交任务 {i+1}: {audio_file.name}")
            task = pool.generate_with_pool(str(audio_file))
            tasks.append(task)
        
        # 等待所有任务完成
        logger.info("等待所有任务完成...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"任务 {i+1} 失败: {result}")
            else:
                logger.success(f"任务 {i+1} 成功完成")
                success_count += 1
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 50)
        logger.info("测试结果统计:")
        logger.info(f"  总任务数: {len(tasks)}")
        logger.info(f"  成功: {success_count}")
        logger.info(f"  失败: {len(tasks) - success_count}")
        logger.info(f"  总耗时: {elapsed_time:.2f}秒")
        if success_count > 0:
            logger.info(f"  平均耗时: {elapsed_time/success_count:.2f}秒/任务")
        logger.info("=" * 50)
        
    finally:
        # 清理资源
        logger.info("清理进程池...")
        pool.cleanup()


async def test_with_transcriber():
    """使用 FunASRTranscriber 测试"""
    logger.info("=" * 50)
    logger.info("使用 FunASRTranscriber 测试进程池")
    logger.info("=" * 50)
    
    from src.core.funasr_transcriber import FunASRTranscriber
    
    # 配置文件路径
    config_path = "config.json"
    
    # 确保配置正确
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置为进程池模式
    config["transcription"]["concurrency_mode"] = "pool"
    config["transcription"]["max_concurrent_tasks"] = 2
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info("创建 FunASRTranscriber 实例...")
    
    # 创建转录器
    transcriber = FunASRTranscriber(config_path)
    await transcriber.initialize()
    
    # 查找测试音频文件
    upload_dir = Path("uploads")
    audio_files = []
    
    if upload_dir.exists():
        audio_files = (list(upload_dir.glob("*.m4a")) + 
                      list(upload_dir.glob("*.mp3")) +
                      list(upload_dir.glob("*.mp4")))
        audio_files = audio_files[:2]  # 测试2个文件
    
    if not audio_files:
        logger.warning("未找到测试音频文件")
        return
    
    # 并发处理
    tasks = []
    for i, audio_file in enumerate(audio_files):
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
    
    # 清理进程池
    if hasattr(transcriber, 'model_pool'):
        transcriber.model_pool.cleanup()


if __name__ == "__main__":
    logger.info("开始测试文件系统进程池")
    
    # 测试单个任务
    asyncio.run(test_single_task())
    
    print("\n" * 2)
    
    # 测试并发任务
    asyncio.run(test_concurrent_tasks())
    
    print("\n" * 2)
    
    # 测试完整转录器
    asyncio.run(test_with_transcriber())