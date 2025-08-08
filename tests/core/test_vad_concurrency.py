"""
测试VAD并发功能 - 验证并发错误修复
"""
import sys
import os
import asyncio
import time
from pathlib import Path
from typing import List
import threading

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from loguru import logger
from src.core.funasr_transcriber import FunASRTranscriber


async def test_single_transcribe(transcriber: FunASRTranscriber, audio_file: str, task_id: int):
    """测试单个转录任务"""
    try:
        logger.info(f"任务 {task_id} 开始处理: {os.path.basename(audio_file)}")
        start_time = time.time()
        
        result = await transcriber.transcribe(
            audio_path=audio_file,
            task_id=f"test_task_{task_id}",
            enable_speaker=True,
            output_format="json"
        )
        
        elapsed = time.time() - start_time
        
        # 解析结果
        if isinstance(result, tuple):
            result_obj, _ = result
            segments_count = len(result_obj.segments)
            logger.success(f"任务 {task_id} 完成: {segments_count} 个片段, 耗时 {elapsed:.2f}秒")
        else:
            logger.success(f"任务 {task_id} 完成, 耗时 {elapsed:.2f}秒")
        
        return True
        
    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {e}")
        return False


async def test_concurrent_transcribe(audio_files: List[str], concurrency: int = 4):
    """测试并发转录"""
    logger.info(f"开始并发测试，并发数: {concurrency}")
    
    # 创建转录器实例
    transcriber = FunASRTranscriber()
    
    # 初始化
    await transcriber.initialize()
    
    # 创建并发任务
    tasks = []
    for i, audio_file in enumerate(audio_files[:concurrency]):
        task = asyncio.create_task(test_single_transcribe(transcriber, audio_file, i + 1))
        tasks.append(task)
    
    # 等待所有任务完成
    logger.info(f"等待 {len(tasks)} 个并发任务完成...")
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r)
    failed_count = len(results) - success_count
    
    logger.info("=" * 50)
    logger.info(f"并发测试完成:")
    logger.info(f"  总任务数: {len(results)}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failed_count}")
    logger.info(f"  总耗时: {total_time:.2f}秒")
    logger.info(f"  平均耗时: {total_time/len(results):.2f}秒/任务")
    logger.info("=" * 50)
    
    return success_count == len(results)


async def test_with_different_modes():
    """测试不同的并发模式"""
    import json
    
    # 查找测试音频文件
    test_files = []
    
    # 查找项目中的测试音频
    test_dirs = [
        "samples/concurrency",
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in Path(test_dir).glob("*.wav"):
                test_files.append(str(file))
            for file in Path(test_dir).glob("*.mp3"):
                test_files.append(str(file))
            for file in Path(test_dir).glob("*.m4a"):
                test_files.append(str(file))
    
    if not test_files:
        # 如果没有找到测试文件，创建一个静音测试文件
        logger.warning("未找到测试音频文件，请将音频文件放在 tests/data 目录下")
        return
    
    logger.info(f"找到 {len(test_files)} 个测试文件")
    
    # 测试线程锁模式
    logger.info("\n" + "=" * 60)
    logger.info("测试线程锁模式 (concurrency_mode: lock)")
    logger.info("=" * 60)
    
    # 确保配置使用lock模式
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    original_mode = config["transcription"].get("concurrency_mode", "lock")
    config["transcription"]["concurrency_mode"] = "lock"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 运行测试
    success = await test_concurrent_transcribe(test_files, concurrency=4)
    
    if success:
        logger.success("线程锁模式测试通过!")
    else:
        logger.error("线程锁模式测试失败!")
    
    # 恢复原始配置
    config["transcription"]["concurrency_mode"] = original_mode
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


async def stress_test():
    """压力测试 - 模拟高并发场景"""
    logger.info("\n" + "=" * 60)
    logger.info("开始压力测试")
    logger.info("=" * 60)
    
    # 查找测试文件
    test_file = None
    test_dirs = ["tests/data", "tests/samples", "uploads", "temp"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in Path(test_dir).glob("*.wav"):
                test_file = str(file)
                break
            if test_file:
                break
            for file in Path(test_dir).glob("*.mp3"):
                test_file = str(file)
                break
            if test_file:
                break
    
    if not test_file:
        logger.warning("未找到测试音频文件，跳过压力测试")
        return
    
    logger.info(f"使用测试文件: {test_file}")
    
    # 创建多个相同文件的副本来测试
    test_files = [test_file] * 10  # 使用同一个文件10次
    
    # 测试不同并发级别
    for concurrency in [2, 4, 6, 8]:
        logger.info(f"\n测试并发级别: {concurrency}")
        success = await test_concurrent_transcribe(test_files[:concurrency], concurrency)
        
        if not success:
            logger.error(f"并发级别 {concurrency} 测试失败")
            break
        
        # 短暂等待，让系统恢复
        await asyncio.sleep(2)
    
    logger.info("\n压力测试完成")


def main():
    """主函数"""
    logger.info("VAD并发测试程序启动")
    
    # 设置日志级别
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    # 运行测试
    asyncio.run(test_with_different_modes())
    
    # 运行压力测试
    # asyncio.run(stress_test())
    
    logger.info("测试完成")


if __name__ == "__main__":
    main()