"""
测试FunASR模型功能
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.core.transcriber import transcriber
from src.core.config import config


async def test_transcription():
    """测试转录功能"""
    # 检查samples目录
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
    
    if not os.path.exists(samples_dir):
        logger.error(f"样本目录不存在: {samples_dir}")
        return
    
    # 获取样本文件
    sample_files = []
    for ext in ['.wav', '.mp3', '.mp4']:
        sample_files.extend([f for f in os.listdir(samples_dir) if f.endswith(ext)])
    
    if not sample_files:
        logger.error("没有找到样本文件")
        return
    
    logger.info(f"找到 {len(sample_files)} 个样本文件")
    
    # 初始化转录器
    try:
        await transcriber.initialize()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return
    
    # 测试每个文件
    for sample_file in sample_files:
        file_path = os.path.join(samples_dir, sample_file)
        logger.info(f"\n{'='*50}")
        logger.info(f"测试文件: {sample_file}")
        
        try:
            # 定义进度回调
            async def progress_callback(progress: float):
                logger.info(f"进度: {progress}%")
            
            # 执行转录
            result = await transcriber.transcribe(
                audio_path=file_path,
                task_id=f"test_{sample_file}",
                progress_callback=progress_callback
            )
            
            # 输出结果
            logger.success(f"转录成功!")
            logger.info(f"文件名: {result.file_name}")
            logger.info(f"时长: {result.duration:.2f}秒")
            logger.info(f"处理时间: {result.processing_time:.2f}秒")
            logger.info(f"说话人: {', '.join(result.speakers)}")
            logger.info(f"片段数: {len(result.segments)}")
            
            # 输出前5个片段
            logger.info("\n前5个转录片段:")
            for i, segment in enumerate(result.segments[:5]):
                logger.info(f"{i+1}. [{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                          f"{segment.speaker}: {segment.text}")
            
            if len(result.segments) > 5:
                logger.info(f"... 还有 {len(result.segments) - 5} 个片段")
            
        except Exception as e:
            logger.error(f"转录失败: {e}")
        
        logger.info(f"{'='*50}\n")


if __name__ == "__main__":
    # 设置日志级别
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    
    # 运行测试
    asyncio.run(test_transcription())