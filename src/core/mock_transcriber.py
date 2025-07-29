"""
Mock转录器 - 用于测试和演示
"""
import time
import os
from typing import Optional
from datetime import datetime
from loguru import logger
from src.models.schemas import TranscriptionSegment, TranscriptionResult
from src.utils.file_utils import get_audio_duration, calculate_file_hash


class MockTranscriber:
    """Mock转录器，用于演示和测试"""
    
    def __init__(self):
        self.is_initialized = False
        
    async def initialize(self):
        """初始化模型"""
        if self.is_initialized:
            return
        
        try:
            logger.info("初始化Mock转录器...")
            # 模拟加载时间
            await asyncio.sleep(2)
            
            self.is_initialized = True
            logger.info("Mock转录器初始化完成")
            
        except Exception as e:
            logger.error(f"Mock转录器初始化失败: {e}")
            raise Exception(f"模型初始化失败: {e}")
    
    async def transcribe(
        self,
        audio_path: str,
        task_id: str,
        progress_callback: Optional[callable] = None
    ) -> TranscriptionResult:
        """转录音频文件"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 获取音频时长
            duration = get_audio_duration(audio_path)
            if duration == 0:
                duration = 30.0  # 默认30秒
                
            logger.info(f"Mock转录开始: {os.path.basename(audio_path)}, 时长: {duration:.2f}秒")
            
            # 模拟转录进度
            if progress_callback:
                await progress_callback(10)
            
            # 模拟处理时间（根据音频长度）
            processing_seconds = min(duration * 0.1, 5.0)  # 最多5秒
            await asyncio.sleep(processing_seconds / 2)
            
            if progress_callback:
                await progress_callback(50)
            
            await asyncio.sleep(processing_seconds / 2)
            
            if progress_callback:
                await progress_callback(90)
            
            # 生成Mock转录结果
            segments = self._generate_mock_segments(duration)
            
            # 提取说话人列表
            speakers = sorted(list(set(seg.speaker for seg in segments)))
            
            # 计算文件哈希
            file_hash = await calculate_file_hash(audio_path)
            
            processing_time = time.time() - start_time
            
            # 构建转录结果
            transcription_result = TranscriptionResult(
                task_id=task_id,
                file_name=os.path.basename(audio_path),
                file_hash=file_hash,
                duration=duration,
                segments=segments,
                speakers=speakers,
                processing_time=processing_time
            )
            
            if progress_callback:
                await progress_callback(100)
            
            logger.info(f"Mock转录完成: {len(segments)}个片段, {len(speakers)}个说话人, 耗时{processing_time:.2f}秒")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Mock转录失败: {e}")
            raise Exception(f"转录失败: {str(e)}")
    
    def _generate_mock_segments(self, duration: float) -> list[TranscriptionSegment]:
        """生成Mock转录片段"""
        logger.debug(f"生成Mock片段，音频时长: {duration}秒")
        segments = []
        
        # 生成示例对话
        mock_texts = [
            "你好，欢迎使用FunASR转录服务",
            "这是一个语音转录的测试",
            "系统正在处理您的音频文件",
            "转录功能包括说话人识别",
            "以及精确的时间戳标注",
            "感谢您的使用",
        ]
        
        # 计算每个片段的时长
        segment_duration = duration / len(mock_texts)
        current_time = 0.0
        
        logger.debug(f"每个片段时长: {segment_duration}秒")
        
        for i, text in enumerate(mock_texts):
            if current_time >= duration:
                logger.debug(f"达到音频时长限制，停止生成片段")
                break
                
            # 交替分配说话人
            speaker = f"Speaker{(i % 2) + 1}"
            
            start_time = current_time
            end_time = min(current_time + segment_duration, duration)
            
            segment = TranscriptionSegment(
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                text=text,
                speaker=speaker,
                confidence=0.95
            )
            segments.append(segment)
            logger.debug(f"添加片段: [{start_time:.2f}s - {end_time:.2f}s] {speaker}: {text}")
            
            current_time = end_time
        
        logger.info(f"生成了 {len(segments)} 个Mock片段")
        return segments


# 导入asyncio（如果需要的话）
import asyncio

# 创建全局Mock转录器实例
mock_transcriber = MockTranscriber()