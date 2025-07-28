"""
FunASR转录核心模块
"""
import os
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from loguru import logger
from funasr import AutoModel
from src.core.config import config
from src.models.schemas import TranscriptionSegment, TranscriptionResult
from src.utils.file_utils import convert_to_wav, get_audio_duration


class FunASRTranscriber:
    """FunASR转录器"""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        
    async def initialize(self):
        """初始化模型"""
        if self.is_initialized:
            return
        
        try:
            logger.info("开始加载FunASR模型...")
            
            # 确保模型目录存在
            Path(config.funasr.model_dir).mkdir(parents=True, exist_ok=True)
            
            # 加载模型
            self.model = AutoModel(
                model=config.funasr.model,
                model_revision="v2.0.4",  # 修改为有效版本
                vad_model=config.funasr.vad_model,
                vad_model_revision="v2.0.4",
                punc_model=config.funasr.punc_model,
                punc_model_revision="v2.0.4",
                spk_model="cam++",
                spk_model_revision="v2.0.2",
                device=config.funasr.device,
                model_hub="ms",  # 使用ModelScope
                disable_update=True  # 禁用版本检查以加快启动
            )
            
            self.is_initialized = True
            logger.info("FunASR模型加载完成")
            
        except Exception as e:
            logger.error(f"加载FunASR模型失败: {e}")
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
            # 转换为WAV格式
            if not audio_path.endswith('.wav'):
                logger.debug(f"转换音频格式: {audio_path}")
                wav_path = convert_to_wav(audio_path)
            else:
                wav_path = audio_path
            
            # 获取音频时长
            duration = get_audio_duration(wav_path)
            logger.info(f"音频时长: {duration:.2f}秒")
            
            # 更新进度
            if progress_callback:
                await progress_callback(10)
            
            # 执行转录
            logger.info(f"开始转录: {os.path.basename(audio_path)}")
            
            result = self.model.generate(
                input=wav_path,
                batch_size_s=config.funasr.batch_size_s,
                batch_size_token_threshold_s=config.funasr.batch_size_token_threshold_s
            )
            
            # 更新进度
            if progress_callback:
                await progress_callback(90)
            
            # 解析结果
            segments = self._parse_result(result)
            
            # 提取说话人列表
            speakers = sorted(list(set(seg.speaker for seg in segments)))
            
            # 计算文件哈希
            from src.utils.file_utils import calculate_file_hash
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
            
            # 更新进度
            if progress_callback:
                await progress_callback(100)
            
            logger.info(f"转录完成: {len(segments)}个片段, {len(speakers)}个说话人, 耗时{processing_time:.2f}秒")
            
            # 清理临时文件
            if wav_path != audio_path and os.path.exists(wav_path):
                os.remove(wav_path)
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise Exception(f"转录失败: {str(e)}")
    
    def _parse_result(self, result: List[Dict[str, Any]]) -> List[TranscriptionSegment]:
        """解析FunASR返回的结果"""
        segments = []
        
        if not result or len(result) == 0:
            return segments
        
        # FunASR返回的是列表，第一个元素包含转录结果
        result_data = result[0]
        
        # 获取句子列表
        sentences = result_data.get('sentence_info', [])
        
        for sentence in sentences:
            # 提取时间戳（毫秒转秒）
            start_time = sentence.get('start', 0) / 1000.0
            end_time = sentence.get('end', 0) / 1000.0
            
            # 提取文本
            text = sentence.get('text', '').strip()
            
            # 提取说话人
            speaker = sentence.get('spk', 'Speaker1')
            if isinstance(speaker, int):
                speaker = f"Speaker{speaker + 1}"
            
            # 提取置信度（如果有）
            confidence = sentence.get('confidence', None)
            
            if text:  # 只添加非空文本
                segment = TranscriptionSegment(
                    start_time=round(start_time, 2),
                    end_time=round(end_time, 2),
                    text=text,
                    speaker=speaker,
                    confidence=confidence
                )
                segments.append(segment)
        
        # 如果sentence_info为空，尝试解析文本
        if not segments and 'text' in result_data:
            # 简单分割文本
            text = result_data['text']
            if text:
                segment = TranscriptionSegment(
                    start_time=0.0,
                    end_time=0.0,
                    text=text,
                    speaker="Speaker1"
                )
                segments.append(segment)
        
        return segments


# 全局转录器实例
transcriber = FunASRTranscriber()