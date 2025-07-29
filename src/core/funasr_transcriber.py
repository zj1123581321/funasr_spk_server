"""
FunASR转录核心模块 - 与测试脚本完全一致的实现
"""
import os
import time
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from loguru import logger
from funasr import AutoModel
from src.models.schemas import TranscriptionSegment, TranscriptionResult
from src.utils.file_utils import convert_to_wav, get_audio_duration


class FunASRTranscriber:
    """FunASR转录器 - 与测试脚本完全一致的实现"""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.cache_dir = "./models"  # 模型缓存目录
        
    async def initialize(self):
        """初始化模型 - 使用与测试脚本完全相同的配置"""
        if self.is_initialized:
            return
        
        try:
            logger.info("开始加载FunASR完整模型（包含说话人识别）...")
            
            # 确保缓存目录存在
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            # 使用与测试脚本完全相同的模型配置
            self.model = AutoModel(
                model="paraformer-zh", 
                model_revision="v2.0.4",
                vad_model="fsmn-vad", 
                vad_model_revision="v2.0.4",
                punc_model="ct-punc-c", 
                punc_model_revision="v2.0.4",
                spk_model="cam++", 
                spk_model_revision="v2.0.2",
                # 添加缓存配置避免重复下载
                cache_dir=self.cache_dir,
                device="cpu",
                disable_update=True,  # 禁用自动更新
                disable_pbar=True     # 禁用进度条
            )
            
            self.is_initialized = True
            logger.info("FunASR完整模型加载成功")
            
        except Exception as e:
            logger.error(f"加载FunASR模型失败: {e}")
            raise Exception(f"模型初始化失败: {e}")
    
    async def transcribe(
        self,
        audio_path: str,
        task_id: str,
        progress_callback: Optional[callable] = None,
        enable_speaker: bool = True
    ) -> TranscriptionResult:
        """转录音频文件 - 使用与测试脚本相同的方法"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        original_path = audio_path
        
        try:
            # 更新进度
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(10)
                else:
                    progress_callback(10)
            
            # 获取音频时长（直接使用原文件）
            duration = get_audio_duration(audio_path)
            logger.info(f"音频时长: {duration:.2f}秒")
            
            logger.info(f"开始转录: {os.path.basename(audio_path)}")
            
            # 使用与测试脚本完全相同的转录参数
            result = self.model.generate(
                input=audio_path,  # 直接使用原始音频文件
                batch_size_s=300, 
                hotword=''  # 与测试脚本保持一致
            )
            
            logger.debug(f"FunASR原始结果: {result}")
            
            # 更新进度
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(90)
                else:
                    progress_callback(90)
            
            # 解析结果并合并相同说话人的连续句子
            segments = self._parse_and_merge_segments(result)
            
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
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(100)
                else:
                    progress_callback(100)
            
            logger.info(f"转录完成: {len(segments)}个片段, {len(speakers)}个说话人, 耗时{processing_time:.2f}秒")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise Exception(f"转录失败: {str(e)}")
    
    def _parse_and_merge_segments(self, result: Any) -> List[TranscriptionSegment]:
        """解析FunASR结果并合并相同说话人的连续句子"""
        segments = []
        
        logger.debug(f"解析结果 - 输入类型: {type(result)}")
        
        # 处理结果格式
        if isinstance(result, list) and len(result) > 0:
            result_data = result[0]
        elif isinstance(result, dict):
            result_data = result
        else:
            logger.warning(f"未知的结果格式: {type(result)}")
            return segments
        
        logger.debug(f"使用数据，键: {list(result_data.keys()) if isinstance(result_data, dict) else 'N/A'}")
        
        # 检查是否有sentence_info（这是成功转录的标志）
        if 'sentence_info' not in result_data:
            logger.warning("结果中没有sentence_info字段，可能转录失败")
            return segments
        
        sentences = result_data.get('sentence_info', [])
        logger.info(f"找到 {len(sentences)} 个句子片段")
        
        # 解析每个句子
        raw_segments = []
        for sentence in sentences:
            # 提取时间戳（毫秒转秒）
            start_time = sentence.get('start', 0) / 1000.0
            end_time = sentence.get('end', 0) / 1000.0
            
            # 提取文本
            text = sentence.get('text', '').strip()
            
            # 提取说话人 - 使用 Speaker1, Speaker2 格式
            speaker_id = sentence.get('spk', 0)
            if isinstance(speaker_id, int):
                speaker = f"Speaker{speaker_id + 1}"
            else:
                speaker = "Speaker1"
            
            if text:  # 只添加非空文本
                segment = TranscriptionSegment(
                    start_time=round(start_time, 2),
                    end_time=round(end_time, 2),
                    text=text,
                    speaker=speaker
                )
                raw_segments.append(segment)
        
        # 合并相同说话人的连续句子（可选，根据需求）
        if self._should_merge_segments():
            segments = self._merge_consecutive_segments(raw_segments)
        else:
            segments = raw_segments
        
        logger.info(f"最终生成 {len(segments)} 个转录片段")
        return segments
    
    def _should_merge_segments(self) -> bool:
        """判断是否应该合并片段 - 启用相同说话人连续句子的合并"""
        # 启用合并功能，将相同说话人的连续句子合并
        return True
    
    def _merge_consecutive_segments(self, segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """合并相同说话人的连续句子"""
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        logger.debug(f"开始合并 {len(segments)} 个片段")
        
        for i in range(1, len(segments)):
            next_seg = segments[i]
            
            # 检查是否为同一说话人且时间间隔较短（小于3秒）
            time_gap = next_seg.start_time - current.end_time
            
            logger.debug(f"片段 {i}: {current.speaker} -> {next_seg.speaker}, 时间间隔: {time_gap:.2f}s")
            
            if (current.speaker == next_seg.speaker and 
                time_gap < 3.0):  # 增加到3秒的间隔阈值
                # 合并文本和时间 - 保留所有标点符号
                merged_text = current.text + next_seg.text
                current = TranscriptionSegment(
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                    text=merged_text,
                    speaker=current.speaker
                )
                logger.debug(f"合并片段: {merged_text[:20]}...")
            else:
                # 添加当前片段，开始新的片段
                merged.append(current)
                logger.debug(f"完成片段: [{current.speaker}] {current.text[:20]}...")
                current = next_seg
        
        # 添加最后一个片段
        merged.append(current)
        logger.debug(f"完成片段: [{current.speaker}] {current.text[:20]}...")
        
        logger.info(f"合并完成: {len(segments)} -> {len(merged)} 个片段")
        return merged


# 全局转录器实例
transcriber = FunASRTranscriber()