"""
FunASR转录核心模块 - 改进版
"""
import os
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from loguru import logger
from funasr import AutoModel
from src.models.schemas import TranscriptionSegment, TranscriptionResult
from src.utils.file_utils import convert_to_wav, get_audio_duration


class FunASRTranscriber:
    """FunASR转录器 - 改进版"""
    
    def __init__(self):
        self.model = None
        self.model_with_spk = None
        self.is_initialized = False
        
    async def initialize(self):
        """初始化模型"""
        if self.is_initialized:
            return
        
        try:
            logger.info("开始加载FunASR模型...")
            
            # 加载基础模型（不含说话人识别，更稳定）
            try:
                logger.info("加载基础ASR模型...")
                self.model = AutoModel(
                    model="paraformer-zh",
                    device="cpu",
                    disable_update=True,
                    disable_pbar=True
                )
                logger.info("基础ASR模型加载成功")
            except Exception as e:
                logger.error(f"基础模型加载失败: {e}")
                raise
            
            # 尝试加载带说话人识别的模型
            try:
                logger.info("尝试加载带说话人识别的模型...")
                self.model_with_spk = AutoModel(
                    model="paraformer-zh",
                    vad_model="fsmn-vad",
                    punc_model="ct-punc",
                    spk_model="cam++",
                    device="cpu",
                    disable_update=True,
                    disable_pbar=True
                )
                logger.info("说话人识别模型加载成功")
            except Exception as e:
                logger.warning(f"说话人识别模型加载失败，将使用基础模型: {e}")
                self.model_with_spk = None
            
            self.is_initialized = True
            logger.info("FunASR模型初始化完成")
            
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
        """转录音频文件"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        wav_path = audio_path
        
        try:
            # 转换为WAV格式
            if not audio_path.endswith('.wav'):
                logger.debug(f"转换音频格式: {audio_path}")
                wav_path = convert_to_wav(audio_path)
            
            # 获取音频时长
            duration = get_audio_duration(wav_path)
            logger.info(f"音频时长: {duration:.2f}秒")
            
            # 更新进度
            if progress_callback:
                await progress_callback(10)
            
            # 选择使用的模型
            use_spk_model = enable_speaker and self.model_with_spk is not None
            model = self.model_with_spk if use_spk_model else self.model
            
            logger.info(f"开始转录: {os.path.basename(audio_path)} (使用{'说话人识别' if use_spk_model else '基础'}模型)")
            
            # 执行转录
            try:
                # 对于较长的音频，使用更大的batch_size_s
                batch_size_s = 300 if duration > 60 else 100
                
                result = model.generate(
                    input=wav_path,
                    batch_size_s=batch_size_s,
                    hotword="",  # 避免热词相关的问题
                    disable_pbar=True
                )
                
            except Exception as e:
                # 如果说话人识别模型失败，回退到基础模型
                if use_spk_model and "math domain error" in str(e):
                    logger.warning(f"说话人识别模型失败，回退到基础模型: {e}")
                    result = self.model.generate(
                        input=wav_path,
                        batch_size_s=300,
                        disable_pbar=True
                    )
                    use_spk_model = False
                else:
                    raise
            
            # 更新进度
            if progress_callback:
                await progress_callback(90)
            
            # 解析结果
            segments = self._parse_result(result, use_spk_model)
            
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
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise Exception(f"转录失败: {str(e)}")
        finally:
            # 清理临时文件
            if wav_path != audio_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except:
                    pass
    
    def _parse_result(self, result: Any, has_speaker_info: bool = True) -> List[TranscriptionSegment]:
        """解析FunASR返回的结果"""
        segments = []
        
        # 处理不同的结果格式
        if isinstance(result, list) and len(result) > 0:
            result_data = result[0]
        elif isinstance(result, dict):
            result_data = result
        else:
            logger.warning(f"未知的结果格式: {type(result)}")
            return segments
        
        # 首先尝试从sentence_info获取
        if has_speaker_info and 'sentence_info' in result_data:
            sentences = result_data.get('sentence_info', [])
            
            for idx, sentence in enumerate(sentences):
                # 提取时间戳（毫秒转秒）
                start_time = sentence.get('start', 0) / 1000.0
                end_time = sentence.get('end', 0) / 1000.0
                
                # 提取文本
                text = sentence.get('text', '').strip()
                
                # 提取说话人
                speaker = sentence.get('spk', f"Speaker1")
                if isinstance(speaker, int):
                    speaker = f"Speaker{speaker + 1}"
                elif speaker == "" or speaker is None:
                    speaker = "Speaker1"
                
                if text:  # 只添加非空文本
                    segment = TranscriptionSegment(
                        start_time=round(start_time, 2),
                        end_time=round(end_time, 2),
                        text=text,
                        speaker=speaker
                    )
                    segments.append(segment)
        
        # 如果没有sentence_info或segments为空，尝试从text和timestamp解析
        if not segments and 'text' in result_data:
            text = result_data.get('text', '')
            timestamps = result_data.get('timestamp', [])
            
            if isinstance(text, str) and text:
                # 如果是单一文本，创建一个segment
                if not timestamps:
                    segment = TranscriptionSegment(
                        start_time=0.0,
                        end_time=0.0,
                        text=text,
                        speaker="Speaker1"
                    )
                    segments.append(segment)
                else:
                    # 尝试按标点符号分割文本
                    import re
                    sentences = re.split(r'[。！？；]', text)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    
                    # 分配时间戳
                    for idx, sent in enumerate(sentences):
                        if idx < len(timestamps):
                            ts = timestamps[idx]
                            start_time = ts[0] / 1000.0 if isinstance(ts, list) and len(ts) > 0 else 0
                            end_time = ts[1] / 1000.0 if isinstance(ts, list) and len(ts) > 1 else start_time + 1
                        else:
                            start_time = 0
                            end_time = 0
                        
                        segment = TranscriptionSegment(
                            start_time=round(start_time, 2),
                            end_time=round(end_time, 2),
                            text=sent,
                            speaker="Speaker1"
                        )
                        segments.append(segment)
        
        # 如果还是没有segments，尝试其他字段
        if not segments:
            # 尝试从其他可能的字段获取
            for key in ['result', 'recognition_result', 'transcript']:
                if key in result_data:
                    text = str(result_data[key])
                    if text and text != 'None':
                        segment = TranscriptionSegment(
                            start_time=0.0,
                            end_time=0.0,
                            text=text,
                            speaker="Speaker1"
                        )
                        segments.append(segment)
                        break
        
        return segments


# 全局转录器实例
transcriber = FunASRTranscriber()