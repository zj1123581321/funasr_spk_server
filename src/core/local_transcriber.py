"""
使用本地模型的FunASR转录核心模块
"""
import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
from funasr import AutoModel
from src.models.schemas import TranscriptionSegment, TranscriptionResult
from src.utils.file_utils import convert_to_wav, get_audio_duration


class LocalFunASRTranscriber:
    """使用本地模型的FunASR转录器"""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        # 本地模型路径
        self.local_model_path = Path("models/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn-onnx")
        
    async def initialize(self):
        """初始化模型"""
        if self.is_initialized:
            return
        
        try:
            logger.info("开始加载本地FunASR模型...")
            
            # 检查本地模型是否存在
            if not self.local_model_path.exists():
                logger.warning(f"本地模型不存在: {self.local_model_path}")
                # 尝试使用在线模型
                logger.info("尝试使用在线模型...")
                self.model = AutoModel(
                    model="paraformer-zh",
                    device="cpu",
                    disable_update=True,
                    disable_pbar=True,
                    cache_dir="./models"
                )
            else:
                # 使用本地ONNX模型
                logger.info(f"使用本地ONNX模型: {self.local_model_path}")
                try:
                    # 尝试加载ONNX模型
                    self.model = AutoModel(
                        model=str(self.local_model_path.absolute()),
                        device="cpu",
                        disable_update=True,
                        disable_pbar=True
                    )
                    logger.info("本地ONNX模型加载成功")
                except Exception as e:
                    logger.warning(f"加载本地ONNX模型失败: {e}")
                    # 回退到在线模型
                    logger.info("回退到在线模型...")
                    self.model = AutoModel(
                        model="paraformer-zh",
                        device="cpu",
                        disable_update=True,
                        disable_pbar=True,
                        cache_dir="./models"
                    )
            
            self.is_initialized = True
            logger.info("FunASR模型初始化完成")
            
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
            
            logger.info(f"开始转录: {os.path.basename(audio_path)}")
            
            # 执行转录
            try:
                # 对于较长的音频，使用更大的batch_size_s
                batch_size_s = 300 if duration > 60 else 100
                
                result = self.model.generate(
                    input=wav_path,
                    batch_size_s=batch_size_s,
                    disable_pbar=True
                )
                
            except Exception as e:
                logger.error(f"转录出错: {e}")
                # 尝试使用更小的batch size
                result = self.model.generate(
                    input=wav_path,
                    batch_size_s=60,
                    disable_pbar=True
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
    
    def _parse_result(self, result: Any) -> List[TranscriptionSegment]:
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
        
        # 从text字段解析
        if 'text' in result_data:
            text = result_data.get('text', '')
            
            if text and text.strip():
                # 清理文本
                text = text.strip()
                # 移除重复的空格
                text = ' '.join(text.split())
                
                # 如果文本看起来是重复的单字（如之前的错误），尝试修复
                words = text.split()
                if len(words) > 10 and len(set(words)) < 5:
                    logger.warning("检测到可能的识别错误（重复单字），返回原始音频内容提示")
                    text = "[音频内容无法正确识别]"
                
                # 创建一个单一的segment
                segment = TranscriptionSegment(
                    start_time=0.0,
                    end_time=result_data.get('duration', 0.0),
                    text=text,
                    speaker="Speaker1"
                )
                segments.append(segment)
        
        return segments


# 全局转录器实例
transcriber = LocalFunASRTranscriber()