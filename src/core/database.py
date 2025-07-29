"""
数据库管理模块
"""
import aiosqlite
import json
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
from loguru import logger
from src.core.config import config
from src.models.schemas import TranscriptionResult


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.db_path = config.database.path
        self._ensure_db_dir()
    
    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def init_db(self):
        """初始化数据库"""
        async with aiosqlite.connect(self.db_path) as db:
            # 创建表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transcription_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    result TEXT NOT NULL,
                    raw_result TEXT,
                    duration REAL,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 检查并添加新列
            try:
                # 检查 raw_result 列是否存在
                cursor = await db.execute("PRAGMA table_info(transcription_cache)")
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                if 'raw_result' not in column_names:
                    await db.execute("ALTER TABLE transcription_cache ADD COLUMN raw_result TEXT")
                    logger.info("添加 raw_result 列")
                
                if 'duration' not in column_names:
                    await db.execute("ALTER TABLE transcription_cache ADD COLUMN duration REAL")
                    logger.info("添加 duration 列")
                
                if 'processing_time' not in column_names:
                    await db.execute("ALTER TABLE transcription_cache ADD COLUMN processing_time REAL")
                    logger.info("添加 processing_time 列")
                    
            except Exception as e:
                logger.error(f"升级数据库表结构失败: {e}")
                # 如果升级失败，删除表重新创建
                await db.execute("DROP TABLE IF EXISTS transcription_cache")
                await db.execute("""
                    CREATE TABLE transcription_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_hash TEXT UNIQUE NOT NULL,
                        file_name TEXT NOT NULL,
                        result TEXT NOT NULL,
                        raw_result TEXT,
                        duration REAL,
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("重新创建数据库表")
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON transcription_cache(file_hash)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON transcription_cache(created_at)
            """)
            
            await db.commit()
            logger.info("数据库初始化完成")
    
    async def get_cached_result(self, file_hash: str, output_format: str = "json") -> Optional[TranscriptionResult]:
        """获取缓存的转录结果"""
        if not config.transcription.cache_enabled:
            return None
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT result, raw_result, file_name, duration FROM transcription_cache WHERE file_hash = ?",
                    (file_hash,)
                )
                row = await cursor.fetchone()
                
                if row:
                    # 更新访问时间
                    await db.execute(
                        "UPDATE transcription_cache SET accessed_at = CURRENT_TIMESTAMP WHERE file_hash = ?",
                        (file_hash,)
                    )
                    await db.commit()
                    
                    result_json, raw_result_json, file_name, duration = row
                    
                    if output_format == "json":
                        # 返回原始的JSON格式结果
                        result_data = json.loads(result_json)
                        return TranscriptionResult(**result_data)
                    elif output_format == "srt" and raw_result_json:
                        # 从原始结果生成SRT格式
                        raw_result = json.loads(raw_result_json)
                        srt_content = self._generate_srt_from_raw_result(raw_result)
                        
                        # 返回包含SRT内容的特殊结果对象
                        return {
                            "format": "srt",
                            "content": srt_content,
                            "file_name": file_name,
                            "file_hash": file_hash,
                            "duration": duration
                        }
                
                return None
        except Exception as e:
            logger.error(f"获取缓存结果失败: {e}")
            return None
    
    async def save_result(self, result: TranscriptionResult, raw_result: dict = None):
        """保存转录结果到缓存"""
        if not config.transcription.cache_enabled:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                result_json = result.json()
                raw_result_json = json.dumps(raw_result) if raw_result else None
                
                await db.execute(
                    """
                    INSERT OR REPLACE INTO transcription_cache 
                    (file_hash, file_name, result, raw_result, duration, processing_time) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (result.file_hash, result.file_name, result_json, raw_result_json, 
                     result.duration, result.processing_time)
                )
                await db.commit()
                logger.debug(f"转录结果已缓存: {result.file_hash}")
        except Exception as e:
            logger.error(f"保存缓存结果失败: {e}")
    
    async def clean_old_cache(self):
        """清理过期的缓存"""
        try:
            cutoff_date = datetime.now() - timedelta(days=config.database.max_cache_days)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM transcription_cache WHERE created_at < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                await db.commit()
                
                if deleted_count > 0:
                    logger.info(f"清理了 {deleted_count} 条过期缓存")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    async def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 总缓存数
                cursor = await db.execute("SELECT COUNT(*) FROM transcription_cache")
                total_count = (await cursor.fetchone())[0]
                
                # 今日缓存数
                today = datetime.now().date()
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM transcription_cache WHERE DATE(created_at) = ?",
                    (today.isoformat(),)
                )
                today_count = (await cursor.fetchone())[0]
                
                # 缓存大小
                cursor = await db.execute(
                    "SELECT SUM(LENGTH(result)) FROM transcription_cache"
                )
                cache_size = (await cursor.fetchone())[0] or 0
                
                return {
                    "total_count": total_count,
                    "today_count": today_count,
                    "cache_size_mb": round(cache_size / 1024 / 1024, 2)
                }
        except Exception as e:
            logger.error(f"获取缓存统计信息失败: {e}")
            return {
                "total_count": 0,
                "today_count": 0,
                "cache_size_mb": 0
            }
    
    def _generate_srt_from_raw_result(self, raw_result: dict) -> str:
        """从原始FunASR结果生成SRT格式字符串"""
        srt_lines = []
        
        # 处理结果格式
        if isinstance(raw_result, list) and len(raw_result) > 0:
            result_data = raw_result[0]
        elif isinstance(raw_result, dict):
            result_data = raw_result
        else:
            logger.warning(f"未知的结果格式: {type(raw_result)}")
            return ""
        
        # 检查是否有sentence_info
        if 'sentence_info' not in result_data:
            logger.warning("结果中没有sentence_info字段，可能转录失败")
            return ""
        
        sentences = result_data.get('sentence_info', [])
        
        # 生成SRT格式
        for idx, sentence in enumerate(sentences, 1):
            # 提取时间戳（毫秒转秒）
            start_ms = sentence.get('start', 0)
            end_ms = sentence.get('end', 0)
            
            # 转换为SRT时间格式 (HH:MM:SS,mmm)
            start_time = self._ms_to_srt_time(start_ms)
            end_time = self._ms_to_srt_time(end_ms)
            
            # 提取文本
            text = sentence.get('text', '').strip()
            
            # 提取说话人
            speaker_id = sentence.get('spk', 0)
            if isinstance(speaker_id, int):
                speaker = f"Speaker{speaker_id + 1}"
            else:
                speaker = "Speaker1"
            
            if text:  # 只添加非空文本
                # SRT格式：序号 -> 时间 -> 文本
                srt_lines.append(f"{idx}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(f"{speaker}:{text}")
                srt_lines.append("")  # 空行分隔
        
        return "\n".join(srt_lines)
    
    def _ms_to_srt_time(self, milliseconds: int) -> str:
        """将毫秒转换为SRT时间格式 (HH:MM:SS,mmm)"""
        seconds = milliseconds / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# 全局数据库管理器实例
db_manager = DatabaseManager()