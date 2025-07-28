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
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transcription_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON transcription_cache(file_hash)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON transcription_cache(created_at)
            """)
            
            await db.commit()
            logger.info("数据库初始化完成")
    
    async def get_cached_result(self, file_hash: str) -> Optional[TranscriptionResult]:
        """获取缓存的转录结果"""
        if not config.transcription.cache_enabled:
            return None
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT result FROM transcription_cache WHERE file_hash = ?",
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
                    
                    result_data = json.loads(row[0])
                    return TranscriptionResult(**result_data)
                
                return None
        except Exception as e:
            logger.error(f"获取缓存结果失败: {e}")
            return None
    
    async def save_result(self, result: TranscriptionResult):
        """保存转录结果到缓存"""
        if not config.transcription.cache_enabled:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                result_json = result.json()
                await db.execute(
                    """
                    INSERT OR REPLACE INTO transcription_cache 
                    (file_hash, file_name, result) 
                    VALUES (?, ?, ?)
                    """,
                    (result.file_hash, result.file_name, result_json)
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


# 全局数据库管理器实例
db_manager = DatabaseManager()