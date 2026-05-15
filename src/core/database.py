"""
数据库管理模块

PR1 变更：
- DatabaseManager 接受可选 db_path（提升可测性）
- transcription_cache 表新增 engine 列
- UNIQUE 约束从 file_hash 单列改为 (file_hash, engine)
- init_db 对旧 schema 自动迁移（加列 + 重建索引）
- get_cached_result / save_result 增加 engine 参数（默认 funasr 保持向后兼容）
"""
import aiosqlite
import json
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
from loguru import logger
from src.core.config import config
from src.models.schemas import TranscriptionResult


# 数据库 schema 当前期望的引擎列默认值（迁移和旧调用兼容时使用）
DEFAULT_ENGINE = "funasr"


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: 可选的数据库文件路径。不传则使用 config.database.path（保持现有单例语义）。
                     传入路径用于测试隔离或多实例场景。
        """
        self.db_path = db_path if db_path is not None else config.database.path
        self._ensure_db_dir()

    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self):
        """初始化数据库（含旧 schema 自动迁移）"""
        async with aiosqlite.connect(self.db_path) as db:
            # 1. 建基础表（已存在则跳过）
            #    注：fresh 创建时这里包含 engine 列；
            #    旧库（无 engine 列）后续会通过 ALTER 补齐。
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transcription_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    result TEXT NOT NULL,
                    raw_result TEXT,
                    duration REAL,
                    processing_time REAL,
                    engine TEXT NOT NULL DEFAULT 'funasr',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. 列升级（兼容旧库）
            cursor = await db.execute("PRAGMA table_info(transcription_cache)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

            if 'raw_result' not in column_names:
                await db.execute("ALTER TABLE transcription_cache ADD COLUMN raw_result TEXT")
                logger.info("数据库迁移：新增 raw_result 列")

            if 'duration' not in column_names:
                await db.execute("ALTER TABLE transcription_cache ADD COLUMN duration REAL")
                logger.info("数据库迁移：新增 duration 列")

            if 'processing_time' not in column_names:
                await db.execute("ALTER TABLE transcription_cache ADD COLUMN processing_time REAL")
                logger.info("数据库迁移：新增 processing_time 列")

            if 'engine' not in column_names:
                # 旧库迁移：加 engine 列 + 已有行回填为 funasr
                await db.execute(
                    "ALTER TABLE transcription_cache ADD COLUMN engine TEXT NOT NULL DEFAULT 'funasr'"
                )
                logger.info("数据库迁移：新增 engine 列，旧数据默认 engine=funasr")

            # 3. 重建索引：丢掉旧的 file_hash 单列 UNIQUE，建 (file_hash, engine) 复合 UNIQUE
            #    SQLite 的 CREATE TABLE IF NOT EXISTS 不会重建 UNIQUE 约束，需要靠索引层处理。
            #    PRAGMA index_list 检查
            cursor = await db.execute("PRAGMA index_list(transcription_cache)")
            indexes = await cursor.fetchall()
            for idx in indexes:
                # idx: (seq, name, unique, origin, partial)
                name = idx[1]
                is_unique = idx[2] == 1
                if not is_unique:
                    continue
                cursor2 = await db.execute(f"PRAGMA index_info({name})")
                idx_cols = [row[2] for row in await cursor2.fetchall()]
                # 旧的单列 file_hash UNIQUE（自动命名 sqlite_autoindex_... 或我们的旧索引）
                if idx_cols == ["file_hash"]:
                    # SQLite 不允许 DROP INDEX 由 UNIQUE 约束自动生成的 sqlite_autoindex_*
                    # 这种情况下需要重建表
                    if name.startswith("sqlite_autoindex_"):
                        await self._rebuild_table_for_unique_migration(db)
                    else:
                        await db.execute(f"DROP INDEX IF EXISTS {name}")
                        logger.info(f"数据库迁移：删除旧 UNIQUE 索引 {name}")

            # 4. 创建新的复合 UNIQUE 索引（已存在则跳过）
            await db.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_cache_hash_engine
                    ON transcription_cache(file_hash, engine)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON transcription_cache(file_hash)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON transcription_cache(created_at)
            """)

            await db.commit()
            logger.info("数据库初始化完成")

    async def _rebuild_table_for_unique_migration(self, db: aiosqlite.Connection):
        """
        旧 schema 的 file_hash UNIQUE 是 CREATE TABLE 内联约束，
        对应 sqlite_autoindex_* 不允许 DROP INDEX。
        策略：拷贝数据到新表，drop 旧表，rename。
        """
        logger.warning("数据库迁移：检测到旧 file_hash UNIQUE 内联约束，开始重建表")
        await db.execute("""
            CREATE TABLE transcription_cache_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                file_name TEXT NOT NULL,
                result TEXT NOT NULL,
                raw_result TEXT,
                duration REAL,
                processing_time REAL,
                engine TEXT NOT NULL DEFAULT 'funasr',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 拷贝时回填 engine='funasr'
        await db.execute("""
            INSERT INTO transcription_cache_new
                (id, file_hash, file_name, result, raw_result, duration, processing_time, engine, created_at, accessed_at)
            SELECT id, file_hash, file_name, result, raw_result, duration, processing_time,
                   COALESCE(engine, 'funasr'), created_at, accessed_at
            FROM transcription_cache
        """)
        await db.execute("DROP TABLE transcription_cache")
        await db.execute("ALTER TABLE transcription_cache_new RENAME TO transcription_cache")
        logger.info("数据库迁移：旧表重建完成，原有数据已保留并填充 engine=funasr")

    async def get_cached_result(
        self,
        file_hash: str,
        output_format: str = "json",
        engine: str = DEFAULT_ENGINE,
    ) -> Optional[TranscriptionResult]:
        """
        获取缓存的转录结果

        Args:
            file_hash: 文件 MD5
            output_format: 返回格式（json / srt），影响返回结构
            engine: ASR 引擎名，作为缓存 key 的一部分（避免不同引擎结果互相覆盖）
        """
        if not config.transcription.cache_enabled:
            return None

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT result, raw_result, file_name, duration "
                    "FROM transcription_cache WHERE file_hash = ? AND engine = ?",
                    (file_hash, engine),
                )
                row = await cursor.fetchone()

                if row:
                    await db.execute(
                        "UPDATE transcription_cache SET accessed_at = CURRENT_TIMESTAMP "
                        "WHERE file_hash = ? AND engine = ?",
                        (file_hash, engine),
                    )
                    await db.commit()

                    result_json, raw_result_json, file_name, duration = row

                    if output_format == "json":
                        result_data = json.loads(result_json)
                        return TranscriptionResult(**result_data)
                    elif output_format == "srt" and raw_result_json:
                        raw_result = json.loads(raw_result_json)
                        srt_content = self._generate_srt_from_raw_result(raw_result)
                        return {
                            "format": "srt",
                            "content": srt_content,
                            "file_name": file_name,
                            "file_hash": file_hash,
                            "duration": duration,
                        }

                return None
        except Exception as e:
            logger.error(f"获取缓存结果失败: {e}")
            return None

    async def save_result(
        self,
        result: TranscriptionResult,
        raw_result: dict = None,
        engine: str = DEFAULT_ENGINE,
    ):
        """
        保存转录结果到缓存

        Args:
            result: TranscriptionResult 对象
            raw_result: FunASR 原始输出（用于格式转换；其它引擎可能传 None）
            engine: ASR 引擎名，作为缓存 key 的一部分
        """
        if not config.transcription.cache_enabled:
            return

        try:
            async with aiosqlite.connect(self.db_path) as db:
                result_json = result.json()
                raw_result_json = json.dumps(raw_result) if raw_result else None

                await db.execute(
                    """
                    INSERT OR REPLACE INTO transcription_cache
                    (file_hash, file_name, result, raw_result, duration, processing_time, engine)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.file_hash,
                        result.file_name,
                        result_json,
                        raw_result_json,
                        result.duration,
                        result.processing_time,
                        engine,
                    ),
                )
                await db.commit()
                logger.debug(f"转录结果已缓存: {result.file_hash} (engine={engine})")
        except Exception as e:
            logger.error(f"保存缓存结果失败: {e}")

    async def clean_old_cache(self):
        """清理过期的缓存"""
        try:
            cutoff_date = datetime.now() - timedelta(days=config.database.max_cache_days)

            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM transcription_cache WHERE created_at < ?",
                    (cutoff_date.isoformat(),),
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
                cursor = await db.execute("SELECT COUNT(*) FROM transcription_cache")
                total_count = (await cursor.fetchone())[0]

                today = datetime.now().date()
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM transcription_cache WHERE DATE(created_at) = ?",
                    (today.isoformat(),),
                )
                today_count = (await cursor.fetchone())[0]

                cursor = await db.execute(
                    "SELECT SUM(LENGTH(result)) FROM transcription_cache"
                )
                cache_size = (await cursor.fetchone())[0] or 0

                return {
                    "total_count": total_count,
                    "today_count": today_count,
                    "cache_size_mb": round(cache_size / 1024 / 1024, 2),
                }
        except Exception as e:
            logger.error(f"获取缓存统计信息失败: {e}")
            return {"total_count": 0, "today_count": 0, "cache_size_mb": 0}

    def _generate_srt_from_raw_result(self, raw_result: dict) -> str:
        """从原始 FunASR 结果生成 SRT 格式字符串"""
        srt_lines = []

        if isinstance(raw_result, list) and len(raw_result) > 0:
            result_data = raw_result[0]
        elif isinstance(raw_result, dict):
            result_data = raw_result
        else:
            logger.warning(f"未知的结果格式: {type(raw_result)}")
            return ""

        if 'sentence_info' not in result_data:
            logger.warning("结果中没有sentence_info字段，可能转录失败")
            return ""

        sentences = result_data.get('sentence_info', [])

        for idx, sentence in enumerate(sentences, 1):
            start_ms = sentence.get('start', 0)
            end_ms = sentence.get('end', 0)
            start_time = self._ms_to_srt_time(start_ms)
            end_time = self._ms_to_srt_time(end_ms)
            text = sentence.get('text', '').strip()
            speaker_id = sentence.get('spk', 0)
            if isinstance(speaker_id, int):
                speaker = f"Speaker{speaker_id + 1}"
            else:
                speaker = "Speaker1"

            if text:
                srt_lines.append(f"{idx}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(f"{speaker}:{text}")
                srt_lines.append("")

        return "\n".join(srt_lines)

    def _ms_to_srt_time(self, milliseconds: int) -> str:
        """将毫秒转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
        seconds = milliseconds / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# 全局数据库管理器实例
db_manager = DatabaseManager()
