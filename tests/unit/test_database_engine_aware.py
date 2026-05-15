"""
PR1 — database engine-aware cache 测试

验证：
1. DatabaseManager 可以接受自定义 db_path（提升可测性）
2. transcription_cache 表 schema 包含 engine 列
3. UNIQUE 约束变为 (file_hash, engine)，同一文件不同引擎可共存
4. get_cached_result 接受 engine 参数，按 engine 区分缓存命中
5. save_result 接受 engine 参数，正确写入
6. init_db 对旧 schema（无 engine 列）执行迁移
"""
import json
import os
from datetime import datetime
from pathlib import Path

import aiosqlite
import pytest

from src.core.database import DatabaseManager
from src.models.schemas import TranscriptionResult, TranscriptionSegment


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """每个测试一个独立 SQLite 文件"""
    return str(tmp_path / "test_cache.db")


@pytest.fixture
async def fresh_db(tmp_db_path: str) -> DatabaseManager:
    """全新 DB（不经迁移），engine 列直接存在"""
    db = DatabaseManager(db_path=tmp_db_path)
    await db.init_db()
    return db


def make_result(file_hash: str = "abc", task_id: str = "t-1") -> TranscriptionResult:
    """造一个最小的 TranscriptionResult，仅用于 save/get 测试"""
    return TranscriptionResult(
        task_id=task_id,
        file_name="test.wav",
        file_hash=file_hash,
        duration=10.0,
        segments=[TranscriptionSegment(start_time=0.0, end_time=1.0, text="你好", speaker="Speaker1")],
        speakers=["Speaker1"],
        processing_time=0.5,
    )


class TestDatabaseManagerAcceptsCustomPath:
    def test_init_with_custom_path(self, tmp_db_path: str):
        db = DatabaseManager(db_path=tmp_db_path)
        assert db.db_path == tmp_db_path

    def test_init_falls_back_to_config_when_no_path(self):
        """不传 db_path 时使用全局 config（兼容旧单例用法）"""
        from src.core.config import config
        db = DatabaseManager()
        assert db.db_path == config.database.path


class TestFreshSchemaHasEngineColumn:
    @pytest.mark.asyncio
    async def test_fresh_db_has_engine_column(self, tmp_db_path: str):
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute("PRAGMA table_info(transcription_cache)")
            cols = {row[1] for row in await cursor.fetchall()}
        assert "engine" in cols, f"engine 列缺失，现有列: {cols}"

    @pytest.mark.asyncio
    async def test_fresh_db_unique_is_file_hash_plus_engine(self, tmp_db_path: str):
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute("PRAGMA index_list(transcription_cache)")
            indexes = await cursor.fetchall()
            unique_indexes = [idx for idx in indexes if idx[2] == 1]
            # 找出包含 file_hash 和 engine 的 UNIQUE 索引
            found = False
            for idx in unique_indexes:
                cursor2 = await conn.execute(f"PRAGMA index_info({idx[1]})")
                cols = {row[2] for row in await cursor2.fetchall()}
                if "file_hash" in cols and "engine" in cols:
                    found = True
                    break
        assert found, f"未找到 (file_hash, engine) UNIQUE 索引: {unique_indexes}"


class TestSaveAndGetWithEngine:
    @pytest.mark.asyncio
    async def test_save_with_engine_then_get_same_engine_hits(self, tmp_db_path: str):
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        result = make_result()
        await db.save_result(result, raw_result=None, engine="funasr")
        cached = await db.get_cached_result(result.file_hash, engine="funasr")
        assert cached is not None
        assert cached.file_hash == result.file_hash

    @pytest.mark.asyncio
    async def test_save_with_engine_then_get_different_engine_misses(self, tmp_db_path: str):
        """同一 file_hash 在 funasr 引擎下有缓存，查 qwen3 引擎应 miss"""
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        result = make_result()
        await db.save_result(result, raw_result=None, engine="funasr")
        cached = await db.get_cached_result(result.file_hash, engine="qwen3")
        assert cached is None, "不同引擎不应命中"

    @pytest.mark.asyncio
    async def test_two_engines_coexist_for_same_file(self, tmp_db_path: str):
        """同一 file_hash 在不同引擎下都能各自缓存"""
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        result_a = make_result(file_hash="dup", task_id="ta")
        result_b = make_result(file_hash="dup", task_id="tb")
        await db.save_result(result_a, raw_result=None, engine="funasr")
        await db.save_result(result_b, raw_result=None, engine="qwen3")

        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute(
                "SELECT engine FROM transcription_cache WHERE file_hash = ?",
                ("dup",),
            )
            engines = {row[0] for row in await cursor.fetchall()}
        assert engines == {"funasr", "qwen3"}


class TestLegacySchemaMigration:
    @pytest.mark.asyncio
    async def test_init_db_migrates_legacy_schema(self, tmp_db_path: str):
        """模拟旧库：file_hash UNIQUE 没有 engine 列；init 后应迁移"""
        # 1. 手工创建旧 schema
        async with aiosqlite.connect(tmp_db_path) as conn:
            await conn.execute("""
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
            # 插入一条旧数据
            await conn.execute(
                """INSERT INTO transcription_cache
                   (file_hash, file_name, result, raw_result, duration, processing_time)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("legacy_hash", "old.wav", "{}", None, 5.0, 0.1),
            )
            await conn.commit()

        # 2. 跑 init_db（迁移）
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()

        # 3. 验证：engine 列存在且旧数据 engine='funasr'
        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute("PRAGMA table_info(transcription_cache)")
            cols = {row[1] for row in await cursor.fetchall()}
            assert "engine" in cols
            cursor = await conn.execute(
                "SELECT engine FROM transcription_cache WHERE file_hash = ?",
                ("legacy_hash",),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "funasr", "旧数据 engine 应默认填充为 funasr"

        # 4. 验证：迁移后能再插入同 file_hash 不同 engine
        result = make_result(file_hash="legacy_hash", task_id="new")
        await db.save_result(result, raw_result=None, engine="qwen3")
        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute(
                "SELECT engine FROM transcription_cache WHERE file_hash = ?",
                ("legacy_hash",),
            )
            engines = {row[0] for row in await cursor.fetchall()}
        assert engines == {"funasr", "qwen3"}

    @pytest.mark.asyncio
    async def test_init_db_is_idempotent(self, tmp_db_path: str):
        """init_db 重复跑不应出错（迁移已完成后再跑）"""
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        await db.init_db()  # 不应抛错


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_get_cached_without_engine_param_defaults_to_funasr(self, tmp_db_path: str):
        """旧调用方未传 engine 时，应当作 funasr 查询（兼容期）"""
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        result = make_result()
        await db.save_result(result, raw_result=None, engine="funasr")
        cached = await db.get_cached_result(result.file_hash)  # 无 engine 参数
        assert cached is not None

    @pytest.mark.asyncio
    async def test_save_without_engine_param_defaults_to_funasr(self, tmp_db_path: str):
        """旧调用方未传 engine 时，save 默认存为 funasr"""
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        result = make_result()
        await db.save_result(result, raw_result=None)  # 无 engine 参数
        async with aiosqlite.connect(tmp_db_path) as conn:
            cursor = await conn.execute(
                "SELECT engine FROM transcription_cache WHERE file_hash = ?",
                (result.file_hash,),
            )
            row = await cursor.fetchone()
        assert row[0] == "funasr"
