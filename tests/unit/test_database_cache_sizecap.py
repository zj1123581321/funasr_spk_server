"""
P4 A1 — DB 缓存 size-cap + clean_old_cache 时间戳格式 regression

eng-review/codex 二审定案:
- size-cap:`clean_old_cache` 在 TTL 清理后,若 COUNT > max_cache_count,按
  accessed_at ASC, id ASC(true LRU + 确定性 tie-breaker)挤到上限。软上限兜底。
- 时间戳 regression(codex):TTL 比较把 Python isoformat()(`T` 分隔)和 DB 的
  CURRENT_TIMESTAMP(空格分隔)做字符串比较,`'T'(0x54) > ' '(0x20)` 会误删
  cutoff 同日、时间更晚(应保留)的行。修:cutoff 用 `%Y-%m-%d %H:%M:%S`(空格)对齐 DB。
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from src.core.config import config
from src.core.database import DatabaseManager


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "sizecap_cache.db")


async def _insert_row(db_path: str, file_hash: str, *, created_at: str, accessed_at: str):
    """直接插一行 cache(绕过 save_result, 精确控制 created_at/accessed_at)。"""
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """INSERT INTO transcription_cache
               (file_hash, file_name, result, raw_result, duration, processing_time,
                engine, created_at, accessed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_hash, f"{file_hash}.wav", "{}", "{}", 1.0, 0.1, "funasr",
             created_at, accessed_at),
        )
        await conn.commit()


async def _count(db_path: str) -> int:
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM transcription_cache")
        return (await cur.fetchone())[0]


async def _hashes(db_path: str) -> set:
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT file_hash FROM transcription_cache")
        return {r[0] for r in await cur.fetchall()}


def _fmt(dt: datetime) -> str:
    """DB CURRENT_TIMESTAMP 同格式(空格分隔)。"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class TestTimestampRegression:
    @pytest.mark.asyncio
    async def test_boundary_row_newer_than_cutoff_is_kept(self, tmp_db_path, monkeypatch):
        """cutoff 同日但时间更晚(实际比 cutoff 新)的行必须保留。

        旧 bug:cutoff 用 isoformat()(T 分隔),与空格分隔的 created_at 字符串比较时
        ' ' < 'T' 把该行误判为更老 → 误删。修后两端都空格分隔,比较正确。
        """
        monkeypatch.setattr(config.database, "max_cache_days", 30)
        monkeypatch.setattr(config.database, "max_cache_count", 0)  # 关 size-cap, 只测 TTL
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()

        cutoff_dt = datetime.now() - timedelta(days=30)
        # 比 cutoff 新 1 秒 → 应保留。多数情况与 cutoff 同日, 正中 bug;
        # 极小概率(cutoff 在 xx:xx:59)跨到次日, 此时仍应保留(更不该删), 测试不会假失败。
        keep_dt = cutoff_dt + timedelta(seconds=1)
        await _insert_row(tmp_db_path, "boundary_keep",
                          created_at=_fmt(keep_dt), accessed_at=_fmt(keep_dt))

        await db.clean_old_cache()
        assert "boundary_keep" in await _hashes(tmp_db_path), \
            "cutoff 同日、时间更晚的行被误删 → 时间戳格式 bug 未修"

    @pytest.mark.asyncio
    async def test_ttl_deletes_old_keeps_new(self, tmp_db_path, monkeypatch):
        monkeypatch.setattr(config.database, "max_cache_days", 30)
        monkeypatch.setattr(config.database, "max_cache_count", 0)
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()

        now = datetime.now()
        await _insert_row(tmp_db_path, "old",
                          created_at=_fmt(now - timedelta(days=40)),
                          accessed_at=_fmt(now - timedelta(days=40)))
        await _insert_row(tmp_db_path, "fresh",
                          created_at=_fmt(now), accessed_at=_fmt(now))

        await db.clean_old_cache()
        h = await _hashes(tmp_db_path)
        assert "old" not in h and "fresh" in h


class TestSizeCap:
    @pytest.mark.asyncio
    async def test_over_cap_trims_to_limit_keeping_recent(self, tmp_db_path, monkeypatch):
        """超上限按 accessed_at LRU 挤,保留最近访问的 N 条。"""
        monkeypatch.setattr(config.database, "max_cache_days", 3650)  # TTL 不触发
        monkeypatch.setattr(config.database, "max_cache_count", 3)
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()

        now = datetime.now()
        # 5 行, accessed_at 递增(r4 最新)
        for i in range(5):
            await _insert_row(
                tmp_db_path, f"r{i}",
                created_at=_fmt(now - timedelta(days=10)),
                accessed_at=_fmt(now - timedelta(hours=(5 - i))),  # r0 最旧, r4 最新
            )

        await db.clean_old_cache()
        assert await _count(tmp_db_path) == 3, "应挤到 max_cache_count=3"
        kept = await _hashes(tmp_db_path)
        # 保留最近访问的 r2/r3/r4, 挤掉最旧的 r0/r1
        assert kept == {"r2", "r3", "r4"}, f"应保留最近访问的 3 条, 实际 {kept}"

    @pytest.mark.asyncio
    async def test_under_cap_no_trim(self, tmp_db_path, monkeypatch):
        monkeypatch.setattr(config.database, "max_cache_days", 3650)
        monkeypatch.setattr(config.database, "max_cache_count", 10)
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        now = datetime.now()
        for i in range(3):
            await _insert_row(tmp_db_path, f"u{i}",
                              created_at=_fmt(now), accessed_at=_fmt(now))
        await db.clean_old_cache()
        assert await _count(tmp_db_path) == 3, "未超上限不该删"

    @pytest.mark.asyncio
    async def test_cap_zero_disables_sizecap(self, tmp_db_path, monkeypatch):
        """max_cache_count<=0 关闭 size-cap(防误配清空), 只走 TTL。"""
        monkeypatch.setattr(config.database, "max_cache_days", 3650)
        monkeypatch.setattr(config.database, "max_cache_count", 0)
        db = DatabaseManager(db_path=tmp_db_path)
        await db.init_db()
        now = datetime.now()
        for i in range(6):
            await _insert_row(tmp_db_path, f"z{i}",
                              created_at=_fmt(now), accessed_at=_fmt(now))
        await db.clean_old_cache()
        assert await _count(tmp_db_path) == 6, "size-cap 关闭时不该按数量删"


class TestConfigField:
    def test_max_cache_count_default(self):
        """DatabaseConfig 有 max_cache_count 字段, 默认 5000。"""
        from src.core.config import DatabaseConfig
        assert DatabaseConfig().max_cache_count == 5000
