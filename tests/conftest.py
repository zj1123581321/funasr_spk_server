"""
pytest 共享 fixtures。

仅放跨 unit/integration 通用的基础设施。具体引擎/数据库 mock 放到各自模块。
"""
import sys
import os
from pathlib import Path

# 让测试代码能 import src.* —— 项目还未发布成包
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"
GOLDEN_DIR = FIXTURES_DIR / "golden"


@pytest.fixture(scope="session")
def audio_dir() -> Path:
    """测试音频目录"""
    return AUDIO_DIR


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """golden baseline 快照目录"""
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def podcast_audio(audio_dir: Path) -> Path:
    """60 秒双人播客音频"""
    p = audio_dir / "podcast_2speakers_60s.wav"
    assert p.exists(), f"测试音频缺失: {p}"
    return p


@pytest.fixture(scope="session")
def tts_audio(audio_dir: Path) -> Path:
    """5 秒单人 TTS 音频"""
    p = audio_dir / "tts_1speaker_5s.wav"
    assert p.exists(), f"测试音频缺失: {p}"
    return p


@pytest.fixture(scope="session")
def silence_audio(audio_dir: Path) -> Path:
    """5 秒静音"""
    p = audio_dir / "silence_5s.wav"
    assert p.exists(), f"测试音频缺失: {p}"
    return p
