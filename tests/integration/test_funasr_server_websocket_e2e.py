"""真 server + 真 websocket client + 真 FunASR 模型端到端集成测试。

补 funasr 端到端覆盖的缺口: 此前 funasr 只有 parity (直调 transcribe), **不经过**
task_manager._process_task 的结果处理层 (save_result + word_align error 提取 + metadata
组装)。2026-06-16 生产事故 ('list' object has no attribute 'get') 正出在那段 —— funasr
的 raw_result 是 list, _process_task JSON 分支误当 dict 调 .get()。parity 抓不到, 只能
生产暴露。本 e2e 走完整 ws → task_manager → _process_task, 把这类"转录后处理"回归挡在
集成层而非生产。

默认 skip (需 FUNASR_RUN_INTEGRATION=1 + FunASR 模型 + 启 subprocess server)。

Cases:
1. JSON 全流程完成 + segments/speakers/metadata 非空 (覆盖出事的 _process_task JSON 分支)
2. SRT 全流程完成 + srt_content 非空 (覆盖 SRT 分支)
3. 同 hash 第二次秒回 cached
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.integration._ws_e2e_common import (
    client_upload_and_wait,
    file_hash_md5,
    free_port,
    wait_for_port,
    websockets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PODCAST = PROJECT_ROOT / "tests" / "fixtures" / "audio" / "podcast_2speakers_60s.wav"

RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="设置 FUNASR_RUN_INTEGRATION=1 启用 (默认 skip, 启 subprocess server + 真 FunASR 模型)",
    ),
    pytest.mark.skipif(websockets is None, reason="websockets 包未安装"),
]


@pytest.fixture(scope="module")
def funasr_server_subprocess(tmp_path_factory):
    """启 run_server.py (engine=funasr) 在临时 port + 临时 data/upload/temp 目录。

    module scope 跨 case 复用 (FunASR 模型加载较贵)。每 case 用唯一 hash audio 避免污染。
    """
    if not PODCAST.exists():
        pytest.skip(f"audio fixture 缺失: {PODCAST}")

    port = free_port()
    tmp_root = tmp_path_factory.mktemp("funasr_ws_e2e")
    data_dir = tmp_root / "data"
    upload_dir = tmp_root / "uploads"
    temp_dir = tmp_root / "temp"
    data_dir.mkdir()
    upload_dir.mkdir()
    temp_dir.mkdir()

    env = os.environ.copy()
    env.update({
        "FUNASR_DEFAULT_ENGINE": "funasr",
        "FUNASR_SERVER_PORT": str(port),
        "FUNASR_SERVER_HOST": "127.0.0.1",
        "FUNASR_DATA_DIR": str(data_dir),
        "FUNASR_UPLOAD_DIR": str(upload_dir),
        "FUNASR_TEMP_DIR": str(temp_dir),
        "TMPDIR": "/tmp",  # 防 ctx-mode 污染
    })

    log_file = tmp_root / "server.log"
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(
        [sys.executable, "run_server.py"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    # FunASR 首次可能需下载模型 → 给宽裕超时
    if not wait_for_port("127.0.0.1", port, timeout=180.0):
        proc.terminate()
        proc.wait(timeout=10)
        log_fh.close()
        pytest.fail(f"server 未起来, 日志尾部:\n{log_file.read_text()[-2000:]}")

    time.sleep(8)  # 让 worker pool ready

    yield f"ws://127.0.0.1:{port}", log_file

    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_fh.close()


def _unique_audio(tmp_path, suffix_bytes: bytes) -> Path:
    """复制 podcast 并 append 唯一字节让 hash 不同。"""
    audio = tmp_path / f"funasr_e2e_{suffix_bytes.hex()}.wav"
    shutil.copy2(PODCAST, audio)
    with open(audio, "ab") as f:
        f.write(suffix_bytes)
    return audio


class TestFunasrRealServerWebSocketE2E:
    @pytest.mark.asyncio
    async def test_json_full_flow_completes(self, funasr_server_subprocess, tmp_path):
        """JSON 全流程: 走完整 _process_task JSON 结果处理层不崩 (回归本次生产事故)。"""
        ws_url, log_file = funasr_server_subprocess
        audio = _unique_audio(tmp_path, b"\x00\x01")

        r = await client_upload_and_wait(ws_url, audio, force_refresh=True, output_format="json")
        assert r["ok"], (
            f"funasr JSON 任务失败: {r.get('error')}\n服务日志尾部:\n"
            f"{log_file.read_text()[-1500:]}"
        )
        segs = r["result"].get("segments", [])
        assert len(segs) > 0, "应有 segments 输出"
        full_text = "".join(s.get("text", "") for s in segs)
        assert full_text.strip(), "转录文本不该全空"
        # metadata 由 _process_task 出口组装 (build_result_metadata 也在出事那段之后)
        meta = r["result"].get("metadata") or {}
        assert meta.get("engine") == "funasr", f"metadata.engine 应为 funasr: {meta}"
        # funasr 无 word_align, 不应有失败残留
        assert meta.get("word_align_error") is None

    @pytest.mark.asyncio
    async def test_srt_full_flow_completes(self, funasr_server_subprocess, tmp_path):
        """SRT 全流程: 覆盖 _process_task 的 SRT 分支不崩。"""
        ws_url, log_file = funasr_server_subprocess
        audio = _unique_audio(tmp_path, b"\x00\x02")

        r = await client_upload_and_wait(ws_url, audio, force_refresh=True, output_format="srt")
        assert r["ok"], (
            f"funasr SRT 任务失败: {r.get('error')}\n服务日志尾部:\n"
            f"{log_file.read_text()[-1500:]}"
        )
        # SRT 结果走 dict (format=srt + content)
        result = r["result"]
        content = result.get("content") if isinstance(result, dict) else None
        assert content and content.strip(), f"SRT content 不该空: {result}"

    @pytest.mark.asyncio
    async def test_cache_hit_second_request(self, funasr_server_subprocess, tmp_path):
        """同 hash 顺序请求: 第 1 次真转录, 第 2 次秒回 cached。"""
        ws_url, _log = funasr_server_subprocess
        audio = _unique_audio(tmp_path, b"\x00\x03")

        r1 = await client_upload_and_wait(ws_url, audio, force_refresh=False, output_format="json")
        assert r1["ok"], f"第 1 次失败: {r1.get('error')}"
        first_wall = r1["wall_time"]

        r2 = await client_upload_and_wait(ws_url, audio, force_refresh=False, output_format="json")
        assert r2["ok"], f"第 2 次失败: {r2.get('error')}"
        assert r2["cached"], f"第 2 次应命中 cache, 实际 cached={r2['cached']}"
        assert r2["wall_time"] < first_wall / 3, (
            f"cache hit 应远快于真转录 (first={first_wall:.2f}s, second={r2['wall_time']:.2f}s)"
        )
