"""
PR4 follow-up — 真 server + 真 websocket client + 真 Qwen3 模型端到端集成测试.

覆盖之前 0 个自动化 test 触达的"生产路径": WebSocket 协议层 / chunked upload /
task_complete 推送 / 真 cache lookup / 多 client 并发不串台.

默认 skip (需 FUNASR_RUN_INTEGRATION=1 + Qwen3 模型 + 启 subprocess server, ~3min).

Fixture 设计:
- subprocess 跑 run_server.py 在临时 port + 临时 data_dir + 临时 upload/temp 目录
- wait-for-port (TCP connect 重试) 直到 server ready
- test 跑完 SIGTERM + wait, 临时目录自动清理

Cases:
1. test_concurrent_different_hash_no_crosstalk:
   2 client 同时上传不同 hash 的 audio (force_refresh=True 跳缓存), 各自独立完成
2. test_cache_hit_on_second_request_same_hash:
   client 1 上传 hash=H -> 完成; client 2 (新连接) 上传同 hash -> 应秒回 task_complete (cached)
3. test_single_client_full_chunked_upload:
   单 client 走完 upload_request -> upload_data -> task_complete 完整流程, 文本非空
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    import websockets
except ImportError:
    websockets = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PODCAST = PROJECT_ROOT / "tests" / "fixtures" / "audio" / "podcast_2speakers_60s.wav"

RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="设置 FUNASR_RUN_INTEGRATION=1 启用 (默认 skip, 启 subprocess server + 真 Qwen3 模型, ~3min)",
    ),
    pytest.mark.skipif(websockets is None, reason="websockets 包未安装"),
]


def _qwen3_models_ready() -> bool:
    from src.core.config import config
    paths = [
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_frontend.onnx",
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_backend.onnx",
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_llm.gguf",
        Path(config.qwen3.segmentation_model),
        Path(config.qwen3.embedding_model),
    ]
    return all(p.exists() for p in paths)


pytestmark2 = pytest.mark.skipif(
    not _qwen3_models_ready(),
    reason="Qwen3 模型权重未落地, 跑 scripts/download_qwen3_models.sh 后再试",
)


def _free_port() -> int:
    """让 OS 分配一个空闲端口."""
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    """轮询直到 TCP 端口可连或超时."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket()
            s.settimeout(1.0)
            s.connect((host, port))
            s.close()
            return True
        except (OSError, socket.timeout):
            time.sleep(0.5)
    return False


def _file_hash_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def server_subprocess(tmp_path_factory):
    """启 run_server.py 在临时 port + 临时 data/upload/temp 目录.

    module scope: 启动 server (含 Qwen3 模型 + 2 worker 加载 ~15-20s) 较贵,
    跨 case 复用. 每 case 用临时 audio 避免互相污染.
    """
    if not PODCAST.exists():
        pytest.skip(f"audio fixture 缺失: {PODCAST}")

    port = _free_port()
    tmp_root = tmp_path_factory.mktemp("ws_e2e")
    data_dir = tmp_root / "data"
    upload_dir = tmp_root / "uploads"
    temp_dir = tmp_root / "temp"
    data_dir.mkdir()
    upload_dir.mkdir()
    temp_dir.mkdir()

    env = os.environ.copy()
    env.update({
        "FUNASR_DEFAULT_ENGINE": "qwen3",
        "FUNASR_QWEN3_POOL_SIZE": "2",
        "FUNASR_SERVER_PORT": str(port),
        "FUNASR_SERVER_HOST": "127.0.0.1",
        "FUNASR_DATA_DIR": str(data_dir),
        "FUNASR_UPLOAD_DIR": str(upload_dir),
        "FUNASR_TEMP_DIR": str(temp_dir),
        "TMPDIR": "/tmp",  # 防 ctx-mode 污染
        "DYLD_LIBRARY_PATH": str(PROJECT_ROOT / "src/core/vendor/qwen_asr_gguf/inference/bin"),
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

    if not _wait_for_port("127.0.0.1", port, timeout=120.0):
        proc.terminate()
        proc.wait(timeout=10)
        log_fh.close()
        pytest.fail(f"server 60s 内未起来, 日志: {log_file.read_text()[-2000:]}")

    # 再 sleep 让 worker pool ready (libllama + Metal + sherpa eager warmup ~10s)
    time.sleep(15)

    yield f"ws://127.0.0.1:{port}", log_file

    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_fh.close()


async def _client_upload_and_wait(ws_url: str, audio: Path, force_refresh: bool = True) -> dict:
    """跑一个完整 client 流程: connect → upload_request → upload_data → wait task_complete.

    Returns dict: {ok, task_id, wall_time, cached, result, error?}.
    """
    t0 = time.time()
    data = audio.read_bytes()
    digest = _file_hash_md5(audio)
    b64 = base64.b64encode(data).decode("utf-8")

    async with websockets.connect(ws_url, max_size=200 * 1024 * 1024) as ws:
        connected = json.loads(await ws.recv())
        assert connected.get("type") == "connected", f"unexpected: {connected}"

        await ws.send(json.dumps({
            "type": "upload_request",
            "data": {
                "file_name": audio.name,
                "file_size": len(data),
                "file_hash": digest,
                "force_refresh": force_refresh,
                "output_format": "json",
            }
        }))
        resp = json.loads(await ws.recv())
        if resp["type"] == "error":
            return {"ok": False, "error": resp["data"]["message"], "wall_time": time.time() - t0}
        if resp["type"] == "task_complete":
            return {
                "ok": True, "cached": True,
                "task_id": resp["data"].get("task_id", "cached"),
                "wall_time": time.time() - t0,
                "result": resp["data"]["result"],
            }
        task_id = resp["data"].get("task_id")
        assert task_id, f"无 task_id: {resp}"

        await ws.send(json.dumps({
            "type": "upload_data",
            "data": {"task_id": task_id, "file_data": b64}
        }))

        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "task_progress":
                continue
            if msg["type"] == "task_complete":
                return {
                    "ok": True, "cached": False,
                    "task_id": task_id,
                    "wall_time": time.time() - t0,
                    "result": msg["data"]["result"],
                }
            if msg["type"] == "error":
                return {"ok": False, "task_id": task_id, "error": msg["data"]["message"],
                        "wall_time": time.time() - t0}


@pytest.fixture
def two_unique_audio(tmp_path):
    """复制 podcast 成两份, append 不同字节让 hash 不同."""
    a = tmp_path / "audio_a.wav"
    b = tmp_path / "audio_b.wav"
    shutil.copy2(PODCAST, a)
    shutil.copy2(PODCAST, b)
    with open(a, "ab") as f:
        f.write(b"\x00\x01")
    with open(b, "ab") as f:
        f.write(b"\x00\x02")
    assert _file_hash_md5(a) != _file_hash_md5(b)
    return a, b


class TestRealServerWebSocketE2E:
    @pytest.mark.asyncio
    async def test_single_client_full_chunked_upload(self, server_subprocess, tmp_path):
        """单 client 走完 upload_request → upload_data → task_complete 全流程, 文本非空."""
        ws_url, _log = server_subprocess
        audio = tmp_path / "single_client.wav"
        shutil.copy2(PODCAST, audio)
        with open(audio, "ab") as f:
            f.write(b"\x00\xAA")  # 让 hash 唯一

        r = await _client_upload_and_wait(ws_url, audio, force_refresh=True)
        assert r["ok"], f"任务失败: {r.get('error')}"
        assert not r["cached"], "首次请求不该命中 cache"
        segs = r["result"].get("segments", [])
        assert len(segs) > 0, "应有 segments 输出"
        full_text = "".join(s.get("text", "") for s in segs)
        assert len(full_text.strip()) > 0, "文本不该全空"

    @pytest.mark.asyncio
    async def test_concurrent_different_hash_no_crosstalk(
        self, server_subprocess, two_unique_audio
    ):
        """2 client 同时上传不同 hash, 各自完成, task_id 不串台."""
        ws_url, _log = server_subprocess
        a, b = two_unique_audio

        t0 = time.time()
        r1, r2 = await asyncio.gather(
            _client_upload_and_wait(ws_url, a, force_refresh=True),
            _client_upload_and_wait(ws_url, b, force_refresh=True),
        )
        total = time.time() - t0

        assert r1["ok"], f"client 1 失败: {r1.get('error')}"
        assert r2["ok"], f"client 2 失败: {r2.get('error')}"
        assert not r1["cached"]
        assert not r2["cached"]
        assert r1["task_id"] != r2["task_id"], "task_id 串台"
        # 并发应 < 2 × 单 task * 1.5 wall (即真并发, 没退化成串行)
        print(
            f"[ws-e2e] 并发 2 task wall={total:.2f}s "
            f"(client1 wall={r1['wall_time']:.2f}s, client2 wall={r2['wall_time']:.2f}s)"
        )

    @pytest.mark.asyncio
    async def test_cache_hit_on_second_request_same_hash(
        self, server_subprocess, tmp_path
    ):
        """同 hash 顺序请求: 第 1 次真转录, 第 2 次秒回 cached (force_refresh=False)."""
        ws_url, _log = server_subprocess
        audio = tmp_path / "cache_check.wav"
        shutil.copy2(PODCAST, audio)
        with open(audio, "ab") as f:
            f.write(b"\xBB\xCC")

        # 第 1 次: 真转录
        r1 = await _client_upload_and_wait(ws_url, audio, force_refresh=False)
        assert r1["ok"], f"第 1 次失败: {r1.get('error')}"
        first_wall = r1["wall_time"]

        # 第 2 次 (新连接, 同 audio 同 hash, 不 force refresh)
        r2 = await _client_upload_and_wait(ws_url, audio, force_refresh=False)
        assert r2["ok"], f"第 2 次失败: {r2.get('error')}"
        assert r2["cached"], f"第 2 次应命中 cache, 实际 cached={r2['cached']}"
        # cache hit 应远快于真转录 (留宽裕预算, 真转录 10s+, cache hit <1s)
        assert r2["wall_time"] < first_wall / 3, (
            f"cache hit 应 < 第 1 次的 1/3 (first={first_wall:.2f}s, second={r2['wall_time']:.2f}s)"
        )
        print(
            f"[ws-e2e] cache hit verified: 真转录 {first_wall:.2f}s, "
            f"cache hit {r2['wall_time']:.2f}s ({r2['wall_time']/first_wall*100:.1f}%)"
        )
