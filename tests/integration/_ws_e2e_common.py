"""真 server + 真 websocket client 端到端测试的共享 helper（引擎无关）。

被 test_funasr_server_websocket_e2e.py 使用。下划线前缀 → pytest 不收集本文件。

NOTE: test_qwen3_server_websocket_e2e.py 目前仍有一份自带的平行 helper（历史），
后续可迁移到本模块统一（DRY）。本模块先服务 funasr e2e，不动既有 qwen3 e2e（避免回归既有测试）。
"""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import time
from pathlib import Path

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None


def free_port() -> int:
    """让 OS 分配一个空闲端口。"""
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for_port(host: str, port: int, timeout: float = 120.0) -> bool:
    """轮询直到 TCP 端口可连或超时。"""
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


def file_hash_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


async def client_upload_and_wait(
    ws_url: str,
    audio: Path,
    *,
    force_refresh: bool = True,
    output_format: str = "json",
    extra_request: dict | None = None,
) -> dict:
    """完整 client 流程: connect → upload_request → upload_data → 等 task_complete。

    走真 server → task_manager → _process_task 结果处理层（这正是 parity 直调
    transcribe 覆盖不到的那段）。

    Returns dict: {ok, task_id, wall_time, cached, result, error?}。
    """
    t0 = time.time()
    data = audio.read_bytes()
    digest = file_hash_md5(audio)
    b64 = base64.b64encode(data).decode("utf-8")

    request_data = {
        "file_name": audio.name,
        "file_size": len(data),
        "file_hash": digest,
        "force_refresh": force_refresh,
        "output_format": output_format,
    }
    if extra_request:
        request_data.update(extra_request)

    async with websockets.connect(ws_url, max_size=200 * 1024 * 1024) as ws:
        connected = json.loads(await ws.recv())
        assert connected.get("type") == "connected", f"unexpected: {connected}"

        await ws.send(json.dumps({"type": "upload_request", "data": request_data}))
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
            "data": {"task_id": task_id, "file_data": b64},
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
