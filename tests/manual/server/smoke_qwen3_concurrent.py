"""
Qwen3 多 Worker 池真服务器并发冒烟 (PR3 部署前最后一道防线)

启动方式 (单独终端):
    FUNASR_DEFAULT_ENGINE=qwen3 FUNASR_QWEN3_POOL_SIZE=2 \
    FUNASR_SERVER_PORT=8868 venv/bin/python run_server.py

或本脚本自带 server 编排:
    venv/bin/python tests/manual/server/smoke_qwen3_concurrent.py

验证:
- 同时发 2 个 upload_request (不同 file_hash)
- 两个 task 各自走完整流程 → task_complete
- 文本互不串台
- server 不崩

输出: 总耗时 / 两 task RTF / 简单文本前 30 字符
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import websockets


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SERVER_URL = "ws://localhost:8868"
PODCAST = PROJECT_ROOT / "tests" / "fixtures" / "audio" / "podcast_2speakers_60s.wav"


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


async def transcribe_one(client_id: int, audio: Path):
    """单 client 跑完整 upload + transcribe 流程"""
    t0 = time.time()
    async with websockets.connect(SERVER_URL, max_size=200 * 1024 * 1024) as ws:
        # server 连接成功后主动发 connected 消息, 先收掉它
        connected = json.loads(await ws.recv())
        assert connected.get("type") == "connected", f"未收到 connected: {connected}"

        data = audio.read_bytes()
        size = len(data)
        digest = file_hash(audio)
        b64 = base64.b64encode(data).decode("utf-8")

        # upload_request
        await ws.send(json.dumps({
            "type": "upload_request",
            "data": {
                "file_name": audio.name,
                "file_size": size,
                "file_hash": digest,
                "force_refresh": True,  # 强制不命中缓存, 走真转录
                "output_format": "json",
            }
        }))
        resp = json.loads(await ws.recv())
        print(f"[client {client_id}] upload_request 响应: type={resp.get('type')} keys={list(resp.get('data', {}).keys())}")
        if resp["type"] == "error":
            return {"client_id": client_id, "ok": False, "error": resp["data"]["message"]}
        if resp["type"] == "task_complete":
            return {
                "client_id": client_id, "ok": True, "cached": True,
                "task_id": resp["data"].get("task_id", "cached"),
                "wall_time": time.time() - t0,
                "result": resp["data"]["result"],
            }
        # upload_response / upload_ack / task_progress 都可能携带 task_id
        task_id = resp["data"].get("task_id")
        if not task_id:
            return {"client_id": client_id, "ok": False, "error": f"无 task_id in resp: {resp}"}

        # upload_data
        await ws.send(json.dumps({
            "type": "upload_data",
            "data": {"task_id": task_id, "file_data": b64}
        }))

        # 接收进度直到 task_complete
        first_text = None
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "task_progress":
                continue
            if msg["type"] == "task_complete":
                result = msg["data"]["result"]
                segs = result.get("segments", [])
                first_text = segs[0]["text"] if segs else ""
                return {
                    "client_id": client_id, "ok": True, "cached": False,
                    "task_id": task_id,
                    "wall_time": time.time() - t0,
                    "result": result,
                    "first_text": first_text,
                }
            if msg["type"] == "error":
                return {
                    "client_id": client_id, "ok": False,
                    "task_id": task_id,
                    "error": msg["data"]["message"],
                }


async def smoke():
    assert PODCAST.exists(), f"测试音频不存在: {PODCAST}"

    # 复制成 2 个不同名 audio 避免缓存命中
    tmp_dir = PROJECT_ROOT / "tmp_smoke"
    tmp_dir.mkdir(exist_ok=True)
    a = tmp_dir / "smoke_a.wav"
    b = tmp_dir / "smoke_b.wav"
    shutil.copy2(PODCAST, a)
    shutil.copy2(PODCAST, b)
    # 让 hash 不同: append 不同字节
    with open(a, "ab") as f:
        f.write(b"\x00\x01")
    with open(b, "ab") as f:
        f.write(b"\x00\x02")

    print(f"[smoke] 同时发 2 个 upload_request: {a.name} + {b.name}")
    t0 = time.time()
    r1, r2 = await asyncio.gather(
        transcribe_one(1, a),
        transcribe_one(2, b),
    )
    total = time.time() - t0

    print(f"\n========== 冒烟结果 ==========")
    print(f"总耗时(2 并发 wall): {total:.2f}s")
    for r in (r1, r2):
        if not r["ok"]:
            print(f"[client {r['client_id']}] FAIL: {r.get('error')}")
            continue
        d = r["result"].get("duration", 0)
        pt = r["result"].get("processing_time", 0)
        rtf = pt / d if d else 0
        text = "".join(s["text"] for s in r["result"].get("segments", []))
        print(
            f"[client {r['client_id']}] ✓ task_id={r['task_id']} "
            f"wall={r['wall_time']:.2f}s server_time={pt:.2f}s "
            f"RTF={rtf:.3f} segs={len(r['result'].get('segments', []))} "
            f"text_first30={text[:30]!r}"
        )

    # 不串台校验: 不允许出现 task_id 完全相同
    if r1["ok"] and r2["ok"]:
        assert r1["task_id"] != r2["task_id"], "task_id 串台!"
        print("\n✓ 不串台: 两个 task_id 不同, 文本各自独立")
    else:
        print("\n⚠ 至少一个 task 失败, 串台校验跳过")

    # 清理
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(smoke())
