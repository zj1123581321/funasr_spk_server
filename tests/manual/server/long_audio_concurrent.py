"""
Qwen3 长音频 + 高并发真服务器冒烟 (补 PR3 task 7 长音频盲区)

启动方式:
    1) 在另一个终端启 server:
       FUNASR_DEFAULT_ENGINE=qwen3 FUNASR_QWEN3_POOL_SIZE=2 \
       FUNASR_SERVER_PORT=8868 venv/bin/python run_server.py
    2) 跑本脚本

输入: tmp_long_audio/ 下两条音频 (83min m4a + 149min mp3)
验证:
- 两 task 并发各自走完完整流程 → task_complete
- 文本 / task_id 不串台
- N=2 worker 总 RSS 不爆 (预期 < 8GB, GGUF mmap 共享)
- 长音频 server 不崩 (Qwen3 LLM 上下文 / Metal 累积 sanity)

报数据: 总 wall / 各 task wall + server_time / RTF / 文本前 30 字 / 内存峰值
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import websockets


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVER_URL = "ws://localhost:8868"
AUDIO_DIR = PROJECT_ROOT / "tmp_long_audio"
AUDIOS = [
    AUDIO_DIR / "audio_83min.m4a",
    AUDIO_DIR / "audio_149min.mp3",
]


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def transcribe_one(client_id: int, audio: Path):
    t0 = time.time()
    print(f"[client {client_id}] connecting → {audio.name} ({audio.stat().st_size / 1024 / 1024:.1f} MB)")
    async with websockets.connect(SERVER_URL, max_size=512 * 1024 * 1024, ping_interval=60, ping_timeout=300) as ws:
        connected = json.loads(await ws.recv())
        assert connected.get("type") == "connected", f"未收到 connected: {connected}"

        data = audio.read_bytes()
        size = len(data)
        digest = file_hash(audio)
        b64 = base64.b64encode(data).decode("utf-8")

        await ws.send(json.dumps({
            "type": "upload_request",
            "data": {
                "file_name": audio.name,
                "file_size": size,
                "file_hash": digest,
                "force_refresh": True,
                "output_format": "json",
            }
        }))
        resp = json.loads(await ws.recv())
        if resp["type"] == "error":
            return {"client_id": client_id, "ok": False, "error": resp["data"]["message"]}
        if resp["type"] == "task_complete":
            return {
                "client_id": client_id, "ok": True, "cached": True,
                "task_id": resp["data"].get("task_id", "cached"),
                "wall_time": time.time() - t0,
                "result": resp["data"]["result"],
            }
        task_id = resp["data"].get("task_id")
        if not task_id:
            return {"client_id": client_id, "ok": False, "error": f"无 task_id: {resp}"}

        print(f"[client {client_id}] uploading {size / 1024 / 1024:.1f}MB (task_id={task_id[:8]}...)")
        upload_start = time.time()
        await ws.send(json.dumps({
            "type": "upload_data",
            "data": {"task_id": task_id, "file_data": b64}
        }))
        print(f"[client {client_id}] upload sent in {time.time() - upload_start:.1f}s, waiting transcribe...")

        last_progress = -1
        while True:
            try:
                msg_text = await asyncio.wait_for(ws.recv(), timeout=1800.0)  # 30 min
            except asyncio.TimeoutError:
                return {"client_id": client_id, "ok": False, "task_id": task_id, "error": "recv timeout 30min"}
            msg = json.loads(msg_text)
            if msg["type"] == "task_progress":
                pct = msg["data"].get("progress", 0)
                # 每 10% 打印一次, 避免刷屏
                if int(pct) // 10 != last_progress // 10:
                    last_progress = int(pct)
                    print(f"[client {client_id}] progress {pct:.0f}% @ wall={time.time() - t0:.0f}s")
                continue
            if msg["type"] == "task_complete":
                result = msg["data"]["result"]
                return {
                    "client_id": client_id, "ok": True, "cached": False,
                    "task_id": task_id,
                    "wall_time": time.time() - t0,
                    "result": result,
                }
            if msg["type"] == "error":
                return {"client_id": client_id, "ok": False, "task_id": task_id, "error": msg["data"]["message"]}


async def rss_sampler(stop_event: asyncio.Event, samples: list):
    """每 5 秒采样 server 主进程 + worker subprocess 总 RSS"""
    while not stop_event.is_set():
        total_mb = 0
        pids = []
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if "qwen3_worker_process.py" in cmdline or "run_server.py" in cmdline:
                    rss_mb = proc.memory_info().rss / 1024 / 1024
                    total_mb += rss_mb
                    pids.append((proc.pid, rss_mb))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        samples.append((time.time(), total_mb, pids))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass


async def main():
    for a in AUDIOS:
        assert a.exists(), f"音频不存在: {a}"

    print("=" * 70)
    print(f"长音频 + 高并发冒烟: {len(AUDIOS)} files concurrent")
    for a in AUDIOS:
        print(f"  - {a.name} ({a.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 70)

    stop = asyncio.Event()
    rss_samples = []
    rss_task = asyncio.create_task(rss_sampler(stop, rss_samples))

    t0 = time.time()
    results = await asyncio.gather(
        *[transcribe_one(i + 1, a) for i, a in enumerate(AUDIOS)],
        return_exceptions=True,
    )
    total = time.time() - t0

    stop.set()
    await rss_task

    print("\n" + "=" * 70)
    print(f"总 wall: {total:.1f}s ({total / 60:.1f} min)")
    for r in results:
        if isinstance(r, Exception):
            print(f"FAIL exception: {r!r}")
            continue
        if not r["ok"]:
            print(f"[client {r['client_id']}] FAIL: {r.get('error')}")
            continue
        res = r["result"]
        d = res.get("duration", 0)
        pt = res.get("processing_time", 0)
        rtf = pt / d if d else 0
        text = "".join(s["text"] for s in res.get("segments", []))
        print(
            f"[client {r['client_id']}] ✓ task_id={r['task_id'][:12]}... "
            f"audio={d:.0f}s ({d/60:.1f}min) "
            f"wall={r['wall_time']:.0f}s server_time={pt:.0f}s RTF={rtf:.3f} "
            f"segs={len(res.get('segments', []))}\n"
            f"   speakers={res.get('speakers')} text[0:50]={text[:50]!r}"
        )

    # RSS 峰值
    if rss_samples:
        peak_total, peak_t, peak_pids = max(rss_samples, key=lambda x: x[1])
        print(f"\nRSS 峰值: {peak_total:.0f}MB @ wall={peak_t - t0:.0f}s")
        for pid, rss in peak_pids:
            print(f"  PID={pid} RSS={rss:.0f}MB")

    # 串台校验
    ok_results = [r for r in results if not isinstance(r, Exception) and r["ok"]]
    if len(ok_results) >= 2:
        task_ids = [r["task_id"] for r in ok_results]
        assert len(set(task_ids)) == len(task_ids), "task_id 串台!"
        print("\n✓ 不串台: 所有 task_id 唯一")


if __name__ == "__main__":
    asyncio.run(main())
