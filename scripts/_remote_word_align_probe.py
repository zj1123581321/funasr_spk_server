"""word_align per-request 开关 — 3060 真机端到端验收 probe (2026-06-16 显存落地)

对真实 server (ws) 顺序跑场景, 覆盖 per-request word_align 的真机切面:

  A. 老客户端不传 word_align 字段 → 默认关 (无 words, metadata.word_align=false, 向后兼容)
  B. word_align=true JSON fresh → CUDA 真算挂词, metadata.word_align=true, 无 word_align_error
  C. word_align=true JSON 无 force → 缓存命中, words 仍在 (决策 B: +wa 行必有词)
  D. word_align=true SRT → 无词 (JSON-only), metadata.word_align=false (delivered), content 非空
  E. word_align=false 显式 → 无 words (即使 server config 兜底开也压过)

用法 (CUDA box, server 先以 FUNASR_PROFILE=cuda_dev 起好):
    venv/bin/python scripts/_remote_word_align_probe.py \
        --server ws://localhost:8867 \
        --audio tests/fixtures/audio/podcast_2speakers_60s.wav

CUDA OOM → CPU fallback + poison 路径不在本 probe (batch=1 + pool=1 稳定不 OOM):
单独用 FUNASR_QWEN3_WORD_ALIGN_CUDA_BATCH_SIZE=4 强制 OOM 起服务 + 看 journalctl 验证.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def request_once(
    server_url: str,
    audio: Path,
    *,
    word_align,  # None=不传字段(老客户端) / True / False
    output_format: str = "json",
    force_refresh: bool = False,
    language: str | None = None,
    diarize: bool = True,
    timeout: float = 600.0,
) -> dict:
    import websockets

    t0 = time.time()
    async with websockets.connect(server_url, max_size=200 * 1024 * 1024) as ws:
        connected = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        assert connected.get("type") == "connected", f"未收到 connected: {connected}"

        data = audio.read_bytes()
        req = {
            "file_name": audio.name,
            "file_size": len(data),
            "file_hash": file_hash(audio),
            "force_refresh": force_refresh,
            "output_format": output_format,
            "diarize": diarize,
        }
        if word_align is not None:
            req["word_align"] = word_align
        if language:
            req["language"] = language
        await ws.send(json.dumps({"type": "upload_request", "data": req}))

        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if resp["type"] == "error":
            raise RuntimeError(f"upload_request error: {resp['data']}")
        if resp["type"] == "task_complete":
            return {"result": resp["data"]["result"], "wall": time.time() - t0, "cached": True}

        task_id = resp["data"]["task_id"]
        await ws.send(json.dumps({
            "type": "upload_data",
            "data": {"task_id": task_id, "file_data": base64.b64encode(data).decode()},
        }))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if msg["type"] in ("task_progress", "upload_complete", "task_queued"):
                continue
            if msg["type"] == "task_complete":
                return {"result": msg["data"]["result"], "wall": time.time() - t0, "cached": False}
            if msg["type"] == "error":
                raise RuntimeError(f"task error: {msg['data']}")


class Check:
    def __init__(self):
        self.failures: list[str] = []

    def ok(self, cond: bool, label: str):
        print(f"    {'✓' if cond else '✗ FAIL'} {label}")
        if not cond:
            self.failures.append(label)


def _words(result: dict) -> int:
    return sum(len(s.get("words") or []) for s in result["segments"])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://localhost:8867")
    ap.add_argument("--audio", default="tests/fixtures/audio/podcast_2speakers_60s.wav")
    ap.add_argument("--language", default="chi")
    args = ap.parse_args()

    src_audio = Path(args.audio)
    assert src_audio.exists(), f"音频不存在: {src_audio}"
    audio = src_audio.parent / f".wa_probe_{uuid.uuid4().hex[:8]}{src_audio.suffix}"
    audio.write_bytes(src_audio.read_bytes() + uuid.uuid4().bytes[:8])
    ck = Check()

    # ---------- A. 老客户端不传 word_align → 默认关 ----------
    print("\n[A] 不传 word_align 字段 (老客户端) → 默认关")
    a = await request_once(args.server, audio, word_align=None, force_refresh=True, language=args.language)
    ra = a["result"]
    md_a = ra.get("metadata") or {}
    print(f"    words={_words(ra)} metadata.word_align={md_a.get('word_align')} wall={a['wall']:.1f}s")
    ck.ok(_words(ra) == 0, "无 words (默认关)")
    ck.ok(md_a.get("word_align") is False, "metadata.word_align=false")

    # ---------- B. word_align=true JSON fresh → CUDA 真算挂词 ----------
    print("\n[B] word_align=true JSON fresh → CUDA 真算挂词")
    b = await request_once(args.server, audio, word_align=True, force_refresh=True, language=args.language)
    rb = b["result"]
    md_b = rb.get("metadata") or {}
    rtf_b = rb["processing_time"] / rb["duration"]
    print(f"    words={_words(rb)} metadata.word_align={md_b.get('word_align')} "
          f"word_align_error={md_b.get('word_align_error')} RTF={rtf_b:.4f} wall={b['wall']:.1f}s")
    ck.ok(_words(rb) > 0, "段挂上 words")
    ck.ok(md_b.get("word_align") is True, "metadata.word_align=true (delivered)")
    ck.ok(md_b.get("word_align_error") is None, "无 word_align_error (CUDA 真算成功)")

    # ---------- C. word_align=true 无 force → 缓存命中, words 仍在 ----------
    print("\n[C] word_align=true JSON 无 force → 缓存命中 words 仍在")
    c = await request_once(args.server, audio, word_align=True, language=args.language)
    rc = c["result"]
    md_c = rc.get("metadata") or {}
    print(f"    cached={c['cached']} words={_words(rc)} metadata.word_align={md_c.get('word_align')}")
    ck.ok(c["cached"], "缓存命中 (未重算)")
    ck.ok(_words(rc) > 0, "缓存命中 words 仍在 (决策 B: +wa 行必有词)")
    ck.ok(md_c.get("word_align") is True, "metadata.word_align=true")

    # ---------- D. word_align=true SRT → 无词 (JSON-only) ----------
    print("\n[D] word_align=true SRT → 无词, delivered=false")
    d = await request_once(args.server, audio, word_align=True, output_format="srt", language=args.language)
    rd = d["result"]
    md_d = rd.get("metadata") or {}
    srt = rd.get("content", "")
    print(f"    cached={d['cached']} blocks={srt.count('-->')} metadata.word_align={md_d.get('word_align')}")
    ck.ok(srt.strip() and "-->" in srt, "SRT 内容非空")
    ck.ok(md_d.get("word_align") is False, "SRT metadata.word_align=false (JSON-only delivered)")

    # ---------- E. word_align=false 显式 → 无 words ----------
    print("\n[E] word_align=false 显式 → 无 words")
    e = await request_once(args.server, audio, word_align=False, force_refresh=True, language=args.language)
    re_ = e["result"]
    md_e = re_.get("metadata") or {}
    print(f"    words={_words(re_)} metadata.word_align={md_e.get('word_align')}")
    ck.ok(_words(re_) == 0, "显式关 → 无 words")
    ck.ok(md_e.get("word_align") is False, "metadata.word_align=false")

    audio.unlink(missing_ok=True)

    print("\n========== word_align e2e probe 汇总 ==========")
    if ck.failures:
        print(f"✗ {len(ck.failures)} 项失败:")
        for x in ck.failures:
            print(f"  - {x}")
        sys.exit(1)
    print("✓ 全部通过")


if __name__ == "__main__":
    asyncio.run(main())
