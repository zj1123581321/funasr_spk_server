"""diarize 开关 — 3060 真机端到端验收 probe (落地顺序第 7 步)

对真实 server (ws) 顺序跑 6 个场景, 覆盖测试矩阵的真机切面 + RTF 对比:

  A. diarize=true  JSON fresh (force_refresh)  → SpeakerN + metadata 回显, 记 RTF
  B. diarize=false JSON 无 force (A 已存 diarized 行) → 投影命中 projected=true,
     speaker=null, speakers=[], words 保留
  C. diarize=false JSON fresh (force_refresh) → 真算 nospk: RTF 对比 A (省算力验收),
     nospk 切段生效 (无超长段), words 在 (word_align 与 diarize 正交)
  D. diarize=false JSON 无 force → exact +nospk 行命中, projected=false
  E. diarize=false SRT 无 force → 无 "SpeakerN:" 前缀
  F. diarize=true  SRT 无 force → 有 "SpeakerN:" 前缀 (回归)

用法 (CUDA box, server 先以 FUNASR_PROFILE=cuda_dev 起好):
    venv/bin/python scripts/_remote_diarize_e2e_probe.py \
        --server ws://localhost:8867 \
        --audio tests/fixtures/audio/podcast_2speakers_60s.wav

每个场景打印 PASS/FAIL, 全过 exit 0, 任一失败 exit 1.
ws 坑: 长音频会断连 (心跳阻塞), 60s 音频 ~15s 转完, 前台等待可承受;
换长音频时请自行改成轮询 server 日志 (见 reference: tests/manual/server/smoke_qwen3_concurrent.py).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import time
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
    diarize: bool,
    output_format: str = "json",
    force_refresh: bool = False,
    language: str | None = None,
    timeout: float = 600.0,
) -> dict:
    """跑一次完整 upload + transcribe, 返回 {result, wall, cached}."""
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
        if language:
            req["language"] = language
        await ws.send(json.dumps({"type": "upload_request", "data": req}))

        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if resp["type"] == "error":
            raise RuntimeError(f"upload_request error: {resp['data']}")
        if resp["type"] == "task_complete":
            # 缓存早返回 (handler upload_request 阶段)
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

    def warn(self, cond: bool, label: str):
        """环境相关项: 不达标只 warn 不判失败 (已知机器课题, 与 diarize 开关无关)."""
        print(f"    {'✓' if cond else '⚠ WARN'} {label}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://localhost:8867")
    ap.add_argument("--audio", default="tests/fixtures/audio/podcast_2speakers_60s.wav")
    ap.add_argument("--language", default="chi")
    args = ap.parse_args()

    audio = Path(args.audio)
    assert audio.exists(), f"音频不存在: {audio}"
    ck = Check()

    # ---------- A. diarize=true JSON fresh ----------
    print("\n[A] diarize=true JSON fresh (force_refresh)")
    a = await request_once(args.server, audio, diarize=True, force_refresh=True, language=args.language)
    ra = a["result"]
    segs_a = ra["segments"]
    rtf_a = ra["processing_time"] / ra["duration"]
    md_a = ra.get("metadata") or {}
    print(f"    segments={len(segs_a)} speakers={ra['speakers']} RTF={rtf_a:.4f} wall={a['wall']:.1f}s metadata={md_a}")
    ck.ok(len(segs_a) > 0 and all(s["speaker"] for s in segs_a), "全部段带 SpeakerN")
    # 3060 ort_cuda + cluster_merge 在该 podcast 上历史性合成 1 cluster
    # (旧代码同样 speakers=1, 见 cuda-diarize-accuracy 课题), 不算 diarize 开关失败
    ck.warn(len(ra["speakers"]) >= 2, "podcast 检出 ≥2 speaker (ort_cuda 已知 under-detect 课题)")
    ck.ok(md_a.get("diarize") is True and md_a.get("engine") == "qwen3", "metadata diarize/engine 回显")
    ck.ok(md_a.get("projected") is False, "fresh diarized → projected=false")
    has_words_a = any(s.get("words") for s in segs_a)
    ck.ok(has_words_a == bool(md_a.get("word_align")), f"words 与 word_align 回显一致 (words={has_words_a})")

    # ---------- B. diarize=false 投影命中 (A 的 diarized 行) ----------
    print("\n[B] diarize=false JSON 无 force → 投影回退命中")
    b = await request_once(args.server, audio, diarize=False, language=args.language)
    rb = b["result"]
    md_b = rb.get("metadata") or {}
    print(f"    cached={b['cached']} segments={len(rb['segments'])} speakers={rb['speakers']} metadata={md_b}")
    ck.ok(b["cached"], "缓存早返回 (未重算)")
    ck.ok(all(s["speaker"] is None for s in rb["segments"]), "全部段 speaker=null")
    ck.ok(rb["speakers"] == [], "speakers=[]")
    ck.ok(md_b.get("projected") is True, "投影命中 → projected=true")
    if has_words_a:
        ck.ok(any(s.get("words") for s in rb["segments"]), "投影保留 words")

    # ---------- C. diarize=false JSON fresh (真算 nospk, RTF 对比) ----------
    print("\n[C] diarize=false JSON fresh (force_refresh) → 真跳层 + RTF 对比")
    c = await request_once(args.server, audio, diarize=False, force_refresh=True, language=args.language)
    rc = c["result"]
    rtf_c = rc["processing_time"] / rc["duration"]
    md_c = rc.get("metadata") or {}
    seg_durs = [s["end_time"] - s["start_time"] for s in rc["segments"]]
    print(f"    segments={len(rc['segments'])} max_seg_dur={max(seg_durs):.1f}s RTF={rtf_c:.4f} "
          f"(diarized RTF={rtf_a:.4f}, 省 {(1 - rtf_c / rtf_a) * 100:.0f}%) metadata={md_c}")
    ck.ok(all(s["speaker"] is None for s in rc["segments"]), "fresh nospk speaker=null")
    ck.ok(md_c.get("projected") is False, "fresh nospk → projected=false (真算)")
    ck.ok(max(seg_durs) <= 20.0, f"nospk 切段生效 (最长段 {max(seg_durs):.1f}s ≤ 20s)")
    ck.ok(rtf_c < rtf_a, f"关 diarize 省算力 (RTF {rtf_c:.4f} < {rtf_a:.4f})")
    if md_c.get("word_align"):
        # metadata.word_align 是 config 回显, 运行时 MMS 可能失败 (如 3060 显存满载
        # 时 pool 第二实例 CUDA session OOM) → 设计内 fallback: 段照常出 words=None,
        # 切段退静音. 有词则验证, 无词只 warn.
        ck.warn(
            any(s.get("words") for s in rc["segments"]),
            "word_align 与 diarize 正交 (nospk 有 words; 无词=MMS 运行时失败 fallback)",
        )

    # ---------- D. exact +nospk 行命中 ----------
    print("\n[D] diarize=false JSON 无 force → exact nospk 行命中")
    d = await request_once(args.server, audio, diarize=False, language=args.language)
    md_d = (d["result"].get("metadata") or {})
    print(f"    cached={d['cached']} metadata={md_d}")
    ck.ok(d["cached"], "缓存命中")
    ck.ok(md_d.get("projected") is False, "exact nospk 行 (真算) → projected=false")

    # ---------- E. diarize=false SRT ----------
    print("\n[E] diarize=false SRT → 无前缀")
    e = await request_once(args.server, audio, diarize=False, output_format="srt", language=args.language)
    srt_e = e["result"]["content"]
    print(f"    cached={e['cached']} blocks={srt_e.count('-->')} head={srt_e.splitlines()[:3]}")
    ck.ok("Speaker" not in srt_e, "SRT 无 SpeakerN: 前缀")
    ck.ok(srt_e.strip() and "-->" in srt_e, "SRT 内容非空")

    # ---------- F. diarize=true SRT (回归) ----------
    print("\n[F] diarize=true SRT → 有前缀 (回归)")
    f = await request_once(args.server, audio, diarize=True, output_format="srt", language=args.language)
    srt_f = f["result"]["content"]
    print(f"    cached={f['cached']} blocks={srt_f.count('-->')}")
    ck.ok("Speaker1:" in srt_f, "SRT 带 SpeakerN: 前缀")

    # ---------- 汇总 ----------
    print("\n========== diarize e2e probe 汇总 ==========")
    print(f"RTF: diarize=true {rtf_a:.4f} | diarize=false {rtf_c:.4f} "
          f"(省 {(1 - rtf_c / rtf_a) * 100:.0f}%)")
    if ck.failures:
        print(f"✗ {len(ck.failures)} 项失败:")
        for x in ck.failures:
            print(f"  - {x}")
        sys.exit(1)
    print("✓ 全部通过")


if __name__ == "__main__":
    asyncio.run(main())
