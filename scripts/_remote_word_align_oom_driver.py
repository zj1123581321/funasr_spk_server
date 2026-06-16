"""逼 CUDA word_align OOM → 验证 CPU fallback + poison (手工, batch>=2 起服务).

连发 N 轮 fresh word_align=true (同 server 生命周期), PoC 表明第二轮起 BFCArena 高水位
会 OOM. 新代码应: 捕获资源错误 → poison pool → dispose → 转 CPU → 仍返回 words.
看 logs/server_cuda.log 的 provider=cpu / POISONED / 资源错误 行为证据.
"""
import asyncio
import sys
import uuid
from pathlib import Path

from scripts._remote_word_align_probe import request_once


async def main():
    src = Path("tests/fixtures/audio/podcast_2speakers_60s.wav")
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for i in range(rounds):
        a = src.parent / (".oom_" + uuid.uuid4().hex[:8] + ".wav")
        a.write_bytes(src.read_bytes() + uuid.uuid4().bytes[:8])
        try:
            r = await request_once("ws://localhost:8867", a, word_align=True,
                                   force_refresh=True, language="chi")
            md = r["result"].get("metadata") or {}
            w = sum(len(s.get("words") or []) for s in r["result"]["segments"])
            print("round {}: words={} word_align={} err={}".format(
                i + 1, w, md.get("word_align"), md.get("word_align_error")))
        except Exception as exc:
            print("round {}: EXCEPTION {}".format(i + 1, exc))
        finally:
            a.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
