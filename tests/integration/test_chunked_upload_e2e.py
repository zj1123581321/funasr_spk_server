"""真 server + 真 websocket client + 真 FunASR 模型的**分片上传**端到端集成测试。

TODOS #24: 长音频主场景走分片上传 (chunked_upload_request → 逐帧 upload_chunk →
finalize), 但此前**端到端零覆盖** —— 单文件 upload_data 路径覆盖不到分片重组 +
offset 写盘那段。本次根路径 ws 事故同型 (主路径没真帧序列测试, 只能生产暴露)。

本文件走真帧序列穿过 server → 分片重组 → finalize → task_manager → _process_task,
把分片重组类回归挡在集成层。逐帧接收的错误分支已由 unit
(test_websocket_chunk_upload.py) + finalize 弹性 (test_websocket_finalize_resilience.py)
覆盖, 此处只验真链路的"正常多分片全流程"和"分片缓存命中"。

默认 skip (需 FUNASR_RUN_INTEGRATION=1 + FunASR 模型 + 启 subprocess server)。
复用 test_funasr_server_websocket_e2e 的 funasr_server_subprocess fixture (import 共享)。
"""
from __future__ import annotations

import os

import pytest

from tests.integration._ws_e2e_common import (
    client_chunked_upload_and_wait,
    websockets,
)
# 复用既有 module 级 server fixture + 唯一 audio helper (import 即注册为本模块 fixture)
from tests.integration.test_funasr_server_websocket_e2e import (  # noqa: F401
    funasr_server_subprocess,
    _unique_audio,
)

RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="设置 FUNASR_RUN_INTEGRATION=1 启用 (默认 skip, 启 subprocess server + 真 FunASR 模型)",
    ),
    pytest.mark.skipif(websockets is None, reason="websockets 包未安装"),
]


class TestChunkedUploadRealServerE2E:
    @pytest.mark.asyncio
    async def test_chunked_json_full_flow_multichunk(self, funasr_server_subprocess, tmp_path):
        """多分片正常全流程: 小 chunk_size 强制拆多片, 验证重组 + offset 写盘 + finalize
        → _process_task 全链路, segments/metadata 非空。"""
        ws_url, log_file = funasr_server_subprocess
        audio = _unique_audio(tmp_path, b"\x10\x01")

        # 256KB chunk → 60s podcast wav 拆成多片 (证明逐帧重组)
        r = await client_chunked_upload_and_wait(
            ws_url, audio, chunk_size=256 * 1024, force_refresh=True, output_format="json")
        assert r["ok"], (
            f"分片 JSON 任务失败: {r.get('error')}\n服务日志尾部:\n"
            f"{log_file.read_text()[-1500:]}"
        )
        assert r["num_chunks"] > 1, f"应拆成多片才有重组意义, 实际 {r['num_chunks']} 片"
        segs = r["result"].get("segments", [])
        assert len(segs) > 0, "应有 segments 输出 (分片重组后文件可被正常转录)"
        full_text = "".join(s.get("text", "") for s in segs)
        assert full_text.strip(), "转录文本不该全空 (重组错位会导致解码乱码/空)"
        meta = r["result"].get("metadata") or {}
        assert meta.get("engine") == "funasr", f"metadata.engine 应为 funasr: {meta}"

    @pytest.mark.asyncio
    async def test_chunked_cache_hit_second_request(self, funasr_server_subprocess, tmp_path):
        """同 hash 分片上传两次: 第 1 次真转录, 第 2 次 finalize 阶段命中缓存秒回。"""
        ws_url, _log = funasr_server_subprocess
        audio = _unique_audio(tmp_path, b"\x10\x02")

        r1 = await client_chunked_upload_and_wait(
            ws_url, audio, chunk_size=256 * 1024, force_refresh=False, output_format="json")
        assert r1["ok"], f"第 1 次分片上传失败: {r1.get('error')}"
        first_wall = r1["wall_time"]

        r2 = await client_chunked_upload_and_wait(
            ws_url, audio, chunk_size=256 * 1024, force_refresh=False, output_format="json")
        assert r2["ok"], f"第 2 次分片上传失败: {r2.get('error')}"
        assert r2["cached"], f"第 2 次应在 finalize 阶段命中 cache, 实际 cached={r2['cached']}"
        assert r2["wall_time"] < first_wall / 3, (
            f"cache hit 应远快于真转录 (first={first_wall:.2f}s, second={r2['wall_time']:.2f}s)"
        )
