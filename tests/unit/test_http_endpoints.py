"""可观测性 (P1) — /health + /metrics HTTP 端点 (websockets 同端口 process_request).

覆盖纯函数 + process_request 路由 + A5 鉴权 + #1/#2 GET/query 解析.
设计定案: docs/开发/2026-06-16-可观测性仪表盘与测试加固-设计定案与落地计划.md
"""
from __future__ import annotations

import http
from unittest.mock import MagicMock, AsyncMock

import pytest

from src.api.http_endpoints import (
    render_prometheus,
    evaluate_health,
    authorize_metrics,
    HttpEndpoints,
)


# ============ render_prometheus (纯函数) ============

SNAP = {
    "queue_size": 3, "max_queue_size": 20, "pending": 2, "processing": 1,
    "tasks_in_memory": 5,
    "terminal_total": {"completed": 10, "failed": 2},
    "errors_total": {"engine_error": 2, "queue_full": 1},
    "task_seconds_ema": 88.5, "engine": "funasr", "pool_size": 2,
}


class TestRenderPrometheus:
    def test_includes_help_and_type_lines(self):
        text = render_prometheus(SNAP, cache_stats=None, vram_mib=None, uptime_s=None)
        assert "# HELP funasr_queue_size" in text
        assert "# TYPE funasr_queue_size gauge" in text

    def test_gauge_values_present(self):
        text = render_prometheus(SNAP, cache_stats=None, vram_mib=None, uptime_s=None)
        assert "funasr_queue_size 3" in text
        assert "funasr_tasks_pending 2" in text
        assert "funasr_tasks_inflight 1" in text

    def test_terminal_total_is_labeled_counter(self):
        text = render_prometheus(SNAP, cache_stats=None, vram_mib=None, uptime_s=None)
        assert 'funasr_tasks_terminal_total{status="completed"} 10' in text
        assert 'funasr_tasks_terminal_total{status="failed"} 2' in text
        assert "# TYPE funasr_tasks_terminal_total counter" in text

    def test_errors_total_labeled(self):
        text = render_prometheus(SNAP, cache_stats=None, vram_mib=None, uptime_s=None)
        assert 'funasr_errors_total{kind="engine_error"} 2' in text

    def test_engine_info_label(self):
        text = render_prometheus(SNAP, cache_stats=None, vram_mib=None, uptime_s=None)
        assert 'funasr_engine_info{engine="funasr"} 1' in text

    def test_vram_omitted_when_none(self):
        text = render_prometheus(SNAP, cache_stats=None, vram_mib=None, uptime_s=None)
        assert "funasr_vram_free_mib" not in text

    def test_vram_present_when_given(self):
        text = render_prometheus(SNAP, cache_stats=None, vram_mib=4096, uptime_s=None)
        assert "funasr_vram_free_mib 4096" in text

    def test_cache_stats_rendered_when_given(self):
        cache = {"total_cached": 100, "hits": 40, "misses": 60, "projected_serves": 7}
        text = render_prometheus(SNAP, cache_stats=cache, vram_mib=None, uptime_s=None)
        assert "funasr_cache_projected_serves 7" in text


# ============ evaluate_health (纯函数, A3-final liveness) ============

class TestEvaluateHealth:
    def test_healthy_when_running_and_loop_alive(self):
        code, body = evaluate_health(is_running=True, maintenance_alive=True)
        assert code == 200
        assert body["status"] == "healthy"
        assert all(body["checks"].values())

    def test_degraded_when_loop_dead(self):
        code, body = evaluate_health(is_running=True, maintenance_alive=False)
        assert code == 503
        assert body["status"] == "degraded"
        assert body["checks"]["maintenance_loop"] is False

    def test_degraded_when_not_running(self):
        code, body = evaluate_health(is_running=False, maintenance_alive=True)
        assert code == 503
        assert body["checks"]["accepting_tasks"] is False


# ============ authorize_metrics (纯函数, A5) ============

class TestAuthorizeMetrics:
    def test_token_set_correct_query_ok(self):
        assert authorize_metrics("/metrics?token=abc", None, configured_token="abc", host="0.0.0.0") is None

    def test_token_set_correct_header_ok(self):
        assert authorize_metrics("/metrics", "Bearer abc", configured_token="abc", host="0.0.0.0") is None

    def test_token_set_wrong_denied(self):
        assert authorize_metrics("/metrics?token=nope", None, configured_token="abc", host="0.0.0.0") == 403

    def test_token_set_missing_denied(self):
        assert authorize_metrics("/metrics", None, configured_token="abc", host="0.0.0.0") == 403

    def test_no_token_on_0000_denied(self):
        # A5: 0.0.0.0 默认 + 无 token → 拒绝(强制设 token 或绑 LAN)
        assert authorize_metrics("/metrics", None, configured_token=None, host="0.0.0.0") == 403

    def test_no_token_on_loopback_ok(self):
        assert authorize_metrics("/metrics", None, configured_token=None, host="127.0.0.1") is None

    def test_no_token_on_lan_ip_ok(self):
        assert authorize_metrics("/metrics", None, configured_token=None, host="192.168.1.10") is None


# ============ process_request 路由 (#1 GET-only doc / #2 query 解析) ============

def _make_endpoints(metrics_enabled=True, metrics_token=None, host="127.0.0.1"):
    tm = MagicMock()
    tm.is_running = True
    tm._maintenance_task = MagicMock()
    tm._maintenance_task.done.return_value = False
    tm.get_metrics_snapshot.return_value = SNAP
    db = MagicMock()
    db.get_cache_stats = AsyncMock(return_value={"total_cached": 1, "hits": 0, "misses": 0, "projected_serves": 0})
    cfg = MagicMock()
    cfg.observability.metrics_enabled = metrics_enabled
    cfg.observability.metrics_token = metrics_token
    cfg.server.host = host
    return HttpEndpoints(task_manager=tm, db_manager=db, config=cfg)


class TestProcessRequestRouting:
    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self):
        ep = _make_endpoints()
        assert await ep.process_request("/ws", {}) is None

    @pytest.mark.asyncio
    async def test_health_returns_200_tuple(self):
        ep = _make_endpoints()
        resp = await ep.process_request("/health", {})
        assert resp is not None
        status, headers, body = resp
        assert int(status) == 200
        assert b"healthy" in body

    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_text(self):
        ep = _make_endpoints(host="127.0.0.1")
        resp = await ep.process_request("/metrics", {})
        status, headers, body = resp
        assert int(status) == 200
        assert b"funasr_queue_size" in body

    @pytest.mark.asyncio
    async def test_metrics_query_string_routes(self):
        # #2: path 带 query, urlsplit 后仍路由到 /metrics
        ep = _make_endpoints(metrics_token="abc", host="0.0.0.0")
        resp = await ep.process_request("/metrics?token=abc", {})
        status, _, body = resp
        assert int(status) == 200

    @pytest.mark.asyncio
    async def test_metrics_denied_no_token_0000(self):
        ep = _make_endpoints(metrics_token=None, host="0.0.0.0")
        resp = await ep.process_request("/metrics", {})
        status, _, body = resp
        assert int(status) == 403

    @pytest.mark.asyncio
    async def test_metrics_body_never_leaks_token(self):
        # A5: /metrics 响应绝不回显 token
        ep = _make_endpoints(metrics_token="sup3rs3cr3t", host="0.0.0.0")
        resp = await ep.process_request("/metrics?token=sup3rs3cr3t", {})
        status, _, body = resp
        assert b"sup3rs3cr3t" not in body

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        ep = _make_endpoints(metrics_enabled=False)
        assert await ep.process_request("/health", {}) is None
        assert await ep.process_request("/metrics", {}) is None


class TestStatusPage:
    """`/` 极简 HTML 状态页 (静态, JS fetch /health + /metrics 渲染)。"""

    @pytest.mark.asyncio
    async def test_root_returns_html_200(self):
        ep = _make_endpoints()
        resp = await ep.process_request("/", {})
        assert resp is not None
        status, headers, body = resp
        assert int(status) == 200
        assert any("text/html" in v for _, v in headers)
        assert b"<!DOCTYPE html" in body or b"<!doctype html" in body

    @pytest.mark.asyncio
    async def test_root_html_references_endpoints(self):
        ep = _make_endpoints()
        _, _, body = await ep.process_request("/", {})
        text = body.decode("utf-8")
        assert "/health" in text
        assert "/metrics" in text

    @pytest.mark.asyncio
    async def test_root_html_is_static_no_token_embedded(self):
        # 页面是静态的, token 由用户在 URL ?token= 提供, 绝不嵌进服务端 HTML
        ep = _make_endpoints(metrics_token="sup3rs3cr3t", host="0.0.0.0")
        _, _, body = await ep.process_request("/?token=sup3rs3cr3t", {})
        assert b"sup3rs3cr3t" not in body

    @pytest.mark.asyncio
    async def test_root_query_string_routes(self):
        ep = _make_endpoints()
        resp = await ep.process_request("/?token=x", {})
        assert resp is not None
        assert int(resp[0]) == 200

    @pytest.mark.asyncio
    async def test_root_disabled_returns_none(self):
        ep = _make_endpoints(metrics_enabled=False)
        assert await ep.process_request("/", {}) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
