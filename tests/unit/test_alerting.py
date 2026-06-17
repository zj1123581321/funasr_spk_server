"""可观测性告警纯逻辑单测 (A: 脚本定时拉 /metrics → 超阈值发企微)。

被测对象 src/observability/alerting.py 是**纯函数**: 输入 probe + 上轮 state +
阈值 + now_ts, 输出 (告警列表, 新 state)。无 I/O、无时钟 (now_ts 注入), 故 100% 可测。
"""
import pytest

from src.observability.alerting import (
    AlertThresholds,
    parse_prometheus,
    evaluate_alerts,
    build_probe,
    load_state,
    save_state,
)


_METRICS_SAMPLE = """\
# HELP funasr_queue_size 当前任务队列深度 (准入控制)
# TYPE funasr_queue_size gauge
funasr_queue_size 18
# HELP funasr_queue_max 任务队列上限
# TYPE funasr_queue_max gauge
funasr_queue_max 20
# HELP funasr_errors_total 错误累计数 (单调, 按 kind)
# TYPE funasr_errors_total counter
funasr_errors_total{kind="engine_error"} 7
funasr_errors_total{kind="queue_full"} 3
# HELP funasr_engine_info 当前引擎
# TYPE funasr_engine_info gauge
funasr_engine_info{engine="funasr"} 1
"""


def _thresholds(**kw):
    base = dict(queue_saturation_ratio=0.8, error_surge_threshold=5, cooldown_seconds=900)
    base.update(kw)
    return AlertThresholds(**base)


def _probe(reachable=True, health_status="healthy", metrics_text=_METRICS_SAMPLE):
    return {
        "reachable": reachable,
        "health_status": health_status,
        "metrics": parse_prometheus(metrics_text) if metrics_text is not None else None,
    }


# ==================== parse_prometheus ====================

class TestParsePrometheus:
    def test_flat_gauge_parsed(self):
        out = parse_prometheus(_METRICS_SAMPLE)
        assert out["flat"]["funasr_queue_size"] == 18.0
        assert out["flat"]["funasr_queue_max"] == 20.0

    def test_labeled_counter_parsed(self):
        out = parse_prometheus(_METRICS_SAMPLE)
        assert out["labeled"]["funasr_errors_total"]["engine_error"] == 7.0
        assert out["labeled"]["funasr_errors_total"]["queue_full"] == 3.0

    def test_comments_and_blank_skipped(self):
        out = parse_prometheus("# comment\n\nfunasr_x 1\n")
        assert out["flat"] == {"funasr_x": 1.0}
        assert out["labeled"] == {}


# ==================== server_down ====================

class TestServerDown:
    def test_unreachable_fires_critical(self):
        res = evaluate_alerts(
            probe={"reachable": False, "health_status": None, "metrics": None},
            prev_state={}, thresholds=_thresholds(), now_ts=1000.0,
        )
        keys = [a["key"] for a in res["alerts"]]
        assert "server_down" in keys
        assert res["alerts"][0]["level"] == "critical"
        assert "server_down" in res["state"]["active"]

    def test_degraded_health_fires(self):
        res = evaluate_alerts(
            probe={"reachable": True, "health_status": "degraded", "metrics": None},
            prev_state={}, thresholds=_thresholds(), now_ts=1000.0,
        )
        assert any(a["key"] == "server_down" for a in res["alerts"])

    def test_recovery_emitted_when_back_healthy(self):
        prev = {"active": {"server_down": {"since": 100.0, "last_notified": 100.0}}}
        res = evaluate_alerts(
            probe=_probe(), prev_state=prev, thresholds=_thresholds(), now_ts=2000.0,
        )
        recovery = [a for a in res["alerts"] if a["key"] == "server_down"]
        assert len(recovery) == 1
        assert recovery[0]["level"] == "recovery"
        assert "server_down" not in res["state"]["active"]

    def test_cooldown_suppresses_repeat(self):
        prev = {"active": {"server_down": {"since": 100.0, "last_notified": 900.0}}}
        # now - last_notified = 500 < 900 cooldown → 不重复告警, 但仍保持 active
        res = evaluate_alerts(
            probe={"reachable": False, "health_status": None, "metrics": None},
            prev_state=prev, thresholds=_thresholds(), now_ts=1400.0,
        )
        assert res["alerts"] == []
        assert "server_down" in res["state"]["active"]

    def test_refires_after_cooldown(self):
        prev = {"active": {"server_down": {"since": 100.0, "last_notified": 100.0}}}
        res = evaluate_alerts(
            probe={"reachable": False, "health_status": None, "metrics": None},
            prev_state=prev, thresholds=_thresholds(cooldown_seconds=900), now_ts=1100.0,
        )
        assert any(a["key"] == "server_down" for a in res["alerts"])
        assert res["state"]["active"]["server_down"]["last_notified"] == 1100.0


# ==================== queue_saturation ====================

class TestQueueSaturation:
    def test_fires_over_ratio(self):
        # 18/20 = 0.9 >= 0.8
        res = evaluate_alerts(
            probe=_probe(), prev_state={}, thresholds=_thresholds(), now_ts=1000.0,
        )
        sat = [a for a in res["alerts"] if a["key"] == "queue_saturation"]
        assert len(sat) == 1
        assert "18" in sat[0]["body"] and "20" in sat[0]["body"]

    def test_no_fire_under_ratio(self):
        text = _METRICS_SAMPLE.replace("funasr_queue_size 18", "funasr_queue_size 5")
        res = evaluate_alerts(
            probe=_probe(metrics_text=text), prev_state={},
            thresholds=_thresholds(), now_ts=1000.0,
        )
        assert not any(a["key"] == "queue_saturation" for a in res["alerts"])

    def test_recovery_when_drains(self):
        prev = {"active": {"queue_saturation": {"since": 100.0, "last_notified": 100.0}}}
        text = _METRICS_SAMPLE.replace("funasr_queue_size 18", "funasr_queue_size 2")
        res = evaluate_alerts(
            probe=_probe(metrics_text=text), prev_state=prev,
            thresholds=_thresholds(), now_ts=2000.0,
        )
        rec = [a for a in res["alerts"] if a["key"] == "queue_saturation"]
        assert len(rec) == 1 and rec[0]["level"] == "recovery"

    def test_zero_queue_max_no_div_error(self):
        text = _METRICS_SAMPLE.replace("funasr_queue_max 20", "funasr_queue_max 0")
        res = evaluate_alerts(
            probe=_probe(metrics_text=text), prev_state={},
            thresholds=_thresholds(), now_ts=1000.0,
        )
        assert not any(a["key"] == "queue_saturation" for a in res["alerts"])


# ==================== error_surge ====================

class TestErrorSurge:
    def test_first_run_no_alert_sets_baseline(self):
        # 首轮无上轮基线 → 不告警, 但记录 error_total_seen = 10
        res = evaluate_alerts(
            probe=_probe(), prev_state={}, thresholds=_thresholds(), now_ts=1000.0,
        )
        assert not any(a["key"] == "error_surge" for a in res["alerts"])
        assert res["state"]["error_total_seen"] == 10.0  # 7 + 3

    def test_surge_fires_on_delta(self):
        # 上轮 total=4, 本轮 10 → delta 6 >= 5
        prev = {"error_total_seen": 4.0}
        res = evaluate_alerts(
            probe=_probe(), prev_state=prev, thresholds=_thresholds(), now_ts=1000.0,
        )
        surge = [a for a in res["alerts"] if a["key"] == "error_surge"]
        assert len(surge) == 1
        assert res["state"]["error_total_seen"] == 10.0

    def test_no_surge_under_threshold(self):
        prev = {"error_total_seen": 8.0}  # delta = 2 < 5
        res = evaluate_alerts(
            probe=_probe(), prev_state=prev, thresholds=_thresholds(), now_ts=1000.0,
        )
        assert not any(a["key"] == "error_surge" for a in res["alerts"])
        assert res["state"]["error_total_seen"] == 10.0  # baseline 始终推进

    def test_baseline_advances_even_when_unreachable(self):
        # server 不可达本轮无 metrics → 不动 error baseline (保留上轮), 不误报 surge
        prev = {"error_total_seen": 4.0}
        res = evaluate_alerts(
            probe={"reachable": False, "health_status": None, "metrics": None},
            prev_state=prev, thresholds=_thresholds(), now_ts=1000.0,
        )
        assert not any(a["key"] == "error_surge" for a in res["alerts"])
        assert res["state"]["error_total_seen"] == 4.0


# ==================== 组合 / state 形状 ====================

class TestStateShape:
    def test_state_always_has_active_and_baseline_keys(self):
        res = evaluate_alerts(
            probe=_probe(), prev_state={}, thresholds=_thresholds(), now_ts=1000.0,
        )
        assert "active" in res["state"]
        assert "error_total_seen" in res["state"]

    def test_multiple_alerts_one_pass(self):
        # 队列 18/20 饱和 + 错误激增 同轮
        prev = {"error_total_seen": 0.0}
        res = evaluate_alerts(
            probe=_probe(), prev_state=prev, thresholds=_thresholds(), now_ts=1000.0,
        )
        keys = {a["key"] for a in res["alerts"]}
        assert "queue_saturation" in keys
        assert "error_surge" in keys


# ==================== build_probe (HTTP 响应 → probe) ====================

class TestBuildProbe:
    def test_healthy_with_metrics(self):
        p = build_probe(health_ok=True, health_body={"status": "healthy"},
                        metrics_text=_METRICS_SAMPLE)
        assert p["reachable"] is True
        assert p["health_status"] == "healthy"
        assert p["metrics"]["flat"]["funasr_queue_size"] == 18.0

    def test_unreachable(self):
        p = build_probe(health_ok=False, health_body=None, metrics_text=None)
        assert p["reachable"] is False
        assert p["health_status"] is None
        assert p["metrics"] is None

    def test_reachable_but_metrics_denied(self):
        # /health 通但 /metrics 鉴权失败 (None text) → reachable, metrics None
        p = build_probe(health_ok=True, health_body={"status": "healthy"},
                        metrics_text=None)
        assert p["reachable"] is True
        assert p["metrics"] is None


# ==================== state 读写 ====================

class TestStateIO:
    def test_load_missing_returns_empty(self, tmp_path):
        assert load_state(str(tmp_path / "nope.json")) == {}

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "state.json")
        state = {"active": {"server_down": {"since": 1.0, "last_notified": 1.0}},
                 "error_total_seen": 42.0}
        save_state(path, state)
        assert load_state(path) == state

    def test_load_corrupt_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert load_state(str(path)) == {}

    def test_save_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "state.json")
        save_state(path, {"active": {}})
        assert load_state(path) == {"active": {}}


# ==================== 跨模块: 真渲染器 → 解析器 round-trip ====================

class TestRenderParseRoundTrip:
    """钉死 parse_prometheus 与 http_endpoints.render_prometheus 输出格式一致,
    防两模块各演化导致告警脚本静默读不到指标。"""

    def test_real_render_parses(self):
        from src.api.http_endpoints import render_prometheus

        snapshot = {
            "queue_size": 18, "max_queue_size": 20, "pending": 3, "processing": 1,
            "tasks_in_memory": 25, "task_seconds_ema": 42.5, "pool_size": 1,
            "engine": "funasr",
            "terminal_total": {"completed": 100, "failed": 4},
            "errors_total": {"engine_error": 7, "queue_full": 3},
        }
        text = render_prometheus(snapshot, cache_stats={"hits": 50}, vram_mib=None)
        parsed = parse_prometheus(text)

        assert parsed["flat"]["funasr_queue_size"] == 18.0
        assert parsed["flat"]["funasr_queue_max"] == 20.0
        assert parsed["labeled"]["funasr_errors_total"]["engine_error"] == 7.0
        assert parsed["labeled"]["funasr_errors_total"]["queue_full"] == 3.0
        # 解析结果直接喂 evaluate 不报错且能识别饱和
        res = evaluate_alerts(
            probe={"reachable": True, "health_status": "healthy", "metrics": parsed},
            prev_state={"error_total_seen": 0.0}, thresholds=_thresholds(), now_ts=1.0,
        )
        keys = {a["key"] for a in res["alerts"]}
        assert "queue_saturation" in keys and "error_surge" in keys
