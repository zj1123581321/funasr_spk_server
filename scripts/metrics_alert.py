#!/usr/bin/env python3
"""可观测性告警: 定时拉 /health + /metrics, 超阈值发企微 (launchd 周期拉起)。

与 server 进程**解耦** —— server 整个挂了, 本脚本拉不到 /health 也能告警 (server
内嵌方案做不到)。纯逻辑在 src/observability/alerting.py (已单测), 本文件只是薄壳:
HTTP 抓取 + 复用 src/utils/notification.send_custom_notification 发送 + state 落盘。

用法 (单次执行, 退出码 0=正常):
    venv/bin/python scripts/metrics_alert.py
launchd 用 StartInterval 周期拉起即可 (见 docs/部署.md 告警节)。

开关: config.observability.alert_enabled (env FUNASR_ALERT_ENABLED=true)。未启用直接退出。
阈值: alert_queue_saturation_ratio / alert_error_surge_threshold / alert_cooldown_seconds。
state: 默认落在 DB 同目录 alert_state.json, env FUNASR_ALERT_STATE_PATH 可覆盖。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

# 保证可从项目根 import src.*
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import httpx
from loguru import logger

from src.core.config import config
from src.observability.alerting import (
    AlertThresholds,
    build_probe,
    evaluate_alerts,
    load_state,
    save_state,
)
from src.utils.notification import send_custom_notification


def _scrape_base() -> str:
    """本机回环 + server 端口 (server.host 可能是 0.0.0.0, 抓取走 127.0.0.1)。"""
    return f"http://127.0.0.1:{config.server.port}"


def _state_path() -> str:
    env = os.getenv("FUNASR_ALERT_STATE_PATH")
    if env:
        return env
    db_dir = os.path.dirname(config.database.path) or "."
    return os.path.join(db_dir, "alert_state.json")


def probe_once(base: str, token, timeout: float) -> dict:
    """抓 /health + /metrics, 组装成 evaluate_alerts 的 probe。

    reachable 看"是否拿到任何 HTTP 响应"(503 degraded 也算可达), 仅连接错误 → 不可达。
    /metrics 仅 200 才取文本 (鉴权失败/错误 → None, 指标型规则本轮跳过)。
    """
    health_ok = False
    health_body = None
    metrics_text = None
    try:
        with httpx.Client(timeout=timeout) as client:
            try:
                r = client.get(base + "/health")
                health_ok = True  # 拿到响应即可达 (200 或 503)
                try:
                    health_body = r.json()
                except Exception:
                    health_body = None
            except httpx.RequestError as exc:
                logger.warning(f"/health 抓取失败 (视为不可达): {exc}")

            try:
                params = {"token": token} if token else None
                r = client.get(base + "/metrics", params=params)
                if r.status_code == 200:
                    metrics_text = r.text
                else:
                    logger.warning(f"/metrics 返回 {r.status_code} (本轮跳过指标型规则)")
            except httpx.RequestError as exc:
                logger.warning(f"/metrics 抓取失败: {exc}")
    except Exception as exc:  # httpx.Client 构造等极端情况
        logger.error(f"探测异常: {exc}")

    return build_probe(health_ok=health_ok, health_body=health_body, metrics_text=metrics_text)


async def _send_all(alerts: list[dict]) -> None:
    """逐条发企微 (复用 notification 的 webhook/重试/设备标识)。"""
    for a in alerts:
        await send_custom_notification(a["title"], a["body"])


def main() -> int:
    obs = config.observability
    if not obs.alert_enabled:
        logger.info("告警未启用 (alert_enabled=false), 跳过。")
        return 0

    thresholds = AlertThresholds(
        queue_saturation_ratio=obs.alert_queue_saturation_ratio,
        error_surge_threshold=obs.alert_error_surge_threshold,
        cooldown_seconds=obs.alert_cooldown_seconds,
    )
    base = _scrape_base()
    probe = probe_once(base, obs.metrics_token, timeout=10.0)

    state_path = _state_path()
    prev_state = load_state(state_path)
    result = evaluate_alerts(
        probe=probe, prev_state=prev_state, thresholds=thresholds, now_ts=time.time()
    )
    save_state(state_path, result["state"])

    alerts = result["alerts"]
    if alerts:
        logger.warning(f"触发 {len(alerts)} 条告警: {[a['key'] for a in alerts]}")
        asyncio.run(_send_all(alerts))
    else:
        logger.debug("本轮无告警。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
