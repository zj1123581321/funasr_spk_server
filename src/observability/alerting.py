"""告警纯逻辑: 把 /metrics + /health 探测结果 → 告警事件列表 + 新 state。

设计 (低成本高价值, 与 server 进程解耦):
  外部脚本 (scripts/metrics_alert.py, launchd 定时拉起) 周期 GET /health + /metrics,
  把结果喂给 evaluate_alerts(纯函数, now_ts 注入), 拿到要发的告警 + 持久化 state,
  发企微 (复用 src/utils/notification.send_custom_notification) 后落盘 state。

为什么纯函数 + state 注入: server_down / queue_saturation 是**状态**(要去抖 + 恢复
通知), error_surge 是**事件**(单调 counter 的 per-轮增量)。两者都需要"上一轮"信息,
但脚本是短命进程 (每轮重新拉起), 故 state 必须外部持久化。纯函数零 I/O / 零时钟,
100% 可测 (见 tests/unit/test_alerting.py)。

告警规则:
  - server_down (critical): 不可达 OR /health != healthy。状态型, 带恢复通知。
  - queue_saturation (warning): queue_size/queue_max >= ratio。状态型, 带恢复通知。
  - error_surge (warning): errors_total 累计本轮较上轮增量 >= threshold。事件型, 无恢复。

去抖: 同一状态型告警在 cooldown_seconds 内不重复通知 (仍保持 active)。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlertThresholds:
    """告警阈值 (来自 ObservabilityConfig)。"""
    queue_saturation_ratio: float
    error_surge_threshold: int
    cooldown_seconds: int


# Prometheus 文本行: `name{labels} value` 或 `name value`
_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)\s*$")
_LABEL_KV = re.compile(r'(\w+)="([^"]*)"')


def parse_prometheus(text: str) -> dict:
    """把 Prometheus 文本解析成 {"flat": {name: float}, "labeled": {name: {labelval: float}}}.

    - 无 label 的指标进 flat (funasr_queue_size → 18.0)。
    - 带 label 的进 labeled, 用**首个** label 值做子键 (errors_total{kind=X} → {X: v}),
      够覆盖本项目所有 label 指标 (errors_total/kind, tasks_terminal_total/status,
      engine_info/engine)。注释 (#) / 空行跳过。无法解析的行静默忽略 (容错优先)。
    """
    flat: dict[str, float] = {}
    labeled: dict[str, dict[str, float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name, labels, val_str = m.group(1), m.group(2), m.group(3)
        try:
            value = float(val_str)
        except ValueError:
            continue
        if labels:
            kvs = _LABEL_KV.findall(labels)
            if not kvs:
                continue
            labeled.setdefault(name, {})[kvs[0][1]] = value
        else:
            flat[name] = value
    return {"flat": flat, "labeled": labeled}


def build_probe(*, health_ok: bool, health_body: Optional[dict],
                metrics_text: Optional[str]) -> dict:
    """把 /health + /metrics 的 HTTP 结果组装成 evaluate_alerts 的 probe 输入 (纯函数)。

    - health_ok: /health 是否拿到 2xx。False → 服务不可达。
    - health_body: /health JSON (含 status 字段), 不可达时 None。
    - metrics_text: /metrics 文本; 拿不到 (鉴权失败/错误) → None → 不评估指标型规则。
    """
    return {
        "reachable": bool(health_ok),
        "health_status": (health_body or {}).get("status") if health_ok else None,
        "metrics": parse_prometheus(metrics_text) if metrics_text is not None else None,
    }


def load_state(path: str) -> dict:
    """读告警 state JSON; 不存在 / 损坏 → 返回 {} (首轮语义, 容错优先)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_state(path: str, state: dict) -> None:
    """落盘告警 state (建父目录)。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _make_alert(key: str, level: str, title: str, body: str) -> dict:
    return {"key": key, "level": level, "title": title, "body": body}


def _process_state_rule(
    *,
    key: str,
    firing: bool,
    level: str,
    fire_title: str,
    fire_body: str,
    recovery_title: str,
    prev_active: dict,
    now_ts: float,
    cooldown_seconds: float,
    emit_recovery: bool,
) -> tuple[list[dict], Optional[dict]]:
    """状态型规则的统一去抖/恢复处理。

    返回 (本轮要发的告警列表, 该规则的新 active entry 或 None)。
      - firing 且此前未 active → 立即告警, 记 since/last_notified。
      - firing 且已 active → 仅当超 cooldown 才重复告警; 否则静默保持 active。
      - 不 firing 且此前 active → (emit_recovery 时) 发恢复通知, 清除 active。
    """
    alerts: list[dict] = []
    was = prev_active.get(key)
    if firing:
        if was is not None:
            entry = dict(was)
            if now_ts - entry.get("last_notified", 0.0) >= cooldown_seconds:
                alerts.append(_make_alert(key, level, fire_title, fire_body))
                entry["last_notified"] = now_ts
            return alerts, entry
        entry = {"since": now_ts, "last_notified": now_ts}
        alerts.append(_make_alert(key, level, fire_title, fire_body))
        return alerts, entry
    # 未 firing
    if was is not None and emit_recovery:
        alerts.append(_make_alert(key, "recovery", recovery_title,
                                  "指标已回落到阈值以下。"))
    return alerts, None


def evaluate_alerts(*, probe: dict, prev_state: dict, thresholds: AlertThresholds,
                    now_ts: float) -> dict:
    """评估一轮探测, 返回 {"alerts": [...], "state": {...}}.

    probe = {
      "reachable": bool,           # /health 能否连上
      "health_status": str | None, # /health 的 status 字段 (healthy/degraded/...)
      "metrics": dict | None,      # parse_prometheus 结果, 不可达时 None
    }
    prev_state = 上一轮返回的 state (首轮传 {})。
    """
    prev_active = dict(prev_state.get("active", {}))
    new_active: dict = {}
    alerts: list[dict] = []
    cooldown = float(thresholds.cooldown_seconds)

    metrics = probe.get("metrics")
    reachable = bool(probe.get("reachable"))
    health_status = probe.get("health_status")

    # ---- 规则 1: server_down (critical, 状态型) ----
    # 不可达 → down; 可达但 status 非 healthy → down; 可达且 healthy → 不 down。
    down = (not reachable) or (health_status != "healthy")
    a, entry = _process_state_rule(
        key="server_down", firing=down, level="critical",
        fire_title="🔴 FunASR 服务异常",
        fire_body=("服务不可达 (/health 连不上)。" if not reachable
                   else f"/health 状态={health_status} (非 healthy)。"),
        recovery_title="🟢 FunASR 服务恢复",
        prev_active=prev_active, now_ts=now_ts, cooldown_seconds=cooldown,
        emit_recovery=True,
    )
    alerts += a
    if entry is not None:
        new_active["server_down"] = entry

    # ---- 规则 2: queue_saturation (warning, 状态型) ----
    # 仅在有 metrics 时评估 (不可达由 server_down 覆盖, 不重复打扰)。
    if metrics is not None:
        flat = metrics.get("flat", {})
        q = flat.get("funasr_queue_size")
        qmax = flat.get("funasr_queue_max")
        saturated = False
        body = ""
        if q is not None and qmax is not None and qmax > 0:
            ratio = q / qmax
            saturated = ratio >= thresholds.queue_saturation_ratio
            body = (f"队列深度 {int(q)}/{int(qmax)} "
                    f"({ratio:.0%} ≥ {thresholds.queue_saturation_ratio:.0%})。")
        a, entry = _process_state_rule(
            key="queue_saturation", firing=saturated, level="warning",
            fire_title="🟠 任务队列接近饱和",
            fire_body=body,
            recovery_title="🟢 任务队列已回落",
            prev_active=prev_active, now_ts=now_ts, cooldown_seconds=cooldown,
            emit_recovery=True,
        )
        alerts += a
        if entry is not None:
            new_active["queue_saturation"] = entry
    else:
        # 无 metrics: 保留上轮 queue_saturation active (不误判恢复)
        if "queue_saturation" in prev_active:
            new_active["queue_saturation"] = prev_active["queue_saturation"]

    # ---- 规则 3: error_surge (warning, 事件型, 无恢复) ----
    # errors_total 是单调 counter, 用本轮 - 上轮的增量判定。baseline 始终推进。
    prev_seen = prev_state.get("error_total_seen")
    error_total_seen = prev_seen  # 默认沿用 (不可达本轮不动)
    if metrics is not None:
        labeled = metrics.get("labeled", {})
        current_total = float(sum(labeled.get("funasr_errors_total", {}).values()))
        if prev_seen is not None:
            delta = current_total - prev_seen
            firing = delta >= thresholds.error_surge_threshold
            a, entry = _process_state_rule(
                key="error_surge", firing=firing, level="warning",
                fire_title="🟠 错误激增",
                fire_body=(f"错误累计本轮新增 {int(delta)} 起 "
                           f"(≥ {thresholds.error_surge_threshold}), 累计 {int(current_total)}。"),
                recovery_title="",
                prev_active=prev_active, now_ts=now_ts, cooldown_seconds=cooldown,
                emit_recovery=False,
            )
            alerts += a
            if entry is not None:
                new_active["error_surge"] = entry
        # 首轮 (prev_seen is None): 不告警, 仅建基线
        error_total_seen = current_total

    state = {"active": new_active}
    if error_total_seen is not None:
        state["error_total_seen"] = error_total_seen
    return {"alerts": alerts, "state": state}
