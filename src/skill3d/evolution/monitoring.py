"""G-37/G-39 离线运行期监控（简化版）：实验后统计 → 触发回滚 / 再审。

§7.1 G-37/G-39 的学术版处置：**不做常驻监控进程**。跑完实验后用
CSV/JSONL 日志计算指标，异常时手动/自动调用已有的 `rollback()`；
必要时离线触发语义审查（v6 §3.4：离线模型 = DeepSeek-V4.1-Flash，写
`DeepSeekGovernanceDecision`，产物为 advisory，不决定 promote/reject）。

监控指标（§8 运行期监控）：成功率 / 几何验证率 / 回退率 / 资源超限 / 答案分布漂移
（KL 散度，相对上一代）。

消融角色：§8.4 的 G2 档（"无监控"）即跳过本模块的判定。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# 触发再审/回滚的起始阈值（TODO_CALIBRATE；全部为参考值）
TH_SUCCESS_DROP = 0.10          # 成功率相对上代下降超过此值 → 告警
TH_GEOMETRY_DROP = 0.10         # 几何验证率下降 → 告警
TH_FALLBACK_RISE = 0.15         # 回退率上升 → 告警
TH_KL_DRIFT = 0.20              # 答案分布 KL 散度超此值 → 漂移告警


@dataclass
class RunHealth:
    """一代（或一次 run）的健康指标。"""

    label: str = ""
    n_episodes: int = 0
    success_rate: Optional[float] = None       # final_state == answer 比例
    answer_rate: Optional[float] = None        # 含 best_effort
    geometry_pass_rate: Optional[float] = None
    fallback_rate: Optional[float] = None
    unavailable_rate: Optional[float] = None
    gate_reject_rate: Optional[float] = None
    mean_wallclock_s: Optional[float] = None
    answer_histogram: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HealthReport:
    """与上一代对比后的监控结论。"""

    current: RunHealth
    previous: Optional[RunHealth] = None
    alerts: list[str] = field(default_factory=list)
    kl_drift: Optional[float] = None
    action: str = "none"            # none | review | rollback
    reason: str = ""

    def to_dict(self) -> dict:
        return {"current": self.current.to_dict(),
                "previous": (self.previous.to_dict() if self.previous else None),
                "alerts": self.alerts, "kl_drift": self.kl_drift,
                "action": self.action, "reason": self.reason}


def _ratio(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def collect_health(outcomes: Sequence, label: str = "") -> RunHealth:
    """从在线链产物聚合健康指标（缺项 None，不臆造 0）。"""
    h = RunHealth(label=label, n_episodes=len(outcomes))
    if not outcomes:
        return h
    n = len(outcomes)
    n_ok = sum(1 for o in outcomes if getattr(o, "final_state", "") == "answer")
    n_best = sum(1 for o in outcomes
                 if getattr(o, "final_state", "") == "answer_best_effort")
    n_unavail = sum(1 for o in outcomes
                    if getattr(o, "final_state", "") == "unavailable")
    verifies = [o for o in outcomes if getattr(o, "verify", None) is not None]
    n_geo = sum(1 for o in verifies if o.verify.passed)
    n_fb = sum(1 for o in outcomes
               if "no_tool_fallback" in (getattr(o, "answer_flags", []) or [])
               or getattr(o, "scene_route", "") == "fallback_2d_only")
    n_gate = sum(1 for o in outcomes
                 # g8_size_reject 仅为兼容旧 trace
                 if any(f in ("g8_size_reject", "g11_measurement_2d_only")
                        for f in (getattr(o, "answer_flags", []) or [])))
    walls = [float(getattr(o.program_trace, "wallclock_s", 0.0) or 0.0)
             for o in outcomes if getattr(o, "program_trace", None) is not None]

    h.success_rate = _ratio(n_ok, n)
    h.answer_rate = _ratio(n_ok + n_best, n)
    h.geometry_pass_rate = _ratio(n_geo, len(verifies))
    h.fallback_rate = _ratio(n_fb, n)
    h.unavailable_rate = _ratio(n_unavail, n)
    h.gate_reject_rate = _ratio(n_gate, n)
    h.mean_wallclock_s = (sum(walls) / len(walls)) if walls else None
    hist: dict[str, int] = {}
    for o in outcomes:
        state = str(getattr(o, "final_state", "unknown"))
        hist[state] = hist.get(state, 0) + 1
    h.answer_histogram = hist
    return h


def _normalize(hist: dict) -> np.ndarray:
    if not hist:
        return np.zeros(0)
    keys = sorted(hist)
    v = np.asarray([hist[k] for k in keys], dtype=float)
    total = v.sum()
    return v / total if total > 0 else v


def answer_distribution_kl(current: dict, previous: dict,
                           eps: float = 1e-9) -> Optional[float]:
    """答案分布漂移：KL(current || previous)（同名状态对齐；无样本返回 None）。"""
    if not current or not previous:
        return None
    keys = sorted(set(current) | set(previous))
    p = np.asarray([current.get(k, 0) for k in keys], dtype=float)
    q = np.asarray([previous.get(k, 0) for k in keys], dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def monitor(current: RunHealth, previous: Optional[RunHealth] = None, *,
            th_success_drop: float = TH_SUCCESS_DROP,
            th_geometry_drop: float = TH_GEOMETRY_DROP,
            th_fallback_rise: float = TH_FALLBACK_RISE,
            th_kl_drift: float = TH_KL_DRIFT,
            monitoring_enabled: bool = True) -> HealthReport:
    """对比两代指标 → 告警与建议动作（G0 全治理；G2 档传 monitoring_enabled=False）。"""
    report = HealthReport(current=current, previous=previous)
    if not monitoring_enabled:
        report.reason = "治理消融 G2：运行期监控关闭（§8.4）"
        return report
    if previous is None:
        report.reason = "无上一代指标：仅记录本代健康度，不判定回滚"
        return report

    def _drop(cur, prev) -> Optional[float]:
        if cur is None or prev is None:
            return None
        return prev - cur

    d_succ = _drop(current.success_rate, previous.success_rate)
    d_geo = _drop(current.geometry_pass_rate, previous.geometry_pass_rate)
    d_fb = _drop(current.fallback_rate, previous.fallback_rate)
    if d_succ is not None and d_succ > th_success_drop:
        report.alerts.append(f"成功率下降 {d_succ:+.3f} > {th_success_drop}")
    if d_geo is not None and d_geo > th_geometry_drop:
        report.alerts.append(f"几何验证率下降 {d_geo:+.3f} > {th_geometry_drop}")
    if d_fb is not None and d_fb > th_fallback_rise:
        report.alerts.append(f"回退率上升 {d_fb:+.3f} > {th_fallback_rise}")
    kl = answer_distribution_kl(current.answer_histogram, previous.answer_histogram)
    report.kl_drift = kl
    if kl is not None and kl > th_kl_drift:
        report.alerts.append(f"答案分布漂移 KL={kl:.3f} > {th_kl_drift}")

    if len(report.alerts) >= 2:
        report.action = "rollback"
        report.reason = "; ".join(report.alerts)
    elif report.alerts:
        report.action = "review"
        report.reason = "; ".join(report.alerts)
    else:
        report.reason = "指标未见异常"
    return report


def write_health_report(report: HealthReport, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def trigger_review(candidate, offline_client, trace_store=None):
    """G-39：监控异常 → 离线触发语义审查（写 `DeepSeekGovernanceDecision`，§3.4）。

    产物是 **advisory**（`advisory_only=True`）：它只解释"为什么告警"，**不**决定
    promote/reject（§3.3：准入由确定性门 + 预注册规则决定）。
    """
    from skill3d.governance.governance_decision import semantic_review

    decision = semantic_review(candidate, offline_client)
    if trace_store is not None:
        trace_store.append("skill_governance_decision", decision.model_dump())
    return decision


def rollback_if_needed(report: HealthReport, store_dir: str | Path,
                       trace_store=None) -> Optional[dict]:
    """G-37：监控判定 rollback 时回滚到**上一个** snapshot（硬约束 12：旧 snapshot 保留）。

    回滚目标是 active snapshot 的 `parent_snapshot_id`（promote 时记录的
    `snapshot_before`）；已处于 genesis 或无父 snapshot 时不动（返回 None 并记 reason）。
    """
    if report.action != "rollback":
        return None
    from skill3d.skills.promote_atomic import read_active_snapshot, rollback

    active = read_active_snapshot(store_dir)
    target = active.get("parent_snapshot_id")
    if not target:
        if trace_store is not None:
            trace_store.append("rollback", {
                "reason": report.reason, "skipped": True,
                "note": f"无可回滚的父 snapshot（active={active.get('snapshot_id')}）"})
        return None
    log: list = []
    try:
        snap = rollback(store_dir, target, promotion_log=log)
    except FileNotFoundError:
        # 初代 promote 的父是虚拟 "genesis"（无 snapshot 文件）→ 回滚 = 清空 active 指针，
        # 使 read_active_snapshot 重新返回 genesis 语义（无任何 Skill 生效）。
        pointer = Path(store_dir) / "active_snapshot.json"
        if pointer.is_file():
            pointer.unlink()
        snap = {"snapshot_id": "genesis", "entries": {}, "parent_snapshot_id": None}
        if trace_store is not None:
            trace_store.append("rollback", {"reason": report.reason,
                                            "rolled_back_to": "genesis",
                                            "note": "清空 active 指针（无父 snapshot 文件）"})
        return snap
    if trace_store is not None:
        trace_store.append("rollback", {"reason": report.reason,
                                        "snapshot_id": snap.get("snapshot_id"),
                                        "rolled_back_to": target})
    return snap
