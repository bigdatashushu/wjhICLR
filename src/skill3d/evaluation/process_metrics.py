"""G-65 过程指标聚合（§16.2 指标表）：从在线链产物聚合论文需要的中间量。

指标（§16.2）：
- 程序执行成功率 / 几何验证通过率 / 回退率（fallback / no-tool 兜底）
- 修复轮数（AST 重生成次数，从 `EpisodeOutcome.states` 的 SYNTHESIZE 重复次数推）
- 工具复用率（被调用 Tool 种类 / 总调用次数；越接近 1 表示编排越集中）
- sample efficiency（达到目标指标所需的 episode 数，越小越好）
- 每代提升 Δ（优化循环每代的指标变化）
- 回归率（优化后性能下降的 episode 比例）
- 发现 bug 数（变形测试/反例挖掘；由离线侧注入）
- 资源成本（wall-clock / tool 调用数；构建成本与推理成本分开报告）

输入为在线链产物（`EpisodeOutcome` 或同构对象）+ 可选的代际曲线；
不依赖具体实验规模，缺项以 `None` 表示"不可算"，不臆造 0。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# TODO_CALIBRATE：几何验证 / 成功率的目标阈值（论文报告用）
DEFAULT_TARGET_MCA = 0.5


@dataclass
class ProcessMetrics:
    """一次评测 run 的过程指标（论文 §16.2）。"""

    n_episodes: int = 0
    n_answer: int = 0
    n_answer_best_effort: int = 0
    n_unanswerable: int = 0
    n_unavailable: int = 0
    program_success_rate: Optional[float] = None      # 沙箱执行成功比例
    geometry_pass_rate: Optional[float] = None        # 几何验证通过比例
    fallback_rate: Optional[float] = None             # 走兜底/受限 Tool 集的比例
    gated_rate: Optional[float] = None                # 被逐题门控拒答的比例（HC33/G11）
    # --- D-3 / §4 M6 字段 9：可靠性指标（"系统可靠性"小节，**不混主表**）---
    refusal_rate: Optional[float] = None              # abstain / unanswerable 占比
    coverage: Optional[float] = None                  # 产出答案的 episode 占比
    coverage_conditioned_mra: Optional[float] = None  # 只在"作答"episode 上算的 MRA
    coverage_conditioned_accuracy: Optional[float] = None
    tool_contract_rate: Optional[float] = None        # 命中 tool_contract 的 episode 占比
    tool_contract_calls: int = 0                      # tool_contract 归因的 Tool 调用数
    tool_contract_recovered: int = 0                  # 回灌/裁剪重生成后恢复的 episode 数
    abstain_rate: Optional[float] = None              # 显式 abstain 占比
    mean_regen_rounds: Optional[float] = None         # AST 重生成轮数均值
    tool_calls_total: int = 0
    unique_tools: int = 0
    tool_reuse_rate: Optional[float] = None           # unique/total（越低越集中）
    llm_calls: int = 0
    stub_program_ratio: Optional[float] = None        # mock_light 占比（诚实性指标）
    wallclock_s_total: float = 0.0
    sample_efficiency: Optional[int] = None           # 达到目标所需 episode 数
    per_generation_delta: list[float] = field(default_factory=list)
    regression_rate: Optional[float] = None
    bugs_found: int = 0
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("notes", None)
        d["per_generation_delta"] = json.dumps(self.per_generation_delta)
        return d


def _ratio(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def _program_ok(outcome) -> Optional[bool]:
    """该 episode 的 program 是否成功执行（无 program 记 None，不参与统计）。"""
    trace = getattr(outcome, "program_trace", None)
    if trace is None:
        return None
    return trace.error_code is None and bool(trace.steps)


def _regen_rounds(outcome) -> int:
    """AST 重生成轮数：states 序列里 SYNTHESIZE_PROGRAM 出现次数 − 1（下界 0）。"""
    states = list(getattr(outcome, "states", []) or [])
    n = sum(1 for s in states if s == "SYNTHESIZE_PROGRAM")
    return max(n - 1, 0)


def sample_efficiency(curve: Sequence[tuple[int, float]],
                      target: float = DEFAULT_TARGET_MCA) -> Optional[int]:
    """达到 `target` 所需的最小 episode 数（曲线须按 episode 数升序）。

    曲线为 `[(n_episodes, metric)]`；始终未达标返回 None（不谎报）。
    """
    for n, metric in sorted(curve, key=lambda t: t[0]):
        if metric >= target:
            return int(n)
    return None


def per_generation_delta(generations: Sequence[tuple[int, float]]) -> list[float]:
    """每代提升 Δ（按代际排序的相邻差；首代为 0 → 不产出）。"""
    ordered = sorted(generations, key=lambda t: t[0])
    return [round(curve_b - curve_a, 6) for (_g1, curve_a), (_g2, curve_b)
            in zip(ordered, ordered[1:])]


def regression_rate(before: Sequence[float], after: Sequence[float]) -> Optional[float]:
    """回归率：优化后指标下降的 episode 比例（逐 episode 配对比较）。"""
    pairs = list(zip(before, after))
    if not pairs:
        return None
    return sum(1 for a, b in pairs if b < a) / len(pairs)


def aggregate_process_metrics(
    outcomes: Sequence,
    *,
    generations: Optional[Sequence[tuple[int, float]]] = None,
    learning_curve: Optional[Sequence[tuple[int, float]]] = None,
    target_metric: float = DEFAULT_TARGET_MCA,
    before_scores: Optional[Sequence[float]] = None,
    after_scores: Optional[Sequence[float]] = None,
    bugs_found: int = 0,
) -> ProcessMetrics:
    """从在线链产物聚合过程指标（§16.2）。所有比例在分母为 0 时为 None。"""
    m = ProcessMetrics(n_episodes=len(outcomes), bugs_found=int(bugs_found))
    if not outcomes:
        m.notes.append("无 episode：所有比例为 None（不臆造 0）")
        return m

    n_prog_ok = n_prog = n_geo_ok = n_geo = 0
    tools: list[str] = []
    regen: list[int] = []
    from_mock = 0
    for o in outcomes:
        state = getattr(o, "final_state", "")
        if state == "answer":
            m.n_answer += 1
        elif state == "answer_best_effort":
            m.n_answer_best_effort += 1
        elif state == "unavailable":
            m.n_unavailable += 1
        else:
            m.n_unanswerable += 1

        ok = _program_ok(o)
        if ok is not None:
            n_prog += 1
            n_prog_ok += int(ok)
        verify = getattr(o, "verify", None)
        if verify is not None:
            n_geo += 1
            n_geo_ok += int(bool(verify.passed))
        trace = getattr(o, "program_trace", None)
        if trace is not None:
            tools.extend(r.tool for r in (trace.results or []))
            m.wallclock_s_total += float(getattr(trace, "wallclock_s", 0.0) or 0.0)
        regen.append(_regen_rounds(o))
        if getattr(o, "synthesis_source", "") == "deterministic_stub":
            from_mock += 1

    m.program_success_rate = _ratio(n_prog_ok, n_prog)
    m.geometry_pass_rate = _ratio(n_geo_ok, n_geo)
    m.mean_regen_rounds = (sum(regen) / len(regen)) if regen else None
    m.tool_calls_total = len(tools)
    m.unique_tools = len(set(tools))
    m.tool_reuse_rate = _ratio(m.unique_tools, m.tool_calls_total)
    m.stub_program_ratio = _ratio(from_mock, len(outcomes))

    # --- D-3/硬约束 23：契约与可靠性指标（与主表分离报告）---
    n_contract = 0
    n_recovered = 0
    contract_calls = 0
    for o in outcomes:
        hits = int(getattr(o, "tool_contract_hits", 0) or 0)
        flags_o = getattr(o, "answer_flags", []) or []
        if hits or "tool_contract" in flags_o:
            n_contract += 1
            if getattr(o, "replay_used", False) or getattr(o, "trimmed_regen_used", False):
                n_recovered += 1
        trace_o = getattr(o, "program_trace", None)
        if trace_o is not None:
            contract_calls += sum(1 for r in (trace_o.results or [])
                                  if getattr(r, "error_code", None) == "tool_contract")
    m.tool_contract_rate = _ratio(n_contract, len(outcomes))
    m.tool_contract_calls = contract_calls
    m.tool_contract_recovered = n_recovered

    abstained = [o for o in outcomes
                 if getattr(o, "abstained", False)
                 or getattr(o, "final_state", "") in ("unanswerable", "abstain")]
    m.abstain_rate = _ratio(
        sum(1 for o in outcomes if getattr(o, "abstained", False)), len(outcomes))
    m.refusal_rate = _ratio(len(abstained), len(outcomes))
    answered = [o for o in outcomes if getattr(o, "final_state", "") in
                ("answer", "answer_best_effort")]
    m.coverage = _ratio(len(answered), len(outcomes))
    # coverage-conditioned：只在真的作答的 episode 上算（与主表口径不同，单独报告）
    ans_mca = [o for o in answered if getattr(o, "is_mca", False)]
    ans_na = [o for o in answered if not getattr(o, "is_mca", False)]
    m.coverage_conditioned_accuracy = _ratio(
        sum(1 for o in ans_mca if getattr(o, "correct", False)), len(ans_mca))
    na_vals = [float(o.mra_value) for o in ans_na
               if getattr(o, "mra_value", None) is not None]
    m.coverage_conditioned_mra = (sum(na_vals) / len(na_vals)) if na_vals else None

    flags = [f for o in outcomes for f in (getattr(o, "answer_flags", []) or [])]
    m.fallback_rate = _ratio(
        sum(1 for o in outcomes
            if "no_tool_fallback" in (getattr(o, "answer_flags", []) or [])
            or getattr(o, "scene_route", "") == "fallback_2d_only"),
        len(outcomes))
    # 含已删除的 g8_size_reject 仅为读旧 trace（新记录不会再产生该 flag）
    m.gated_rate = _ratio(sum(1 for f in flags if f in ("g8_size_reject",
                                                       "g11_measurement_2d_only")),
                          len(outcomes))
    if generations:
        m.per_generation_delta = per_generation_delta(generations)
    if before_scores is not None and after_scores is not None:
        m.regression_rate = regression_rate(before_scores, after_scores)
    if learning_curve:
        m.sample_efficiency = sample_efficiency(learning_curve, target_metric)
        if m.sample_efficiency is None:
            m.notes.append(f"sample_efficiency=None：未在曲线上达到 target={target_metric}")
    if m.stub_program_ratio:
        m.notes.append("含 deterministic_stub program：mock_light 仅管道验证，"
                       "不得作为结果（§9.2）")
    if m.tool_contract_rate:
        m.notes.append(
            f"tool_contract 命中 {n_contract}/{len(outcomes)} 个 episode（"
            f"{contract_calls} 次 Tool 调用）：abstain 主榜按错计，"
            "本组比例只在'系统可靠性'小节报告（§4 M6 字段 9）")
    return m


def write_process_metrics_csv(metrics: ProcessMetrics, out_path: str | Path) -> Path:
    """过程指标落 CSV（论文表格直接可用；CSV/JSONL 足以支撑论文，§13 G-06）。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = metrics.to_row()
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return out


def build_learning_curve(outcomes: Sequence, *, metric: str = "mca",
                         curve_csv: str | Path = "") -> list[tuple[int, float]]:
    """由 episode 结果序列构造学习曲线 `[(n_episodes, 累计指标)]`。

    metric ∈ {mca, combined}：mca 只累计选择题；combined 为 MCA/WRONG 与 MRA 的混合
    （与在线 runner 的评分口径一致：MCA → 1/0，NA → MRA）。
    """
    curve: list[tuple[int, float]] = []
    total = 0.0
    for i, o in enumerate(outcomes, start=1):
        is_mca = bool(getattr(o, "is_mca", False))
        if is_mca:
            total += 1.0 if getattr(o, "correct", False) else 0.0
        else:
            total += float(getattr(o, "mra_value", 0.0) or 0.0)
        if metric == "combined" or is_mca:
            curve.append((i, total / i))
    if curve_csv:
        out = Path(curve_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_episodes", "metric"])
            w.writerows(curve)
    return curve
