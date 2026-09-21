"""v6 §18 实验协议 + §19.4 split 访问审计 + §22 红线 9（Phase 6 的协议内核）。

本模块把第 18 章的纪律变成**可判定的代码**，而不是文档承诺：

1. **三级隔离（§18.1）**：`inner` 可反复迭代、`outer` 只跑一次（第二次要显式
   override + 决策记录）、`final` 完全隔离且仅盲评一次。跨进程可查的 run ledger
   落盘在调用方给的目录；`final` 在 ledger 缺失/损坏时 **fail-closed 拒绝**
   （绝不默认放行"没记录就是没跑过"）。
2. **样本量三档（§18.2）**：16/32/32+ 题 × 3/3/5 seed、每题权重、二值
   SE = 0.5/√N；池 < 32 题/题型时用**池内全部题**（不子采样到 4），不足的运行次数
   靠增加 seed 补足，并如实报告真实池大小。
3. **噪声底（§18.3）**：同配置重复 ≥3 seed（paper ≥5）下的逐题一致率与 per-task
   波动；多副本轮询**只准用于吞吐**，提供显式 assert 拦住"拿轮询做对比"。
4. **paired A/B（§18.4）**：先断言两臂同 frames/同重建产物/同输入/同模型/同模板
   版本/同硬件/同 EvidenceProfile，再出 McNemar（或 paired BCa bootstrap）+ Cliff's
   delta / Cohen's d + Bonferroni；退化样本归一化为 `p=None` 并注
   "退化样本 → 判为不显著"，**不得**呈现为显著。
5. **报告（§18.5）**：每题型 95% CI（不只点估计）、按证据状态分组的选择率 /
   覆盖率 / 胜负（`evidence_state_breakdown`）。
6. **paper_eligible（§18.6）**：数据隔离 + ≥3 seed（paper ≥5）+ 统计门 + 非 mock；
   并且四个 `[待实验]` 项未提供 PoC 证据时一律拒绝。
7. **红线 9（§22）/ §19.4**：策略/阈值只能 `derived_from=inner`；outer 调出来的策略
   进论文一律拒绝；读 outer 必须留理由、策略/阈值文件带 `derived_from` + 哈希、
   跨 split 改动需显式决策记录。

纪律（与全系统一致）：**不臆造任何统计量** —— 算不出来就返回 `None`/`False` 并把
原因写进 `notes`/`reasons`；mock 与退化样本永远不得被表述为已验证事实。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from skill3d.evolution.paired_score import (
    DEGENERATE_NOTE,
    N_BOOTSTRAP,
    bonferroni as _bonferroni,
    cliffs_delta,
    cohens_d_paired,
    mcnemar_test,
    normalize_p,
    paired_bootstrap_ci_detailed,
)
from skill3d.schemas.evidence import CAPABILITIES

from .multi_seed_aggregator import (
    ALPHA,
    MIN_SEEDS_FOR_MAIN_TABLE,
    MIN_SEEDS_PAPER_ELIGIBLE,
    per_question_agreement,
    seed_fluctuation,
)

# ------------------------------------------------------------------ 常量 ----

# §18.3 噪声底：同配置重复最少 seed 数（与主表门槛同源）
MIN_SEEDS_NOISE_FLOOR = MIN_SEEDS_FOR_MAIN_TABLE
# §18.6 paper-eligible 最少 seed 数
MIN_SEEDS_PAPER = MIN_SEEDS_PAPER_ELIGIBLE
# §18.2：完整档位要求的每题型题数（决定"池不足"判定）
MIN_POOL_FOR_FULL_TIER = 32
# §18.2：池不足时用池内全部题，且**明确禁止**子采样到这个数量级（v5 的 4 题/题型）
FORBIDDEN_SUBSAMPLE_PER_TASK = 4
# §18.5：置信水平
CI_LEVEL = 0.95
# 容忍的浮点误差（比值比较）
_EPS = 1e-12

# 多副本轮询**只**允许的用途（§18.3：不得用于任何对比实验）
THROUGHPUT_ONLY_PURPOSES: frozenset[str] = frozenset(
    {"throughput", "throughput_only", "smoke", "throughput_smoke", "pipeline_check"})

# §19.3 synthesis_source 六类 + 显式非论文值 `mock_stub`
SYNTHESIS_SOURCES_V6: tuple[str, ...] = (
    "vllm_ok", "vllm_parse_error", "vllm_service_error",
    "m8_parse_recovered", "direct_answer_fallback", "partial_tool_recovery",
    "mock_stub",
)
# v5 历史取值 → v6 归一（只用于**回读旧落盘**；新记录不得再写 v5 名）
V5_SYNTHESIS_SOURCE_ALIASES: dict[str, str] = {
    "vllm": "vllm_ok",
    "vllm_parse_error": "vllm_parse_error",
    "vllm_service_error": "vllm_service_error",
    "m8_parse_recovered": "m8_parse_recovered",
    "direct_answer_fallback": "direct_answer_fallback",
    "partial_tool_recovery": "partial_tool_recovery",
    "deterministic_stub": "mock_stub",
    "mock_stub": "mock_stub",
    "none": "",
    "": "",
}
# mock 血统的 synthesis_source（两者都不得进主表，HC24）
MOCK_SYNTHESIS_SOURCES: frozenset[str] = frozenset({"mock_stub", "deterministic_stub"})
# v5 answer_source → v6 四值（§5.3）。未识别的取值原样保留（不猜）
V5_ANSWER_SOURCE_ALIASES: dict[str, str] = {
    "program": "tool_program",
    "direct_vlm": "direct_vlm_routed",
    "tool_program": "tool_program",
    "direct_vlm_routed": "direct_vlm_routed",
    "abstain": "abstain",
    "tool_contract": "tool_contract",
}
# 真正算"选了程序路径"的 answer_source
PROGRAM_ANSWER_SOURCES: frozenset[str] = frozenset({"tool_program"})

# §18.6 四个 `[待实验]` 项：未提供 PoC 证据 → 不得进论文主表
PENDING_POC_ITEMS: tuple[str, ...] = (
    "metric_scale_fusion",
    "moge2_effect",
    "track_consensus_counting_effect",
    "program_path_beats_direct_answer",
)


# ------------------------------------------------------------- 异常族 ----

class ProtocolError(RuntimeError):
    """违反 §18 实验协议的通用错误。"""


class IsolationViolation(ProtocolError):
    """三级隔离被违反（第二次 outer/final、final 触碰离线模型等，§18.1）。"""


class LedgerCorruptError(IsolationViolation):
    """run ledger 缺失/损坏 → 无法证明隔离状态（fail-closed）。"""


class AuditError(ProtocolError):
    """§19.4 split 访问审计违规（读 outer 不留理由、跨 split 改动无决策记录）。"""


class PairedABViolation(ProtocolError):
    """paired A/B 两臂不可比（§16.3/§18.4）。"""


class RedLineViolation(ProtocolError):
    """红线 9：用 outer 调出来的策略不得进论文（§22）。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Mapping 或对象统一取值（trace 记录两种形态都要能读）。"""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _canonical(value: Any) -> str:
    """把任意值压成稳定的比较串（大对象走 sha256，避免把 artifact 打进报告）。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:  # noqa: BLE001 - 非 pydantic 对象走 str
            value = str(value)
    if isinstance(value, Mapping):
        blob = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        blob = str(value)
    if len(blob) <= 120:
        return blob
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ============================================================ §18.1 三级隔离

class Stage(str, Enum):
    """三级隔离的 stage（§18.1）。取值与 §18.2 三档表一一对应。"""

    INNER = "inner"
    OUTER = "outer"
    FINAL = "final"


# stage ↔ 数据 split 名（`adapters/vsibench_loader.py` 的四层切分）
SPLIT_OF_STAGE: dict[Stage, str] = {
    Stage.INNER: "inner_validation",
    Stage.OUTER: "outer_holdout",
    Stage.FINAL: "final_test",
}
STAGE_OF_SPLIT: dict[str, Stage] = {
    "inner": Stage.INNER, "inner_validation": Stage.INNER, "dev": Stage.INNER,
    "outer": Stage.OUTER, "outer_holdout": Stage.OUTER, "holdout": Stage.OUTER,
    "final": Stage.FINAL, "final_test": Stage.FINAL, "blind": Stage.FINAL,
    # 在线 CLI 用 `--split test` 指 final_test（硬约束 9）
    "test": Stage.FINAL,
}

# 每个 stage 的**可执行操作**规则（§18.1/§16.2/§22 红线 9）
# - iterations: unlimited（inner 可反复）/ once（outer 只跑一次）/ once_blind（final 仅盲评一次）
# - allows_strategy_change: 只有 inner 可以在看到结果后改策略/阈值
# - final 不允许非确定性/外部痕迹：离线模型、Memory/Skill 写入、沙箱容器
STAGE_RULES: dict[Stage, dict] = {
    Stage.INNER: {
        "iterations": "unlimited",
        "allows_strategy_change": True,
        "allows_offline_model": True,
        "allows_memory_write": True,
        "allows_skill_write": True,
        "allows_sandbox_container": True,
        "requires_decision_note": False,
        "purpose": "调路由策略/阈值，看趋势",
    },
    Stage.OUTER: {
        "iterations": "once",
        "allows_strategy_change": False,      # Outer 失败即 reject，不允许基于 Outer 失败再修订
        "allows_offline_model": True,
        "allows_memory_write": True,
        "allows_skill_write": True,
        "allows_sandbox_container": True,
        "requires_decision_note": True,
        "purpose": "paired A/B 验证一次（程序 vs 直答）",
    },
    Stage.FINAL: {
        "iterations": "once_blind",
        "allows_strategy_change": False,
        "allows_offline_model": False,        # 不接触离线强模型（红线 1）
        "allows_memory_write": False,
        "allows_skill_write": False,
        "allows_sandbox_container": False,    # 不进沙箱容器
        "requires_decision_note": True,
        "purpose": "论文主表：预注册、paper-eligible、仅盲评一次",
    },
}

# 操作名 → STAGE_RULES 里的许可位（与 stage 规则一一对应，未知操作名 fail-closed）
_OPERATION_BITS: dict[str, str] = {
    "offline_model": "allows_offline_model",
    "memory_write": "allows_memory_write",
    "skill_write": "allows_skill_write",
    "sandbox_container": "allows_sandbox_container",
    "strategy_change": "allows_strategy_change",
    "threshold_change": "allows_strategy_change",
    "rerun": "iterations",
}


def coerce_stage(value: Any) -> Stage:
    """把 stage / split 名统一成 `Stage`（未知取值 raise，不猜）。"""
    if isinstance(value, Stage):
        return value
    key = str(value or "").strip().lower()
    if key in STAGE_OF_SPLIT:
        return STAGE_OF_SPLIT[key]
    raise ProtocolError(
        f"未知 stage/split {value!r}（取值域：inner|outer|final 及对应 split 名）")


def isolation_violations(stage: Any, operations: Sequence[str] = ()) -> list[str]:
    """列出在 `stage` 上做这些操作违反了哪些 §18.1 规则（空列表 = 合规）。

    `operations` 取值见 `_OPERATION_BITS`（offline_model / memory_write /
    skill_write / sandbox_container / strategy_change / threshold_change / rerun）。
    **未知操作名一律算违规**（fail-closed：不认识的操作不能默认放行）。
    """
    st = coerce_stage(stage)
    rules = STAGE_RULES[st]
    out: list[str] = []
    for op in operations:
        key = str(op)
        if key not in _OPERATION_BITS:
            out.append(f"{st.value}: 未知操作 {key!r} → 未声明许可位，按违规处理")
            continue
        bit = _OPERATION_BITS[key]
        if bit == "iterations":
            if rules["iterations"] != "unlimited":
                out.append(f"{st.value}: 不允许重跑（{rules['iterations']}，§18.1）")
        elif not rules.get(bit, False):
            out.append(f"{st.value}: 不允许 {key}（§18.1{'/红线 1' if key == 'offline_model' else ''}）")
    return out


def assert_isolation(stage: Any, operations: Sequence[str] = ()) -> None:
    """`isolation_violations` 的抛错版（有违规即 raise）。"""
    bad = isolation_violations(stage, operations)
    if bad:
        raise IsolationViolation("；".join(bad))


# --------------------------------------------------- §18.1 run ledger ----

@dataclass
class LedgerEntry:
    """run ledger 的一条记录（一次真实启动的 stage 运行）。"""

    stage: str
    run_id: str = ""
    timestamp: str = ""
    split: str = ""
    n_seeds: int = 0
    n_per_task: int = 0
    config_hash: str = ""
    override: bool = False
    note: str = ""
    actor: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage, "run_id": self.run_id, "timestamp": self.timestamp,
            "split": self.split, "n_seeds": int(self.n_seeds),
            "n_per_task": int(self.n_per_task), "config_hash": self.config_hash,
            "override": bool(self.override), "note": self.note, "actor": self.actor,
            "extra": dict(self.extra),
        }


@dataclass
class RunPermission:
    """一次 stage 运行的许可判定结果（`allowed=False` 时必须带原因）。"""

    allowed: bool
    stage: str
    code: str
    reason: str
    requires_override: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"allowed": bool(self.allowed), "stage": self.stage, "code": self.code,
                "reason": self.reason, "requires_override": bool(self.requires_override),
                "notes": list(self.notes)}


class RunLedger:
    """跨进程可查的 run ledger（§18.1：outer 只跑一次、final 仅盲评一次）。

    落盘形态（JSON）：

    ```json
    {"schema": "skill3d-experiment-run-ledger-v1", "created_at": "...",
     "entries": [...], "blocked_attempts": [...]}
    ```

    读取纪律（fail-closed）：

    - **ledger 缺失**：`final` 一律拒绝（`ledger_missing_fail_closed`）——
      "没记录"不等于"没跑过"，不允许拿缺失当许可；`inner`/`outer` 的首次运行允许
      并在记录时创建 ledger；
    - **ledger 损坏/结构非法**：所有 stage 一律拒绝（`ledger_corrupt`）；
    - **final 即使带 override 也拒绝**（§18.1：完全隔离、仅盲评一次）。
    """

    SCHEMA = "skill3d-experiment-run-ledger-v1"
    FILENAME = "experiment_run_ledger.json"

    def __init__(self, ledger_dir: str | Path) -> None:
        self.dir = Path(ledger_dir)
        self.path = self.dir / self.FILENAME

    # ------------------------------------------------------------ 读写 ----
    def exists(self) -> bool:
        return self.path.is_file()

    def initialize(self, *, actor: str = "", note: str = "",
                   campaign: str = "") -> dict:
        """显式初始化 ledger（实验开始时就建，使 `final` 的首次运行可被验证）。

        已存在则 raise —— 不允许"初始化"覆盖已有记录（那正是掩盖二次运行的手法）。
        """
        if self.exists():
            raise IsolationViolation(
                f"run ledger 已存在（{self.path}）：不得覆盖既有运行记录（§18.1）")
        data = {"schema": self.SCHEMA, "created_at": _now(), "campaign": campaign,
                "actor": actor, "note": note, "entries": [], "blocked_attempts": []}
        self._write(data)
        return data

    def _write(self, data: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> dict:
        """读 ledger；缺失 → `FileNotFoundError`，损坏/结构非法 → `LedgerCorruptError`。"""
        if not self.exists():
            raise FileNotFoundError(f"run ledger 不存在: {self.path}")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise LedgerCorruptError(
                f"run ledger 无法解析（{self.path}）：{exc} → fail-closed（§18.1）") from exc
        if (not isinstance(data, dict) or data.get("schema") != self.SCHEMA
                or not isinstance(data.get("entries"), list)):
            raise LedgerCorruptError(
                f"run ledger 结构非法（{self.path}）：缺 schema/entries → fail-closed（§18.1）")
        return data

    # ------------------------------------------------------------ 查询 ----
    def entries(self, stage: Any = None) -> list[dict]:
        """已记录的真实运行（可按 stage 过滤）；ledger 缺失返回空列表。"""
        if not self.exists():
            return []
        data = self.load()
        items = [e for e in data["entries"] if isinstance(e, dict)]
        if stage is None:
            return items
        st = coerce_stage(stage)
        return [e for e in items if str(e.get("stage", "")) == st.value]

    def count(self, stage: Any) -> int:
        return len(self.entries(stage))

    def blocked_attempts(self, stage: Any = None) -> list[dict]:
        if not self.exists():
            return []
        data = self.load()
        items = [e for e in (data.get("blocked_attempts") or []) if isinstance(e, dict)]
        if stage is None:
            return items
        st = coerce_stage(stage)
        return [e for e in items if str(e.get("stage", "")) == st.value]

    def summary(self) -> dict:
        """给审计/报告的汇总（不解析失败就 raise —— 损坏时不得给出乐观摘要）。"""
        counts = {st.value: 0 for st in Stage}
        overrides = 0
        blocked = 0
        for st in Stage:
            ents = self.entries(st)
            counts[st.value] = len(ents)
            overrides += sum(1 for e in ents if e.get("override"))
        if self.exists():
            blocked = len(self.blocked_attempts())
        return {
            "path": str(self.path), "exists": self.exists(), "counts": counts,
            "outer_rerun_override_used": bool(overrides),
            "final_runs": counts[Stage.FINAL.value],
            "blocked_attempts": blocked,
        }

    def check(self, stage: Any, *, override: bool = False,
              note: str = "") -> RunPermission:
        """判断该 stage 能不能再跑一次（不改动 ledger）。"""
        st = coerce_stage(stage)
        if not self.exists():
            if st is Stage.FINAL:
                return RunPermission(
                    False, st.value, "ledger_missing_fail_closed",
                    f"run ledger 缺失（{self.path}）：无法证明 final 尚未盲评过 → "
                    "fail-closed 拒绝（§18.1/红线 5）。请先在实验开始时 "
                    "`RunLedger(<dir>).initialize()` 建账。",
                    notes=["缺失 ≠ 未跑过；final 不得依赖'没记录'放行"])
            return RunPermission(
                True, st.value, "ledger_missing_first_run",
                f"run ledger 尚不存在：视为 {st.value} 首次运行（记录时创建 ledger）")
        try:
            self.load()
        except LedgerCorruptError as exc:
            return RunPermission(False, st.value, "ledger_corrupt",
                                 f"{exc} → 所有 stage 一律拒绝（fail-closed）")
        n = self.count(st)
        rules = STAGE_RULES[st]
        if rules["iterations"] == "unlimited":
            return RunPermission(True, st.value, "inner_iteration",
                                 f"inner 可反复迭代（已记录 {n} 次）")
        if rules["iterations"] == "once":
            if n == 0:
                return RunPermission(True, st.value, "outer_first_run",
                                     "outer 首次运行（只跑一次，§18.1）")
            if not override:
                return RunPermission(
                    False, st.value, "outer_already_run",
                    f"outer 已跑过 {n} 次：只允许一次；第二次必须显式 override 并留"
                    "决策记录（§18.1：Outer 失败即 reject，不允许基于 Outer 失败再修订）",
                    requires_override=True)
            if not str(note or "").strip():
                return RunPermission(
                    False, st.value, "outer_override_needs_note",
                    "outer 重跑 override 必须带非空决策记录（note）——"
                    "没有决策记录的重跑等于偷偷二次运行（§18.1/§19.4）",
                    requires_override=True)
            return RunPermission(
                True, st.value, "outer_rerun_override",
                f"outer 已被显式 override 重跑（第 {n + 1} 次），决策记录已留痕；"
                "该结果不得作为 paper-eligible 的验证依据（§18.1/红线 9）",
                requires_override=True,
                notes=["基于 outer 结果修订策略 = 违反红线 9"])
        if n == 0:
            return RunPermission(True, st.value, "final_first_run",
                                 "final 首次盲评（ledger 存在且可验证）")
        return RunPermission(
            False, st.value, "final_already_run",
            f"final 已跑过 {n} 次：完全隔离、仅盲评一次（§18.1/红线 5）→ 拒绝第二次"
            "（override 不适用）")

    # ------------------------------------------------------------ 记录 ----
    def record(self, stage: Any, run_id: str = "", *, override: bool = False,
               note: str = "", actor: str = "", n_seeds: int = 0,
               n_per_task: int = 0, config_hash: str = "",
               extra: Optional[Mapping] = None) -> LedgerEntry:
        """判定 + 落盘一次运行；不允许时抛 `IsolationViolation`（并留 blocked 留痕）。"""
        st = coerce_stage(stage)
        perm = self.check(st, override=override, note=note)
        if not perm.allowed:
            self._record_blocked(st, run_id, perm, override=override, note=note,
                                 actor=actor)
            raise IsolationViolation(f"{perm.reason} [{perm.code}]")
        data = self._load_or_create(st)
        entry = LedgerEntry(
            stage=st.value, run_id=str(run_id), timestamp=_now(),
            split=SPLIT_OF_STAGE[st], n_seeds=int(n_seeds),
            n_per_task=int(n_per_task), config_hash=str(config_hash),
            override=bool(override), note=str(note), actor=str(actor),
            extra=dict(extra or {}))
        data["entries"].append(entry.as_dict())
        self._write(data)
        return entry

    def begin_run(self, stage: Any, run_id: str = "", **kwargs) -> RunPermission:
        """`record` 的语义化别名：拿到许可（否则 raise）并落盘，返回本次授权记录。"""
        entry = self.record(stage, run_id, **kwargs)
        return RunPermission(True, entry.stage, "recorded",
                             f"已记录 {entry.stage} 运行（{entry.timestamp}，"
                             f"override={entry.override}）",
                             requires_override=bool(entry.override))

    def _load_or_create(self, st: Stage) -> dict:
        if self.exists():
            return self.load()
        self.dir.mkdir(parents=True, exist_ok=True)
        return {"schema": self.SCHEMA, "created_at": _now(), "campaign": "",
                "actor": "", "note": f"auto-created at first {st.value} run",
                "entries": [], "blocked_attempts": []}

    def _record_blocked(self, st: Stage, run_id: str, perm: RunPermission, *,
                        override: bool, note: str, actor: str) -> None:
        """被拒的尝试也留痕（审计价值最高的一类记录）；写失败不得掩盖隔离错误。"""
        if not self.exists():
            return
        try:
            data = self.load()
        except LedgerCorruptError:
            return
        data.setdefault("blocked_attempts", []).append({
            "stage": st.value, "run_id": str(run_id), "timestamp": _now(),
            "code": perm.code, "reason": perm.reason, "override": bool(override),
            "note": str(note), "actor": str(actor)})
        try:
            self._write(data)
        except OSError:
            return


# ==================================================== §18.2 样本量三档 ----

@dataclass(frozen=True)
class SampleTier:
    """§18.2 的一档样本量要求。"""

    stage: str
    n_per_task: int                 # 每题型题数（final 为 32+，此处记下界 32）
    n_seeds: int
    reported_weight: float          # 报告版式里的每题权重（分）
    reported_se_points: float        # 报告版式里的二值 SE（分）
    purpose: str

    @property
    def weight_per_question_max(self) -> float:
        """每题权重上界（100 / n_per_task 分）。"""
        return 100.0 / self.n_per_task

    @property
    def se_points_max(self) -> float:
        """二值 SE 上界（100 × 0.5/√n_per_task 分）。"""
        return binary_se_points(self.n_per_task)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage, "n_per_task": self.n_per_task,
            "n_seeds": self.n_seeds, "weight_per_question": self.weight_per_question_max,
            "binary_se_points": self.se_points_max,
            "reported_weight": self.reported_weight,
            "reported_se_points": self.reported_se_points,
            "purpose": self.purpose,
        }


# §18.2 样本量三档（不再 4 题/题型）
SAMPLE_TIERS: dict[Stage, SampleTier] = {
    Stage.INNER: SampleTier(stage="inner", n_per_task=16, n_seeds=3,
                            reported_weight=6.25, reported_se_points=12.5,
                            purpose="调路由策略/阈值，看趋势"),
    Stage.OUTER: SampleTier(stage="outer", n_per_task=32, n_seeds=3,
                            reported_weight=3.125, reported_se_points=8.8,
                            purpose="paired A/B 程序 vs 直答"),
    Stage.FINAL: SampleTier(stage="final", n_per_task=32, n_seeds=5,
                            reported_weight=3.125, reported_se_points=8.8,
                            purpose="预注册、paper-eligible（论文主表）"),
}


def binary_se(n: int) -> float:
    """逐题二值结果的 SE（§18.2）：`SE = 0.5/√N`（0–1 量纲）。"""
    n = int(n)
    if n < 1:
        raise ValueError("binary_se 需要 N ≥ 1")
    return 0.5 / math.sqrt(n)


def binary_se_points(n: int) -> float:
    """逐题二值结果的 SE（分数制，满分 100）：`100 × 0.5/√N`。"""
    return 100.0 * binary_se(n)


def weight_per_question(n: int) -> float:
    """每题权重（分）：`100/N`。"""
    n = int(n)
    if n < 1:
        raise ValueError("weight_per_question 需要 N ≥ 1")
    return 100.0 / n


def required_seeds_for_pool(stage: Any, pool_size: int) -> int:
    """§18.2：池不足时，用池内**全部题**、靠 seed 补足运行次数所需的 seed 数。

    运行次数目标 = 档位题数 × 档位 seed 数（如 outer：32 × 3 = 96）；池只有
    `pool_size` 题时，seed 数应至少 `ceil(目标 / pool_size)`。返回值不再低于档位
    自身要求（outer ≥3 / final ≥5）。
    """
    st = coerce_stage(stage)
    tier = SAMPLE_TIERS[st]
    pool = int(pool_size)
    if pool < 1:
        raise ValueError("pool_size 必须 ≥ 1")
    needed = math.ceil((tier.n_per_task * tier.n_seeds) / pool)
    floor = tier.n_seeds
    if st is Stage.FINAL:
        floor = max(floor, MIN_SEEDS_PAPER)
    return max(floor, needed)


@dataclass
class SampleSizeReport:
    """`check_sample_size` 的结论（`ok=False` 时 `shortfalls` 逐条说明原因）。"""

    stage: str
    ok: bool
    n_per_task: int
    n_seeds: int
    effective_n_per_task: int
    required_n_per_task: int
    required_n_seeds: int
    weight_per_question: Optional[float]
    binary_se: Optional[float]
    binary_se_points: Optional[float]
    pool_limited: bool = False
    true_pool_size: Optional[int] = None
    required_runs: int = 0
    planned_runs: int = 0
    shortfalls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage, "ok": self.ok, "n_per_task": self.n_per_task,
            "n_seeds": self.n_seeds, "effective_n_per_task": self.effective_n_per_task,
            "required_n_per_task": self.required_n_per_task,
            "required_n_seeds": self.required_n_seeds,
            "weight_per_question": self.weight_per_question,
            "binary_se": self.binary_se, "binary_se_points": self.binary_se_points,
            "pool_limited": self.pool_limited, "true_pool_size": self.true_pool_size,
            "required_runs": self.required_runs, "planned_runs": self.planned_runs,
            "shortfalls": list(self.shortfalls), "notes": list(self.notes),
        }


def check_sample_size(stage: Any, n_per_task: int, n_seeds: int, *,
                      pool_size: Optional[int] = None) -> SampleSizeReport:
    """§18.2 样本量校验：档位题数/seed 数 + "池 < 32 就用全池" 规则。

    - 池 ≥ 档位要求而 `n_per_task` 偏小 → 不许缩（池里明明有题）；
    - 池 < 档位要求 → **必须用池内全部题**（`n_per_task == pool_size`，不得子采样，
      尤其不得缩到 `FORBIDDEN_SUBSAMPLE_PER_TASK`=4），运行次数靠 increase seed 补足；
      报告里给出真实池大小与所需 seed 数，论文必须如实报告池大小。
    """
    st = coerce_stage(stage)
    tier = SAMPLE_TIERS[st]
    n_per_task = int(n_per_task)
    n_seeds = int(n_seeds)
    shortfalls: list[str] = []
    notes: list[str] = []
    pool_limited = False
    true_pool: Optional[int] = None
    eff_n = n_per_task
    required_seeds = tier.n_seeds

    if pool_size is not None:
        true_pool = int(pool_size)
        if true_pool < 1:
            shortfalls.append("pool_size < 1：题池为空，无法定义任何样本量结论")
        elif true_pool < tier.n_per_task:
            pool_limited = True
            eff_n = true_pool
            required_seeds = max(required_seeds, required_seeds_for_pool(st, true_pool))
            notes.append(
                f"§18.2 池不足：每题型真实池只有 {true_pool} 题（< {tier.n_per_task}）"
                f"→ 用池内全部题（不子采样），运行次数 {tier.n_per_task * tier.n_seeds} "
                f"次靠 {required_seeds} seed 补足；论文必须如实报告池大小 {true_pool}")
            if n_per_task > true_pool:
                shortfalls.append(
                    f"n_per_task={n_per_task} > 池大小 {true_pool}：池里没有这么多题"
                    "（不得重复计题）")
            elif n_per_task < true_pool:
                shortfalls.append(
                    f"题池只有 {true_pool} 题/题型（< {tier.n_per_task}）：必须用"
                    f"**池内全部题**，不得子采样到 {n_per_task}（§18.2）")
            if true_pool <= FORBIDDEN_SUBSAMPLE_PER_TASK:
                notes.append(
                    f"警告：池只有 {true_pool} 题/题型，接近 v5 的 "
                    f"{FORBIDDEN_SUBSAMPLE_PER_TASK} 题/题型（SE≈25 分）—— "
                    "此时任何 per-task 差异都不能当增益，只能报为探索性结果")
        else:
            if n_per_task < tier.n_per_task:
                shortfalls.append(
                    f"{st.value} 要求每题型 ≥{tier.n_per_task} 题；当前 {n_per_task}"
                    f"（池内有 {true_pool} 题，足够）→ 不得缩减题数（§18.2）")
            if n_per_task < FORBIDDEN_SUBSAMPLE_PER_TASK + 1:
                shortfalls.append(
                    f"n_per_task={n_per_task} ≤ {FORBIDDEN_SUBSAMPLE_PER_TASK}："
                    "v5 的 4 题/题型（一题 25 分、SE 25 分）已被 §18.2 明确废止")
    elif n_per_task < tier.n_per_task:
        shortfalls.append(
            f"{st.value} 要求每题型 ≥{tier.n_per_task} 题；当前 {n_per_task}"
            f"（每题 {weight_per_question(max(n_per_task, 1)):.2f} 分）")

    if n_seeds < required_seeds:
        shortfalls.append(
            f"{st.value} 要求 ≥{required_seeds} seed（§18.2 档位"
            f"{'，含 paper-eligible ≥5' if st is Stage.FINAL else ''}）；当前 {n_seeds}")

    eff = eff_n if eff_n > 0 else 0
    se = binary_se(eff) if eff > 0 else None
    return SampleSizeReport(
        stage=st.value, ok=not shortfalls, n_per_task=n_per_task, n_seeds=n_seeds,
        effective_n_per_task=eff, required_n_per_task=tier.n_per_task,
        required_n_seeds=required_seeds,
        weight_per_question=(weight_per_question(eff) if eff > 0 else None),
        binary_se=se, binary_se_points=(None if se is None else 100.0 * se),
        pool_limited=pool_limited, true_pool_size=true_pool,
        required_runs=tier.n_per_task * tier.n_seeds,
        planned_runs=eff * n_seeds, shortfalls=shortfalls, notes=notes)


def format_sample_tiers() -> str:
    """§18.2 三档表的文本版（报告/README 直接可用）。"""
    lines = [f"{'stage':8s}{'题/题型':>8s}{'seeds':>7s}{'每题权重':>10s}"
             f"{'二值SE':>9s}  用途"]
    for st in (Stage.INNER, Stage.OUTER, Stage.FINAL):
        t = SAMPLE_TIERS[st]
        lines.append(f"{t.stage:8s}{t.n_per_task:>8d}{t.n_seeds:>7d}"
                     f"{t.weight_per_question_max:>10.3f}{t.se_points_max:>9.2f}"
                     f"  {t.purpose}")
    lines.append("SE = 0.5/√N；若 benchmark 每题型池 < 32：用池内全部题（不子采样到 4），"
                 "不足部分靠增加 seed 补足，论文如实报告池大小（§18.2）")
    return "\n".join(lines)


# ==================================================== §18.3 噪声底协议 ----

@dataclass
class NoiseFloorReport:
    """§18.3 噪声底：seed 门槛 + 逐题一致率 + per-task 波动。"""

    n_seeds: int
    ok_noise_floor: bool
    ok_paper: bool
    agreement: dict = field(default_factory=dict)
    per_task_fluctuation: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def per_question_agreement_rate(self) -> Optional[float]:
        return self.agreement.get("agreement_rate")

    def as_dict(self) -> dict:
        return {"n_seeds": self.n_seeds, "ok_noise_floor": self.ok_noise_floor,
                "ok_paper": self.ok_paper, "agreement": self.agreement,
                "per_task_fluctuation": self.per_task_fluctuation,
                "per_question_agreement_rate": self.per_question_agreement_rate,
                "notes": list(self.notes)}


def check_round_robin_usage(round_robin_used: bool, purpose: str) -> tuple[bool, str]:
    """§18.3：多副本轮询只准用于吞吐，不得用于任何对比实验。

    返回 `(ok, reason)`；`ok=False` 时 reason 说明违规点。
    """
    if not round_robin_used:
        return True, "未使用多副本轮询"
    key = str(purpose or "").strip().lower()
    if key in THROUGHPUT_ONLY_PURPOSES:
        return True, f"多副本轮询仅用于吞吐（purpose={key}）"
    if not key:
        return False, ("使用了多副本轮询但未声明 purpose → fail-closed："
                       "无法证明它只用于吞吐（§18.3）")
    return False, (f"多副本轮询被用于 {key!r}：只允许用于吞吐"
                   f"（{sorted(THROUGHPUT_ONLY_PURPOSES)}），不得用于任何对比实验（§18.3）")


def assert_round_robin_not_for_comparison(round_robin_used: bool, purpose: str) -> None:
    """`check_round_robin_usage` 的抛错版。"""
    ok, reason = check_round_robin_usage(round_robin_used, purpose)
    if not ok:
        raise ProtocolError(reason)


def noise_floor_report(per_question: Mapping[str, Sequence[Optional[bool]]], *,
                       per_task_scores: Optional[Mapping[str, Sequence[float]]] = None,
                       n_seeds: Optional[int] = None,
                       round_robin_used: bool = False,
                       round_robin_purpose: str = "") -> NoiseFloorReport:
    """§18.3 噪声底报告：同配置重复 ≥3 seed（paper ≥5）下的稳定度。

    - `per_question[qa_id]` = 各 seed 在该题上的二值结果（`None` = 该 seed 未产出）；
    - `per_task_scores[task]` = 各 seed 在该题型上的分数（0–1 或 0–100 均可，只做
      相对波动比较）；
    - 多副本轮询若被声明，会在此校验其用途（不得用于对比）。
    """
    notes: list[str] = []
    if n_seeds is None:
        implied = [len([v for v in (vals or []) if v is not None])
                   for vals in per_question.values()] or [0]
        n_seeds = int(max(implied))
        notes.append(f"未显式给 n_seeds：由逐题取值推断为 {n_seeds}（请显式传入）")
    agreement = per_question_agreement(per_question)
    fluctuation = seed_fluctuation(per_task_scores or {})
    ok_floor = int(n_seeds) >= MIN_SEEDS_NOISE_FLOOR
    ok_paper = int(n_seeds) >= MIN_SEEDS_PAPER
    if not ok_floor:
        notes.append(f"仅 {n_seeds} seed < {MIN_SEEDS_NOISE_FLOOR}：不满足 §18.3 噪声底，"
                     "任何差异都不能当增益")
    elif not ok_paper:
        notes.append(f"{n_seeds} seed 满足噪声底（≥{MIN_SEEDS_NOISE_FLOOR}）但不足 "
                     f"paper-eligible（≥{MIN_SEEDS_PAPER}，§18.6）")
    notes.extend(agreement.get("notes", []))
    ok_rr, rr_reason = check_round_robin_usage(round_robin_used, round_robin_purpose)
    if not ok_rr:
        notes.append(rr_reason)
    return NoiseFloorReport(n_seeds=int(n_seeds), ok_noise_floor=bool(ok_floor),
                            ok_paper=bool(ok_paper and ok_rr), agreement=agreement,
                            per_task_fluctuation=fluctuation, notes=notes)


# ==================================================== §18.4 paired A/B ----

# 两臂必须一致的字段（§16.3/§18.4）；值为该字段的候选键名（容忍不同落盘形态）
PAIRED_AB_FIELDS: dict[str, tuple[str, ...]] = {
    "frame_set_hash": ("frame_set_hash",),
    "recon_artifact": ("recon_artifact_hash", "reconstruction_hash",
                       "reconstruction_artifact", "artifact_ref"),
    "model": ("model", "vllm_model", "synthesizer_model"),
    "template_version": ("template_version", "prompt_version"),
    "hardware": ("hardware", "docker_digest", "gpu", "inference_env"),
    "evidence_signature": ("evidence_signature", "evidence_states", "evidence_profile"),
}


@dataclass
class PairedABCheck:
    """两臂可比性的判定结果（`compatible=False` 时 `mismatches` 逐条列出）。"""

    compatible: bool
    shared: dict = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"compatible": self.compatible, "shared": self.shared,
                "mismatches": list(self.mismatches), "missing": list(self.missing)}


def _paired_ab_value(arm: Any, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = _field(arm, key, None)
        if value is None:
            continue
        if key == "evidence_profile" and isinstance(value, Mapping) and "capabilities" not in value:
            # §5.4 画像 dict → 只取 8 项能力，避免 version/subvalues 干扰签名比较
            only = {c: value[c] for c in CAPABILITIES if c in value}
            value = only or value
        canon = _canonical(value)
        if canon != "":
            return canon
    return None


def check_paired_ab(arm_a: Any, arm_b: Any) -> PairedABCheck:
    """§16.3/§18.4 硬约束：两臂必须**同 frames / 同输入 / 同模型 / 同模板版本 /
    同硬件 / 同 EvidenceProfile**，且使用**完全相同的重建产物**。

    任一项缺失或不同 → `compatible=False`（缺失即不可比，fail-closed：无法证明
    相同就不得当作相同的 A/B）。
    """
    shared: dict = {}
    mismatches: list[str] = []
    missing: list[str] = []
    for field_name, keys in PAIRED_AB_FIELDS.items():
        va = _paired_ab_value(arm_a, keys)
        vb = _paired_ab_value(arm_b, keys)
        if va is None or vb is None:
            missing.append(field_name)
            mismatches.append(
                f"{field_name}: 两臂未同时提供（{'A 缺失' if va is None else ''}"
                f"{'/' if va is None and vb is None else ''}"
                f"{'B 缺失' if vb is None else ''}）→ 无法证明相同（fail-closed）")
            continue
        if va != vb:
            mismatches.append(f"{field_name}: A={va!r} ≠ B={vb!r}（§16.3 要求完全相同）")
        shared[field_name] = va
    return PairedABCheck(compatible=not mismatches, shared=shared,
                         mismatches=mismatches, missing=missing)


def assert_paired_ab(arm_a: Any, arm_b: Any) -> PairedABCheck:
    """`check_paired_ab` 的抛错版（不可比即拒绝，不允许"先跑了再说"）。"""
    chk = check_paired_ab(arm_a, arm_b)
    if not chk.compatible:
        raise PairedABViolation("paired A/B 不可比 → " + "；".join(chk.mismatches))
    return chk


def paired_statistics(program: Sequence[Optional[bool]],
                      direct: Sequence[Optional[bool]], *,
                      n_resamples: int = N_BOOTSTRAP, seed: int = 0,
                      alpha: float = ALPHA, n_comparisons: int = 1,
                      primary: str = "mcnemar") -> dict:
    """§18.4 paired A/B 统计：McNemar（或 paired BCa bootstrap）+ 效应量 + Bonferroni。

    - `program` / `direct`：两臂**逐题二值**结果，按 qa 顺序对齐（`None` 成对丢弃）；
    - 退化样本（完全相同组 / 差值零方差）→ 所有 p 归一化为 `None`，
      `degenerate_note` 写 "退化样本 → 判为不显著"，`significant` 恒 False；
    - `primary="bootstrap"` 时以 BCa CI 下界 > 0 判显著（McNemar p 仍然报出）。
    """
    if primary not in ("mcnemar", "bootstrap"):
        raise ProtocolError(f"未知 primary={primary!r}（取值域：mcnemar|bootstrap）")
    if len(program) != len(direct):
        raise ProtocolError(
            f"paired A/B 两臂长度不一致：{len(program)} vs {len(direct)}")
    pairs = [(pv, dv) for pv, dv in zip(program, direct)
             if not _is_missing(pv) and not _is_missing(dv)]
    p_arr = np.asarray([_to_binary(pv) for pv, _ in pairs], dtype=float)
    d_arr = np.asarray([_to_binary(dv) for _, dv in pairs], dtype=float)
    n = int(p_arr.shape[0])
    n_dropped = len(program) - n
    mc = mcnemar_test(d_arr, p_arr)          # control=direct, treatment=program
    if n:
        diffs = p_arr - d_arr
        degenerate = bool(np.all(diffs == 0.0))
        zero_variance = bool(np.all(diffs == diffs[0]))
    else:
        degenerate = True
        zero_variance = True
    boot = paired_bootstrap_ci_detailed(d_arr, p_arr, n_resamples=n_resamples,
                                        seed=seed)
    lo = boot["lo"] if math.isfinite(boot["lo"]) else None
    hi = boot["hi"] if math.isfinite(boot["hi"]) else None
    delta = cliffs_delta(d_arr, p_arr)
    cohen = cohens_d_paired(d_arr, p_arr)
    notes: list[str] = []
    if degenerate:
        notes.append(DEGENERATE_NOTE)
    if zero_variance and not degenerate:
        notes.append("差值零方差（常数差）→ bootstrap 退化，仅 McNemar 精确检验有效")
    if mc.get("p") is None and not degenerate:
        notes.append(mc.get("note") or "McNemar p 不可用 → 判为不显著")
    m = max(int(n_comparisons), 1)
    p_bonf = _bonferroni(mc.get("p"), m)
    alpha_corrected = float(alpha) / m
    if degenerate:
        significant = False
    elif primary == "bootstrap":
        significant = bool(lo is not None and lo > 0 and not zero_variance)
    else:
        significant = bool(p_bonf is not None and p_bonf <= alpha
                           and mc["b"] > mc["c"])
    direction = ("program" if mc["b"] > mc["c"]
                 else ("direct" if mc["c"] > mc["b"] else "tie"))
    if direction == "direct":
        notes.append("方向为直答更优 → 不得表述为程序路径增益（§16.4：直答不算工具贡献）")
    return {
        "n_pairs": n,
        "n_program_win": mc["b"],
        "n_direct_win": mc["c"],
        "n_tie": mc["n_tie"],
        "n_dropped": int(n_dropped),
        "degenerate": degenerate,
        "zero_variance": zero_variance,
        "degenerate_note": (DEGENERATE_NOTE if degenerate else ""),
        "primary_test": primary,
        "mcnemar_p": normalize_p(mc.get("p")),
        "mcnemar_statistic": mc.get("statistic"),
        "mcnemar_method": mc.get("method"),
        "p_bonferroni": p_bonf,
        "n_comparisons": m,
        "alpha": float(alpha),
        "alpha_corrected": alpha_corrected,
        "significant": significant,
        "direction": direction,
        "bootstrap_ci_lo": lo,
        "bootstrap_ci_hi": hi,
        "bootstrap_method": boot["method"],
        "bootstrap_degenerate": bool(boot["degenerate"]),
        "cliffs_delta": (None if delta is None or not math.isfinite(delta)
                         else float(delta)),
        "cohens_d": (None if cohen is None or not math.isfinite(cohen)
                     else float(cohen)),
        "notes": notes,
    }


def paired_statistics_by_task(program: Sequence[Optional[bool]],
                              direct: Sequence[Optional[bool]],
                              task_types: Sequence[str], **kwargs) -> dict:
    """按题型切片的 §18.4 统计（Bonferroni 校正按**切片数**）。"""
    if len(task_types) != len(program) or len(program) != len(direct):
        raise ProtocolError("program/direct/task_types 长度必须一致")
    tasks = sorted(set(str(t) for t in task_types))
    out: dict = {"n_comparisons": max(len(tasks), 1), "by_task": {}}
    for task in tasks:
        idx = [i for i, t in enumerate(task_types) if str(t) == task]
        out["by_task"][task] = paired_statistics(
            [program[i] for i in idx], [direct[i] for i in idx],
            n_comparisons=max(len(tasks), 1), **kwargs)
    out["overall"] = paired_statistics(program, direct, **kwargs)
    return out


def _is_missing(value: Any) -> bool:
    """成对丢弃判定：`None` 或 `nan`（缺结果），其余视为有效取值。"""
    if value is None:
        return True
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _to_binary(value: Any) -> float:
    """二值归一：bool/0/1 → 0.0/1.0；其他取值 raise（不静默取整）。"""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    f = float(value)
    if f not in (0.0, 1.0):
        raise ProtocolError(f"paired A/B 只接受二值逐题结果，收到 {value!r}")
    return f


# ======================================================== §18.5 报告要求 ----

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% CI（§18.5：每题型不只报点估计）。

    `z=1.96` 对应 95%（正态近似分位点；小样本下 Wilson 比 Wald 稳，n=0 时无定义 →
    raise，不返回"0 宽度"的假 CI）。
    """
    n = int(n)
    if n < 1:
        raise ValueError("wilson_ci 需要 n ≥ 1（分母为 0 时 CI 无定义）")
    k = int(k)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def ci95_of_seed_values(values: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    """多 seed 均值的 95% CI（t 近似用 1.96 保守替代；n<2 → `(None, None)`）。"""
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if len(vals) < 2:
        return None, None
    arr = np.asarray(vals, dtype=float)
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)


def per_task_ci95(per_task_correct: Mapping[str, Sequence[Optional[bool]]]) -> dict:
    """§18.5：每题型报 95% CI + 样本量 + 二值 SE（不只点估计）。"""
    out: dict[str, dict] = {}
    for task, values in (per_task_correct or {}).items():
        vals = [bool(v) for v in (values or []) if v is not None]
        n = len(vals)
        k = sum(1 for v in vals if v)
        if n == 0:
            out[str(task)] = {"n": 0, "n_correct": 0, "accuracy": None,
                              "ci95": None, "binary_se": None, "se_points": None,
                              "note": "无有效样本 → CI 不可算（不臆造）"}
            continue
        acc = k / n
        lo, hi = wilson_ci(k, n)
        se = binary_se(n)
        out[str(task)] = {"n": n, "n_correct": k, "accuracy": acc, "ci95": [lo, hi],
                          "ci_width": hi - lo, "binary_se": se,
                          "se_points": 100.0 * se}
    return out


def normalize_synthesis_source(value: Any) -> str:
    """synthesis_source → v6 归一（§19.3 六类 + `mock_stub`）。

    v5 名（`vllm` / `deterministic_stub` / `none`）只作**历史回读**；不认识的取值
    原样返回并单独成桶 —— 绝不静默归到 `vllm_ok`（那会把 mock/未知来源洗成"模型
    正常产出"）。
    """
    v = str(value or "").strip()
    if v in V5_SYNTHESIS_SOURCE_ALIASES:
        return V5_SYNTHESIS_SOURCE_ALIASES[v]
    return v


def normalize_answer_source(value: Any) -> str:
    """answer_source → v6 四值（§5.3）；v5 名（program/direct_vlm）作历史回读。"""
    v = str(value or "").strip()
    return V5_ANSWER_SOURCE_ALIASES.get(v, v)


def evidence_states_of(record: Any) -> dict[str, str]:
    """从 trace 记录取 8 项能力的三值快照（`EpisodeTrace.evidence_states` 优先，
    退化到 `TraceRecord.evidence_profile`）；取不到返回 `{}`（不猜）。"""
    states = _field(record, "evidence_states", None)
    if isinstance(states, Mapping) and states:
        return {str(k): str(v) for k, v in states.items()}
    profile = _field(record, "evidence_profile", None)
    if profile is not None:
        if isinstance(profile, Mapping):
            data = profile
        else:
            data = {c: _field(profile, c, None) for c in CAPABILITIES}
        only = {c: str(data[c]) for c in CAPABILITIES
                if c in data and data[c] is not None}
        if only:
            return only
    return {}


# 无证据快照的桶名（fail-closed：不得当作"全 available"混进分组）
MISSING_SIGNATURE = "(no-evidence-states)"


def signature_key(states: Mapping[str, str]) -> str:
    """证据签名 → 稳定分组键（按 `CAPABILITIES` 顺序；缺失能力记 `missing`）。"""
    if not states:
        return MISSING_SIGNATURE
    return "|".join(f"{c}={states.get(c, 'missing')}" for c in CAPABILITIES)


@dataclass
class TraceView:
    """trace 记录的统一视图（`EpisodeTrace` / `TraceRecord` / dict 都归一到这里）。"""

    qa_id: str
    evidence_states: dict = field(default_factory=dict)
    answer_source: str = ""
    synthesis_source: str = ""
    abstained: bool = False
    final_state: str = ""
    scene_route: str = ""
    question_tool_scope: str = ""
    used_result_ids: list = field(default_factory=list)
    recovery_count: int = 0
    partial_tool_recovery: bool = False
    failure_code: Optional[str] = None
    frame_set_hash: str = ""

    @property
    def signature(self) -> str:
        return signature_key(self.evidence_states)

    @property
    def selected_program(self) -> bool:
        """是否真的选了程序路径（`tool_program`；direct/abstain 均不算）。"""
        return self.answer_source in PROGRAM_ANSWER_SOURCES

    @property
    def covered(self) -> Optional[bool]:
        """该题是否有可用 Skill/Tool 覆盖（§16.5 覆盖率）。

        优先看 `question_tool_scope`（v6 逐题授权后的工具域）；退化到
        `answer_source != abstain`（§16.6：无可用 Skill 才 abstain）。两者都缺 →
        `None`（不可算，不进分母，绝不默认算作"已覆盖"）。
        """
        scope = str(self.question_tool_scope or "").strip()
        if scope:
            return scope not in ("none", "no_tool", "empty", "-")
        src = str(self.answer_source or "").strip()
        if src:
            return src != "abstain"
        return None


def trace_view(record: Any) -> TraceView:
    """把 `EpisodeTrace` / `TraceRecord` / dict 归一为 `TraceView`（v5 取值同时回读）。"""
    qa_id = str(_field(record, "qa_id", "") or _field(record, "episode_id", "") or "")
    raw_scope = _field(record, "question_tool_scope", "")
    if isinstance(raw_scope, (list, tuple, set)):
        scope = ",".join(sorted(str(x) for x in raw_scope))
    else:
        scope = str(raw_scope or "")
    return TraceView(
        qa_id=qa_id,
        evidence_states=evidence_states_of(record),
        answer_source=normalize_answer_source(_field(record, "answer_source", "")),
        synthesis_source=normalize_synthesis_source(_field(record, "synthesis_source", "")),
        abstained=bool(_field(record, "abstained", False)),
        final_state=str(_field(record, "final_state", "") or ""),
        scene_route=str(_field(record, "scene_route", "") or ""),
        question_tool_scope=scope,
        used_result_ids=list(_field(record, "used_result_ids", []) or []),
        recovery_count=int(_field(record, "recovery_count", 0) or 0),
        partial_tool_recovery=bool(_field(record, "partial_tool_recovery", False)),
        failure_code=_field(record, "failure_code", None),
        frame_set_hash=str(_field(record, "frame_set_hash", "") or ""),
    )


@dataclass
class RateGroup:
    """一个分组（按证据签名 / 题型 / Skill 路径）的选择率·覆盖率·胜负。"""

    key: str
    n: int = 0
    n_selected: int = 0
    selection_rate: Optional[float] = None
    n_covered: int = 0
    coverage: Optional[float] = None
    n_coverage_unknown: int = 0
    n_abstain: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    n_paired: int = 0
    win_rate: Optional[float] = None
    answer_source_counts: dict = field(default_factory=dict)
    synthesis_source_counts: dict = field(default_factory=dict)
    failure_code_counts: dict = field(default_factory=dict)
    evidence_states: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "key": self.key, "n": self.n, "n_selected": self.n_selected,
            "selection_rate": self.selection_rate, "n_covered": self.n_covered,
            "coverage": self.coverage, "n_coverage_unknown": self.n_coverage_unknown,
            "n_abstain": self.n_abstain, "wins": self.wins, "losses": self.losses,
            "ties": self.ties, "n_paired": self.n_paired, "win_rate": self.win_rate,
            "answer_source_counts": dict(self.answer_source_counts),
            "synthesis_source_counts": dict(self.synthesis_source_counts),
            "failure_code_counts": dict(self.failure_code_counts),
            "evidence_states": dict(self.evidence_states),
        }


def _bump(counter: dict, key: Any) -> None:
    k = str(key) if key not in (None, "") else "(none)"
    counter[k] = counter.get(k, 0) + 1


def _fill_rates(group: RateGroup) -> None:
    """填 选择率 / 覆盖率（分母为 0 → None，不臆造）。"""
    group.selection_rate = (group.n_selected / group.n) if group.n else None
    known = group.n - group.n_coverage_unknown
    group.coverage = (group.n_covered / known) if known else None


def _rate_by_dimension(records: Sequence[Any], key_of, *,
                       correct_by_qa: Optional[Mapping[str, Any]] = None,
                       direct_correct_by_qa: Optional[Mapping[str, Any]] = None,
                       view_cache: Optional[dict] = None) -> dict:
    """按维度分组算选择率 / 覆盖率 / 胜负（§16.5/§18.5 的报告内核）。"""
    groups: dict[str, RateGroup] = {}
    cache = view_cache if view_cache is not None else {}
    for rec in records:
        view = cache.get(id(rec))
        if view is None:
            view = trace_view(rec)
            cache[id(rec)] = view
        key = str(key_of(view) or "(unknown)")
        g = groups.get(key)
        if g is None:
            g = RateGroup(key=key, evidence_states=dict(view.evidence_states))
            groups[key] = g
        g.n += 1
        if view.selected_program:
            g.n_selected += 1
        covered = view.covered
        if covered is None:
            g.n_coverage_unknown += 1
        elif covered:
            g.n_covered += 1
        if view.abstained or view.answer_source == "abstain":
            g.n_abstain += 1
        _bump(g.answer_source_counts, view.answer_source or "(none)")
        _bump(g.synthesis_source_counts, view.synthesis_source or "(none)")
        if view.failure_code:
            _bump(g.failure_code_counts, view.failure_code)
        if correct_by_qa is not None and direct_correct_by_qa is not None:
            qa = view.qa_id
            pc = correct_by_qa.get(qa) if qa else None
            dc = direct_correct_by_qa.get(qa) if qa else None
            if pc is not None and dc is not None:
                g.n_paired += 1
                if bool(pc) and not bool(dc):
                    g.wins += 1
                elif bool(dc) and not bool(pc):
                    g.losses += 1
                else:
                    g.ties += 1
    for g in groups.values():
        _fill_rates(g)
        decided = g.wins + g.losses
        if g.n_paired == 0:
            g.win_rate = None
        elif decided == 0:
            g.win_rate = None            # 无判别对 → 胜负率没有定义（不臆造 0/1）
        else:
            g.win_rate = g.wins / decided
    return {k: v.as_dict() for k, v in sorted(groups.items())}


def evidence_state_breakdown(records: Sequence[Any], *,
                             correct_by_qa: Optional[Mapping[str, Any]] = None,
                             direct_correct_by_qa: Optional[Mapping[str, Any]] = None,
                             task_by_qa: Optional[Mapping[str, str]] = None,
                             skill_path_by_qa: Optional[Mapping[str, str]] = None,
                             ) -> dict:
    """§18.5/§16.5：按 **EvidenceProfile 状态**分组报选择率 / 覆盖率 / 胜负（论文
    "程序路径为什么、在什么条件下赢过直答"的核心证据）。

    输入为真实 `EpisodeTrace` / `TraceRecord`（或同构 dict）；`qa_id` 上的正确性、
    题型、Skill 路径由调用方从评测产物映射进来（trace 本身不含正确性）。

    返回：
    - `by_evidence_state`：按 8 项能力三值签名分组（含样本量、选择率、覆盖率、
      胜负、answer_source / synthesis_source / failure_code 分布）；
    - `by_task` / `by_skill_path`：同样的三个率换行分组（Skill 路径未显式给出时退化
      为 `question_tool_scope`，并在 notes 里说明是退化口径）；
    - `notes`：不可算的口径逐条说明（分母 0 / 缺正确性 / 缺证据快照）。
    """
    records = list(records or [])
    cache: dict[int, TraceView] = {}
    notes: list[str] = []
    by_state = _rate_by_dimension(
        records, lambda v: v.signature, correct_by_qa=correct_by_qa,
        direct_correct_by_qa=direct_correct_by_qa, view_cache=cache)
    by_task: dict = {}
    if task_by_qa:
        by_task = _rate_by_dimension(
            records, lambda v: task_by_qa.get(v.qa_id, "(unknown-task)"),
            correct_by_qa=correct_by_qa, direct_correct_by_qa=direct_correct_by_qa,
            view_cache=cache)
    skill_path: dict = {}
    if skill_path_by_qa:
        skill_path = _rate_by_dimension(
            records, lambda v: skill_path_by_qa.get(v.qa_id, "(unknown-skill-path)"),
            correct_by_qa=correct_by_qa, direct_correct_by_qa=direct_correct_by_qa,
            view_cache=cache)
        notes.append("by_skill_path 来自调用方提供的 Skill 路径映射")
    else:
        skill_path = _rate_by_dimension(
            records, lambda v: v.question_tool_scope or "(no-scope)",
            correct_by_qa=correct_by_qa, direct_correct_by_qa=direct_correct_by_qa,
            view_cache=cache)
        notes.append("by_skill_path 未提供显式 Skill 路径映射 → 退化为 "
                     "question_tool_scope（口径已在报告中标明，不得当作 Skill 级归因）")
    n_missing = sum(1 for v in cache.values() if not v.evidence_states)
    if n_missing:
        notes.append(f"{n_missing}/{len(records)} 条 trace 没有 evidence_states 快照 "
                     f"→ 归入 {MISSING_SIGNATURE} 桶（不得当作'全 available'）")
    if correct_by_qa is None or direct_correct_by_qa is None:
        notes.append("未提供两臂逐题正确性 → 胜负率一律 None（不臆造）；"
                     "选择率/覆盖率不受影响")
    return {"n_records": len(records), "by_evidence_state": by_state,
            "by_task": by_task, "by_skill_path": skill_path, "notes": notes}


# ================================================== §18.6 paper_eligible ----

@dataclass
class RequirementCheck:
    """单项要求（§18.6 四要件之一 / 四个 `[待实验]` 项之一）的判定。"""

    name: str
    ok: bool
    reason: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "reason": self.reason,
                "detail": dict(self.detail)}


@dataclass
class PaperEligibility:
    """paper_eligible 的完整判定（`eligible=False` 时 `reasons` 逐条说明）。"""

    eligible: bool
    checks: dict = field(default_factory=dict)
    pending_poc: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "eligible": self.eligible,
            "checks": {k: v.as_dict() for k, v in self.checks.items()},
            "pending_poc": {k: v.as_dict() for k, v in self.pending_poc.items()},
            "reasons": list(self.reasons),
        }


def check_data_isolation(scene_ids_by_split: Optional[Mapping[str, Sequence[str]]]
                         ) -> RequirementCheck:
    """要件 1：数据隔离（scene 级互斥，红线 6）。"""
    if not scene_ids_by_split:
        return RequirementCheck(
            "data_isolation", False,
            "未提供 scene 级 split 划分 → 无法证明互斥（数据隔离要件不满足）")
    sets: dict[str, set] = {}
    for split, ids in scene_ids_by_split.items():
        vals = {str(s) for s in (ids or []) if str(s).strip()}
        if not vals:
            return RequirementCheck(
                "data_isolation", False, f"split {split!r} 的 scene 列表为空 → 不可判定")
        sets[str(split)] = vals
    if len(sets) < 2:
        return RequirementCheck(
            "data_isolation", False,
            f"只提供了 {sorted(sets)} 一个 split → 无法证明互斥")
    overlaps: list[str] = []
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(sets[a] & sets[b])
            if shared:
                overlaps.append(f"{a}∩{b}={shared[:5]}{'…' if len(shared) > 5 else ''}"
                                f"(共{len(shared)})")
    if overlaps:
        return RequirementCheck("data_isolation", False,
                                "scene 级不互斥：" + "；".join(overlaps),
                                {"overlaps": overlaps})
    return RequirementCheck(
        "data_isolation", True,
        f"scene 级互斥 OK（{ {k: len(v) for k, v in sets.items()} }）",
        {"splits": {k: len(v) for k, v in sets.items()}})


def check_statistical_gate(stat_gate: Any) -> RequirementCheck:
    """要件 3：统计门（§18.4 的检验结论必须是"显著且非退化"）。"""
    if stat_gate is None:
        return RequirementCheck("statistical_gate", False,
                                "未提供统计检验结果 → 不得声称增益（§18.5）")
    if isinstance(stat_gate, bool):
        return RequirementCheck(
            "statistical_gate", bool(stat_gate),
            "统计门显式给定为 " + ("True（需附 §18.4 检验产物以便复核）"
                                   if stat_gate else "False"))
    if isinstance(stat_gate, Mapping):
        degenerate = bool(stat_gate.get("degenerate", False))
        sig = bool(stat_gate.get("significant", False))
        p = stat_gate.get("p_bonferroni", stat_gate.get("mcnemar_p"))
        if degenerate:
            return RequirementCheck(
                "statistical_gate", False,
                "统计检验为退化样本 → 判为不显著，不得进论文主表（§18.4）",
                {"degenerate": True})
        if not sig:
            return RequirementCheck(
                "statistical_gate", False,
                f"统计门未通过（significant=False, p={p!r}）→ 不报未检验的增益（§18.5）",
                {"p": p})
        if p is None:
            return RequirementCheck(
                "statistical_gate", False,
                "significant=True 但 p 值为 None（退化样本）→ 拒绝（不得凭空声称显著）")
        return RequirementCheck("statistical_gate", True,
                                f"统计门通过（p_bonferroni={p!r}，§18.4）",
                                {"p": p, "direction": stat_gate.get("direction")})
    return RequirementCheck("statistical_gate", False,
                            f"统计门结果类型不可识别：{type(stat_gate).__name__}")


def check_non_mock(modes: Sequence[str], records: Sequence[Any] = ()) -> RequirementCheck:
    """要件 4：非 mock 证据（HC24：mock/合成不得进主表）。"""
    ms = [str(m or "") for m in (modes or [])]
    mock_qa: list[str] = []
    for rec in (records or []):
        view = trace_view(rec)
        if view.synthesis_source in MOCK_SYNTHESIS_SOURCES:
            mock_qa.append(view.qa_id or "(unknown)")
    if not ms:
        return RequirementCheck("non_mock_evidence", False,
                                "未提供 run 的 mode → 来源不明按不可信处理（fail-closed）")
    bad = sorted({m or "(未记录)" for m in ms} - {"real"})
    detail = {"modes": sorted(set(ms)), "mock_records": len(mock_qa)}
    if bad or mock_qa:
        return RequirementCheck(
            "non_mock_evidence", False,
            f"存在非 real 模式（{bad}）/ mock synthesis_source 的 episode"
            f"（{len(mock_qa)} 条）→ 按 HC24 不得进主表", detail)
    return RequirementCheck("non_mock_evidence", True,
                            f"全部 {len(ms)} 个 run 为 real 模式且无 mock episode", detail)


def check_poc_evidence(item: str, poc_evidence: Optional[Mapping[str, Any]]) -> RequirementCheck:
    """§18.6 末条：`[待实验]` 项必须有**自己的 PoC 证据**，否则不得进论文主表。

    只接受带 receipt/evidence 引用的通过记录：**裸布尔值不算证据**（无法复核，
    且"未跑的 PoC 一律不得写成 True"）。
    """
    ev = (poc_evidence or {}).get(item)
    if ev is None:
        return RequirementCheck(item, False,
                                f"{item} 是 [待实验] 项：未提供 PoC 证据 → 不得进论文主表")
    if isinstance(ev, bool):
        return RequirementCheck(
            item, False,
            f"{item} 只给了布尔值（{ev}）而无 receipt/evidence 引用 → 不可复核，"
            "不得进论文主表")
    if isinstance(ev, Mapping):
        passed = bool(ev.get("passed", False))
        ref = str(ev.get("evidence_ref", "") or ev.get("receipt_ref", "") or "").strip()
        if not passed:
            return RequirementCheck(item, False,
                                    f"{item} 的 PoC 未通过（passed=False）→ 不得进论文主表")
        if not ref:
            return RequirementCheck(
                item, False,
                f"{item} 声称 PoC 通过但无 receipt/evidence 引用 → 不可复核，拒绝")
        return RequirementCheck(item, True, f"{item} 已提供 PoC 证据：{ref}",
                                {"evidence_ref": ref})
    return RequirementCheck(item, False,
                            f"{item} 的 PoC 证据形态不可识别：{type(ev).__name__}")


def check_isolation_ledger(ledger: Optional[RunLedger]) -> RequirementCheck:
    """附加要件（§18.1/§18.6 联动）：paper-eligible 只能来自那次唯一的 `final` 盲评。

    - outer 用过 override 重跑 → 验证依据不再干净（§18.1：Outer 失败即 reject）；
    - `final` 必须恰好记录过 **1 次**（0 次 = 没有盲评记录；≥2 次已被 ledger 拒，但
      账本可能由外部写入，故不假设）。
    """
    if ledger is None:
        return RequirementCheck("isolation_ledger", False, "未提供 run ledger（未参与判定）",
                                {"provided": False})
    try:
        summary = ledger.summary()
    except (LedgerCorruptError, OSError) as exc:
        return RequirementCheck("isolation_ledger", False,
                                f"run ledger 不可读（{exc}）→ fail-closed 拒绝（§18.1）")
    reasons: list[str] = []
    if summary.get("outer_rerun_override_used"):
        reasons.append("outer 被 override 重跑过 → 该结果不得作为 paper-eligible 的验证依据"
                       "（§18.1/红线 9）")
    final_runs = int(summary.get("final_runs", 0))
    if final_runs != 1:
        reasons.append(f"final 盲评记录为 {final_runs} 次（要求恰好 1 次，§18.1/红线 5）")
    if reasons:
        return RequirementCheck("isolation_ledger", False, "；".join(reasons),
                                {"final_runs": final_runs, "summary": summary})
    return RequirementCheck("isolation_ledger", True,
                            "outer 无 override 重跑，且 final 恰好盲评 1 次（§18.1）",
                            {"final_runs": final_runs})


def check_paper_eligible(*, scene_ids_by_split: Optional[Mapping[str, Sequence[str]]] = None,
                         n_seeds: int = 0, paper: bool = True,
                         stat_gate: Any = None,
                         modes: Sequence[str] = (), records: Sequence[Any] = (),
                         poc_evidence: Optional[Mapping[str, Any]] = None,
                         ledger: Optional[RunLedger] = None) -> PaperEligibility:
    """§18.6 paper_eligible 判定（四要件 + 四个 `[待实验]` 项）。

    四要件（任一不满足即 `eligible=False`）：
    1. **数据隔离**：`scene_ids_by_split` 的 scene 级互斥（红线 6）；
    2. **seed**：≥3（`paper=True` 时 ≥5，§18.6）；
    3. **统计门**：`stat_gate` 为 §18.4 检验产物且 `significant=True` 且非退化；
    4. **非 mock 证据**：`modes` 全为 real 且 trace 里无 mock episode（HC24）。

    另外：四个 `[待实验]` 项（度量尺度融合 / MoGe-2 效果 / track 共识计数效果 /
    程序路径赢直答）必须在 `poc_evidence` 里各自给出带 receipt 引用的通过证据，
    否则一律拒绝 —— 未跑 PoC 的内容不得进论文主表。

    传入 `ledger` 时额外要求（§18.1 联动）：outer 没有被 override 重跑过，且 `final`
    恰好盲评一次。
    """
    checks: dict[str, RequirementCheck] = {}
    checks["data_isolation"] = check_data_isolation(scene_ids_by_split)
    required = MIN_SEEDS_PAPER if paper else MIN_SEEDS_NOISE_FLOOR
    n_seeds = int(n_seeds or 0)
    checks["seed_requirement"] = RequirementCheck(
        "seed_requirement", n_seeds >= required,
        (f"{n_seeds} seed ≥ {required}（§18.6）" if n_seeds >= required
         else f"{n_seeds} seed < {required}：不满足 §18.6 paper-eligible 的 seed 要求"),
        {"n_seeds": n_seeds, "required": required, "paper": bool(paper)})
    checks["statistical_gate"] = check_statistical_gate(stat_gate)
    checks["non_mock_evidence"] = check_non_mock(modes, records)
    if ledger is not None:
        checks["isolation_ledger"] = check_isolation_ledger(ledger)

    poc: dict[str, RequirementCheck] = {}
    for item in PENDING_POC_ITEMS:
        poc[item] = check_poc_evidence(item, poc_evidence)

    reasons = [f"{name}: {c.reason}" for name, c in checks.items() if not c.ok]
    reasons += [f"{name}([待实验]): {c.reason}" for name, c in poc.items() if not c.ok]
    return PaperEligibility(eligible=not reasons, checks=checks, pending_poc=poc,
                            reasons=reasons)


def assert_paper_eligible(**kwargs) -> PaperEligibility:
    """`check_paper_eligible` 的抛错版（写主表前的硬门）。"""
    verdict = check_paper_eligible(**kwargs)
    if not verdict.eligible:
        raise ProtocolError("paper_eligible 未满足 → " + "；".join(verdict.reasons))
    return verdict


# ================================== §22 红线 9 + §19.4 split 访问审计 ----

def coerce_split(value: Any) -> str:
    """split 名归一（stage 名 → 对应 split；未知但非空的名字原样保留）。"""
    if isinstance(value, Stage):
        return SPLIT_OF_STAGE[value]
    if isinstance(value, Enum):
        value = value.value
    v = str(value or "").strip()
    if not v:
        raise AuditError("split 名不得为空（§19.4）")
    if v.lower() in STAGE_OF_SPLIT:
        return SPLIT_OF_STAGE[STAGE_OF_SPLIT[v.lower()]]
    return v


def is_outer_split(split: Any) -> bool:
    """是否属于 outer（读它必须留理由，§19.4）。"""
    try:
        return coerce_stage(coerce_split(split)) is Stage.OUTER
    except ProtocolError:
        return False


class SplitAccessLog:
    """§19.4 split 访问审计（JSONL 追加，跨进程可读）。

    - **读 outer 必须留理由**：`log_access` 无 reason 直接 raise；
    - **跨 split 改动需显式决策记录**：`record_cross_split_decision` 先落决策，
      `assert_cross_split_change` 找不到"已批准"决策即拒绝。
    """

    ACCESS_FILENAME = "split_access_log.jsonl"
    DECISION_FILENAME = "cross_split_decisions.jsonl"

    def __init__(self, audit_dir: str | Path) -> None:
        self.dir = Path(audit_dir)
        self.access_path = self.dir / self.ACCESS_FILENAME
        self.decision_path = self.dir / self.DECISION_FILENAME

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"error": "unparsable_line", "raw": line[:200]})
        return rows

    # ------------------------------------------------------------ 读留痕 ----
    def log_access(self, split: Any, reason: str, *, actor: str = "",
                   run_id: str = "", purpose: str = "", at: str = "",
                   stage: Any = None) -> dict:
        """记录一次 split 访问；**reason 必填**（读 outer 必须留理由，§19.4）。"""
        name = coerce_split(split)
        why = str(reason or "").strip()
        if not why:
            raise AuditError(
                f"读 split={name} 必须留理由（reason 不得为空，§19.4）"
                + ("：outer 是验证集，无理由的读取等于泄漏" if is_outer_split(name) else ""))
        if stage is not None:
            stage_value = coerce_stage(stage).value
        else:
            try:
                stage_value = coerce_stage(name).value
            except ProtocolError:
                stage_value = ""
        record = {"split": name, "reason": why, "actor": str(actor),
                  "run_id": str(run_id), "purpose": str(purpose),
                  "at": str(at or _now()), "stage": stage_value}
        self._append(self.access_path, record)
        return record

    def accesses(self, split: Any = None) -> list[dict]:
        rows = self._read(self.access_path)
        if split is None:
            return rows
        name = coerce_split(split)
        return [r for r in rows if str(r.get("split")) == name]

    def reads_of(self, split: Any) -> int:
        return len(self.accesses(split))

    def last_access(self, split: Any) -> Optional[dict]:
        rows = self.accesses(split)
        return rows[-1] if rows else None

    def accesses_before(self, split: Any, at: str) -> list[dict]:
        """`at` 之前的访问（ISO 时间串比较）：用于判断"策略是否在读过 outer 之后生成"。"""
        return [r for r in self.accesses(split) if str(r.get("at", "")) <= str(at)]

    # -------------------------------------------------------- 跨 split 决策 ----
    def record_cross_split_decision(self, decision_id: str, from_split: Any,
                                   to_split: Any, decision: str, note: str, *,
                                   actor: str = "", at: str = "") -> dict:
        """落一条跨 split 改动决策（`decision ∈ {approved, rejected}`；note 必填）。"""
        f, t = coerce_split(from_split), coerce_split(to_split)
        dec = str(decision or "").strip().lower()
        if dec not in ("approved", "rejected"):
            raise AuditError(f"decision 必须是 approved/rejected，收到 {decision!r}")
        why = str(note or "").strip()
        if not why:
            raise AuditError(f"跨 split 改动（{f} → {t}）必须带决策理由（note，§19.4）")
        record = {"decision_id": str(decision_id), "from_split": f, "to_split": t,
                  "decision": dec, "note": why, "actor": str(actor),
                  "at": str(at or _now())}
        self._append(self.decision_path, record)
        return record

    def cross_split_decisions(self) -> list[dict]:
        return self._read(self.decision_path)

    def approved_decision(self, from_split: Any, to_split: Any) -> Optional[dict]:
        f, t = coerce_split(from_split), coerce_split(to_split)
        for row in self.cross_split_decisions():
            if (str(row.get("from_split")) == f and str(row.get("to_split")) == t
                    and str(row.get("decision")) == "approved"):
                return row
        return None

    def check_cross_split_change(self, from_split: Any, to_split: Any
                                 ) -> tuple[bool, str]:
        """跨 split 改动是否被显式批准（返回 `(ok, reason)`）。"""
        f, t = coerce_split(from_split), coerce_split(to_split)
        if f == t:
            return True, f"同一 split（{f}）内部改动，不需要跨 split 决策记录"
        dec = self.approved_decision(f, t)
        if dec is None:
            return False, (f"跨 split 改动 {f} → {t} 缺少显式决策记录（§19.4）；"
                           "先 `record_cross_split_decision(..., 'approved', note=…)`")
        return True, f"跨 split 改动 {f} → {t} 已有批准决策 {dec.get('decision_id')}"

    def assert_cross_split_change(self, from_split: Any, to_split: Any) -> None:
        ok, reason = self.check_cross_split_change(from_split, to_split)
        if not ok:
            raise AuditError(reason)


def strategy_content_hash(path_or_bytes: Any) -> str:
    """策略/阈值文件内容的 sha256（§19.4：文件带哈希，改动可查）。"""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return hashlib.sha256(bytes(path_or_bytes)).hexdigest()
    p = Path(path_or_bytes)
    if not p.is_file():
        raise AuditError(f"策略文件不存在：{p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass
class StrategyRecord:
    """一条策略/阈值登记的元数据（§19.4 + 红线 9）。

    `derived_from` 记录**这条策略是在哪个 stage 上调出来的**：只有 `inner` 可以，
    outer/final 上得出的策略一律不得进论文（红线 9）。
    """

    name: str
    derived_from: str
    content_hash: str = ""
    source_path: str = ""
    decision_note: str = ""
    thresholds: dict = field(default_factory=dict)
    created_at: str = ""
    run_id: str = ""
    actor: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "derived_from": self.derived_from,
                "content_hash": self.content_hash, "source_path": self.source_path,
                "decision_note": self.decision_note, "thresholds": dict(self.thresholds),
                "created_at": self.created_at, "run_id": self.run_id,
                "actor": self.actor}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StrategyRecord":
        return cls(name=str(data.get("name", "")),
                   derived_from=str(data.get("derived_from", "")),
                   content_hash=str(data.get("content_hash", "")),
                   source_path=str(data.get("source_path", "")),
                   decision_note=str(data.get("decision_note", "")),
                   thresholds=dict(data.get("thresholds") or {}),
                   created_at=str(data.get("created_at", "")),
                   run_id=str(data.get("run_id", "")),
                   actor=str(data.get("actor", "")))


def strategy_record_from_file(path: Any, *, name: str, derived_from: Any,
                              note: str = "", thresholds: Optional[Mapping] = None,
                              run_id: str = "", actor: str = "",
                              created_at: str = "") -> StrategyRecord:
    """由策略/阈值文件生成登记记录（自动算 sha256；`derived_from` 必填且须可识别）。"""
    st = coerce_stage(derived_from)
    return StrategyRecord(name=str(name), derived_from=st.value,
                          content_hash=strategy_content_hash(path),
                          source_path=str(path), decision_note=str(note),
                          thresholds=dict(thresholds or {}), created_at=str(created_at or _now()),
                          run_id=str(run_id), actor=str(actor))


@dataclass
class StrategyVerdict:
    """红线 9 判定（`paper_allowed=False` 时 `reasons` 逐条说明）。"""

    paper_allowed: bool
    derived_from: str
    name: str = ""
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"paper_allowed": self.paper_allowed, "derived_from": self.derived_from,
                "name": self.name, "reasons": list(self.reasons),
                "notes": list(self.notes)}


def check_strategy_provenance(record: StrategyRecord, *,
                              access_log: Optional[SplitAccessLog] = None,
                              verify_hash: bool = True) -> StrategyVerdict:
    """红线 9（§22.9）：策略/阈值只能 `derived_from=inner`；outer 调出来的一律拒绝。

    额外检查（§19.4）：
    - `derived_from` 缺失/不可识别 → fail-closed 拒绝；
    - `source_path` 存在但内容哈希与登记不符 → 拒绝（文件在登记后被改动）；
    - 若审计日志显示策略生成**晚于**一次 outer 读取，且登记没有决策记录 → 拒绝
      （跨 split 改动需显式决策记录）。
    """
    reasons: list[str] = []
    notes: list[str] = []
    raw = str(record.derived_from or "").strip()
    if not raw:
        return StrategyVerdict(False, "", record.name,
                               ["未声明 derived_from → fail-closed 拒绝（红线 9）"],
                               notes)
    try:
        derived = coerce_stage(raw)
    except ProtocolError as exc:
        return StrategyVerdict(False, raw, record.name,
                               [f"derived_from={raw!r} 不可识别：{exc}"], notes)
    if derived is not Stage.INNER:
        reasons.append(
            f"策略 derived_from={derived.value}：只有 inner 上定策略/阈值（§16.2），"
            "outer/final 上得出的策略不得进论文（红线 9）")
    else:
        notes.append("derived_from=inner ✓（可用作论文方法）")
    if verify_hash and record.source_path and record.content_hash:
        try:
            actual = strategy_content_hash(record.source_path)
        except AuditError as exc:
            reasons.append(f"无法校验策略文件哈希：{exc}")
        else:
            if actual != record.content_hash:
                reasons.append(
                    f"策略文件哈希不符（登记 {record.content_hash[:16]}… ≠ 实算 "
                    f"{actual[:16]}…）→ 文件在登记后被改动，须重新登记（§19.4）")
    if access_log is not None and record.created_at:
        outer_reads = access_log.accesses_before(Stage.OUTER.value, record.created_at)
        if outer_reads and not str(record.decision_note or "").strip():
            reasons.append(
                f"策略生成于 outer 读取之后（此前的 {len(outer_reads)} 次 outer 访问）且无"
                "决策记录 → 无法排除用 outer 调策略（§19.4/红线 9）")
        elif outer_reads:
            notes.append(f"策略生成晚于 {len(outer_reads)} 次 outer 访问，但已留决策记录："
                         f"{record.decision_note}")
    return StrategyVerdict(not reasons, derived.value, record.name, reasons, notes)


def assert_strategy_paper_eligible(record: StrategyRecord, *,
                                   access_log: Optional[SplitAccessLog] = None) -> None:
    """红线 9 的抛错版。"""
    verdict = check_strategy_provenance(record, access_log=access_log)
    if not verdict.paper_allowed:
        raise RedLineViolation("；".join(verdict.reasons))


__all__ = [
    # 异常
    "AuditError", "IsolationViolation", "LedgerCorruptError", "PairedABViolation",
    "ProtocolError", "RedLineViolation",
    # §18.1 三级隔离
    "SPLIT_OF_STAGE", "STAGE_OF_SPLIT", "STAGE_RULES", "LedgerEntry", "RunLedger",
    "RunPermission", "Stage", "assert_isolation", "coerce_stage",
    "isolation_violations",
    # §18.2 样本量三档
    "FORBIDDEN_SUBSAMPLE_PER_TASK", "MIN_POOL_FOR_FULL_TIER", "SAMPLE_TIERS",
    "SampleSizeReport", "SampleTier", "binary_se", "binary_se_points",
    "check_sample_size", "format_sample_tiers", "required_seeds_for_pool",
    "weight_per_question",
    # §18.3 噪声底
    "MIN_SEEDS_NOISE_FLOOR", "MIN_SEEDS_PAPER", "NoiseFloorReport",
    "THROUGHPUT_ONLY_PURPOSES", "assert_round_robin_not_for_comparison",
    "check_round_robin_usage", "noise_floor_report",
    # §18.4 paired A/B
    "PAIRED_AB_FIELDS", "PairedABCheck", "assert_paired_ab", "check_paired_ab",
    "paired_statistics", "paired_statistics_by_task",
    # §18.5 报告
    "MISSING_SIGNATURE", "RateGroup", "TraceView", "ci95_of_seed_values",
    "evidence_state_breakdown", "evidence_states_of", "normalize_answer_source",
    "normalize_synthesis_source", "MOCK_SYNTHESIS_SOURCES", "SYNTHESIS_SOURCES_V6",
    "per_task_ci95", "signature_key", "trace_view", "wilson_ci",
    # §18.6 paper_eligible
    "PENDING_POC_ITEMS", "PaperEligibility", "RequirementCheck",
    "assert_paper_eligible", "check_data_isolation", "check_isolation_ledger",
    "check_non_mock", "check_paper_eligible", "check_poc_evidence",
    "check_statistical_gate",
    # §19.4 + 红线 9
    "SplitAccessLog", "StrategyRecord", "StrategyVerdict", "coerce_split",
    "assert_strategy_paper_eligible", "check_strategy_provenance", "is_outer_split",
    "strategy_content_hash", "strategy_record_from_file",
]
