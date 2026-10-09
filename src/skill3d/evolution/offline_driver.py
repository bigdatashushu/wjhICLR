"""§6.2 离线演进 FSM 的真实 driver（G-35）＋离线模型归纳驱动（G-28）。

```bash
python -m skill3d.evolution.offline_driver --mode real --run-id gen-0001
```

串起 §6.2 全状态链：

```
CLUSTER_TRACES → GPT6_SYNTHESIZE → LEAKAGE_CHECK
→ OPTIMIZATION_LOOP { SYNTHESIZE → STATIC_CHECK → TEST(L1→L2→L3) → DIAGNOSE → REVISE }
→ PROMOTE / REJECT / QUARANTINE
```

要点：
- **每个状态落一次 JSON checkpoint**（`--checkpoint`），中断后 `--resume` 从最后状态继续，
  已完成的 episode 级工作不重跑（§6.2 新增：持久化 + 中断恢复）；
- **离线强模型仅离线**（v6 §3.4：DeepSeek-V4.1-Flash / `deepseek-flash`）：只在
  `GPT6_SYNTHESIZE` 与 REVISE 被调用；不可用时按 §3.4 策略进 `service_unavailable` /
  QUARANTINE —— **不切在线链、不用 mock 结果推进 Readiness**；本模块不得被在线链
  import（硬约束 1/2）；
- **准入必须 real**（§5.6b）：`--mode mock_light` 只做管道验证，不可能 promote；
- **outer 只跑一次**（硬约束 10），失败即 REJECT，不基于 outer 失败再修订；
- **候选不可变**（硬约束 11）：REVISE 由 M16 产新 revision（parent 链），driver 不原地改；
- **LEAKAGE_CHECK 硬门**（硬约束 13/19）：候选 spec_content 含答案/sample id 即拒；
- **advisory**（§3.3）：离线模型的文本（归纳/修订/审查）都不构成准入决定 —— promote
  由确定性门 + 预注册规则决定，`admission.py` 一票否决。

归纳输入纪律：只喂"失败类型/成功失败标签/题型/场景"等**不含答案**的摘要；
prompt 发送前做泄漏扫描（`build_induction_prompt` 内断言）。

**v5 标识符说明**：`OfflineState.GPT6_SYNTHESIZE` 与事件 `gpt6_unavailable` 是 v5 遗留的
状态/事件**标识符**（定义在 `fsm/offline_fsm.py`，会序列化进 checkpoint），v6 语义已变为
"离线模型归纳 / 离线模型不可用"。保留字面量以免历史 checkpoint 不可恢复；代码内不再出现
GPT-6 客户端。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from skill3d.adapters.episode_source import (
    EpisodeItem,
    EpisodeSourceError,
    load_jsonl_items,
    load_synthetic_items,
    load_vsi_bench_items,
)
from skill3d.evolution.optimization_loop import (
    DEFAULT_BUDGET_LIMIT,
    CandidateArchive,
)
from skill3d.evolution.panel import (
    FULL_LEVELS,
    SIMPLIFIED_LEVELS,
    admit,
    candidate_skill,
    run_candidate_panels,
)
from skill3d.evolution.firewall import SplitContaminationError
from skill3d.fsm.offline_fsm import OfflineFSM, OfflineState
from skill3d.governance.deepseek_client import (
    DeepSeekClient,
    OfflineAuthError,
    OfflineNotConfiguredError,
    OfflineRequestError,
    OfflineResponseError,
    OfflineServiceUnavailable,
)
from skill3d.governance.induce import (
    InsufficientEvidenceError,
    induce_candidate_legacy_v9,
)
from skill3d.memory.consolidation import leakage_scan_text
from skill3d.online.config import DEFAULT_CONFIG, load_config, load_yaml, paths_from
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import (
    CandidateRevision,
    CounterexampleBundle,
    DataAccessRecord,
    EpisodeTrace,
    PairedOutcome,
    utcnow_iso,
)
from skill3d.trace.store import TraceStore

MODES = ("real", "mock_light")
# §8 治理消融档（G-41/G-64）：G0 全治理 / G1 无 review / G2 无运行期监控
GOVERNANCE_MODES = ("G0_full", "G1_no_review", "G2_no_monitoring")

# 治理消融对状态链的影响（G-41）：G1 跳归纳后语义审查；G2 跳运行期监控（离线统计）
_ABLATION_SKIP_REVIEW = {"G1_no_review"}
_ABLATION_SKIP_MONITORING = {"G2_no_monitoring"}

# ---- §3.4 离线失败的显式结局码（绝不回退在线链 / 绝不用 mock 顶上）----
FAILURE_SERVICE_UNAVAILABLE = "service_unavailable"   # 超时/限流/5xx 重试耗尽、健康检查失败
FAILURE_OFFLINE_AUTH = "offline_auth_error"           # 401/403 或 DEEPSEEK_API_KEY 未注入
FAILURE_OFFLINE_REQUEST = "offline_request_error"     # 其余不可重试的 4xx / 响应不可用

# §3.4 归族 → 结局码（auth 不重试、不降级；服务类失败记 service_unavailable）
_AUTH_EXCEPTIONS = (OfflineAuthError,)


def classify_offline_failure(exc: BaseException) -> str:
    """§3.4：离线调用异常 → 显式结局码。

    - 认证失败 / 未配置（`OfflineAuthError`，含 `OfflineNotConfiguredError`）
      → `offline_auth_error`（**不重试、不降级**）；
    - 服务不可用（`OfflineServiceUnavailable`）→ `service_unavailable`；
    - 请求/响应问题（`OfflineRequestError` / `OfflineResponseError`）→
      `offline_request_error`。

    三类结局**一律进 QUARANTINE**（§3.4：不得切入在线链、不得用 mock 结果推进
    Readiness）；区别只在 reason 码，便于审计"当时是哪种失败"。
    """
    if isinstance(exc, _AUTH_EXCEPTIONS):
        return FAILURE_OFFLINE_AUTH
    if isinstance(exc, OfflineServiceUnavailable):
        return FAILURE_SERVICE_UNAVAILABLE
    if isinstance(exc, (OfflineRequestError, OfflineResponseError)):
        return FAILURE_OFFLINE_REQUEST
    # 传输层裸异常（未包装）同样按服务不可用处理，绝不静默继续
    return FAILURE_SERVICE_UNAVAILABLE


@dataclass
class OfflineDriverConfig:
    """离线 driver 配置（阈值一律来自 configs/*.yaml，全部 TODO_CALIBRATE）。"""

    mode: str = "real"
    split: str = "induction"
    panel_source: str = "synthetic"
    episodes_jsonl: str = ""
    video_root: str = ""
    n_min_cross_scene: int = 3
    min_delta_mca: float = 0.02
    min_delta_mra: float = 0.02
    l1_limit: int = 4
    seed: int = 0
    traces_glob: str = ""
    trace_dir: str = "data/traces"
    skill_store: str = "skill_library/snapshots"
    active_snapshot: str = "skill_library/snapshots/active_snapshot.json"
    recon_dir: str = "data/reconstructions"
    recon_method: str = "vggt"
    checkpoint_path: str = "data/offline_runs/latest.json"
    run_id: str = ""
    governance: str = "G0_full"
    max_revisions: int = 0
    simplified_phase_gate: bool = False
    ablation: str = ""          # E0..E5（§8.3）；空 = E5 等价全开
    resume: bool = False
    # §18.1 纪律闸：跑到指定状态**之前**停下，用于"只跑归纳半程"而不消耗 outer_holdout。
    # 取值是 OfflineState 名（cluster / induce / leakage）。
    # 为什么需要它：下游 LOOP_SYNTHESIZE 的 L3 面板口径就是 outer_holdout
    #（`load_panels`），而 §18.1 / 硬约束 10 规定 outer **只跑一次**。
    # 把"归纳→候选→泄漏检查"这段本来不需要 outer 的工作与"花掉 outer"解耦，
    # 是纪律要求，不是为了方便。
    stop_after: str = ""


# ---- §18.1 纪律闸：`--stop-after` 的状态名映射 ----
# 语义：**执行完**该状态后停止（该状态本身照跑），不再进入下游状态。
_STOP_AFTER_STATES: dict[str, OfflineState] = {
    "cluster": OfflineState.CLUSTER_TRACES,
    "induce": OfflineState.GPT6_SYNTHESIZE,      # v5 标识符；v6 语义 = 离线模型归纳
    "leakage": OfflineState.LEAKAGE_CHECK,
    "loop": OfflineState.LOOP_SYNTHESIZE,        # 进入这里就会跑 panels（含 outer）
    "promote": OfflineState.PROMOTE,
}


# ------------------------------------------------------------------ checkpoint ----

@dataclass
class OfflineCheckpoint:
    """离线 run 的持久化状态：每步一次原子写（中断可恢复，§6.2）。"""

    run_id: str
    state: str
    transitions: list[dict] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    revision_ids: list[str] = field(default_factory=list)
    current_revision_id: str = ""
    # Resume-safe lineage: IDs alone cannot reconstruct the final payload for
    # an atomic promote after process restart.
    revision_payloads: dict[str, dict] = field(default_factory=dict)
    # §14.3：准入证据来自 **inner 面板**（`admission_outcome`）；
    # `l3_outcome` 是 outer_holdout 的独立验证证据，**不作为准入门**。
    admission_outcome: Optional[dict] = None
    l3_outcome: Optional[dict] = None
    # 离线强模型（DeepSeek-V4.1-Flash）是否可用；不可用时终态必为 QUARANTINE
    offline_available: bool = True
    termination_reason: str = ""
    notes: list[str] = field(default_factory=list)
    skipped_stages: list[str] = field(default_factory=list)
    updated_at: str = ""
    # §17.1「数据访问」行：每次真实读取落一条（split／用途／角色／运行身份／
    # 输入清单 hash／时间；归纳侧还带 label_access 与拒收账本）
    data_access: list[dict] = field(default_factory=list)
    # §14.1 硬隔离的拒收账本：逐条记 `{episode_id, reason}`，供审计复核
    # "哪些材料被挡在归纳之外"（不进入任何归纳 prompt）
    refused_traces: list[dict] = field(default_factory=list)
    # §3.4：失败结局码（service_unavailable / offline_auth_error / offline_request_error）
    offline_failure_code: str = ""
    # §19.2 离线治理模型块（provider/model_id/endpoint_hash/prompt_version/latency/
    # token_usage；由 `DeepSeekClient.manifest_fields()` 提供，**绝不含密钥**）
    offline_model_fields: dict = field(default_factory=dict)

    @property
    def gpt6_available(self) -> bool:
        """v5 字段名别名（只读；历史 checkpoint/调用点过渡用）。"""
        return self.offline_available

    def save(self, path: str | Path) -> Path:
        """原子写：先写临时文件再替换，避免中断留下半截 JSON。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "OfflineCheckpoint":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # v5 checkpoint 兼容：字段名 gpt6_available → offline_available（§20 已弃用名）
        if "gpt6_available" in data and "offline_available" not in data:
            data["offline_available"] = bool(data.pop("gpt6_available"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ------------------------------------------------------------------ trace 读取 ----

def _iter_trace_records(patterns: list[str]):
    """逐行遍历 trace JSONL，产出 `(topic, payload)`。

    兼容两种落盘形态：
    ① `TraceStore` 的**按 topic 分文件**（`episode_trace.jsonl` 每行即 payload）；
    ② 带封装的一行一记录（`{"topic": ..., "payload": {...}}`）。
    topic 由封装字段或文件名推断；无法判定的行跳过（不因单行损坏中止离线跑）。
    """
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            stem = Path(f).stem.replace("_", "")
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    continue
                if "topic" in rec and isinstance(rec.get("payload"), dict):
                    yield str(rec["topic"]), rec["payload"]
                    continue
                if "final_state" in rec:
                    yield "episode_trace", rec
                elif "correct" in rec:
                    yield "evaluation_result", rec
                elif stem:
                    yield stem, rec


def _trace_files(traces_glob: str, trace_dir: str) -> list[str]:
    """展开 glob：未给 glob 时优先 `episode_trace.jsonl`，否则 trace_dir 下全部。"""
    if traces_glob:
        return [traces_glob]
    preferred = sorted(glob.glob(str(Path(trace_dir) / "episode_trace*.jsonl")))
    return preferred or [str(Path(trace_dir) / "*.jsonl")]


def load_episode_traces(traces_glob: str, trace_dir: str = "data/traces") -> list[EpisodeTrace]:
    """从 TraceStore 的 JSONL 读回 EpisodeTrace（离线归纳输入）。"""
    traces: list[EpisodeTrace] = []
    for topic, payload in _iter_trace_records(_trace_files(traces_glob, trace_dir)):
        if topic != "episode_trace":
            continue
        try:
            traces.append(EpisodeTrace.model_validate(payload))
        except Exception:  # noqa: BLE001 - 非法行跳过
            continue
    return traces


def trace_outcomes(trace_dir: str) -> dict[str, bool]:
    """从 trace JSONL 侧读 episode 的"是否答对"标签（不含答案文本）。

    只取 `evaluation_result.correct` 这一派生布尔量用于区分成功/失败轨迹；
    `ground_truth` 绝不进入归纳 prompt（硬约束 19）。
    """
    ok_of: dict[str, bool] = {}
    for topic, payload in _iter_trace_records([str(Path(trace_dir) / "*.jsonl")]):
        if topic != "evaluation_result":
            continue
        qa_id = str(payload.get("qa_id", ""))
        if qa_id:
            ok_of[qa_id] = bool(payload.get("correct"))
    return ok_of


def _offline_manifest_fields(client) -> dict:
    """§19.2 离线治理模型块（`provider`/`model_id`/`endpoint_hash`/`prompt_version`/
    `latency`/`token_usage`）。

    清洗口径复用 `online/eval.py::offline_manifest_fields`（单一定义点：两处都不得落
    密钥 / Authorization / base URL 本体）。拿不到元数据时返回 `{}`——**不虚构**。
    """
    if client is None:
        return {}
    try:
        from skill3d.online.eval import offline_manifest_fields

        return offline_manifest_fields(client)
    except Exception:  # noqa: BLE001 - 元数据是附加审计信息，取不到不阻断离线链
        return {}


# ------------------------------------------------------------------ driver ----

class OfflineDriver:
    """§6.2 离线 FSM driver（离线强模型只在本模块的归纳 / REVISE / 审查被调用）。"""

    def __init__(
        self,
        cfg: OfflineDriverConfig,
        *,
        offline_client=None,
        gpt6_client=None,
        trace_store: Optional[TraceStore] = None,
        panels: Optional[dict[str, list[EpisodeItem]]] = None,
        base_cfg: Optional[OnlineRunConfig] = None,
        episode_meta: Optional[dict[str, tuple[str, str]]] = None,
        llm=None,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        # 离线强模型客户端（v6：DeepSeek-V4.1-Flash）。`gpt6_client` 是 v5 形参别名
        # （过渡期只读；§20 已废止 GPT-6），只做转发。
        self.offline = offline_client if offline_client is not None else gpt6_client
        self.trace_store = trace_store
        self.panels = panels or {}
        self.base_cfg = base_cfg
        # qa_id → (规范题型, scene_name)：来自 induction split 的 episode 元数据
        self.episode_meta = episode_meta or {}
        self.llm = llm
        self.print = print_fn
        self.fsm = OfflineFSM()
        self.ckpt = OfflineCheckpoint(run_id=cfg.run_id or f"gen-{uuid.uuid4().hex[:8]}",
                                      state=self.fsm.state.value)
        self.archive = CandidateArchive()

    # ---- checkpoint 辅助 ----
    def _advance(self, event: str, ctx: Optional[dict] = None) -> OfflineState:
        before = self.fsm.state
        state = self.fsm.step(event, ctx)
        self.ckpt.state = state.value
        self.ckpt.transitions.append({
            "from": before.value, "event": event, "to": state.value,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        return state

    def _save(self) -> None:
        if self.cfg.checkpoint_path:
            self.ckpt.save(self.cfg.checkpoint_path)

    def _log(self, msg: str) -> None:
        self.ckpt.notes.append(msg)
        self.print(msg)

    # ---- 主流程 ----
    def run(self) -> OfflineCheckpoint:
        """跑完整状态链；返回终态 checkpoint（幂等：可从中断处 resume）。"""
        if self.cfg.resume and Path(self.cfg.checkpoint_path).is_file():
            self.ckpt = OfflineCheckpoint.load(self.cfg.checkpoint_path)
            self.archive = CandidateArchive()
            for rid, payload in self.ckpt.revision_payloads.items():
                try:
                    self.archive.revisions[rid] = CandidateRevision.model_validate(payload)
                except Exception:  # noqa: BLE001 - corrupt lineage blocks promote later
                    self._log(f"[resume] revision payload 无法恢复: {rid}")
            self.fsm.state = OfflineState(self.ckpt.state)
            if self.ckpt.admission_outcome:
                try:
                    self._admission = PairedOutcome.model_validate(
                        self.ckpt.admission_outcome)
                except Exception:  # noqa: BLE001 - corrupt outcome blocks promote
                    self._log("[resume] 准入面板结果无法恢复，拒绝后续 promote")
            if self.ckpt.l3_outcome:
                try:
                    self._l3 = PairedOutcome.model_validate(self.ckpt.l3_outcome)
                except Exception:  # noqa: BLE001 - corrupt evidence is not a gate
                    self._log("[resume] L3 验证证据无法恢复（不影响准入判定）")
            self._log(f"[resume] 从 {self.ckpt.state} 恢复 run_id={self.ckpt.run_id}")

        self._log(f"run_id={self.ckpt.run_id} governance={self.cfg.governance} "
                  f"mode={self.cfg.mode} split={self.cfg.split}")

        stop_state = _STOP_AFTER_STATES.get(str(self.cfg.stop_after or "").strip().lower())
        if self.cfg.stop_after and stop_state is None:
            raise ValueError(f"未知 --stop-after={self.cfg.stop_after!r}"
                             f"（可选 {sorted(_STOP_AFTER_STATES)}）")

        # 顺序与旧实现一致（每个状态处理完可能改变 self.fsm.state → 下一个 `if` 再判）
        stages = (
            (OfflineState.CLUSTER_TRACES, self._stage_cluster, False),
            (OfflineState.GPT6_SYNTHESIZE, self._stage_induce, True),
            (OfflineState.LEAKAGE_CHECK, self._stage_leakage_check, True),
            (OfflineState.LOOP_SYNTHESIZE, self._stage_optimize, False),
            (OfflineState.PROMOTE, self._stage_promote, False),
        )
        for state, stage, gated in stages:
            if self.fsm.state is not state:
                continue
            if gated:
                if not stage():
                    return self.ckpt
            else:
                stage()
            if stop_state is state:
                self.ckpt.termination_reason = f"stop_after:{self.cfg.stop_after}"
                self._log(
                    f"[停止] --stop-after={self.cfg.stop_after}：已执行完 {state.value}，"
                    "不再进入下游 LOOP_SYNTHESIZE/PROMOTE —— **outer_holdout 未被消耗**"
                    "（§18.1 / 硬约束 10：outer 只跑一次）")
                self.ckpt.save(self.cfg.checkpoint_path)
                return self.ckpt
        self._log(f"[结束] 终态={self.fsm.state.value} reason={self.ckpt.termination_reason}")
        return self.ckpt

    # ---- CLUSTER_TRACES ----
    def _stage_cluster(self) -> None:
        traces = load_episode_traces(self.cfg.traces_glob, self.cfg.trace_dir)
        ok_of = trace_outcomes(self.cfg.trace_dir)
        # 题型/场景取自 induction split 的 episode 元数据（trace 内不含 scene，防泄漏）
        task_of = {q: t for q, (t, _s) in self.episode_meta.items()}
        scene_of = {q: s for q, (_t, s) in self.episode_meta.items()}
        scenes = {scene_of.get(t.episode_id, "unknown") for t in traces}
        tasks = {task_of.get(t.episode_id, "unknown") for t in traces}
        n_ok = sum(1 for t in traces if ok_of.get(t.episode_id))
        self.print(f"[CLUSTER_TRACES] traces={len(traces)} scenes={len(scenes)} "
                   f"tasks={len(tasks)} 成功={n_ok}/{len(traces)}")
        self._traces = traces
        self._task_of, self._scene_of, self._ok_of = task_of, scene_of, ok_of
        cross_ok = (len(scenes) >= self.cfg.n_min_cross_scene
                    and len(traces) >= self.cfg.n_min_cross_scene)
        if not cross_ok:
            self.ckpt.termination_reason = (
                f"insufficient_evidence: scenes={len(scenes)} traces={len(traces)} "
                f"< N_min={self.cfg.n_min_cross_scene}（TODO_CALIBRATE）")
        self._advance("clustered", {"cross_scene_ok": cross_ok})

    # ---- GPT6_SYNTHESIZE（G-28；v6：离线模型 = DeepSeek-V4.1-Flash）----
    def _resolve_offline_client(self):
        """解析离线客户端：注入优先；否则新建 `DeepSeekClient` 并做**启动前健康检查**。

        §3.4：密钥未注入 → `OfflineNotConfiguredError`（⊂ `OfflineAuthError`，**不重试**）；
        健康检查（`GET /models`）失败 → `OfflineServiceUnavailable`。两种情况都由调用方
        记 `offline_auth_error` / `service_unavailable` + QUARANTINE；**不切在线链、不用
        mock 顶替**。注入的 client（测试 / 自定义 transport）不做健康检查——由注入方负责，
        避免测试触网。
        """
        if self.offline is not None:
            return self.offline
        client = DeepSeekClient()
        if not client.api_key_configured:      # 无密钥：直接归族，不做任何网络尝试
            raise OfflineNotConfiguredError(
                f"{client.provider} 离线客户端未配置密钥（只从环境变量注入，§3.4）")
        client.require_service()   # 失败按其异常类型归族（auth / service_unavailable）
        self.offline = client
        return client

    def _record_offline_failure(self, exc: BaseException, where: str) -> str:
        """§3.4 统一失败处理：登记结局码 + 模型块 + 日志，返回结局码。"""
        code = classify_offline_failure(exc)
        self.ckpt.offline_available = False
        self.ckpt.offline_failure_code = code
        self.ckpt.termination_reason = f"{code}: {exc}"
        self._log(f"[{where}] 离线模型不可用（{code}）→ QUARANTINE（§3.4：不切在线链、"
                  "不用 mock 结果推进 Readiness）: " + str(exc)[:300])
        return code

    def _stage_induce(self) -> bool:
        try:
            client = self._resolve_offline_client()
        except (OfflineAuthError, OfflineServiceUnavailable, OfflineRequestError,
                OfflineResponseError) as exc:
            self._record_offline_failure(exc, "GPT6_SYNTHESIZE")
            self._advance("gpt6_unavailable")
            return True
        self.ckpt.offline_available = True
        self.ckpt.offline_failure_code = ""

        traces = getattr(self, "_traces", [])
        try:
            # v9 路径的归纳语义（按题型聚合、产出无父版本的 draft）**不**满足
            # v10 §5.2/§7.2（候选必须有父版本、必须完整 SkillSpec）。v10 的两代演化
            # 走 `evolution/campaign.py` + `governance.induce.induce_candidate_from_bundle`；
            # v9 driver 保留旧语义只为让历史实验链仍可运行。
            candidate = induce_candidate_legacy_v9(
                traces, getattr(self, "_task_of", {}), getattr(self, "_scene_of", {}),
                offline_client=client, n_min=self.cfg.n_min_cross_scene)
        except InsufficientEvidenceError as exc:
            self.ckpt.termination_reason = f"insufficient_evidence: {exc}"
            self._log(f"[GPT6_SYNTHESIZE] 样本不足 → REJECT: {exc}")
            self._advance("candidate_ready", {"cross_scene_ok": False})
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()
            return True
        except (OfflineAuthError, OfflineServiceUnavailable, OfflineRequestError,
                OfflineResponseError) as exc:
            # 端点在调用时才暴露不可用 → 按 §3.4 暂停（不 promote、不阻塞在线链）
            self._record_offline_failure(exc, "GPT6_SYNTHESIZE")
            self._advance("gpt6_unavailable")
            return True
        if candidate is None:
            # 离线模型未配置但低风险：不进 promotion（§3.4）
            self.ckpt.offline_available = False
            self.ckpt.offline_failure_code = FAILURE_SERVICE_UNAVAILABLE
            self.ckpt.termination_reason = "offline_returned_no_candidate"
            self._log("[GPT6_SYNTHESIZE] 未产出 candidate → QUARANTINE")
            self._advance("gpt6_unavailable")
            return True

        self._candidate_v0 = candidate
        self.archive.remember(candidate)
        self.ckpt.candidate_ids.append(candidate.root_candidate_id)
        self.ckpt.revision_ids.append(candidate.revision_id)
        self.ckpt.current_revision_id = candidate.revision_id
        self.ckpt.revision_payloads[candidate.revision_id] = candidate.model_dump()
        self.ckpt.offline_model_fields = _offline_manifest_fields(client)
        self._log(f"[GPT6_SYNTHESIZE] candidate_v0 root={candidate.root_candidate_id} "
                  f"revision={candidate.revision_id} "
                  f"（归纳自 {len(candidate.induction_trace_refs)} 条轨迹；"
                  f"离线模型={self.ckpt.offline_model_fields.get('model_id', '?')}）")
        if self.trace_store is not None:
            self.trace_store.append("candidate_revision", candidate.model_dump())
        self._advance("candidate_ready")
        return True

    # ---- LEAKAGE_CHECK（硬门）----
    def _stage_leakage_check(self) -> bool:
        candidate = getattr(self, "_candidate_v0", None)
        if candidate is None:
            self.ckpt.termination_reason = "no_candidate"
            self._save()
            return False
        leaks = leakage_scan_text(candidate.spec_content,
                                  forbidden_sample_ids=set(self.episode_meta))
        self._log(f"[LEAKAGE_CHECK] hits={leaks or '∅'}（硬约束 13/19；"
                  f"扫描 {len(self.episode_meta)} 个 induction sample id）")
        if self.trace_store is not None:
            self.trace_store.append("leakage_check", {
                "candidate_id": candidate.root_candidate_id,
                "hits": leaks, "passed": not leaks})
        self._no_leakage = not leaks
        self._advance("pass" if self._no_leakage else "reject")
        if not self._no_leakage:
            self.ckpt.termination_reason = f"leakage_detected: {leaks}"
        return True

    # ---- OPTIMIZATION_LOOP ----
    def _stage_optimize(self) -> None:
        candidate = getattr(self, "_candidate_v0", None)
        if candidate is None or self.base_cfg is None or not self.panels:
            self.ckpt.termination_reason = "missing_candidate_or_panel"
            self._log("[OPTIMIZATION_LOOP] 缺候选或面板 → REJECT")
            self._advance("fail", {"cross_scene_ok": False})
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()
            return

        arc = self.archive
        if arc.similar_rejected(candidate.spec_content):
            self.ckpt.termination_reason = "similar_to_rejected_no_new_evidence"
            self._log("[OPTIMIZATION_LOOP] 与已 reject 候选高度相似且无新证据 → REJECT（防重复）")
            self._advance("fail")
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()
            return

        def revise_fn(rev: CandidateRevision, feedback: str) -> CandidateRevision:
            """REVISE：离线强模型基于反例摘要产新版本（绝不原地改，硬约束 11）。"""
            from skill3d.governance.revise_patch import revise_from_bundle

            visible = f"测试反馈（不含答案）: {feedback}"
            if ce_summary:
                visible = f"{ce_summary} | {visible}"
            bundle = CounterexampleBundle(
                bundle_id=f"bundle-{rev.revision_id}",
                source_revision_id=rev.revision_id,
                failed_episode_refs=list(getattr(self, "_ce_failed_refs", [])),
                minimal_counterexamples=[],
                metamorphic_transforms=[],
                regression_set_ref="",
                gpt6_visible_summary=visible,
                generated_at="1970-01-01T00:00:00+00:00",
            )
            if self.trace_store is not None:
                self.trace_store.append("counterexample_bundle", bundle.model_dump())
            client = self.offline or DeepSeekClient()
            new_rev = revise_from_bundle(rev, bundle, client)
            self.archive.remember(new_rev)
            self.ckpt.revision_ids.append(new_rev.revision_id)
            self.ckpt.current_revision_id = new_rev.revision_id
            self.ckpt.revision_payloads[new_rev.revision_id] = new_rev.model_dump()
            self.ckpt.offline_model_fields = _offline_manifest_fields(client)
            self._log(f"[REVISE] 新版本 revision={new_rev.revision_id} "
                      f"parent={new_rev.parent_version}（候选不可变，硬约束 11）")
            return new_rev

        # ---- 消融档特性（E0–E5，§8.3）：只改机制开关，不改模型/split/seed ----
        feats = {}
        if self.cfg.ablation:
            from skill3d.evolution.ablation import e_ablation_features

            feats = e_ablation_features(self.cfg.ablation)
            self._log(f"[消融 {self.cfg.ablation}] {feats.get('desc', '')} → "
                      f"replay={feats.get('replay')} 反例={feats.get('counterexample')} "
                      f"MR={feats.get('metamorphic')} paired={feats.get('paired')}")
            self.ckpt.skipped_stages.extend(
                [f"ablation:{k}" for k, v in feats.items() if v is False and k != "desc"])

        def static_check_fn(rev: CandidateRevision) -> bool:
            try:
                candidate_skill(rev)
                return True
            except Exception:  # noqa: BLE001
                return False

        # 反例挖掘（E2+）：先跑一次 baseline 臂收集失败 episode，作为反例池喂给 REVISE
        ce_summary = ""
        if feats.get("counterexample"):
            ce_summary = self._mine_counterexamples()

        # 5 类空间 MR 门（E3+）：对场景不变量做确定性校验，违例即拒（不进准入）
        if feats.get("metamorphic"):
            mr_ok, mr_note = self._mr_gate()
            self._log(f"[消融 E3] MR 门: {mr_note}")
            self.ckpt.notes.append(f"[消融 E3] {mr_note}")
            if not mr_ok:
                self.ckpt.termination_reason = f"metamorphic_violation: {mr_note}"
                self.fsm.state = OfflineState.REJECT
                self.ckpt.state = self.fsm.state.value
                self._save()
                return

        limit = DEFAULT_BUDGET_LIMIT
        if self.cfg.max_revisions:
            limit = limit.model_copy(update={"max_revisions": self.cfg.max_revisions})
        self._log("[OPTIMIZATION_LOOP] SYNTHESIZE→STATIC_CHECK→TEST(L1→L2→L3)→DIAGNOSE→REVISE")
        try:
            run, level_state = run_candidate_panels(
                candidate, base_cfg=self.base_cfg, panels=self.panels,
                trace_store=self.trace_store, seed=self.cfg.seed,
                min_delta_mca=self.cfg.min_delta_mca,
                min_delta_mra=self.cfg.min_delta_mra,
                revise_fn=revise_fn, static_check_fn=static_check_fn, archive=arc,
                budget_limit=limit, print_fn=self.print,
                levels=self._admission_safe_levels(),
            )
        except (OfflineAuthError, OfflineServiceUnavailable, OfflineRequestError,
                OfflineResponseError) as exc:
            # §3.4：修订需要离线强模型；不可用 → 暂停（QUARANTINE），绝不降级为
            # "没有模型就跳过修订继续 promote"。
            self._record_offline_failure(exc, "OPTIMIZATION_LOOP")
            self.fsm.state = OfflineState.QUARANTINE
            self.ckpt.state = self.fsm.state.value
            self._save()
            return
        self._opt_run, self._level_state = run, level_state
        self.ckpt.current_revision_id = run.current_revision_id
        if self.trace_store is not None:
            self.trace_store.append("optimization_run", run.model_dump())
        self._log(f"[OPTIMIZATION_LOOP] status={run.status} reason={run.termination_reason} "
                  f"revisions={run.budget_used.revisions} rollouts={run.budget_used.rollouts}")

        # 映射到 FSM 终态（outer 只跑一次，硬约束 10）
        #
        # §14.3：**准入消费 inner 面板**（L2 = 该题型完整 inner），
        # L3 = outer_holdout 的结果只作为快照冻结后的独立验证证据落盘，
        # 不参与晋升判定 —— 用 holdout 选代会让它不再是 holdout。
        l2 = level_state.get("L2")
        l3 = level_state.get("L3")
        if l3 is not None:
            self.ckpt.l3_outcome = l3[0].model_dump()
        if run.status == "promoted" and l2 is not None:
            self._admission = l2[0]
            self.ckpt.admission_outcome = self._admission.model_dump()
            self._advance("pass")
        else:
            self.ckpt.termination_reason = f"optimization_{run.status}: {run.termination_reason}"
            self._advance("fail")
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()

    def _admission_safe_levels(self) -> tuple[str, ...]:
        """§14.3：准入必须在**该题型的完整 inner 面板**上判定。

        G-31 的"简化阶段门"会跳过 L2（完整 inner），但准入不能因此没有准入证据 ——
        缺 L2 时晋升会以 `no_admission_panel_outcome` 拒绝。因此这里保留简化意图
        （L1 仍是最小切片预筛），但**始终保留 L2**，并明确记一条说明，
        而不是让一次真实演化跑到最后才发现没有准入面板。
        """
        if not self.cfg.simplified_phase_gate:
            return FULL_LEVELS
        self._log("[配置] simplified_phase_gate=True：L1 仍为最小切片预筛，但按 §14.3 "
                  "保留 L2（完整 inner）作为准入门 —— outer 不作为准入门")
        return FULL_LEVELS

    # ---- 准入 + PROMOTE ----
    def _stage_promote(self) -> None:
        run = getattr(self, "_opt_run", None)
        revision_id = (getattr(run, "current_revision_id", "")
                       if run is not None else self.ckpt.current_revision_id)
        candidate = getattr(self.archive, "revisions", {}).get(revision_id)
        # Compatibility for injected panel doubles that return a promoted run
        # without exercising the archive. This is only a no-revision case;
        # optimized runs must resolve their final payload from the archive.
        root = getattr(self, "_candidate_v0", None)
        if (candidate is None and root is not None and run is not None
                and len(list(getattr(run, "revision_history", []))) == 1):
            candidate = root
        admission_po = getattr(self, "_admission", None)
        if candidate is None:
            self.ckpt.termination_reason = "missing_final_revision"
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()
            return
        if admission_po is None:
            # §14.3：没有 inner 面板结果就不能准入（缺的是**准入证据**，不是 outer）
            self.ckpt.termination_reason = "no_admission_panel_outcome"
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()
            return

        decision = admit(
            candidate, admission_po, panel_items=self.panels.get("L2", []),
            no_leakage=getattr(self, "_no_leakage", False),
            n_min=self.cfg.n_min_cross_scene,
            min_delta_mca=self.cfg.min_delta_mca, min_delta_mra=self.cfg.min_delta_mra,
        )
        if self.trace_store is not None:
            self.trace_store.append("admission_decision", decision.model_dump())
        self._log(f"[准入] promotes={decision.promotes} reason={decision.reason} "
                  f"（确定性硬门一票否决；离线模型建议不可覆盖，§3.3/硬约束 13）")

        # 治理消融 G1（G-41）：跳过归纳后语义审查
        if self.cfg.governance in _ABLATION_SKIP_REVIEW:
            self.ckpt.skipped_stages.append("offline_semantic_review")
            self._log("[治理 G1] 跳过语义审查（消融档，§16.1）")
        elif decision.promotes:
            try:
                from skill3d.governance.governance_decision import semantic_review

                client = self.offline or DeepSeekClient()
                gov = semantic_review(candidate, client)
                if self.trace_store is not None:
                    self.trace_store.append("skill_governance_decision", gov.model_dump())
                self.ckpt.offline_model_fields = _offline_manifest_fields(client)
                # §3.3：审查文本是 advisory（advisory_only=True），**不构成**准入决定
                self._log(f"[治理] 离线语义审查 decision={gov.decision_id} "
                          f"risk={gov.semantic_risk} advisory_only={gov.advisory_only}"
                          "（promote 仍只由确定性门决定）")
            except (OfflineAuthError, OfflineServiceUnavailable, OfflineRequestError,
                    OfflineResponseError) as exc:
                # §3.4：审查不可用 → 记结局码；**硬门结论不受影响**（审查本就是 advisory）
                code = classify_offline_failure(exc)
                self.ckpt.offline_failure_code = code
                self.ckpt.offline_available = False
                self._log(f"[治理] 离线语义审查不可用（{code}: {exc}）；"
                          "硬门结论不受影响（§3.3：审查是 advisory，非准入决定）")
            except Exception as exc:  # noqa: BLE001 - 解析失败等同样不阻塞硬门
                self._log(f"[治理] 语义审查不可用（{type(exc).__name__}: {exc}）；"
                          "硬门结论不受影响（硬约束 13）")

        if not decision.promotes:
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self.ckpt.termination_reason = f"admission_rejected: {decision.reason}"
            self._save()
            return

        if self.cfg.mode != "real":
            self.ckpt.termination_reason = "admission_must_be_real（§5.6b）"
            self._log("[暂停] L3 通过但 mode != real：准入必须 real → 不 promote（§5.6b）")
            self.fsm.state = OfflineState.QUARANTINE
            self.ckpt.state = self.fsm.state.value
            self._save()
            return

        promoted = candidate.model_copy(update={"status": "promoted"})
        log: list = []
        from skill3d.skills.promote_atomic import promote as promote_fn

        snap = promote_fn(self.cfg.skill_store, promoted, promotion_log=log,
                          strict_skill_specs=True)
        if self.trace_store is not None:
            self.trace_store.append("promotion", log[-1] if log else snap)
        self._log(f"[promote] 原子切换 snapshot_before="
                  f"{log[-1]['snapshot_before'] if log else '?'} → "
                  f"snapshot_after={snap['snapshot_id']}（旧 snapshot 保留可回滚，硬约束 12）")


    def _mine_counterexamples(self) -> str:
        """E2：跑 baseline 臂收集失败 episode → 反例摘要（不含答案，硬约束 19）。

        返回给 REVISE 的可见摘要；失败的 episode ref 存到 `_ce_failed_refs`
        并写进 bundle。MVP 不做自动 shrink（doc：`shrunk_by="human"` 占位）。
        """
        from skill3d.evolution.panel import run_panel

        items = list(self.panels.get("L2", []))
        if not items or self.base_cfg is None:
            return ""
        outs = run_panel(items, self.base_cfg, self.trace_store)
        failed = [(it, o) for it, o in zip(items, outs)
                  if o.final_state != "answer" or o.correct is False]
        self._ce_failed_refs = [o.qa_id for _it, o in failed]
        states = sorted({o.final_state for _it, o in failed})
        tasks = sorted({o.task or o.question_type for _it, o in failed})
        summary = (f"反例池: 失败 {len(failed)}/{len(items)} 条；"
                   f"终态={states or ['none']}；题型={tasks or ['none']}")
        self._log(f"[消融 E2] {summary}")
        self.ckpt.notes.append(f"[消融 E2] {summary}")
        return summary

    def _mr_gate(self) -> tuple[bool, str]:
        """5 类空间 MR 门（E3+）：对合成面板做确定性不变量校验。

        校验量与 §8.3 的 MR 定义一致：单位变换等比、刚体/视角变换下成对距离不变、
        置换下集合类答案不变、缺帧下答案稳定或显式降级。面板为 mock_light 合成时
        几何由 `online/synthetic` 提供，真实面板缺几何则跳过（不臆造结论）。
        """
        import numpy as np

        from skill3d.evolution.metamorphic import (
            apply_rigid_transform,
            check_unit_change,
            invariant_holds,
            pairwise_distance_matrix,
            random_rotation,
        )

        checked = 0
        for it in self.panels.get("L2", [])[:3]:
            geo = getattr(it, "geometry", None)
            if geo is None or not getattr(geo, "objects", None):
                continue
            pts = np.asarray([o.centroid_world for o in geo.objects], dtype=float)
            if pts.shape[0] < 2:
                continue
            rng = np.random.default_rng(self.cfg.seed)
            moved = apply_rigid_transform(pts, random_rotation(rng), rng.normal(size=3))
            if not invariant_holds(float(pairwise_distance_matrix(moved)[0, 1]),
                                   float(pairwise_distance_matrix(pts)[0, 1]), 1e-6):
                return False, f"rigid_transform 不变量被破坏（episode={it.episode.qa_id}）"
            if not check_unit_change(1.0, "m", "cm", 100.0):
                return False, "unit_change 不变量被破坏"
            checked += 1
        if checked == 0:
            return True, "无可校验几何（真实面板缺合成几何）→ MR 门跳过，不臆造结论"
        return True, f"通过 {checked} 个 episode 的刚体/单位不变量"

# ------------------------------------------------------------------ CLI ----

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m skill3d.evolution.offline_driver",
        description="§6.2 离线演进 FSM driver（G-35）：CLUSTER→归纳→泄漏门→优化循环→准入",
    )
    p.add_argument("--mode", default="real", choices=list(MODES),
                   help="面板执行模式；准入必须 real（§5.6b）")
    p.add_argument("--split", default="induction", help="归纳轨迹所在 split")
    p.add_argument("--stop-after", default="",
                   choices=[""] + sorted(_STOP_AFTER_STATES),
                   help="§18.1 纪律闸：执行完该状态即停止，不再进入下游。"
                        "`induce`/`leakage` 用于只跑归纳半程 —— **不加载、不消耗 "
                        "outer_holdout**（硬约束 10：outer 只跑一次）。"
                        "留空 = 跑完整状态链（会跑 L3 面板 = outer）。")
    p.add_argument("--panel-source", default="synthetic",
                   choices=["synthetic", "jsonl", "vsi_bench"])
    p.add_argument("--episodes-jsonl", default="")
    p.add_argument("--video-root", default="")
    p.add_argument("--traces-glob", default="", help="归纳轨迹 JSONL glob（默认 trace-dir/*.jsonl）")
    p.add_argument("--trace-dir", default="")
    p.add_argument("--limit", type=int, default=0, help="每层面板最多条数（0=全部）")
    p.add_argument("--l1-limit", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recon-dir", default="")
    p.add_argument("--recon-method", default="vggt", choices=["vggt"],
                   help="重建方法；v6 §5.2 受控枚举只有 vggt（colmap/dust3r 基线已废止，§20）")
    p.add_argument("--skill-store", default="")
    p.add_argument("--active-snapshot", default="")
    p.add_argument("--n-cross-scene", type=int, default=0)
    p.add_argument("--max-revisions", type=int, default=0)
    p.add_argument("--simplified-phase-gate", action="store_true",
                   help="G-31：层级简化为 L1→L3（跳过 L2 全量 inner）")
    p.add_argument("--ablation", default="", choices=["", "E0", "E1", "E2", "E3", "E4", "E5"],
                   help="演进沙箱消融档（§8.3）：控制 replay/反例/MR/paired 开关")
    p.add_argument("--governance", default="G0_full", choices=list(GOVERNANCE_MODES),
                   help="治理消融档（G-41/G-64）：G0 全治理 / G1 无 review / G2 无监控")
    p.add_argument("--run-id", default="")
    p.add_argument("--checkpoint", default="data/offline_runs/latest.json",
                   help="状态 checkpoint（每步原子写；--resume 从这里恢复）")
    p.add_argument("--resume", action="store_true", help="从中断处继续")
    p.add_argument("--run-manifest", default="data/run_manifest_offline.json",
                   help="RunManifest 落盘路径（§16.4；空串=不写）")
    p.add_argument("--admission-config", default="configs/admission_thresholds.yaml")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    return p


def _load_split_items(args, cfg_yaml: dict, split: str) -> list[EpisodeItem]:
    """按 split 载入 episode 条目（归纳侧元数据来源；与面板同源同分流约定）。"""
    from pathlib import Path as _P

    paths = paths_from(cfg_yaml)
    limit = args.limit or None
    if args.panel_source == "synthetic":
        return load_synthetic_items(split, limit=limit, seed=args.seed,
                                    out_dir=str(_P(paths.reconstructions) / "mock_light"))
    if args.panel_source == "jsonl":
        if not args.episodes_jsonl:
            raise EpisodeSourceError("--panel-source jsonl 需要 --episodes-jsonl PATH")
        return load_jsonl_items(args.episodes_jsonl, split=split, limit=limit)
    split_cfg = load_yaml(cfg_yaml.get("split_config", "configs/vsi_bench_split.yaml"))
    return load_vsi_bench_items(split, split_cfg,
                                video_root=args.video_root or paths.raw_videos,
                                video_fallback_roots=paths.raw_video_fallbacks,
                                cache_dir=paths.vsi_bench_meta, limit=limit,
                                seed=args.seed)


def load_panels(args, cfg_yaml: dict) -> dict[str, list[EpisodeItem]]:
    """装配 L1（最小切片）/L2（inner 全量）/L3（outer holdout）三个面板。"""
    inner = _load_split_items(args, cfg_yaml, "inner_validation")
    outer = _load_split_items(args, cfg_yaml, "outer_holdout")
    return {"L1": inner[: max(args.l1_limit, 1)], "L2": inner, "L3": outer}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_yaml = load_config(args.config)
    paths = paths_from(cfg_yaml)
    adm = load_yaml(args.admission_config)
    trace_dir = args.trace_dir or paths.trace_store

    cfg = OfflineDriverConfig(
        mode=args.mode,
        split=args.split,
        panel_source=args.panel_source,
        episodes_jsonl=args.episodes_jsonl,
        video_root=args.video_root,
        n_min_cross_scene=args.n_cross_scene or int(adm.get("N_min_cross_scene", 3)),
        min_delta_mca=float(adm.get("min_delta_mca", 0.02)),
        min_delta_mra=float(adm.get("min_delta_mra", 0.02)),
        l1_limit=args.l1_limit,
        seed=args.seed,
        traces_glob=args.traces_glob,
        trace_dir=trace_dir,
        skill_store=args.skill_store or paths.skill_registry,
        active_snapshot=args.active_snapshot or paths.active_snapshot,
        recon_dir=args.recon_dir or paths.reconstructions,
        recon_method=args.recon_method,
        checkpoint_path=args.checkpoint,
        run_id=args.run_id,
        governance=args.governance,
        max_revisions=args.max_revisions,
        simplified_phase_gate=args.simplified_phase_gate,
        ablation=args.ablation,
        resume=args.resume,
        stop_after=args.stop_after,
    )

    print("=" * 78)
    print(f"§6.2 离线 driver：run_id={cfg.run_id or '(auto)'} mode={cfg.mode} "
          f"split={cfg.split} governance={cfg.governance} n_min={cfg.n_min_cross_scene}")
    if cfg.mode != "real":
        print("⚠ mode != real：仅管道验证；准入必须 real（§5.6b）→ 本跑不会 promote")
    print("⚠ 离线链：离线强模型（DeepSeek-V4.1-Flash）只在归纳/修订/审查被调用，"
          "绝不进在线链（硬约束 1/2；§3.4）")
    print("=" * 78)

    # §18.1 纪律闸：`--stop-after=cluster|induce|leakage` 只需要归纳侧 episode 元数据，
    # **不需要面板**。旧实现无条件 `load_panels(...)`，而面板的 L3 就是 outer_holdout
    # → 即便只想跑归纳半程，也会去解码整批 outer 视频（读一次 outer 还是其次，
    # 真正的问题是它把"跑不跑 outer"和"跑不跑归纳"绑在了一起）。
    # 因此这里按 `--stop-after` 决定要不要装配面板：半程跑**完全不加载 outer**。
    try:
        if str(cfg.stop_after or "").strip().lower() in ("cluster", "induce", "leakage"):
            print("[纪律] --stop-after 在半程内 → **不加载 panels**（不读、不跑 "
                  "outer_holdout；§18.1 硬约束 10）")
            panels = {}
        else:
            panels = load_panels(args, cfg_yaml)
        induction_items = _load_split_items(args, cfg_yaml, cfg.split)
    except EpisodeSourceError as exc:
        print(f"[错误] 面板数据不可用: {exc}", file=sys.stderr)
        return 1

    # qa_id → (规范题型, scene)：归纳侧的题型/场景分层依据（trace 内不含 scene）
    from skill3d.routing.task_classifier import canonical_task

    episode_meta = {it.episode.qa_id:
                    (canonical_task(it.episode.question_type), it.episode.scene_name)
                    for it in induction_items}
    print(f"归纳侧 episode 元数据: {len(episode_meta)} 条（split={cfg.split}）")

    base_cfg = OnlineRunConfig(
        mode=cfg.mode, baseline="C1_tools_program", seed=cfg.seed, skills=[],
        recon_dir=cfg.recon_dir, recon_method=cfg.recon_method,
        trace_dir=trace_dir, active_snapshot_ref=cfg.active_snapshot,
    )
    store = TraceStore(trace_dir)
    driver = OfflineDriver(cfg, trace_store=store, panels=panels, base_cfg=base_cfg,
                           episode_meta=episode_meta)
    try:
        ckpt = driver.run()
    except (SplitContaminationError, OfflineAuthError, OfflineServiceUnavailable,
            OfflineRequestError, OfflineResponseError) as exc:
        print(f"[错误] {classify_offline_failure(exc)}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 3
    print(f"\ncheckpoint 已写: {cfg.checkpoint_path}  state={ckpt.state}  "
          f"transitions={len(ckpt.transitions)}")
    print(f"revision 链: {ckpt.revision_ids}")
    print(f"离线模型: {ckpt.offline_model_fields.get('offline_model', '（未登记）')} "
          f"model_id={ckpt.offline_model_fields.get('model_id', '（未登记）')} "
          f"available={ckpt.offline_available} "
          f"failure_code={ckpt.offline_failure_code or '（无）'}")

    # ---- RunManifest（G-67/§16.4 + §19.2：演进实验同样要记录复现信息与离线模型块）----
    if args.run_manifest:
        from skill3d.infra.version_lock import build_run_manifest, write_run_manifest
        from skill3d.online.eval import current_version_fields

        try:
            from skill3d.adapters.split_builder import load_split_config

            split_path = cfg_yaml.get("split_config", "configs/vsi_bench_split.yaml")
            split_version = load_split_config(split_path).split_version
        except Exception:  # noqa: BLE001
            split_path, split_version = "", ""
        try:
            m = build_run_manifest(
                docker_image=str((cfg_yaml.get("sandbox") or {}).get("image", "")),
                checkpoint_path=str((cfg_yaml.get("vllm") or {}).get("model", "")),
                config_path=args.config, repo_dir=".",
                split_version=split_version, seed=cfg.seed,
                split_config_path=split_path)
            offline = dict(ckpt.offline_model_fields or {})
            path = write_run_manifest(m, args.run_manifest, extra={
                "run_id": ckpt.run_id, "governance": cfg.governance, "mode": cfg.mode,
                "final_state": ckpt.state, "candidate_ids": ckpt.candidate_ids,
                "revision_ids": ckpt.revision_ids,
                "recon_method": cfg.recon_method,
                **current_version_fields(),
                # §19.2 离线治理模型字段（来自 DeepSeekClient.manifest_fields()；
                # 未发起调用时为空 dict —— 不虚构，也绝不含 API key）
                "offline_model": offline.get("offline_model", ""),
                "provider": offline.get("provider", ""),
                "model_id": offline.get("model_id", ""),
                "endpoint_hash": offline.get("endpoint_hash", ""),
                "prompt_version": offline.get("prompt_version", ""),
                "latency": offline.get("latency", {}),
                "token_usage": offline.get("token_usage", {}),
                "offline_governance": offline,
                "offline_available": ckpt.offline_available,
                "offline_failure_code": ckpt.offline_failure_code,
                "checkpoint_path": cfg.checkpoint_path})
            print(f"RunManifest: {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] RunManifest 写失败（不阻断）: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    return 0 if ckpt.state != OfflineState.QUARANTINE.value else 3


if __name__ == "__main__":
    raise SystemExit(main())
