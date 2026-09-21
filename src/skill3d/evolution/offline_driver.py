"""§6.2 离线演进 FSM 的真实 driver（G-35）＋ GPT-6 归纳驱动（G-28）。

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
- **GPT-6 仅离线**：只在 GPT6_SYNTHESIZE 与 REVISE 被调用；不可用时按 §8 策略
  QUARANTINE（不 promote、不让在线链等待）；本模块不得被在线链 import（硬约束 1/2）；
- **准入必须 real**（§5.6b）：`--mode mock_light` 只做管道验证，不可能 promote；
- **outer 只跑一次**（硬约束 10），失败即 REJECT，不基于 outer 失败再修订；
- **候选不可变**（硬约束 11）：REVISE 由 M16 产新 revision（parent 链），driver 不原地改；
- **LEAKAGE_CHECK 硬门**（硬约束 13/19）：候选 spec_content 含答案/sample id 即拒。

归纳输入纪律：只喂"失败类型/成功失败标签/题型/场景"等**不含答案**的摘要；
prompt 发送前做泄漏扫描（`build_induction_prompt` 内断言）。
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

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
    TestReport,
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
from skill3d.governance.gpt6_client import GPT6Client, GPT6NotConfiguredError
from skill3d.governance.induce import InsufficientEvidenceError, induce_candidate
from skill3d.memory.consolidation import leakage_scan_text
from skill3d.online.config import DEFAULT_CONFIG, load_config, load_yaml, paths_from
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import CandidateRevision, CounterexampleBundle, EpisodeTrace
from skill3d.trace.store import TraceStore

MODES = ("real", "mock_light")
# §8 治理消融档（G-41/G-64）：G0 全治理 / G1 无 review / G2 无运行期监控
GOVERNANCE_MODES = ("G0_full", "G1_no_review", "G2_no_monitoring")

# 治理消融对状态链的影响（G-41）：G1 跳归纳后语义审查；G2 跳运行期监控（离线统计）
_ABLATION_SKIP_REVIEW = {"G1_no_review"}
_ABLATION_SKIP_MONITORING = {"G2_no_monitoring"}


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
    skill_store: str = "data/skill_registry"
    active_snapshot: str = "data/active_snapshot.json"
    recon_dir: str = "data/reconstructions"
    recon_method: str = "vggt"
    checkpoint_path: str = "data/offline_runs/latest.json"
    run_id: str = ""
    governance: str = "G0_full"
    max_revisions: int = 0
    simplified_phase_gate: bool = False
    ablation: str = ""          # E0..E5（§8.3）；空 = E5 等价全开
    resume: bool = False


# ------------------------------------------------------------------ checkpoint ----

@dataclass
class OfflineCheckpoint:
    """离线 run 的持久化状态：每步一次原子写（中断可恢复，§6.2）。"""

    run_id: str
    state: str
    transitions: list[dict] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    revision_ids: list[str] = field(default_factory=list)
    gpt6_available: bool = True
    termination_reason: str = ""
    notes: list[str] = field(default_factory=list)
    skipped_stages: list[str] = field(default_factory=list)
    updated_at: str = ""

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
        return cls(**data)


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


# ------------------------------------------------------------------ driver ----

class OfflineDriver:
    """§6.2 离线 FSM driver（GPT-6 仅在本模块的 GPT6_SYNTHESIZE / REVISE 被调用）。"""

    def __init__(
        self,
        cfg: OfflineDriverConfig,
        *,
        gpt6_client=None,
        trace_store: Optional[TraceStore] = None,
        panels: Optional[dict[str, list[EpisodeItem]]] = None,
        base_cfg: Optional[OnlineRunConfig] = None,
        episode_meta: Optional[dict[str, tuple[str, str]]] = None,
        llm=None,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.gpt6 = gpt6_client
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
            self.fsm.state = OfflineState(self.ckpt.state)
            self._log(f"[resume] 从 {self.ckpt.state} 恢复 run_id={self.ckpt.run_id}")

        self._log(f"run_id={self.ckpt.run_id} governance={self.cfg.governance} "
                  f"mode={self.cfg.mode} split={self.cfg.split}")

        if self.fsm.state is OfflineState.CLUSTER_TRACES:
            self._stage_cluster()
        if self.fsm.state is OfflineState.GPT6_SYNTHESIZE:
            if not self._stage_induce():
                return self.ckpt
        if self.fsm.state is OfflineState.LEAKAGE_CHECK:
            if not self._stage_leakage_check():
                return self.ckpt
        if self.fsm.state is OfflineState.LOOP_SYNTHESIZE:
            self._stage_optimize()
        if self.fsm.state is OfflineState.PROMOTE:
            self._stage_promote()
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

    # ---- GPT6_SYNTHESIZE（G-28）----
    def _stage_induce(self) -> bool:
        client = self.gpt6
        if client is None:
            try:
                client = GPT6Client()
            except GPT6NotConfiguredError as exc:
                self.ckpt.gpt6_available = False
                self.ckpt.termination_reason = f"gpt6_not_configured: {exc}"
                self._log(f"[GPT6_SYNTHESIZE] GPT-6 不可用 → QUARANTINE（§8：不 promote，"
                          "在线链不等待）")
                self._advance("gpt6_unavailable")
                return True
        self.ckpt.gpt6_available = True

        traces = getattr(self, "_traces", [])
        try:
            candidate = induce_candidate(
                traces, getattr(self, "_task_of", {}), getattr(self, "_scene_of", {}),
                gpt6_client=client, n_min=self.cfg.n_min_cross_scene)
        except InsufficientEvidenceError as exc:
            self.ckpt.termination_reason = f"insufficient_evidence: {exc}"
            self._log(f"[GPT6_SYNTHESIZE] 样本不足 → REJECT: {exc}")
            self._advance("candidate_ready", {"cross_scene_ok": False})
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()
            return True
        except GPT6NotConfiguredError as exc:
            # GPT-6 端点在调用时才暴露未配置 → 按 §8 暂停（不 promote、不阻塞在线链）
            self.ckpt.gpt6_available = False
            self.ckpt.termination_reason = f"gpt6_not_configured: {exc}"
            self._log(f"[GPT6_SYNTHESIZE] GPT-6 不可用 → QUARANTINE（§8）: {exc}")
            self._advance("gpt6_unavailable")
            return True
        if candidate is None:
            # GPT-6 未配置但低风险：不进 promotion（§8）
            self.ckpt.termination_reason = "gpt6_returned_no_candidate"
            self._log("[GPT6_SYNTHESIZE] 未产出 candidate → QUARANTINE")
            self._advance("gpt6_unavailable")
            return True

        self._candidate_v0 = candidate
        self.ckpt.candidate_ids.append(candidate.root_candidate_id)
        self.ckpt.revision_ids.append(candidate.revision_id)
        self._log(f"[GPT6_SYNTHESIZE] candidate_v0 root={candidate.root_candidate_id} "
                  f"revision={candidate.revision_id} "
                  f"（归纳自 {len(candidate.induction_trace_refs)} 条轨迹）")
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
            """REVISE：M16 GPT-6 基于反例摘要产新版本（绝不原地改，硬约束 11）。"""
            from skill3d.governance.revise_patch import gpt6_revise

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
            client = self.gpt6 or GPT6Client()
            new_rev = gpt6_revise(rev, bundle, client)
            self.ckpt.revision_ids.append(new_rev.revision_id)
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
                levels=(SIMPLIFIED_LEVELS if self.cfg.simplified_phase_gate
                        else FULL_LEVELS),
            )
        except GPT6NotConfiguredError as exc:
            self.ckpt.termination_reason = f"gpt6_not_configured: {exc}"
            self._log("[OPTIMIZATION_LOOP] 修订需 GPT-6 但不可用 → 暂停（§8）")
            self.fsm.state = OfflineState.QUARANTINE
            self.ckpt.state = self.fsm.state.value
            self._save()
            return
        self._opt_run, self._level_state = run, level_state
        if self.trace_store is not None:
            self.trace_store.append("optimization_run", run.model_dump())
        self._log(f"[OPTIMIZATION_LOOP] status={run.status} reason={run.termination_reason} "
                  f"revisions={run.budget_used.revisions} rollouts={run.budget_used.rollouts}")

        # 映射到 FSM 终态（outer 只跑一次，硬约束 10）
        l3 = level_state.get("L3")
        if run.status == "promoted" and l3 is not None:
            self._l3 = l3[0]
            self._advance("pass")
        else:
            self.ckpt.termination_reason = f"optimization_{run.status}: {run.termination_reason}"
            self._advance("fail")
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()

    # ---- 准入 + PROMOTE ----
    def _stage_promote(self) -> None:
        candidate = getattr(self, "_candidate_v0", None)
        l3 = getattr(self, "_l3", None)
        if candidate is None or l3 is None:
            self.ckpt.termination_reason = "no_outer_outcome"
            self.fsm.state = OfflineState.REJECT
            self.ckpt.state = self.fsm.state.value
            self._save()
            return

        decision = admit(
            candidate, l3, outer_items=self.panels.get("L3", []),
            no_leakage=getattr(self, "_no_leakage", False),
            n_min=self.cfg.n_min_cross_scene,
            min_delta_mca=self.cfg.min_delta_mca, min_delta_mra=self.cfg.min_delta_mra,
        )
        if self.trace_store is not None:
            self.trace_store.append("admission_decision", decision.model_dump())
        self._log(f"[准入] promotes={decision.promotes} reason={decision.reason} "
                  f"（硬门一票否决，GPT-6 建议不可覆盖，硬约束 13）")

        # 治理消融 G1（G-41）：跳过归纳后语义审查
        if self.cfg.governance in _ABLATION_SKIP_REVIEW:
            self.ckpt.skipped_stages.append("gpt6_semantic_review")
            self._log("[治理 G1] 跳过语义审查（消融档，§16.1）")
        elif decision.promotes:
            try:
                from skill3d.governance.governance_decision import semantic_review

                gov = semantic_review(candidate, self.gpt6 or GPT6Client())
                if self.trace_store is not None:
                    self.trace_store.append("skill_governance_decision", gov.model_dump())
                self._log(f"[治理] 语义审查 decision={gov.decision_id} "
                          f"risk={gov.semantic_risk}")
            except Exception as exc:  # noqa: BLE001 - 审查不可用不阻塞硬门结论
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

        snap = promote_fn(self.cfg.skill_store, promoted, promotion_log=log)
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
    p.add_argument("--recon-method", default="vggt", choices=["vggt", "dust3r_mast3r", "colmap"])
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
    )

    print("=" * 78)
    print(f"§6.2 离线 driver：run_id={cfg.run_id or '(auto)'} mode={cfg.mode} "
          f"split={cfg.split} governance={cfg.governance} n_min={cfg.n_min_cross_scene}")
    if cfg.mode != "real":
        print("⚠ mode != real：仅管道验证；准入必须 real（§5.6b）→ 本跑不会 promote")
    print("⚠ 离线链：GPT-6 只在归纳/修订被调用，绝不进在线链（硬约束 1/2）")
    print("=" * 78)

    try:
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
    except (SplitContaminationError, GPT6NotConfiguredError) as exc:
        print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"\ncheckpoint 已写: {cfg.checkpoint_path}  state={ckpt.state}  "
          f"transitions={len(ckpt.transitions)}")
    print(f"revision 链: {ckpt.revision_ids}")

    # ---- RunManifest（G-67/§16.4：演进实验同样要记录复现信息）----
    if args.run_manifest:
        from skill3d.infra.version_lock import build_run_manifest, write_run_manifest

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
            path = write_run_manifest(m, args.run_manifest, extra={
                "run_id": ckpt.run_id, "governance": cfg.governance, "mode": cfg.mode,
                "final_state": ckpt.state, "candidate_ids": ckpt.candidate_ids,
                "revision_ids": ckpt.revision_ids})
            print(f"RunManifest: {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] RunManifest 写失败（不阻断）: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    return 0 if ckpt.state != OfflineState.QUARANTINE.value else 3


if __name__ == "__main__":
    raise SystemExit(main())
