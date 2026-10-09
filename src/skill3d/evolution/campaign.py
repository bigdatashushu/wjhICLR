"""P2 / v10 §11：`EvolutionCampaign` —— 两代演化的**唯一调度入口**。

规范原文（§11.1 状态机）：

```
INIT
→ RUN_PARENT_LEARNING
→ BUILD_EXPERIENCE_BUNDLE
→ GENERATE_CANDIDATE
→ STATIC_VALIDATE
→ BUILD_CANDIDATE_SNAPSHOT
→ RUN_INNER_SEED_0
→ RUN_INNER_SEED_1
→ DECIDE
→ PUBLISH_OR_REJECT
→ VERIFY_POST_PUBLISH_USE
→ NEXT_GENERATION
→ COMPLETE
```

"任何状态失败都写 checkpoint；恢复时不得重复消费已经完成的模型调用、面板或发布动作。"

本模块回答 §2.2 的三个问题，并把 §14.1 的十类收据逐件落盘：

1. **积累是否发生** —— RUN_PARENT_LEARNING 用**正常检索**跑父快照，经验按
   `skill_id@version` 建 ExperienceEvent / ExperienceBundle（P1 的
   `evolution/experience.py`）；
2. **迭代是否发生** —— GENERATE_CANDIDATE 只接受"有父版本声明"的完整候选
   （`SkillCandidate`，§7.2）；
3. **复用是否发生** —— VERIFY_POST_PUBLISH_USE 用**新快照 + 正常检索**证明新版本被
   检索、被交付、正文 hash 一致，并产出指向新版本的经验事件；不满足就只能记
   `promoted_not_observed` 且**不得**开始下一代归纳（§10.2）。

两代定义（§11.3）：第二代必须从**第一代之后**产生的真实运行经验出发；不能把第一代
同一批 trace 再提交一次，也不能把同一候选内部的多次 revision 算作第二代。本实现用
"每代自己的 run_id + trace 目录 + snapshot" 把这件事变成结构性事实，而不是纪律口号。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from skill3d.evolution.experience import (
    usage_state_of,
    EpisodeEvidence,
    ExperienceLedgerError,
    SPLIT_ALIASES,
    build_experience_bundle,
    build_experience_events,
    evidence_from_trace_store,
    write_bundle_json,
    write_events_jsonl,
)
from skill3d.evolution.panel import (
    decide_promotion,
    run_fixed_skill_evaluation,
)
from skill3d.governance.induce import (
    INDUCE_PROMPT_VERSION,
    CandidateStaticCheckError,
    InsufficientEvidenceError,
    build_induction_prompt,
    induce_candidate_from_bundle,
    validate_candidate_against_parent,
)
from skill3d.online.runner import OnlineRunConfig, run_episode
from skill3d.routing.skill_retriever import canonical_question_type
from skill3d.schemas import (
    CampaignDecision,
    ExperienceBundle,
    EvolutionCampaign,
    PairedPanelReceipt,
    PostPublishUseReceipt,
    PromotionReceipt,
    RejectionReceipt,
    SkillCandidate,
    SkillEvaluationBinding,
    SkillSpec,
    StaticValidationReceipt,
)
from skill3d.skills.delivery import skill_content_sha256
from skill3d.skills.promote_atomic import (
    build_candidate_snapshot,
    canonical_digest,
    publish_candidate_snapshot,
    read_active_snapshot,
)
from skill3d.skills.registry import (
    active_snapshot_provenance,
    load_legacy_active_skills,
)
from skill3d.trace.store import TraceStore

# 收据文件名（§14.1 逐项对应；缺任一 → 该代 `incomplete`）
RECEIPTS: tuple[str, ...] = (
    "parent_run_manifest.json", "experience_events.jsonl", "experience_bundle.json",
    "inducer_prompt.txt", "inducer_receipt.json",
    "candidate.json", "static_validation.json", "candidate_snapshot.json",
    "paired_seed_0.json", "paired_seed_1.json", "decision.json", "promotion.json",
    "post_publish_use.json",
)


def _offline_failure_classes() -> tuple:
    """延迟导入：campaign 不 import 在线/离线客户端以外的治理实现细节。"""
    from skill3d.governance.deepseek_client import (
        OfflineAuthError,
        OfflineRequestError,
        OfflineResponseError,
        OfflineServiceUnavailable,
    )

    return (OfflineAuthError, OfflineServiceUnavailable, OfflineRequestError,
            OfflineResponseError)


def classify_offline_failure(exc: BaseException) -> str:
    """§3.4 结局码（与 v9 driver 同族命名，便于跨版本对账）。"""
    from skill3d.governance.deepseek_client import (
        OfflineAuthError,
        OfflineRequestError,
        OfflineResponseError,
        OfflineServiceUnavailable,
    )

    if isinstance(exc, OfflineAuthError):
        return "offline_auth_error"
    if isinstance(exc, OfflineServiceUnavailable):
        return "service_unavailable"
    if isinstance(exc, (OfflineRequestError, OfflineResponseError)):
        return "offline_request_error"
    return "service_unavailable"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# §3.4 的离线失败族（服务不可用 / 鉴权 / 请求 / 响应）——统一映射为 blocked 结局码
OFFLINE_FAILURES: tuple = _offline_failure_classes()


def _promotion_outcome(payload: dict) -> str:
    """`promotion.json` 的形态判定（§14.1 的 promote / reject 两态）。

    历史兼容：本轮早期版本把 reject 写成 `{"promoted": false, ...}`（没有 `outcome`
    字段）——那种记录同样按 reject 解释，不因为字段缺失就当成 promote。
    """
    outcome = str(payload.get("outcome", "") or "")
    if outcome in ("promoted", "rejected"):
        return outcome
    return "rejected" if payload.get("promoted") is False else "promoted"


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _offline_identity(client) -> dict:
    """离线模型身份（§19.2 口径；**绝不含密钥/URL 本体**）。缺字段就不编造。"""
    fields: dict = {}
    try:
        raw = client.manifest_fields() if hasattr(client, "manifest_fields") else {}
        if isinstance(raw, dict):
            fields = {k: v for k, v in raw.items()
                      if k in ("offline_model", "provider", "model_id", "endpoint_hash",
                               "prompt_version", "latency", "token_usage")}
    except Exception:  # noqa: BLE001 - 身份元数据缺失不得阻断演化
        fields = {}
    return fields


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (value.model_dump(mode="json") if hasattr(value, "model_dump") else value)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@dataclass
class CampaignConfig:
    """一次两代 campaign 的配置（阈值一律来自 configs/*.yaml）。"""

    campaign_id: str = ""
    target_question_type: str = "object_counting"
    target_skill_id: str = "S01"
    # §14.2/§14.3：真正读取配置里的轮数与 seed 集合（v9 的"死配置"问题）
    max_generations: int = 2
    validation_seeds: tuple[int, ...] = (0, 1)
    mode: str = "real"                       # real | mock_light（mock 不得 promote）
    # 目标 Skill 库（含 snapshots/ manifests/）。**不要**指向仓库的 S0 库：
    # campaign 在自己的库里演进，S0 的 `--check` 不变量保持不变。
    library_root: str = "data/evolution/library"
    run_root: str = "data/evolution/runs"
    learning_split: str = "induction"
    inner_split: str = "inner_validation"
    learning_limit: int = 32                 # 每次 learning 真实运行的题数上限
    post_publish_probe_limit: int = 4        # §10.2 发布后验证用的小批新 episode
    inner_panel_limit: int = 16              # 每个 inner 子面板题数上限
    min_eligible_experiences: int = 3        # N_min（TODO_CALIBRATE）
    method_context_max_chars: int = 8000
    max_competing_per_lineage: int = 2
    max_revisions: int = 2                   # §7.4 超过最大 revision 数 → reject
    # §7.4：最大 revision（归纳/修订）次数；超过即 reject
    max_candidate_attempts: int = 3
    # 首期纪律：只用 learning/induction 与 inner_validation，不跑 outer/final（§0/§8.5）
    allow_outer_holdout: bool = False
    resume: bool = True
    # §13.5：检索策略（top-k / 排序权重 / 方法上下文上限）必须来自**冻结配置**且在
    # 演化开始前定好；None = 配置缺省值（检索记录里会如实记 config_source="default"）。
    retrieval_policy: object = None

    def resolved_id(self) -> str:
        return self.campaign_id or f"camp-{uuid.uuid4().hex[:10]}"


class CampaignBlocked(RuntimeError):
    """campaign 无法继续（离线模型不可用、经验不足、面板缺失等）→ blocked 终态。"""


class CampaignRuntime:
    """真实运行实现（默认）。测试可注入替代实现，避免真跑模型/工具。"""

    def run_learning(self, *, items, run_cfg: OnlineRunConfig, trace_dir: str,
                     llm=None):
        """用**正常检索**跑一批 learning episode（返回 outcomes）。

        `llm` 必须一路传进 `run_episode`：§8.4 的两阶段版本选择要用它发选择请求，
        而 `OnlineRunConfig` 里没有 endpoint 时 runner 不会自建客户端。
        """
        store = TraceStore(trace_dir)
        return [run_episode(it.episode, it.pixels, run_cfg, geometry=it.geometry,
                            trace_store=store, llm=llm) for it in items]

    def run_inner(self, *, campaign_id: str, generation: int, seed: int, panel_id: str,
                  items, parent_spec: SkillSpec, candidate_spec: SkillSpec,
                  base_cfg: OnlineRunConfig, trace_store=None, llm=None):
        """§8.2 固定注入 A/B（父臂 / 候选臂）。"""
        return run_fixed_skill_evaluation(
            campaign_id=campaign_id, generation=generation, seed=seed,
            panel_id=panel_id, items=items, parent_spec=parent_spec,
            candidate_spec=candidate_spec, base_cfg=base_cfg, trace_store=trace_store,
            llm=llm)


class EvolutionCampaignRunner:
    """§11 状态机的执行体。`run()` 返回终态 `EvolutionCampaign`。"""

    def __init__(self, cfg: CampaignConfig, *, runtime=None, offline_client=None,
                 offline_client_factory: Optional[Callable[[], object]] = None,
                 llm=None, panels_provider: Optional[Callable[[str, int], list]] = None,
                 tool_names: Sequence[str] = (), qa_ids_by_split: Optional[dict] = None,
                 print_fn: Callable[[str], None] = print):
        self.cfg = cfg
        self.campaign_id = cfg.resolved_id()
        self.runtime = runtime or CampaignRuntime()
        self.offline_client = offline_client
        self.offline_client_factory = offline_client_factory
        self.llm = llm
        self.panels_provider = panels_provider
        self.tool_names = tuple(tool_names)
        self.qa_ids_by_split = dict(qa_ids_by_split or {})
        self.print = print_fn
        self.root = Path(cfg.run_root) / self.campaign_id
        self.store_dir = Path(cfg.library_root) / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self._generation = 0
        self._stop_reason = ""
        self._completion_note = ""
        self._receipts_by_gen: dict[int, list[str]] = {}
        # 上一代晋升的版本键（§11.3：下一代的父版本 = 上一代晋升的那个版本）
        self._promoted_skill_key = ""
        self._promoted_generations: list[int] = []
        self._campaign = EvolutionCampaign(
            campaign_id=self.campaign_id,
            target_question_type=cfg.target_question_type,
            target_skill_id=cfg.target_skill_id,
            max_generations=int(cfg.max_generations),
            current_generation=0,
            initial_snapshot_id="",
            current_parent_snapshot_id="",
            state="INIT",
        )

    # ------------------------------------------------------------------ 基础设施
    def _checkpoint_path(self) -> Path:
        return self.root / "campaign.json"

    def _gen_dir(self, generation: int) -> Path:
        return self.root / f"gen{generation}"

    def _receipt(self, generation: int, name: str) -> Path:
        return self._gen_dir(generation) / name

    def _set_state(self, state: str, **ctx) -> None:
        self._campaign.state = state
        self._save_checkpoint()
        detail = " ".join(f"{k}={v}" for k, v in ctx.items())
        self.print(f"[{self.campaign_id}] {state} {detail}".rstrip())

    def _save_checkpoint(self) -> None:
        self._campaign.current_generation = int(self._generation)
        payload = {
            "campaign": self._campaign.model_dump(mode="json"),
            "stop_reason": self._stop_reason,
            "completion_note": self._completion_note,
            "internal": {
                "generation": self._generation,
                "receipts": {str(k): list(v) for k, v in self._receipts_by_gen.items()},
            },
        }
        _write_json(self._checkpoint_path(), payload)

    def _load_checkpoint(self) -> bool:
        payload = _read_json(self._checkpoint_path())
        if not payload or "campaign" not in payload:
            return False
        try:
            self._campaign = EvolutionCampaign.model_validate(payload["campaign"])
        except Exception as exc:  # noqa: BLE001 - 损坏的 checkpoint 不得静默继续
            raise CampaignBlocked(f"checkpoint 损坏，拒绝恢复: {exc}") from exc
        self._stop_reason = str(payload.get("stop_reason", ""))
        self._completion_note = str(payload.get("completion_note", ""))
        self._generation = int(payload.get("internal", {}).get("generation",
                                                              self._campaign.current_generation))
        for gen, names in (payload.get("internal", {}).get("receipts") or {}).items():
            self._receipts_by_gen[int(gen)] = list(names)
        return True

    def _record_receipt(self, generation: int, name: str) -> None:
        names = self._receipts_by_gen.setdefault(int(generation), [])
        if name not in names:
            names.append(name)

    def _has_receipt(self, generation: int, name: str) -> bool:
        return self._receipt(generation, name).is_file()

    def _finish(self, status: str, note: str = "") -> EvolutionCampaign:
        """收口：写终态与**完成说明**。

        `_stop_reason` 只承载"阻断性原因"（blocked/failed/发布不可见…）——它是**恢复时
        的闸门**，一旦被写上就再也不会继续跑代。完成说明另存 `completion_note`：
        否则"两代都未晋升"这类**正常结论**会在 resume 时被当成阻断原因，把第二代
        挡在门外（本轮真实踩到过）。
        """
        self._campaign.completion_status = status  # type: ignore[assignment]
        self._campaign.state = "COMPLETE"
        self._completion_note = note or self._completion_note
        self._campaign.final_snapshot_id = str(
            read_active_snapshot(self.store_dir).get("snapshot_id"))
        self._save_checkpoint()
        self.print(f"[{self.campaign_id}] COMPLETE status={status} {note}".rstrip())
        return self._campaign

    # ------------------------------------------------------------------ 面板
    def _items(self, kind: str, generation: int) -> list:
        if self.panels_provider is None:
            raise CampaignBlocked(
                f"缺少 panels_provider：{kind} 面板必须由调用方冻结后注入（§8.5）")
        items = list(self.panels_provider(kind, int(generation)) or [])
        if not items:
            raise CampaignBlocked(f"{kind} 面板为空（§8.5：面板清单必须先冻结）")
        return items

    def _parent_skill(self) -> SkillSpec:
        """本代的父 Skill：**最新晋升的在线版本**（§11.3）。

        §11.3 规定第二代"以其（晋升后版本的）经验修订为 S01@1.2.0"。晋升后新旧版本
        共同竞争（§5.3），快照里同时存在 1.0.0 与 1.1.0，所以"父版本"必须由本代明确
        指认为**上一代晋升的那个版本**；没有晋升记录时（第一代）取谱系内最高版本的
        在线条目（确定性：版本号只用于选父，不参与检索排序，§5.3-6）。
        """
        skills, warnings, snapshot_id = load_legacy_active_skills(self.store_dir)
        candidates = [s for s in skills if s.skill_id == self.cfg.target_skill_id]
        if not candidates:
            raise CampaignBlocked(
                f"active 快照 {snapshot_id} 里没有 {self.cfg.target_skill_id}"
                f"（warnings={warnings}）")
        for spec in candidates:
            if list(spec.applicable_question_types) != [self.cfg.target_question_type]:
                raise CampaignBlocked(
                    f"{spec.skill_id}@{spec.version} 的题型 "
                    f"{spec.applicable_question_types} ≠ 目标题型 "
                    f"{self.cfg.target_question_type}（§8.5）")
        if self._promoted_skill_key:
            for spec in candidates:
                if f"{spec.skill_id}@{spec.version}" == self._promoted_skill_key:
                    return spec
            raise CampaignBlocked(
                f"上一代晋升的 {self._promoted_skill_key} 不在 active 快照 "
                f"{snapshot_id} 里（不得改用别的版本充当父版本）")
        return max(candidates, key=lambda s: _parse_version(s.version))

    def _base_cfg(self) -> OnlineRunConfig:
        from skill3d.routing.retrieval_policy import RetrievalPolicy

        snapshot_id, manifest_hash = active_snapshot_provenance(self.store_dir)
        return OnlineRunConfig(
            mode=self.cfg.mode, baseline="C1_tools_program",
            active_snapshot_ref=snapshot_id,
            active_snapshot_manifest_sha256=str(manifest_hash or ""),
            trace_dir=str(self.root / "traces"),
            retrieval_policy=(self.cfg.retrieval_policy or RetrievalPolicy()),
        )

    # ------------------------------------------------------------------ 主流程
    def run(self) -> EvolutionCampaign:
        active = read_active_snapshot(self.store_dir)
        if active.get("schema_version") == "runtime-skill-snapshot/2.0":
            raise CampaignBlocked(
                "v10 evolution campaign 不支持当前 SkillSpecV11 快照；"
                "请等待 v11 四阶段演化迁移，不得用旧 JSON SkillSpec 流程修改 active")
        if self.cfg.resume and self._load_checkpoint():
            self.print(f"[{self.campaign_id}] 从 checkpoint 恢复 state="
                       f"{self._campaign.state} generation={self._generation}")
        else:
            snap = active
            self._campaign.initial_snapshot_id = str(snap.get("snapshot_id", ""))
            self._campaign.current_parent_snapshot_id = str(snap.get("snapshot_id", ""))
            self._set_state("INIT", snapshot=self._campaign.initial_snapshot_id)

        for generation in range(1, int(self.cfg.max_generations) + 1):
            if self._stop_reason:
                break
            self._generation = generation
            self._run_generation(generation)

        if self._stop_reason:
            status = "blocked" if self._stop_reason.startswith("blocked") else (
                "failed" if self._stop_reason.startswith("failed")
                else "completed_with_rejection")
            return self._finish(status, self._stop_reason)
        if not self._promoted_generations:
            return self._finish(
                "completed_with_rejection",
                "两代都未晋升：闭环跑通但未实现版本积累（§17.2）")
        if (self._campaign.current_generation >= int(self.cfg.max_generations)
                and int(self.cfg.max_generations) >= 2):
            return self._finish("completed_two_generations",
                                "两代流程走完（见各代收据与决定）")
        # 只跑了部分代（如先用 --max-generations 1 验证第一代）：如实记 running，
        # 不把"跑了一代"说成"两代完成"。
        return self._finish(
            "running",
            f"已晋升但只跑到第 {self._campaign.current_generation} 代"
            f"（max_generations={self.cfg.max_generations}）；用 --max-generations 继续")
        return self._finish("running", "未达 max_generations 但已无待执行代")

    # ---- 单代 ----
    def _run_generation(self, generation: int) -> None:
        parent_snapshot = str(read_active_snapshot(self.store_dir).get("snapshot_id"))
        parent_spec = self._parent_skill()
        parent_key = f"{parent_spec.skill_id}@{parent_spec.version}"
        self._campaign.current_parent_snapshot_id = parent_snapshot
        self.print(f"===== generation {generation}: parent snapshot={parent_snapshot} "
                   f"skill={parent_key} =====")

        # ---- RUN_PARENT_LEARNING ----
        self._set_state("RUN_PARENT_LEARNING", generation=generation)
        learning = self._run_parent_learning(generation, parent_snapshot)

        # ---- BUILD_EXPERIENCE_BUNDLE ----
        self._set_state("BUILD_EXPERIENCE_BUNDLE", generation=generation)
        bundle = self._build_bundle(generation, parent_snapshot, parent_key, learning)
        if bundle.n_eligible < int(self.cfg.min_eligible_experiences):
            self._stop_reason = (
                f"blocked:insufficient_experience eligible={bundle.n_eligible} "
                f"< N_min={self.cfg.min_eligible_experiences}（§6.2）")
            self._save_checkpoint()
            return

        # ---- GENERATE_CANDIDATE ----
        self._set_state("GENERATE_CANDIDATE", generation=generation)
        candidate = self._generate_candidate(generation, parent_snapshot, parent_spec, bundle)
        if candidate is None:
            return

        # ---- STATIC_VALIDATE ----
        self._set_state("STATIC_VALIDATE", generation=generation)
        receipt = self._static_validate(generation, parent_spec, candidate, bundle)
        if receipt is None or not receipt.passed:
            self._stop_reason = "blocked:candidate_static_validation_failed"
            self._save_checkpoint()
            return

        # ---- BUILD_CANDIDATE_SNAPSHOT ----
        self._set_state("BUILD_CANDIDATE_SNAPSHOT", generation=generation)
        snapshot_path = self._receipt(generation, "candidate_snapshot.json")
        if not snapshot_path.is_file():
            snapshot, _change = build_candidate_snapshot(
                read_active_snapshot(self.store_dir), candidate,
                max_competing_per_lineage=int(self.cfg.max_competing_per_lineage))
            _write_json(snapshot_path, snapshot)
            self._record_receipt(generation, "candidate_snapshot.json")
        self._save_checkpoint()

        # ---- RUN_INNER_SEED_* ----
        inner_items = self._items("inner", generation)
        receipts: list[PairedPanelReceipt] = []
        for seed in self.cfg.validation_seeds:
            self._set_state(f"RUN_INNER_SEED_{seed}", generation=generation)
            receipts.append(self._run_inner_seed(
                generation, int(seed), inner_items, parent_spec, candidate))
        # 严格按 §8.6 的 seed 集合校验（配置里给几个 seed 就必须有几个结果）
        if sorted(r.seed for r in receipts) != sorted(int(s) for s in self.cfg.validation_seeds):
            self._stop_reason = "failed:inner_seed_receipts_incomplete"
            self._save_checkpoint()
            return

        # ---- DECIDE ----
        self._set_state("DECIDE", generation=generation)
        decision = self._decide(generation, candidate, receipts)

        # ---- PUBLISH_OR_REJECT ----
        self._set_state("PUBLISH_OR_REJECT", generation=generation)
        promoted = self._publish_or_reject(generation, candidate, decision)

        # ---- VERIFY_POST_PUBLISH_USE ----
        self._set_state("VERIFY_POST_PUBLISH_USE", generation=generation)
        use = self._verify_post_publish(generation, parent_spec, candidate, promoted)
        self._campaign.current_generation = generation

        if promoted and use.status != "observed":
            self._stop_reason = (
                "promoted_not_observed: 发布成功但新版本未在后续真实运行中被检索+交付"
                "（§10.2 不得开始下一代归纳）")
            self._save_checkpoint()
            return

        # ---- NEXT_GENERATION ----
        self._set_state("NEXT_GENERATION", generation=generation)
        ref = str(self._gen_dir(generation))
        if ref not in self._campaign.generation_receipt_refs:
            self._campaign.generation_receipt_refs.append(ref)
        if not promoted:
            self.print(f"[{self.campaign_id}] 第 {generation} 代 reject："
                       "下一代将基于父版本的新经验重试（§11.3）")
        self._save_checkpoint()

    # ---- 各状态实现 ----
    def _run_parent_learning(self, generation: int, parent_snapshot: str) -> dict:
        """RUN_PARENT_LEARNING：父快照 + **正常检索**的真实运行（§8.4）。"""
        manifest_path = self._receipt(generation, "parent_run_manifest.json")
        trace_dir = str(self.root / "traces" / f"gen{generation}" / "learning")
        manifest = _read_json(manifest_path)
        if manifest is not None:
            self.print(f"[{self.campaign_id}] 复用已完成的 learning 运行 "
                       f"run_id={manifest.get('run_id')}（不重复消费面板）")
            self._record_receipt(generation, "parent_run_manifest.json")
            return manifest
        items = self._items("learning", generation)
        run_id = f"{self.campaign_id}-g{generation}-learning"
        cfg = self._base_cfg()
        skills, warnings, snapshot_id = load_legacy_active_skills(self.store_dir)
        if snapshot_id != parent_snapshot:
            raise CampaignBlocked(
                f"learning 运行前的 active 快照 {snapshot_id} ≠ 期望父快照 {parent_snapshot}")
        cfg = _replace_cfg(cfg, skills=skills, trace_dir=trace_dir)
        if warnings:
            self.print(f"[{self.campaign_id}] loader 警告: {warnings[:3]}")
        outcomes = self.runtime.run_learning(items=items, run_cfg=cfg,
                                             trace_dir=trace_dir, llm=self.llm)
        manifest = {
            "run_id": run_id,
            "campaign_id": self.campaign_id,
            "generation": int(generation),
            "split": self.cfg.learning_split,
            "trace_dir": trace_dir,
            "snapshot_id": snapshot_id,
            "manifest_hash": self._manifest_hash(),
            "n_items": len(items),
            "qa_ids": [str(it.episode.qa_id) for it in items],
            # 逐题 scene 身份随运行清单落盘：经验包的"跨 scene 样本数"必须来自这次
            # 真实运行的题目清单，而不是外部注入的映射（否则可能是别的运行的事实）。
            "episodes": [{"qa_id": str(it.episode.qa_id),
                          "scene_id": str(it.episode.scene_name)} for it in items],
            "panel_hash": _items_hash(items),
            "skills_loaded": [f"{s.skill_id}@{s.version}" for s in skills],
            "mode": self.cfg.mode,
            "seed": int(self._base_cfg().seed),
            "retrieval_policy_version": cfg.retrieval_policy.version(),
            "retrieval_policy_sha256": cfg.retrieval_policy.sha256(),
            "code_state": code_state(),
            "created_at": _now_iso(),
        }
        _write_json(manifest_path, manifest)
        self._record_receipt(generation, "parent_run_manifest.json")
        self._save_checkpoint()
        self.print(f"[{self.campaign_id}] learning 运行完成 run_id={run_id} "
                   f"n={len(items)} snapshot={snapshot_id}")
        return manifest

    def _build_bundle(self, generation: int, parent_snapshot: str, parent_key: str,
                      manifest: dict) -> ExperienceBundle:
        bundle_path = self._receipt(generation, "experience_bundle.json")
        existing = _read_json(bundle_path)
        if existing is not None:
            bundle = ExperienceBundle.model_validate(existing)
            self._record_receipt(generation, "experience_events.jsonl")
            self._record_receipt(generation, "experience_bundle.json")
            self.print(f"[{self.campaign_id}] 复用经验包 {bundle.bundle_id} "
                       f"eligible={bundle.n_eligible}")
            return bundle
        trace_dir = Path(manifest.get("trace_dir")
                         or (self.root / "traces" / f"gen{generation}" / "learning"))
        episodes_meta = list(manifest.get("episodes") or [])
        qa_ids = [str(q) for q in (manifest.get("qa_ids")
                                   or [m.get("qa_id") for m in episodes_meta])]
        split = str(manifest.get("split") or self.cfg.learning_split)
        canon = SPLIT_ALIASES.get(split, split)
        split_of = {qa: canon for qa in qa_ids}
        evidence = evidence_from_trace_store(
            trace_dir, split_of=split_of,
            snapshot_id=str(manifest.get("snapshot_id", "")),
            skills_in_scope=[parent_key])
        scene_of = {str(m.get("qa_id")): str(m.get("scene_id"))
                    for m in episodes_meta}
        for ev in evidence:
            ev.scene_id = str(scene_of.get(ev.episode_id, ev.scene_id))
        events = build_experience_events(
            campaign_id=self.campaign_id, generation=int(generation),
            parent_snapshot_id=parent_snapshot, skills=[parent_key], episodes=evidence)
        write_events_jsonl(events, self._receipt(generation, "experience_events.jsonl"))
        bundle = build_experience_bundle(
            campaign_id=self.campaign_id, generation=int(generation),
            parent_snapshot_id=parent_snapshot, parent_skill_key=parent_key,
            canonical_question_type=self.cfg.target_question_type, events=events,
            source_manifest_hash=str(manifest.get("manifest_hash", "")),
            evidence_failure_codes={ev.episode_id: ev.failure_code for ev in evidence},
            evidence_mra={ev.episode_id: ev.mra_value for ev in evidence
                          if ev.mra_value is not None})
        write_bundle_json(bundle, bundle_path)
        self._record_receipt(generation, "experience_events.jsonl")
        self._record_receipt(generation, "experience_bundle.json")
        self._save_checkpoint()
        self.print(f"[{self.campaign_id}] 经验包 eligible={bundle.n_eligible} "
                   f"scenes={bundle.scene_count} 成功={bundle.success_count} "
                   f"失败={bundle.failure_count} 排除={bundle.exclusion_summary}")
        return bundle

    def _offline(self):
        if self.offline_client is not None:
            return self.offline_client
        if self.offline_client_factory is not None:
            self.offline_client = self.offline_client_factory()
            return self.offline_client
        return None

    def _generate_candidate(self, generation: int, parent_snapshot: str,
                            parent_spec: SkillSpec,
                            bundle: ExperienceBundle) -> Optional[SkillCandidate]:
        path = self._receipt(generation, "candidate.json")
        existing = _read_json(path)
        if existing is not None:
            candidate = SkillCandidate.model_validate(existing)
            self._record_receipt(generation, "candidate.json")
            self.print(f"[{self.campaign_id}] 复用候选 {candidate.candidate_id} "
                       f"→ {candidate.candidate_skill_version}")
            return candidate
        client = self._offline()
        if client is None:
            self._stop_reason = (
                "blocked:offline_model_unavailable（§3.4：不创建伪候选、不用 mock 推进）")
            self._save_checkpoint()
            return None
        major, minor, _patch = _parse_version(parent_spec.version)
        candidate_version = f"{parent_spec.skill_id}@{major}.{minor + 1}.0"
        # §7.1 的输入清单必须可审计：把**这次归纳调用**的 prompt 与模型身份写进收据
        # （prompt 只含父 Skill + 经验包摘要 + 工具名，绝不含答案；正文另存文件并记 hash）。
        prompt = build_induction_prompt(parent_spec, bundle, self.tool_names)
        _write_text(self._receipt(generation, "inducer_prompt.txt"), prompt)
        # §7.4："Schema 不合法：生成结构化错误反馈，产生新 revision；超过最大 revision
        # 数：reject"。因此静态检查失败**不是**一次就停：把问题清单回灌给归纳器重试，
        # 最多 `max_revisions` 次；每次尝试的收据都落盘（attemptN），用尽即 blocked。
        attempts = max(1, int(self.cfg.max_candidate_attempts))
        feedback = ""
        candidate = None
        last_exc: Optional[CandidateStaticCheckError] = None
        for attempt in range(1, attempts + 1):
            try:
                candidate = induce_candidate_from_bundle(
                    bundle=bundle, parent_spec=parent_spec,
                    parent_snapshot_id=parent_snapshot, campaign_id=self.campaign_id,
                    generation=int(generation),
                    candidate_skill_version=candidate_version,
                    offline_client=client, tool_names=self.tool_names,
                    parent_content_sha256=skill_content_sha256(parent_spec),
                    method_context_max_chars=int(self.cfg.method_context_max_chars),
                    qa_ids_to_avoid=set(self.qa_ids_by_split.get("learning") or []),
                    n_min=int(self.cfg.min_eligible_experiences),
                    static_feedback=feedback)
                break
            except InsufficientEvidenceError as exc:
                self._stop_reason = f"blocked:insufficient_evidence {exc}"
                self._save_checkpoint()
                return None
            except OFFLINE_FAILURES as exc:
                # §3.4：离线模型不可用（超时/限流/鉴权）→ 记结局码 + blocked，
                # **不**切在线链、**不**用 mock 顶替、也不让异常把进程打崩
                # （本轮真实运行里一次 DeepSeek 超时就是这样把 campaign 打崩的）。
                code = classify_offline_failure(exc)
                self._stop_reason = f"blocked:{code}: {type(exc).__name__}: {exc}"
                _write_json(self._receipt(generation, "offline_failure.json"), {
                    "campaign_id": self.campaign_id, "generation": int(generation),
                    "failure_code": code, "exception": type(exc).__name__,
                    "message": str(exc)[:500], "created_at": _now_iso()})
                self._record_receipt(generation, "offline_failure.json")
                self._save_checkpoint()
                self.print(f"[{self.campaign_id}] 离线归纳不可用（{code}）→ blocked"
                           "（§3.4：不切在线链、不用 mock 推进）")
                return None
            except CandidateStaticCheckError as exc:
                last_exc = exc
                attempt_path = self._receipt(
                    generation, f"static_validation_attempt{attempt}.json")
                _write_json(attempt_path, exc.receipt)
                self._record_receipt(generation, attempt_path.name)
                self.print(f"[{self.campaign_id}] 候选静态检查失败（第 {attempt}/"
                           f"{attempts} 次）：{exc.receipt.problems} → 回灌反馈重试")
                feedback = "\n".join(f"- {p}" for p in exc.receipt.problems)
                _write_text(self._receipt(generation, "inducer_prompt.txt"),
                            build_induction_prompt(parent_spec, bundle, self.tool_names,
                                                   feedback=feedback))
        if candidate is None:
            _write_json(self._receipt(generation, "static_validation.json"),
                        last_exc.receipt if last_exc is not None else {})
            self._record_receipt(generation, "static_validation.json")
            self._stop_reason = (
                "blocked:candidate_static_validation_failed "
                f"（{attempts} 次尝试均未通过，§7.4 超过最大 revision 数 → reject）")
            self._save_checkpoint()
            return None
        _write_json(path, candidate)
        _write_json(self._receipt(generation, "inducer_receipt.json"), {
            "campaign_id": self.campaign_id,
            "generation": int(generation),
            "candidate_id": candidate.candidate_id,
            "prompt_version": INDUCE_PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "prompt_ref": "inducer_prompt.txt",
            "bundle_id": bundle.bundle_id,
            "parent_skill_version": candidate.parent_skill_version,
            "offline_model": _offline_identity(client),
            "created_at": _now_iso(),
        })
        self._record_receipt(generation, "inducer_receipt.json")
        self._record_receipt(generation, "inducer_prompt.txt")
        self._record_receipt(generation, "candidate.json")
        self._save_checkpoint()
        self.print(f"[{self.campaign_id}] 候选 {candidate.candidate_id} "
                   f"{candidate.parent_skill_version} → {candidate.candidate_skill_version} "
                   f"diff 字段={[d['field'] for d in candidate.structured_diff]}")
        return candidate

    def _static_validate(self, generation: int, parent_spec: SkillSpec,
                         candidate: SkillCandidate,
                         bundle: ExperienceBundle) -> Optional[StaticValidationReceipt]:
        path = self._receipt(generation, "static_validation.json")
        existing = _read_json(path)
        if existing is not None and str(existing.get("candidate_id", "")) == \
                candidate.candidate_id:
            self._record_receipt(generation, "static_validation.json")
            receipt = StaticValidationReceipt.model_validate(existing)
            self.print(f"[{self.campaign_id}] 复用静态检查 passed={receipt.passed}")
            return receipt
        if existing is not None:
            # 收据属于**另一个**候选（例如上一次尝试失败留下的）→ 不当作本次结论
            self.print(f"[{self.campaign_id}] 忽略属于其它候选的静态检查收据："
                       f"{existing.get('candidate_id')!r} ≠ {candidate.candidate_id!r}")
        receipt = validate_candidate_against_parent(
            candidate.full_skill_spec, parent_spec,
            parent_content_sha256=skill_content_sha256(parent_spec),
            claimed_parent_content_sha256=skill_content_sha256(parent_spec),
            canonical_question_type=bundle.canonical_question_type,
            tool_names=self.tool_names,
            forbidden_sample_ids=set(self.qa_ids_by_split.get("learning") or []),
            method_context_max_chars=int(self.cfg.method_context_max_chars))
        receipt.candidate_id = candidate.candidate_id
        receipt.campaign_id = self.campaign_id
        receipt.generation = int(generation)
        _write_json(path, receipt)
        self._record_receipt(generation, "static_validation.json")
        self._save_checkpoint()
        self.print(f"[{self.campaign_id}] 静态检查 passed={receipt.passed} "
                   f"problems={receipt.problems}")
        return receipt

    def _run_inner_seed(self, generation: int, seed: int, items, parent_spec: SkillSpec,
                        candidate: SkillCandidate) -> PairedPanelReceipt:
        path = self._receipt(generation, f"paired_seed_{seed}.json")
        existing = _read_json(path)
        if existing is not None:
            self._record_receipt(generation, f"paired_seed_{seed}.json")
            receipt = PairedPanelReceipt.model_validate(existing)
            self.print(f"[{self.campaign_id}] 复用 seed {seed} 面板结果 "
                       f"delta={receipt.delta:+.4f}")
            return receipt
        base_cfg = self._base_cfg()
        trace_dir = str(self.root / "traces" / f"gen{generation}" / f"inner_seed{seed}")
        store = TraceStore(trace_dir)
        receipt, _outs = self.runtime.run_inner(
            campaign_id=self.campaign_id, generation=int(generation), seed=int(seed),
            # §8.5 的面板命名：inner_g0 / inner_g1 是**两代各自的独立子面板**
            # （generation 从 1 起，面板从 g0 起 —— 标签必须与 panels_provider 给的内容
            # 一致，否则收据上的面板名会指错批次）。
            panel_id=f"inner_g{int(generation) - 1}", items=items,
            parent_spec=parent_spec,
            candidate_spec=candidate.full_skill_spec, base_cfg=base_cfg,
            trace_store=store, llm=self.llm)
        _write_json(path, receipt)
        self._record_receipt(generation, f"paired_seed_{seed}.json")
        self._save_checkpoint()
        return receipt

    def _decide(self, generation: int, candidate: SkillCandidate,
                receipts: Sequence[PairedPanelReceipt]) -> CampaignDecision:
        path = self._receipt(generation, "decision.json")
        existing = _read_json(path)
        if existing is not None:
            self._record_receipt(generation, "decision.json")
            decision = CampaignDecision.model_validate(existing)
            self.print(f"[{self.campaign_id}] 复用决定 promote={decision.promote}")
            return decision
        decision = decide_promotion(
            receipts, required_seeds=[int(s) for s in self.cfg.validation_seeds],
            candidate_delivered_eligible=bool(candidate.candidate_id))
        decision.candidate_id = candidate.candidate_id
        decision.campaign_id = self.campaign_id
        decision.generation = int(generation)
        _write_json(path, decision)
        self._record_receipt(generation, "decision.json")
        self._save_checkpoint()
        self.print(f"[{self.campaign_id}] 决定 promote={decision.promote} "
                   f"reasons={decision.reasons}")
        return decision

    def _publish_or_reject(self, generation: int, candidate: SkillCandidate,
                           decision: CampaignDecision) -> bool:
        path = self._receipt(generation, "promotion.json")
        existing = _read_json(path)
        if existing is not None:
            self._record_receipt(generation, "promotion.json")
            outcome = _promotion_outcome(existing)
            if outcome != "promoted":
                self.print(f"[{self.campaign_id}] 复用 reject 记录（未发布）")
                return False
            promotion = PromotionReceipt.model_validate(
                {k: v for k, v in existing.items() if k != "outcome"})
            self._promoted_skill_key = promotion.skill_version
            if int(generation) not in self._promoted_generations:
                self._promoted_generations.append(int(generation))
            self.print(f"[{self.campaign_id}] 复用发布快照 {promotion.snapshot_after}")
            return True
        if not decision.promote:
            self._record_receipt(generation, "promotion.json")  # 明确标记"未发布"
            _write_json(path, RejectionReceipt(
                campaign_id=self.campaign_id, generation=int(generation),
                candidate_id=candidate.candidate_id, reasons=list(decision.reasons),
                snapshot_before=str(read_active_snapshot(self.store_dir).get("snapshot_id")),
                created_at=_now_iso()).model_dump(mode="json"))
            self._save_checkpoint()
            self.print(f"[{self.campaign_id}] reject：active 指针保持不变"
                       f"（{decision.reasons}）")
            return False
        if self.cfg.mode != "real":
            self._stop_reason = "blocked:admission_must_be_real（§5.6b）"
            self._save_checkpoint()
            return False
        snapshot, receipt = publish_candidate_snapshot(
            self.store_dir, candidate,
            max_competing_per_lineage=int(self.cfg.max_competing_per_lineage),
            method_context_max_chars=int(self.cfg.method_context_max_chars))
        _write_json(path, {"outcome": "promoted", **receipt.model_dump(mode="json")})
        self._record_receipt(generation, "promotion.json")
        self._save_checkpoint()
        self.print(f"[{self.campaign_id}] promote：{receipt.snapshot_before} → "
                   f"{receipt.snapshot_after}（竞争集合={receipt.competing_versions}，"
                   f"历史={receipt.historical_versions}）")
        self._promoted_skill_key = receipt.skill_version
        if int(generation) not in self._promoted_generations:
            self._promoted_generations.append(int(generation))
        return True

    def _verify_post_publish(self, generation: int, parent_spec: SkillSpec,
                             candidate: SkillCandidate,
                             promoted: bool) -> PostPublishUseReceipt:
        path = self._receipt(generation, "post_publish_use.json")
        existing = _read_json(path)
        if existing is not None:
            self._record_receipt(generation, "post_publish_use.json")
            receipt = PostPublishUseReceipt.model_validate(existing)
            self.print(f"[{self.campaign_id}] 复用发布后验证 status={receipt.status}")
            return receipt
        new_key = candidate.candidate_skill_version
        if not promoted:
            receipt = PostPublishUseReceipt(
                campaign_id=self.campaign_id, generation=int(generation),
                snapshot_id=str(read_active_snapshot(self.store_dir).get("snapshot_id")),
                new_skill_version=new_key, status="promoted_not_observed",
                created_at=_now_iso())
            _write_json(path, receipt)
            self._record_receipt(generation, "post_publish_use.json")
            return receipt
        # §10.2：必须运行**新的 learning episode**（新快照 + 正常检索）。
        items = self._items("post_publish", generation)
        trace_dir = str(self.root / "traces" / f"gen{generation}" / "post_publish")
        snapshot_id, manifest_hash = active_snapshot_provenance(self.store_dir)
        skills, warnings, loaded_snapshot = load_legacy_active_skills(self.store_dir)
        cfg = _replace_cfg(self._base_cfg(), skills=skills, trace_dir=trace_dir)
        self.runtime.run_learning(items=items, run_cfg=cfg, trace_dir=trace_dir,
                                  llm=self.llm)
        # 检索 / 交付事实一律从**落盘的 trace** 读（收据为准，不依赖内存对象）——
        # 这样恢复重跑与真实运行走同一条判定路径。
        evidence = evidence_from_trace_store(
            trace_dir, split_of={str(it.episode.qa_id): "learning" for it in items},
            snapshot_id=snapshot_id, skills_in_scope=[new_key])
        retrieved = delivered = False
        delivered_sha = ""
        for ev in evidence:
            state = usage_state_of(ev, new_key)["state"]
            if state != "not_retrieved":
                retrieved = True
            if state in ("delivered_not_used", "usage_supported"):
                delivered = True
                delivered_sha = delivered_sha or str(
                    usage_state_of(ev, new_key)["delivery_sha256"])
        snapshot_sha = _snapshot_content_sha(self.store_dir, new_key)
        events = build_experience_events(
            campaign_id=self.campaign_id, generation=int(generation),
            parent_snapshot_id=snapshot_id, skills=[new_key], episodes=evidence)
        eligible = [e for e in events if e.eligible_for_induction]
        event_refs = [e.experience_id for e in eligible]
        match = bool(delivered_sha and snapshot_sha and delivered_sha == snapshot_sha)
        observed = bool(retrieved and delivered and match and event_refs)
        receipt = PostPublishUseReceipt(
            campaign_id=self.campaign_id, generation=int(generation),
            snapshot_id=str(snapshot_id), new_skill_version=new_key,
            episodes_run=len(items), retrieved=retrieved, delivered=delivered,
            delivered_content_sha256=delivered_sha, snapshot_content_sha256=snapshot_sha,
            request_content_sha256_match=match, experience_event_refs=event_refs,
            status="observed" if observed else "promoted_not_observed",
            created_at=_now_iso())
        _write_json(path, receipt)
        self._record_receipt(generation, "post_publish_use.json")
        if warnings:
            self.print(f"[{self.campaign_id}] 发布后 loader 警告: {warnings[:2]}")
        self._save_checkpoint()
        self.print(f"[{self.campaign_id}] 发布后验证 status={receipt.status} "
                   f"retrieved={retrieved} delivered={delivered} hash_match={match} "
                   f"eligible_events={len(event_refs)}")
        return receipt

    def _manifest_hash(self) -> str:
        payload = _read_json(Path(self.cfg.library_root) / "manifests" / "library_manifest.json")
        return str((payload or {}).get("manifest_sha256", ""))


def _replace_cfg(cfg: OnlineRunConfig, **updates) -> OnlineRunConfig:
    from dataclasses import replace

    return replace(cfg, **updates)


def _parse_version(version: str) -> tuple[int, int, int]:
    parts = str(version).split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise CampaignBlocked(f"父版本 {version!r} 不是 MAJOR.MINOR.PATCH")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _items_hash(items: Sequence) -> str:
    keys = sorted(f"{it.episode.qa_id}:{it.episode.scene_name}" for it in items)
    return canonical_digest(keys)


def _snapshot_content_sha(store_dir: Path, skill_version: str) -> str:
    """快照里该版本的**正文 hash**（与请求里的交付 hash 同口径，用于 §10.2 核对）。"""
    snap = read_active_snapshot(store_dir)
    entry = (snap.get("entries") or {}).get(skill_version) or {}
    raw = str(entry.get("spec_content") or "")
    if not raw:
        return ""
    try:
        spec = SkillSpec.model_validate(json.loads(raw))
    except Exception:  # noqa: BLE001 - 损坏内容不得被当成匹配
        return ""
    return skill_content_sha256(spec)


def _code_commit() -> str:
    """当前代码提交（读 .git；读不到返回空串，不编造）。"""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, cwd=str(Path(__file__).resolve().parents[3]))
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# v10 演化链的关键源码（身份要能复算：commit + 这些文件的 sha256）
_FROZEN_SOURCES: tuple[str, ...] = (
    "src/skill3d/evolution/campaign.py",
    "src/skill3d/evolution/experience.py",
    "src/skill3d/evolution/panel.py",
    "src/skill3d/governance/induce.py",
    "src/skill3d/governance/revise_patch.py",
    "src/skill3d/skills/promote_atomic.py",
    "src/skill3d/skills/registry.py",
    "src/skill3d/routing/skill_retriever.py",
    "src/skill3d/online/runner.py",
    "src/skill3d/schemas/experience.py",
    "src/skill3d/schemas/evolution.py",
    "src/skill3d/schemas/retrieval.py",
    "scripts/run_evolution_campaign.py",
)


def code_state() -> dict:
    """§17.1-1"锁定代码"：commit + 关键源码的内容摘要（工作区脏也能复算身份）。

    只凭 commit 无法锁定"当时跑的是哪份代码"（本轮 v10 改动都在工作区）。
    因此额外给出这些文件的 sha256 与一个聚合摘要；`dirty` 表明工作区未提交。
    """
    import subprocess

    repo = Path(__file__).resolve().parents[3]
    digests: dict[str, str] = {}
    for rel in _FROZEN_SOURCES:
        path = repo / rel
        if path.is_file():
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    aggregate = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(digests.items())).encode("utf-8")
    ).hexdigest()
    dirty = False
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                             text=True, timeout=15, cwd=str(repo))
        dirty = bool(out.stdout.strip())
    except Exception:  # noqa: BLE001 - 读不到就如实留 False，不猜
        dirty = False
    return {
        "commit": _code_commit(),
        "dirty_worktree": dirty,
        "sources": digests,
        "sources_sha256": aggregate,
    }


def seed_campaign_library(source_library_root: str | Path,
                          target_library_root: str | Path) -> dict:
    """把 S0 库的**快照与 manifest** 复制成 campaign 自己的库（不动仓库的 S0 库）。

    仓库里的 `skill_library/` 有 `--check` 不变量（S0 快照 digest 与 active 指针）；
    演化必须发生在**另一个**库里，否则一次 promote 就会让 S0 的校验失败。本函数把
    父快照、指针与 manifest 原样复制过去，campaign 之后只在自己的库里发布。
    """
    src = Path(source_library_root)
    dst = Path(target_library_root)
    copied: list[str] = []
    for rel in ("snapshots", "manifests", "validation"):
        s = src / rel
        if not s.exists():
            continue
        d = dst / rel
        d.mkdir(parents=True, exist_ok=True)
        for item in sorted(s.iterdir()):
            if item.is_dir():
                shutil.copytree(item, d / item.name, dirs_exist_ok=True)
            else:
                shutil.copyfile(item, d / item.name)
            copied.append(f"{rel}/{item.name}")
    return {"source": str(src), "target": str(dst), "copied": copied}


__all__ = [
    "RECEIPTS",
    "CampaignBlocked",
    "CampaignConfig",
    "CampaignRuntime",
    "EvolutionCampaignRunner",
    "seed_campaign_library",
]
