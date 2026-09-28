"""P1 / v10 §6：经验资格与经验包（ExperienceEvent / ExperienceBundle 的确定性构造）。

规范原文（§6.2）——只有同时满足以下条件的 learning episode，才能作为某个 Skill 的
**主要修订经验**：

    「1. episode 使用父快照运行；
    2. Skill 被正常检索；
    3. Skill 正文真正进入模型请求；
    4. 至少存在可观察的程序使用线索；
    5. trace、结果和版本身份完整；
    6. 不属于 inner/final；
    7. 没有 Schema 损坏、缺失映射或无法确认快照身份。

    `delivered_not_used` 只能用于改进描述和检索适用性；`not_retrieved` 只能用于识别
    覆盖缺口，不能冒充该 Skill 的执行经验。」

本模块把上面七条实现成**确定性函数**：输入是 episode 的 trace 事实（检索记录 /
交付 hash / 使用线索 / 结果），输出是逐 (episode, skill 版本) 的 `ExperienceEvent`
与按 `skill_id@version` 聚合的 `ExperienceBundle`。归纳器不再自己去读一整个目录的
trace —— 它只收经验包（§6.3），于是"这批经验是从哪条 Skill 的哪些题来的、被排除了
什么"都可回答。

**确定性规则：`usage_supported`（§6.2-4 的"可观察的程序使用线索"）**

一条已交付经验记 `usage_supported` 当且仅当该版本的交付记录里存在至少一条**可观察**
线索：

- `declared_in_program=true`：程序文本里字面出现了该 `skill_id@version`（或 skill_id）；
- `template_tool_overlap` 非空：该 Skill 模板点名的 Tool，与程序源码里字面出现的 Tool
  有交集。

两者都是**机械交叉检查**（`online/runner._skill_usage_clues`），不是因果贡献的证明；
它们只用于"这条经验是否算该 Skill 的执行经验"这一资格判断，不用于给 Skill 记功
（§13.6 口径不变：模型自称不是方法成功或因果贡献的充分证明）。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from skill3d.schemas import ExperienceBundle, ExperienceEvent

# 规范层的 learning 层在本仓库的四层 split 里叫 `induction`（M1 的 5:3:2:留出）。
# 这里做一次**显式**映射，而不是把 induction 静默当成 learning。
SPLIT_ALIASES: dict[str, str] = {
    "induction": "learning",
    "learning": "learning",
    "inner_validation": "inner_validation",
}


class ExperienceLedgerError(RuntimeError):
    """经验账本构造失败（身份不完整、结构损坏等硬错误）。"""


@dataclass
class EpisodeEvidence:
    """一个 episode 在"经验资格"视角下的事实（不做任何推断）。"""

    episode_id: str
    scene_id: str = ""
    # 规范层 split（learning / inner_validation / 其它原样保留，由资格判定拒绝）
    split: str = ""
    source_split: str = ""
    snapshot_id: str = ""
    # 检索与交付事实：直接用落盘的 `EpisodeTrace` 载荷（dict）或对象
    trace: Any = None
    answer_correct: Optional[bool] = None
    mra_value: Optional[float] = None
    failure_categories: tuple[str, ...] = ()
    failure_code: str = ""
    outcome_ref: str = ""
    # §6.2-7：结构 / 映射损坏的显式标记（调用方如实填写，不由本模块猜测）
    schema_ok: bool = True
    mapping_ok: bool = True
    skills_in_scope: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    def canonical_split(self) -> str:
        """规范层 split（未知取值原样返回 → 资格判定按"不在 learning"拒绝）。"""
        key = str(self.source_split or self.split or "")
        return SPLIT_ALIASES.get(key, key)


def _trace_field(trace: Any, name: str, default=None):
    if trace is None:
        return default
    if isinstance(trace, dict):
        return trace.get(name, default)
    return getattr(trace, name, default)


def _records_of(trace: Any) -> list[dict]:
    out: list[dict] = []
    for rec in (_trace_field(trace, "retrieval_records", None) or []):
        if hasattr(rec, "model_dump"):
            out.append(rec.model_dump(mode="json"))
        elif isinstance(rec, dict):
            out.append(rec)
    return out


def _row_of(record: dict, skill_version: str) -> Optional[dict]:
    for row in (record.get("candidates") or []):
        if str(row.get("skill_version", "")) == skill_version:
            return row
    return None


def _delivery_sha(record: dict, skill_version: str) -> str:
    return str((record.get("delivered_content_sha256") or {}).get(skill_version, "") or "")


def _usage_clues(record: dict, skill_version: str) -> list[dict]:
    return [dict(c) for c in (record.get("usage_clues") or [])
            if str(c.get("skill_version", "")) == skill_version]


def clue_is_usage_support(clue: dict) -> bool:
    """单条线索是否构成 §6.2-4 的"可观察使用线索"（确定性判定）。"""
    if bool(clue.get("declared_in_program")):
        return True
    return bool(clue.get("template_tool_overlap"))


def usage_state_of(evidence: EpisodeEvidence, skill_version: str) -> dict:
    """§6.2-2/3/4：判定一个 episode 对某版本的检索 / 交付 / 使用状态。

    返回 `{state, reasons, delivery_sha256, record_ref, clue_refs}`：

    - `not_retrieved`：没有任何检索记录提到它（且它也没被交付过）；
    - `retrieved_not_delivered`：进了 top-k 但正文没进任何实际发出的请求；
    - `delivered_not_used`：正文进了请求但没有可观察使用线索；
    - `usage_supported`：正文进了请求且至少一条可观察使用线索。

    细节：多轮 / 证据更新后的重检索会留下**多条**记录，只要有任意一条记录显示该版本
    被交付，就按交付处理（交付事实不会被后来的重检索抹掉，这与 trace 的"逐次记录"口径
    一致）；使用线索同理按并集取。
    """
    version = str(skill_version)
    records = _records_of(evidence.trace)
    reasons: list[str] = []
    delivery_sha = ""
    record_ref = ""
    clues: list[dict] = []
    ever_selected = False
    ever_eligible = False
    seen = False
    for idx, record in enumerate(records, start=1):
        row = _row_of(record, version)
        if row is None:
            continue
        seen = True
        ever_eligible = ever_eligible or bool(row.get("hard_filter_passed"))
        ever_selected = ever_selected or bool(row.get("selected"))
        sha = _delivery_sha(record, version)
        if row.get("delivered") and sha:
            delivery_sha = sha
            record_ref = f"retrieval_record#{record.get('retrieval_index', idx)}"
        clues.extend(_usage_clues(record, version))
        if not ever_selected and not row.get("hard_filter_passed"):
            reasons.append(f"record#{idx}:{row.get('reason_code', '')}")
    if not seen or not ever_eligible:
        return {"state": "not_retrieved", "reasons": reasons or ["未出现在任何检索记录中"],
                "delivery_sha256": "", "record_ref": "", "clue_refs": []}
    if not delivery_sha:
        return {"state": "retrieved_not_delivered", "reasons": ["检索选中但正文未进入请求"],
                "delivery_sha256": "", "record_ref": record_ref, "clue_refs": []}
    support = [c for c in clues if clue_is_usage_support(c)]
    clue_refs: list[str] = []
    for clue in clues:
        kinds: list[str] = []
        if bool(clue.get("declared_in_program")):
            kinds.append("declared_in_program")
        if bool(clue.get("template_tool_overlap")):
            kinds.append("template_tool_overlap")
        for kind in kinds:
            clue_refs.append(f"{record_ref or 'retrieval_record'}:{kind}")
    if not support:
        return {"state": "delivered_not_used", "reasons": ["已交付但无可观察使用线索"],
                "delivery_sha256": delivery_sha, "record_ref": record_ref,
                "clue_refs": clue_refs}
    return {"state": "usage_supported", "reasons": [], "delivery_sha256": delivery_sha,
            "record_ref": record_ref, "clue_refs": clue_refs}


def experience_id_of(campaign_id: str, generation: int, episode_id: str,
                     skill_version: str) -> str:
    """经验事件的稳定 id（同一次真实运行重复构建 → 同一 id）。"""
    payload = "|".join([str(campaign_id), str(int(generation)), str(episode_id),
                        str(skill_version)])
    return "exp-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def parent_snapshot_matches(evidence: EpisodeEvidence, parent_snapshot_id: str) -> bool:
    """§6.2-1：episode 是否在**父快照**上运行（按快照 id 归一比较，不比路径）。"""
    seen = str(evidence.snapshot_id or "")
    want = str(parent_snapshot_id or "")
    if not seen or not want:
        return False
    return Path(seen).stem.lstrip("snapshot_") == Path(want).stem.lstrip("snapshot_") \
        or seen == want


def build_experience_events(
    *,
    campaign_id: str,
    generation: int,
    parent_snapshot_id: str,
    skills: Sequence[str],
    episodes: Iterable[EpisodeEvidence],
    label_access: bool = True,
) -> list[ExperienceEvent]:
    """逐 (episode, skill 版本) 生成经验事件（§6.1 + §6.2 七条资格判定）。

    `skills` 是本代**在范围内**的 skill 版本清单（父版本 + 竞争版本）。给一条与
    episode 无关的版本不会凭空产出事件 —— 只有出现在该 episode 检索记录里的版本才会
    有事件（`not_retrieved` 也只在它确实在候选池里而未被检索到时才记）。
    """
    events: list[ExperienceEvent] = []
    for ev in episodes:
        split = ev.canonical_split()
        snapshot_ok = parent_snapshot_matches(ev, parent_snapshot_id)
        for key in skills:
            skill_id, _, version = str(key).partition("@")
            if not version:
                raise ExperienceLedgerError(f"skill 版本必须是 skill_id@version：{key!r}")
            state = usage_state_of(ev, key)
            exclusions: list[str] = []
            if not ev.schema_ok:
                exclusions.append("schema_corrupted")
            if not ev.mapping_ok:
                exclusions.append("skill_identity_unmapped")
            if split != "learning":
                exclusions.append("split_not_learning" if split else "split_unknown")
            if not snapshot_ok:
                exclusions.append(
                    "snapshot_identity_unconfirmed"
                    if not (ev.snapshot_id and parent_snapshot_id)
                    else "not_parent_snapshot")
            if state["state"] == "not_retrieved":
                exclusions.append("retrieval_record_missing"
                                  if not state["reasons"] or state["reasons"] == [
                                      "未出现在任何检索记录中"]
                                  else "skill_not_retrieved")
            elif state["state"] == "retrieved_not_delivered":
                exclusions.append("skill_not_delivered")
            elif state["state"] == "delivered_not_used":
                exclusions.append("delivered_but_no_usage_clue")
            # §6.2-5：结果身份完整 = 有可用的判分事实（MCA 的对错或 NA 的 MRA）。
            if ev.answer_correct is None and ev.mra_value is None:
                exclusions.append("result_identity_missing")
            eligible = not exclusions
            events.append(ExperienceEvent(
                experience_id=experience_id_of(campaign_id, generation,
                                               ev.episode_id, key),
                campaign_id=str(campaign_id),
                generation=int(generation),
                episode_id=str(ev.episode_id),
                scene_id=str(ev.scene_id or ""),
                split=split,  # type: ignore[arg-type]
                source_split=str(ev.source_split or ev.split or ""),
                snapshot_id=str(ev.snapshot_id or ""),
                skill_id=skill_id,
                skill_version=str(key),
                retrieval_state=state["state"],  # type: ignore[arg-type]
                retrieval_record_ref=str(state["record_ref"]),
                delivery_content_sha256=(state["delivery_sha256"] or None),
                usage_clue_refs=list(state["clue_refs"]),
                outcome_ref=str(ev.outcome_ref or f"evaluation_result:{ev.episode_id}"),
                failure_categories=sorted(set(ev.failure_categories)),
                answer_correct=ev.answer_correct,
                # §6.4："learning 的答对/答错和误差可由离线归纳器使用，但必须记录
                # label_access=true" —— **误差**（MRA 题型的分值）同样是标签信号：
                # 只有"进了归纳输入的合格经验"才记 label_access=true（被排除的经验
                # 不在经验包里，它的标签没有被归纳器看到）。
                label_access=bool(
                    label_access and split == "learning" and eligible
                    and (ev.answer_correct is not None or ev.mra_value is not None)),
                eligible_for_induction=eligible,
                exclusion_reasons=sorted(set(exclusions)),
            ))
    return events


def build_experience_bundle(
    *,
    campaign_id: str,
    generation: int,
    parent_snapshot_id: str,
    parent_skill_key: str,
    canonical_question_type: str,
    events: Sequence[ExperienceEvent],
    source_manifest_hash: str = "",
    evidence_failure_codes: Optional[dict[str, str]] = None,
    evidence_mra: Optional[dict[str, float]] = None,
) -> ExperienceBundle:
    """§6.3：按 `skill_id@version` 聚合经验包（禁止混题型 / 混父版本）。

    `evidence_failure_codes` 是 `episode_id → failure_code` 的可选映射（来自 trace；
    `ExperienceEvent` 本身不带 failure_code，因为它不是 §6.1 的字段）。缺省 → 空表，
    **不编造**代码。

    `evidence_mra` 是 `episode_id → mra_value` 的可选映射：MRA 题型（`object_counting`
    等）的判分是相对精度而不是对错布尔量，`answer_correct_rate` 只对**有判分布尔量**的
    题给值（分母 `n_answer_graded`），MRA 题型另给 `mra_mean` —— 两者不互相顶替。
    """
    evidence_failure_codes = dict(evidence_failure_codes or {})
    parent_id, _, parent_version = str(parent_skill_key).partition("@")
    scoped = [e for e in events if e.skill_version == parent_skill_key]
    if not scoped:
        raise ExperienceLedgerError(
            f"经验包里没有属于 {parent_skill_key} 的经验事件（§6.3 禁止混桶）")
    wrong_type = sorted({e.skill_id for e in scoped if e.skill_id != parent_id})
    if wrong_type:
        raise ExperienceLedgerError(
            f"经验事件跨谱系：{wrong_type} ≠ {parent_id}（§6.3）")

    eligible = [e for e in scoped if e.eligible_for_induction]
    excluded = [e for e in scoped if not e.eligible_for_induction]
    exclusion_summary = dict(sorted(Counter(
        r for e in excluded for r in e.exclusion_reasons).items()))
    scenes = sorted({e.scene_id for e in eligible if e.scene_id})
    success = [e for e in eligible if e.answer_correct is True]
    failure = [e for e in eligible if e.answer_correct is False]
    n = len(eligible)
    # MRA 题型（如 `object_counting`：官方判分是相对精度）没有对错布尔量 ——
    # 这时 `answer_correct_rate` 的分母只能是**有判分布尔量**的题，绝不能把
    # "没测过对错" 记成 0%（那会把 MRA 题型说成"全错"）。
    graded = len(success) + len(failure)
    # 行为摘要描述的是**合格经验集**，所以 MRA 也只统计合格经验对应的 episode
    # （否则会把被排除的题混进"这批经验的表现"里，n_eligible=4 却报 8 条 MRA）。
    eligible_ids = {e.episode_id for e in eligible}
    mra_values = [float(v) for ep, v in (evidence_mra or {}).items()
                  if ep in eligible_ids and v is not None]

    behavior = {
        "n_eligible": n,
        "n_usage_supported": sum(1 for e in eligible
                                 if e.retrieval_state == "usage_supported"),
        "n_delivered_not_used": sum(1 for e in eligible
                                    if e.retrieval_state == "delivered_not_used"),
        "n_retrieved_not_delivered": sum(1 for e in eligible
                                         if e.retrieval_state == "retrieved_not_delivered"),
        "n_answer_correct": len(success),
        "n_answer_wrong": len(failure),
        "n_run_error": sum(1 for e in eligible if "run_error"
                           in set(e.failure_categories)),
        "n_abstain": sum(1 for e in eligible if "evaluator_noanswer"
                         in set(e.failure_categories)),
        "n_partial_tool_recovery": sum(1 for e in eligible
                                       if "tool_contract" in set(e.failure_categories)),
        # 线索类计数直接从事件的 `usage_clue_refs` 数出来（refs 自带线索种类后缀，
        # 因此"哪几条经验是模型自称用了方法、哪几条是模板工具交叠"可复算）。
        "n_declared_in_program": sum(
            1 for e in eligible
            if any(r.endswith(":declared_in_program") for r in e.usage_clue_refs)),
        "n_template_tool_overlap": sum(
            1 for e in eligible
            if any(r.endswith(":template_tool_overlap") for r in e.usage_clue_refs)),
        "n_answer_graded": graded,
        "answer_correct_rate": (len(success) / graded) if graded else None,
        "n_mra_graded": len(mra_values),
        "mra_mean": (sum(mra_values) / len(mra_values)) if mra_values else None,
        "run_error_rate": ((sum(1 for e in eligible if "run_error"
                                in set(e.failure_categories)) / n) if n else None),
    }
    failures = Counter(c for e in eligible for c in e.failure_categories)
    codes = Counter(evidence_failure_codes.get(e.episode_id, "")
                    for e in eligible if evidence_failure_codes.get(e.episode_id))
    failure_summary = {
        "by_category": dict(sorted(failures.items())),
        "by_failure_code": dict(sorted(codes.items())),
        "n_without_failure": sum(1 for e in eligible if not e.failure_categories),
    }
    bundle_id = "bundle-" + hashlib.sha256(
        "|".join([str(campaign_id), str(int(generation)), parent_skill_key,
                  canonical_question_type,
                  ",".join(sorted(e.experience_id for e in eligible))]).encode("utf-8")
    ).hexdigest()[:20]
    return ExperienceBundle(
        bundle_id=bundle_id,
        campaign_id=str(campaign_id),
        generation=int(generation),
        parent_snapshot_id=str(parent_snapshot_id),
        parent_skill_id=parent_id,
        parent_skill_version=parent_version,
        canonical_question_type=str(canonical_question_type),
        eligible_experience_refs=sorted(e.experience_id for e in eligible),
        excluded_experience_refs=sorted(e.experience_id for e in excluded),
        exclusion_summary=exclusion_summary,
        scene_count=len(scenes),
        success_count=len(success),
        failure_count=len(failure),
        behavior_summary=behavior,
        failure_summary=failure_summary,
        source_manifest_hash=str(source_manifest_hash or ""),
    )


# --------------------------------------------------------------------------- #
# 落盘 / 读取（收据）
# --------------------------------------------------------------------------- #

def write_events_jsonl(events: Sequence[ExperienceEvent], path: str | Path) -> Path:
    """写 `experience_events.jsonl`（§14.1；逐行 JSON，便于按行核对）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(event.model_dump_json() + "\n")
    return p


def write_bundle_json(bundle: ExperienceBundle, path: str | Path) -> Path:
    """写 `experience_bundle.json`（§14.1）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False,
                            indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_events_jsonl(path: str | Path) -> list[ExperienceEvent]:
    """读 `experience_events.jsonl`（恢复路径用；不解释、不修补）。"""
    p = Path(path)
    if not p.is_file():
        return []
    return [ExperienceEvent.model_validate_json(line)
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_from_trace_store(trace_dir: str | Path,
                              *,
                              split_of: dict[str, str],
                              scene_of: Optional[dict[str, str]] = None,
                              snapshot_id: str = "",
                              skills_in_scope: Sequence[str] = ()) -> list[EpisodeEvidence]:
    """从落盘的 trace 目录重建 episode 证据（**恢复**路径用，不重跑在线链）。

    只读两个 topic：`episode_trace.jsonl`（检索 / 交付 / 失败类别）与
    `evaluation_result.jsonl`（答对与否）。**ground truth 不被读入** —— 经验事件与
    经验包里只有 `answer_correct` 布尔量（§6.4 标签边界）。
    """
    root = Path(trace_dir)
    traces: dict[str, dict] = {}
    episode_trace = root / "episode_trace.jsonl"
    if episode_trace.is_file():
        for line in episode_trace.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qa = str(row.get("qa_id") or row.get("episode_id") or "")
            if qa:
                traces[qa] = row
    correct: dict[str, Optional[bool]] = {}
    mra: dict[str, Optional[float]] = {}
    evaluation = root / "evaluation_result.jsonl"
    if evaluation.is_file():
        for line in evaluation.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qa = str(row.get("qa_id", ""))
            if qa:
                value = row.get("correct")
                correct[qa] = None if value is None else bool(value)
                raw_mra = row.get("mra_value")
                mra[qa] = None if raw_mra is None else float(raw_mra)
    out: list[EpisodeEvidence] = []
    for qa, row in traces.items():
        failure = row.get("failure") or {}
        categories = tuple(str(c) for c in (failure.get("categories") or []))
        source_split = str(split_of.get(qa, ""))
        out.append(EpisodeEvidence(
            episode_id=qa,
            scene_id=str((scene_of or {}).get(qa, "")),
            split=SPLIT_ALIASES.get(source_split, source_split),
            source_split=source_split,
            snapshot_id=str(row.get("active_snapshot_ref", "") or snapshot_id),
            trace=row,
            answer_correct=correct.get(qa),
            mra_value=mra.get(qa),
            failure_categories=categories,
            failure_code=str(row.get("failure_code", "") or ""),
            outcome_ref=f"evaluation_result:{qa}",
            schema_ok=str(row.get("schema_version", "")) != "",
            mapping_ok=not (row.get("skill_mapping_misses") or []),
            skills_in_scope=tuple(skills_in_scope),
        ))
    return out


__all__ = [
    "SPLIT_ALIASES",
    "EpisodeEvidence",
    "ExperienceLedgerError",
    "build_experience_bundle",
    "build_experience_events",
    "clue_is_usage_support",
    "evidence_from_trace_store",
    "experience_id_of",
    "parent_snapshot_matches",
    "read_events_jsonl",
    "usage_state_of",
    "write_bundle_json",
    "write_events_jsonl",
]
