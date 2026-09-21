"""Skill Retriever（§4 M7 + v6 §17.1/§17.2）：题型 + EvidenceProfile 证据签名硬匹配。

v6 的检索硬条件只有三条：

1. **题型匹配**：`question_type ∈ skill.applicable_question_types`（§17.1）；题型未知/无法
   规范化 → 一律不检索（fail-closed，上层走空 Skill baseline）；
2. **证据签名匹配**：`scene.evidence_profile.satisfies(skill.required_evidence_signature)`
   （能力 → 最低可接受状态）；无签名声明（空 dict）= 不额外增加证据硬条件，任意证据状态可检索；
3. **米制 Skill 双重 fail-closed**（§13.6）：`requires_metric_evidence=True` 时**再**校验
   `MetricEvidenceGateResult.gate_passed=True` **且**
   `applicable_gate_version == gate.gate_version`，不匹配 → 不检索、不执行；
   校验结果记进 `RetrievedSkill.gate_version_matched`。

v5 的 `requires_artifacts` / `minimum_quality` / `metric_scale_required` / route /
`scale_confidence` 硬过滤**已删除**：v6 里"产物是否可用"由 EvidenceProfile 的能力三值表达，
单项失败只收回依赖该项的 Skill（§7.2/§17.2），不再有"按 route 一刀切"。

设计纪律：

- **不用 VLM 裁决**（§16.1）：检索只看可观测证据（EvidenceProfile + grounding 信号 +
  question_type），没有任何"让模型挑 Skill"的路径；
- `matched_evidence_signature` 记录"这条 Skill 是在哪种证据状态下被选中的"（§17.2：不同签名
  分开积累、互不污染）——值是当前画像在 Skill 签名键上的**投影**，不是 Skill 的声明；
- 排序权重是**模块级冻结常量** `RANK_WEIGHTS`（§16.1：inner 定策略 / outer 只验 / final 冻结）。

语义排序（G-20）：优先 LanceDB 向量候选 + `bge-reranker-v2-m3`；两者不可用时使用
`memory/vector_index.py` 的确定性 hashing 嵌入余弦（无需模型下载、可复现）。`rerank=False`
时改用关键词打分（消融"有/无语义排序"）。无命中 → 返回空列表。
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple, Optional, Sequence

from skill3d.schemas import RetrievedSkill, SceneState, SkillSpec
from skill3d.schemas.evidence import CAPABILITIES, EvidenceProfile

from .task_classifier import TASK_TYPES, canonical_task

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3        # TODO_CALIBRATE：每题进 prompt 的 Skill 条数
DEFAULT_CANDIDATES = 50  # TODO_CALIBRATE：向量候选池大小

# Skill 排序权重（§16.1：排序权重在 inner 上选定后**冻结**，outer 只验一次、final 不改）。
# 全部为模块级常量：在线链里**不得**按题目/按场景临时调参，否则 outer 上验证的策略失效。
RANK_WEIGHTS: dict[str, float] = {
    "semantic": 1.0,              # 语义排序主分权重（rerank=True）
    "keyword": 1.0,               # 关键词主分权重（rerank=False 的消融口径）
    "semantic_mix_keyword": 0.0,  # 语义路径里混入的关键词分权重（0=纯语义，保持消融可比）
}  # TODO_CALIBRATE

# 允许进入检索的 Skill 状态（§7：未 promoted 的候选不得生效）。
# registry/active snapshot 已按状态过滤；这里是**防御性二次确认**（duck-typing：若上游把
# 状态快照挂在 SkillSpec 对象上，检索层同样遵守）。
_ACTIVE_STATES: tuple[str, ...] = ("promoted", "consolidated")


class RetrievalDecision(NamedTuple):
    """单条 Skill 的检索判定结果（可审计：为什么检索/为什么被拦下）。"""

    ok: bool
    matched_signature: dict[str, str]
    gate_version_matched: Optional[bool]
    reason: str


def canonical_question_type(value: Optional[str]) -> Optional[str]:
    """原始 question_type / 规范题型 → 规范题型（8 类之一）；空或未知 → None。

    None = 题型不可知 → 调用方按 fail-closed 处理（不检索任何 Skill，§17.1）。
    未知题型只告警不抛错：上游数据的题型错字不应让整条在线链崩掉。
    """
    if not value:
        return None
    try:
        return canonical_task(str(value))
    except Exception as exc:  # noqa: BLE001 - 未知题型 → 走 fail-closed
        logger.warning("M7 检索：未知 question_type=%r（%s）→ 不检索任何 Skill",
                       value, type(exc).__name__)
        return None


def skill_question_types(skill: SkillSpec) -> set[str]:
    """Skill 声明的适用题型（规范题型集合；无法规范化的条目按原样归一化保留）。"""
    out: set[str] = set()
    for qt in (getattr(skill, "applicable_question_types", None) or []):
        out.add(canonical_question_type(qt) or str(qt).strip().lower())
    return out


def evidence_profile_of(scene: SceneState) -> Optional[EvidenceProfile]:
    """取当前场景的 EvidenceProfile（无画像 → None，签名非空时按 fail-closed 处理）。"""
    return getattr(scene, "evidence_profile", None)


def metric_evidence_usable(scene: SceneState) -> bool:
    """§13.2/§13.8：米制证据是否可用（单一事实源 = `EvidenceProfile.metric_scale`）。

    `metric_scale` 的三值由 `MetricEvidenceGateResult` 决定（6 项全过 = available）。
    无画像 → 判不可用（fail-closed：无法证明自己有米制证据）。
    """
    profile = evidence_profile_of(scene)
    if profile is None:
        return False
    return profile.state("metric_scale") == "available"


def evidence_signature_match(skill: SkillSpec, scene: SceneState) -> bool:
    """§17.1：`evidence_profile.satisfies(skill.required_evidence_signature)`。

    - 无签名声明（空 dict）→ True（任意证据状态可检索；"无条件 Skill"是显式选择）；
    - 有签名但当前无 EvidenceProfile → False（fail-closed）；
    - 签名含未知能力名/非法状态值 → `EvidenceProfile.satisfies` 已返回 False。
    """
    signature = dict(getattr(skill, "required_evidence_signature", None) or {})
    if not signature:
        return True
    profile = evidence_profile_of(scene)
    if profile is None:
        return False
    return bool(profile.satisfies(signature))


def signature_unmet(skill: SkillSpec, scene: SceneState) -> list[str]:
    """签名未达标的能力（诊断文本用；无画像时把所有声明的键标为未达标）。"""
    signature = dict(getattr(skill, "required_evidence_signature", None) or {})
    profile = evidence_profile_of(scene)
    if profile is None:
        return [f"{c}(无证据画像)" for c in signature]
    return profile.unmet(signature)


def matched_evidence_signature(skill: SkillSpec, scene: SceneState) -> dict[str, str]:
    """命中签名 = 当前画像在 Skill 签名键上的**投影**（§17.2 审计口径）。

    只投影 Skill 自己声明的能力键：于是"这条 Skill 是在哪种证据状态下被选中的"与它的签名
    一一对应，可据此按签名分桶积累 Skill，互不污染。未知能力名不投影（签名匹配已 fail-closed）。
    """
    signature = dict(getattr(skill, "required_evidence_signature", None) or {})
    profile = evidence_profile_of(scene)
    out: dict[str, str] = {}
    for cap in signature:
        if cap not in CAPABILITIES:
            continue
        out[cap] = profile.state(cap) if profile is not None else "unavailable"
    return out


def metric_gate_match(skill: SkillSpec, scene: SceneState) -> tuple[bool, Optional[bool]]:
    """§13.6 米制 Skill 的 gate 双重校验 → `(是否放行, gate_version_matched)`。

    - 非米制 Skill → `(True, None)`（无 gate 语义，字段留空）；
    - 米制 Skill → 要求 `gate_passed=True` **且** 版本逐字相等；缺 gate / 版本不符一律拦下。
    """
    if not bool(getattr(skill, "requires_metric_evidence", False)):
        return True, None
    gate = getattr(scene, "metric_evidence_gate_result", None)
    if gate is None:
        return False, False       # 连 gate 结果都没有 = 无法证明米制证据有效
    version_ok = bool(
        getattr(skill, "applicable_gate_version", None)
        and str(gate.gate_version) == str(skill.applicable_gate_version))
    return bool(gate.gate_passed) and version_ok, version_ok


def retrieval_decision(skill: SkillSpec, scene: SceneState, task_type: str) -> RetrievalDecision:
    """检索判定（§17.1/§13.6 的唯一事实源）：题型 ∧ 签名 ∧ 米制 gate（∧ 状态防御）。

    `task_type` 必须是**规范题型**（调用方已用 `canonical_question_type` 归一）。
    `hard_filter` 与 `retrieve` 共用本函数，保证"入口判定"与"实际检索"永不分叉。
    """
    signature = matched_evidence_signature(skill, scene)
    if task_type not in skill_question_types(skill):
        return RetrievalDecision(
            False, signature, None,
            f"题型不匹配（题目={task_type}，Skill 适用="
            f"{sorted(skill_question_types(skill))}）")
    if not evidence_signature_match(skill, scene):
        return RetrievalDecision(
            False, signature, None,
            f"证据签名不满足（未达标={signature_unmet(skill, scene)}）")
    gate_ok, version_ok = metric_gate_match(skill, scene)
    if not gate_ok:
        gate = getattr(scene, "metric_evidence_gate_result", None)
        return RetrievalDecision(
            False, signature, version_ok,
            "米制证据门未通过或 gate 版本不匹配"
            f"（Skill 要求={getattr(skill, 'applicable_gate_version', None)}，"
            f"当前={getattr(gate, 'gate_version', None)}，"
            f"gate_passed={bool(gate is not None and gate.gate_passed)}）")
    state = getattr(skill, "state", None)
    if state is not None and str(state) not in _ACTIVE_STATES:
        return RetrievalDecision(False, signature, version_ok, f"状态 {state} 非 active")
    return RetrievalDecision(True, signature, version_ok, "命中")


def hard_filter(
    skill: SkillSpec,
    scene: SceneState,
    scene_quality: Optional[float] = None,
    task_type: Optional[str] = None,
    *,
    question_type: Optional[str] = None,
) -> bool:
    """§17.1/§13.6 检索硬条件：**题型匹配 ∧ 证据签名匹配**（米制 Skill 再叠 gate 双重校验）。

    `question_type` 是原始官方取值（如 `object_rel_direction_hard`，与 `episode.question_type`
    同源），`task_type` 是规范题型；两者都给时以 `question_type` 为准。两者都缺或无法规范化
    → fail-closed 返回 False（题型不可知就不该检索任何 Skill）。

    `scene_quality` 是 **v5 遗留的兼容形参**：v6 的 SkillSpec 已无 `minimum_quality`，质量差异
    统一由 `EvidenceProfile` 表达，故该参数**不参与判定**（保留只为不改既有调用点）。
    """
    effective = canonical_question_type(
        question_type if question_type is not None else task_type)
    if not effective:
        return False
    return retrieval_decision(skill, scene, effective).ok


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _keyword_score(question: str, skill: SkillSpec) -> float:
    """退化打分：question 与 (适用题型 + description) 的 token 重叠率。"""
    q = _tokenize(question)
    if not q:
        return 0.0
    types = " ".join(str(t) for t in (skill.applicable_question_types or []))
    s = _tokenize(f"{types} {skill.description}")
    return len(q & s) / len(q)


def skill_text(skill: SkillSpec) -> str:
    """索引文本：适用题型 + 描述 + 模板签名（模板签名让"怎么做"参与检索）。"""
    types = " ".join(str(t) for t in (skill.applicable_question_types or []))
    tmpl = str(getattr(skill, "call_graph_template", "") or "")
    return f"{types} {skill.description} {tmpl}"


def build_index(skills: Sequence[SkillSpec], *, embedder=None):
    """为候选 Skill 建向量索引（内存；可选 `to_lancedb` 持久化）。

    元数据保留 `task_type`（首个适用题型 / 族名，兼容 LanceDB `where` 过滤），并带上
    `skill_family` / `requires_metric_evidence` / `applicable_gate_version`，
    使"米制 Skill 的 gate 版本"在检索层也可审计（§13.6）。
    """
    from skill3d.memory.vector_index import SkillVectorIndex

    idx = SkillVectorIndex(embedder)
    for s in skills:
        key = f"{s.skill_id}@{s.version}"
        idx.add_text(key, skill_text(s), task_type=s.task_type,
                     skill_family=str(s.skill_family),
                     question_types=list(s.applicable_question_types),
                     requires_metric_evidence=bool(s.requires_metric_evidence),
                     applicable_gate_version=str(s.applicable_gate_version or ""),
                     state=str(getattr(s, "state", "promoted")))
    return idx


def _try_lancedb_candidates(question: str, index, limit: int = DEFAULT_CANDIDATES,
                            table_path: str = "") -> Optional[list[str]]:
    """LanceDB 向量检索候选 id（lazy import；不可用返回 None 走内存索引）。"""
    from skill3d.memory.vector_index import lancedb_available

    if index is None or not lancedb_available() or not table_path:
        return None
    try:
        import lancedb  # noqa: F401

        db = lancedb.connect(str(table_path))
        from skill3d.memory.vector_index import _table_names

        if "skills" not in _table_names(db):
            return None
        qv = index.embedder.embed(question)
        hits = db.open_table("skills").search(qv.tolist()).limit(limit).to_list()
        return [str(h["entry_id"]) for h in hits]
    except Exception as exc:  # noqa: BLE001 - 表结构/版本差异 → 退化
        logger.warning("LanceDB 检索失败（退化到内存索引）: %s", exc)
        return None


def _hit(skill: SkillSpec, score: float, decision: RetrievalDecision) -> RetrievedSkill:
    """构造检索输出：硬条件全过 + 命中签名 + gate 版本匹配结果（§17.2/§13.6 可审计）。"""
    return RetrievedSkill(
        skill_id=skill.skill_id,
        skill_version=f"{skill.skill_id}@{skill.version}",
        score=float(score),
        hard_filter_passed=True,
        matched_evidence_signature=dict(decision.matched_signature),
        gate_version_matched=decision.gate_version_matched,
    )


def _rank_key(skill: SkillSpec, score: float) -> tuple:
    """确定性排序键：分数降序 + `skill_id@version` 升序（并列时结果可复现）。"""
    return (-float(score), f"{skill.skill_id}@{skill.version}")


def retrieve(
    question: str,
    scene: SceneState,
    skills: Sequence[SkillSpec],
    question_type: Optional[str] = None,
    scene_quality: Optional[float] = None,
    top_k: int = DEFAULT_TOP_K,
    *,
    embedder=None,
    rerank: bool = True,
    lancedb_path: str = "",
) -> list[RetrievedSkill]:
    """retrieve(question, scene) -> list[RetrievedSkill]；无命中返回空列表。

    硬条件（§17.1/§13.6）：题型匹配 ∧ 证据签名匹配 ∧（米制 Skill）gate 通过且版本匹配。
    `rerank=False` 时使用关键词打分（消融"有/无语义排序"，§8.1 的
    "Ours w/o LanceDB reranker" 行）；`scene_quality` 为 v5 兼容形参，不参与判定。
    """
    task_type = canonical_question_type(question_type)
    if not task_type:
        return []  # 题型不可知 → 不检索任何 Skill（fail-closed → 空 Skill baseline）

    decisions: list[tuple[SkillSpec, RetrievalDecision]] = []
    for s in skills:
        d = retrieval_decision(s, scene, task_type)
        if d.ok:
            decisions.append((s, d))
        elif s.requires_metric_evidence:
            # 米制 Skill 被拦下是可审计事件（§13.6 trace 口径），单独告警
            logger.info("M7 米制 Skill %s@%s 不可检索：%s",
                        s.skill_id, s.version, d.reason)
        else:
            logger.debug("M7 Skill %s@%s 不可检索：%s", s.skill_id, s.version, d.reason)
    if not decisions:
        return []  # 无 Skill 命中 → 上层走空 Skill（baseline direct program generation）

    candidates = [s for s, _ in decisions]
    hits_decision = [d for _, d in decisions]
    if not rerank:
        scores = [RANK_WEIGHTS["keyword"] * _keyword_score(question, s) for s in candidates]
    else:
        from skill3d.memory.vector_index import rerank_scores

        docs = [skill_text(s) for s in candidates]
        try:
            sem = rerank_scores(question, docs, embedder=embedder)
        except Exception as exc:  # noqa: BLE001 - reranker 异常不得阻断在线链
            logger.warning("语义排序失败（退化到关键词打分）: %s", exc)
            sem = [_keyword_score(question, s) for s in candidates]
        if len(sem) != len(candidates):
            logger.warning("语义排序返回 %d 个分数（期望 %d），退化到关键词打分",
                           len(sem), len(candidates))
            sem = [_keyword_score(question, s) for s in candidates]
        w_sem = float(RANK_WEIGHTS["semantic"])
        w_mix = float(RANK_WEIGHTS["semantic_mix_keyword"])
        scores = []
        for s, sc in zip(candidates, sem):
            total = w_sem * float(sc)
            if w_mix:
                total += w_mix * _keyword_score(question, s)
            scores.append(total)

    ranked = sorted(zip(candidates, hits_decision, scores),
                    key=lambda t: _rank_key(t[0], t[2]))
    return [_hit(s, sc, d) for s, d, sc in ranked[:top_k]]


def keyword_only_ranking(question: str, skills: Sequence[SkillSpec]) -> list[str]:
    """仅关键词排序的 skill 键（`skill_id@version`）序列（消融对照与测试用）。"""
    scored = [(f"{s.skill_id}@{s.version}", _keyword_score(question, s)) for s in skills]
    return [k for k, _ in sorted(scored, key=lambda t: (-t[1], t[0]))]


def all_task_types() -> tuple[str, ...]:
    """规范题型列表（供 Skill 编写者对齐 `applicable_question_types`）。"""
    return TASK_TYPES
