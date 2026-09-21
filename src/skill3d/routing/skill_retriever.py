"""Skill Retriever（§4 M7）：硬过滤 + 语义排序（G-20 接线）。

硬过滤为真实逻辑：
- requires_artifacts 满足（由 scene.route 推导可用产物集合）
- minimum_quality 达标（scene 质量分）
- supported_coordinate_frames 含 scene.frame
- metric_scale_required == True 时 scene.scale_known 必须为 True
- `state` 非 promoted（若 SkillSpec 带该字段）不出现在结果中

语义排序（G-20）：优先 LanceDB 向量候选 + `bge-reranker-v2-m3`；两者不可用时
使用 `memory/vector_index.py` 的确定性 hashing 嵌入余弦（无需模型下载，可复现）。
**排序口径与关键词打分不同**（字符 n-gram 参与），因此消融"有/无语义排序"可比较。

无命中 → 返回空列表（上层走空 Skill baseline：无模板直生成 program）。
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional, Sequence

from skill3d.schemas import RetrievedSkill, SceneState, SkillSpec
from skill3d.tools.contract import (
    ARTIFACT_DEPTH,
    ARTIFACT_FRAMES,
    ARTIFACT_INTRINSICS,
    ARTIFACT_OBJECTS,
    ARTIFACT_POINT_CLOUD,
    ARTIFACT_POSES,
    ARTIFACT_SCALE,
    ROUTE_ARTIFACTS,
)

from .task_classifier import TASK_TYPES, canonical_task

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3  # TODO_CALIBRATE
DEFAULT_CANDIDATES = 50  # TODO_CALIBRATE：向量候选池大小

# route → 可用重建产物集合：与 `tools.contract.ROUTE_ARTIFACTS` **同一份判定**
# （硬约束 16：路由在重建之后；D-3：静态裁剪与执行期 fail-closed 共用一套映射）
_ROUTE_ARTIFACTS = {route: set(arts) for route, arts in ROUTE_ARTIFACTS.items()}

# Skill 的 requires_artifacts 词汇与 Tool 产物词汇表不同源，这里做归一别名，
# 使人工/离线归纳写出的 Skill 仍能对齐 route 映射（§9 术语表口径）。
_ARTIFACT_ALIASES: dict[str, str] = {
    "point_map": ARTIFACT_POINT_CLOUD,
    "point_cloud": ARTIFACT_POINT_CLOUD,
    "c2w": ARTIFACT_POSES,
    "poses": ARTIFACT_POSES,
    "depth_maps": ARTIFACT_DEPTH,
    "depth": ARTIFACT_DEPTH,
    "confidence": ARTIFACT_SCALE,   # 置信度/尺度同属"重建可信度"产物族
    "scale": ARTIFACT_SCALE,
    "objects": ARTIFACT_OBJECTS,
    "frames": ARTIFACT_FRAMES,
    "intrinsics": ARTIFACT_INTRINSICS,
}

# 允许进入检索的 Skill 状态（§7：未 promoted 的候选不得生效）
_ACTIVE_STATES = ("promoted", "consolidated")
# G-11/D-2：metric 尺度类 Skill 只接受 medium/high 档（low 一律硬过滤掉）
ACCEPTED_SCALE_CONFIDENCE: tuple[str, ...] = ("medium", "high")


def normalize_artifact_name(name: str) -> str:
    """Skill 声明的产物名 → 规范产物词汇（未知名原样返回，fail-closed 下会被过滤）。"""
    key = str(name).strip().lower()
    return _ARTIFACT_ALIASES.get(key, key)


def metric_scale_usable(scene: SceneState) -> bool:
    """G-11/D-2/v4 HC33：米制 Skill 的尺度可用性（单一事实源）。

    v4 口径：以 `scene.allowed_metric_tasks` 为准（逐题型授权）——**至少授权一个
    米制题型**才算尺度可用。**不得**只凭全局 `scale_known` 或粗粒度
    `scale_confidence` 放行（§3 M7 明文）。
    """
    if not scene.scale_known:
        return False
    tasks = {str(t) for t in (getattr(scene, "allowed_metric_tasks", None) or set())}
    if tasks:
        return True
    # 兼容路径：老 SceneState 无 allowed_metric_tasks 字段时退回置信档判据
    # （None / low 一律视为不可用，fail-closed）
    return scene.scale_confidence in ACCEPTED_SCALE_CONFIDENCE


def metric_task_authorized(scene: SceneState, task_type: Optional[str]) -> bool:
    """v4 HC33：该题型是否被本次尺度评估授权（米制 Skill 的逐题硬过滤）。

    `task_type` 为空 = 尚未分类 → 不授权（米制 Skill 不得在不知道题型时进入候选）。
    """
    if not task_type:
        return False
    tasks = {str(t) for t in (getattr(scene, "allowed_metric_tasks", None) or set())}
    if not tasks:
        return metric_scale_usable(scene)     # 兼容路径（见上）
    return str(task_type) in tasks


def hard_filter(
    skill: SkillSpec,
    scene: SceneState,
    scene_quality: Optional[float] = None,
    task_type: Optional[str] = None,
) -> bool:
    """§4 M7 伪代码 hard_filter 的真实实现。"""
    available = (_ROUTE_ARTIFACTS.get(scene.route, set())
                 | {normalize_artifact_name(a) for a in (scene.available_artifacts or set())})
    requires = {normalize_artifact_name(a) for a in skill.requires_artifacts}
    if not requires.issubset(available):
        return False
    if task_type is not None and skill.task_type != task_type:
        return False
    if scene_quality is not None and scene_quality < skill.minimum_quality:
        return False
    if scene.frame not in skill.supported_coordinate_frames:
        return False
    if skill.metric_scale_required:
        # v4 HC33：米制 Skill **逐题型**硬过滤，不得只凭一个全局 scale_known /
        # scale_confidence 粗粒度放行（§3 M7）
        effective_task = task_type or getattr(skill, "task_type", None)
        if not metric_task_authorized(scene, effective_task):
            return False
    # 状态门（duck-typing：SkillSpec 本身无 state，registry 层已过滤）
    state = getattr(skill, "state", None)
    if state is not None and str(state) not in _ACTIVE_STATES:
        return False
    return True


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _keyword_score(question: str, skill: SkillSpec) -> float:
    """退化打分：question 与 (task_type + description) 的 token 重叠率。"""
    q = _tokenize(question)
    if not q:
        return 0.0
    s = _tokenize(f"{skill.task_type} {skill.description}")
    return len(q & s) / len(q)


def skill_text(skill: SkillSpec) -> str:
    """索引文本：题型 + 描述 + 模板签名（模板签名让"怎么做"参与检索）。"""
    tmpl = str(getattr(skill, "call_graph_template", "") or "")
    return f"{skill.task_type} {skill.description} {tmpl}"


def build_index(skills: Sequence[SkillSpec], *, embedder=None):
    """为候选 Skill 建向量索引（内存；可选 `to_lancedb` 持久化）。"""
    from skill3d.memory.vector_index import SkillVectorIndex

    idx = SkillVectorIndex(embedder)
    for s in skills:
        key = f"{s.skill_id}@{s.semver}"
        idx.add_text(key, skill_text(s), task_type=s.task_type,
                     metric_scale_required=bool(s.metric_scale_required),
                     minimum_quality=float(s.minimum_quality),
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

    `rerank=False` 时使用关键词打分（消融"有/无语义排序"，§8.1 的
    "Ours w/o LanceDB reranker" 行）。
    """
    task_type: Optional[str] = None
    if question_type is not None:
        task_type = canonical_task(question_type)   # 规范题型（用于与 SkillSpec 对齐）

    hard = [s for s in skills
            if hard_filter(s, scene, scene_quality=scene_quality, task_type=task_type)]
    if not hard:
        return []  # 无 Skill 命中 → 上层走空 Skill（baseline direct program generation）

    if not rerank:
        scored = [(s, _keyword_score(question, s)) for s in hard]
        ranked = sorted(scored, key=lambda t: (-t[1], f"{t[0].skill_id}@{t[0].semver}"))
        return [RetrievedSkill(skill_semver=f"{s.skill_id}@{s.semver}", score=float(sc),
                               hard_filter_passed=True) for s, sc in ranked[:top_k]]

    from skill3d.memory.vector_index import rerank_scores

    docs = [skill_text(s) for s in hard]
    try:
        scores = rerank_scores(question, docs, embedder=embedder)
    except Exception as exc:  # noqa: BLE001 - reranker 异常不得阻断在线链
        logger.warning("语义排序失败（退化到关键词打分）: %s", exc)
        scores = [_keyword_score(question, s) for s in hard]
    if len(scores) != len(hard):
        logger.warning("语义排序返回 %d 个分数（期望 %d），退化到关键词打分",
                       len(scores), len(hard))
        scores = [_keyword_score(question, s) for s in hard]

    ranked = sorted(zip(hard, scores),
                    key=lambda t: (-float(t[1]), f"{t[0].skill_id}@{t[0].semver}"))
    return [
        RetrievedSkill(skill_semver=f"{s.skill_id}@{s.semver}", score=float(sc),
                       hard_filter_passed=True)
        for s, sc in ranked[:top_k]
    ]


def keyword_only_ranking(question: str, skills: Sequence[SkillSpec]) -> list[str]:
    """仅关键词排序的 skill semver 序列（消融对照与测试用）。"""
    scored = [(f"{s.skill_id}@{s.semver}", _keyword_score(question, s)) for s in skills]
    return [k for k, _ in sorted(scored, key=lambda t: (-t[1], t[0]))]


def all_task_types() -> tuple[str, ...]:
    """规范题型列表（供 Skill 编写者对齐 task_type）。"""
    return TASK_TYPES
