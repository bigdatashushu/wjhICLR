"""M16 GPT-6 离线归纳：从 induction split 轨迹聚类归纳 candidate_v0。

纪律（§7 / 硬约束 2/19）：
- 仅使用 induction split 轨迹；按 题型+scene 聚类；
- ≥ N_min 跨 scene 样本才归纳（TODO_CALIBRATE），禁单题成 Skill；
- 构造 GPT-6 prompt 只含失败类型摘要与输入特征，绝不含 ground truth；
- 产出 CandidateRevision（created_by="gpt6_induction"，parent_version=None）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from skill3d.schemas import CandidateRevision, EpisodeTrace
from skill3d.evolution.firewall import scan_prompt_for_leakage

# TODO_CALIBRATE：跨 scene 最小样本数（与 configs/admission_thresholds.yaml 对齐）
N_MIN_CROSS_SCENE = 3

GPT6_INDUCE_PROMPT_VERSION = "induce-v1"  # prompt 模板版本化


class InsufficientEvidenceError(ValueError):
    """跨 scene 样本不足 N_min，不归纳。"""


def cluster_traces(traces: list[EpisodeTrace],
                   task_type_of: dict[str, str],
                   scene_of: dict[str, str]) -> dict[str, list[EpisodeTrace]]:
    """按 (题型, scene) 聚类 induction 轨迹。key = "task_type|scene"。"""
    clusters: dict[str, list[EpisodeTrace]] = {}
    for t in traces:
        key = f"{task_type_of.get(t.episode_id, 'unknown')}|{scene_of.get(t.episode_id, 'unknown')}"
        clusters.setdefault(key, []).append(t)
    return clusters


def build_induction_prompt(failure_summaries: list[str],
                           input_features: list[str]) -> str:
    """构造归纳 prompt：只给失败类型摘要与输入特征，绝不含 ground truth（硬约束 19）。"""
    lines = [
        "你是离线 Skill 归纳器。基于以下失败类型摘要与输入特征，",
        "归纳一个跨场景可泛化的题型级程序合成模板（不得包含任何具体题目答案）。",
        "## 失败类型摘要",
        *[f"- {s}" for s in failure_summaries],
        "## 输入特征",
        *[f"- {f}" for f in input_features],
        "## 输出",
        "输出 candidate_v0 的 spec_content（模板文本）。",
    ]
    prompt = "\n".join(lines)
    scan_prompt_for_leakage(prompt)  # 发送前泄漏扫描断言（硬约束 13/19）
    return prompt


def induce_candidate(traces: list[EpisodeTrace],
                     task_type_of: dict[str, str],
                     scene_of: dict[str, str],
                     gpt6_client=None,
                     n_min: int = N_MIN_CROSS_SCENE) -> CandidateRevision | None:
    """从 induction 轨迹归纳 candidate_v0。

    - 跨 scene 样本 < n_min → 抛 InsufficientEvidenceError；
    - gpt6_client 为 None 时（未配置）返回 None：调用方决定 shadow/暂停（M16 降级策略）。
    """
    clusters = cluster_traces(traces, task_type_of, scene_of)
    covered_scenes = {key.split("|", 1)[1] for key in clusters}
    if len(covered_scenes) < n_min or len(traces) < n_min:
        raise InsufficientEvidenceError(
            f"跨 scene 样本不足: scenes={len(covered_scenes)}, traces={len(traces)}, "
            f"N_min={n_min}（TODO_CALIBRATE）"
        )

    failure_summaries = sorted({
        ",".join(t.failure.categories) for t in traces if t.failure is not None
    })
    input_features = sorted({task_type_of.get(t.episode_id, "unknown") for t in traces})
    prompt = build_induction_prompt(failure_summaries, input_features)

    if gpt6_client is None:
        return None  # GPT-6 不可用：低风险候选可 shadow，高风险暂停（不得让在线链等）
    spec_content = gpt6_client.chat(prompt)

    return CandidateRevision(
        revision_id=f"rev-{uuid.uuid4().hex[:12]}",
        root_candidate_id=f"cand-{uuid.uuid4().hex[:12]}",
        parent_version=None,  # v0 无父版本
        candidate_type="skill",
        spec_content=spec_content,
        status="draft",
        induction_trace_refs=[t.episode_id for t in traces],
        evidence_lineage_ref="",
        created_by="gpt6_induction",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
