"""M16 离线归纳：从 induction split 轨迹聚类归纳 candidate_v0（v6：DeepSeek-V4.1-Flash）。

纪律（§3.3 / §3.4 / 硬约束 2/19）：

- 仅使用 induction split 轨迹；按 题型+scene 聚类；
- ≥ N_min 跨 scene 样本才归纳（TODO_CALIBRATE），禁单题成 Skill；
- 构造 prompt 只含失败类型摘要与输入特征，绝不含 ground truth；
- 产出 CandidateRevision（parent_version=None）；**离线模型文本不得直接修改
  active Skill/Memory，也不得决定 promote/reject**（§3.3）——本模块只产 draft 候选，
  是否准入由 `evolution/admission.py` 的确定性门 + 预注册规则决定。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from skill3d.schemas import CandidateRevision, EpisodeTrace
from skill3d.evolution.firewall import scan_prompt_for_leakage

# TODO_CALIBRATE：跨 scene 最小样本数（与 configs/admission_thresholds.yaml 对齐）
N_MIN_CROSS_SCENE = 3

# prompt 模板版本（冻结配置项，进 RunManifest 的 prompt_version；§3.4）
INDUCE_PROMPT_VERSION = "induce-v1"
# v5 名称别名（过渡期只读；新代码一律用 INDUCE_PROMPT_VERSION）
GPT6_INDUCE_PROMPT_VERSION = INDUCE_PROMPT_VERSION

# `CandidateRevision.created_by` 的受控枚举定义在 `schemas/evolution.py`（本模块不拥有
# 该文件，其字面量仍是 v5 取值）。集中在这里，避免 v5 名称散落在治理链各处。
CREATED_BY_OFFLINE_INDUCTION = "gpt6_induction"


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
    """构造归纳 prompt：只给失败类型摘要与输入特征，绝不含 ground truth（硬约束 19）。

    §3.3 纪律同时写进 system 提示：离线模型**不得**执行实验 / 评分 / 决定 promote。
    """
    lines = [
        "你是 harness3D 的离线 Skill 归纳器（Offline Inducer，DeepSeek-V4.1-Flash）。",
        "基于以下失败类型摘要与输入特征，归纳一个跨场景可泛化的题型级程序合成模板",
        "（不得包含任何具体题目答案）。",
        "你**不**执行实验、不评分、不决定 promote/reject —— 准入由确定性门决定。",
        "## 失败类型摘要",
        *[f"- {s}" for s in failure_summaries],
        "## 输入特征",
        *[f"- {f}" for f in input_features],
        "## 输出",
        "输出 candidate_v0 的 spec_content（模板文本，JSON）。",
    ]
    prompt = "\n".join(lines)
    scan_prompt_for_leakage(prompt)  # 发送前泄漏扫描断言（硬约束 13/19）
    return prompt


def induce_candidate(traces: list[EpisodeTrace],
                     task_type_of: dict[str, str],
                     scene_of: dict[str, str],
                     offline_client=None,
                     n_min: int = N_MIN_CROSS_SCENE,
                     *,
                     gpt6_client=None) -> CandidateRevision | None:
    """从 induction 轨迹归纳 candidate_v0。

    - 跨 scene 样本 < n_min → 抛 InsufficientEvidenceError；
    - `offline_client` 为 None → 返回 None：调用方决定 shadow / 暂停（M16 降级策略）。
      这里的 None **不是**"可以用 mock 顶替"的许可（§3.4：服务不可用不得用 mock 结果
      推进 Readiness）；driver 侧统一记 `service_unavailable` / quarantine。
    - `gpt6_client` 是 v5 形参别名（过渡期只读，§20 已废止 GPT-6），仅做转发。
    """
    if offline_client is None and gpt6_client is not None:
        offline_client = gpt6_client
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

    if offline_client is None:
        # 离线模型不可用：低风险候选可 shadow，高风险暂停（不得让在线链等）
        return None
    spec_content = offline_client.chat(prompt)

    return CandidateRevision(
        revision_id=f"rev-{uuid.uuid4().hex[:12]}",
        root_candidate_id=f"cand-{uuid.uuid4().hex[:12]}",
        parent_version=None,  # v0 无父版本
        candidate_type="skill",
        spec_content=spec_content,
        status="draft",
        induction_trace_refs=[t.episode_id for t in traces],
        evidence_lineage_ref="",
        created_by=CREATED_BY_OFFLINE_INDUCTION,  # type: ignore[arg-type]
        created_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = [
    "CREATED_BY_OFFLINE_INDUCTION",
    "GPT6_INDUCE_PROMPT_VERSION",
    "INDUCE_PROMPT_VERSION",
    "N_MIN_CROSS_SCENE",
    "InsufficientEvidenceError",
    "build_induction_prompt",
    "cluster_traces",
    "induce_candidate",
]
