"""M15 paired A/B 编排接口（硬约束 18）：

同一 episode 的 A/B 两臂必须使用**完全相同**的 ReconstructionArtifact，
以隔离 Skill 的因果增益。本模块只做编排：构造分支对 → 调 evolution sandbox
执行 → 收 PairedOutcome。评分见 evolution/paired_score.py。
"""

from __future__ import annotations

from typing import Callable, Sequence

from skill3d.schemas import EnvironmentSnapshot, ExperimentBranch


class PairedArtifactMismatchError(AssertionError):
    """A/B 两臂 reconstruction_artifact_ref 不一致（违反硬约束 18）。"""


class PairedFrameSetMismatchError(AssertionError):
    """A/B 两臂 frame_set_hash 不一致（违反硬约束 18/21：禁止双帧集）。"""


def assert_paired_outcomes_share_artifact(outs_a: Sequence, outs_b: Sequence) -> None:
    """逐 episode 断言：A/B 两臂同 artifact ref 且同 frame_set_hash（硬约束 18/21）。

    两臂的 scene 不同会各自有不同帧集，因此判定必须**逐 episode 配对**做，
    不能在整批层面比单一哈希。
    """
    if len(outs_a) != len(outs_b):
        raise PairedArtifactMismatchError(
            f"A/B 两臂 episode 数不同: {len(outs_a)} vs {len(outs_b)}")
    for oa, ob in zip(outs_a, outs_b):
        ra = str(getattr(oa, "artifact_ref", "") or "")
        rb = str(getattr(ob, "artifact_ref", "") or "")
        if ra and rb and ra != rb:
            raise PairedArtifactMismatchError(
                f"episode {getattr(oa, 'qa_id', '?')} 两臂 artifact 不同: {ra} vs {rb}"
                "（硬约束 18：paired A/B 必须复用同一 ReconstructionArtifact）")
        ha = str(getattr(oa, "frame_set_hash", "") or "")
        hb = str(getattr(ob, "frame_set_hash", "") or "")
        if ha and hb and ha != hb:
            raise PairedFrameSetMismatchError(
                f"episode {getattr(oa, 'qa_id', '?')} 两臂 frame_set_hash 不同:"
                f" {ha[:12]} vs {hb[:12]}（硬约束 21：禁止双帧集）")


def assert_same_reconstruction_artifact(branch_a: ExperimentBranch,
                                        branch_b: ExperimentBranch,
                                        snapshot: EnvironmentSnapshot) -> None:
    """硬约束 18 断言：paired 两臂同源重建 artifact、同 snapshot。"""
    if branch_a.snapshot_id != branch_b.snapshot_id:
        raise PairedArtifactMismatchError(
            f"A/B 分支 snapshot 不同: {branch_a.snapshot_id} vs {branch_b.snapshot_id}"
        )
    if branch_a.snapshot_id != snapshot.snapshot_id:
        raise PairedArtifactMismatchError("分支与 snapshot 不匹配")
    # snapshot 内 reconstruction_artifact_ref 唯一 → 两臂必然同源；此处显式留审计断言
    if not snapshot.reconstruction_artifact_ref:
        raise PairedArtifactMismatchError("snapshot 缺少 reconstruction_artifact_ref")


def run_paired_ab(branch_a: ExperimentBranch, branch_b: ExperimentBranch,
                  snapshot: EnvironmentSnapshot,
                  episode_refs: list[str],
                  run_branch_fn: Callable[[ExperimentBranch, list[str]], dict]) -> dict:
    """paired A/B 编排：先断言同源 artifact，再依次执行两臂并返回配对结果。

    run_branch_fn：由 evolution sandbox 提供的分支执行接口（本模块不实现沙箱）。
    返回 {"arm_a": ..., "arm_b": ..., "reconstruction_artifact_ref": ...}。
    """
    assert_same_reconstruction_artifact(branch_a, branch_b, snapshot)
    result_a = run_branch_fn(branch_a, episode_refs)
    result_b = run_branch_fn(branch_b, episode_refs)
    return {
        "arm_a": result_a,
        "arm_b": result_b,
        "reconstruction_artifact_ref": snapshot.reconstruction_artifact_ref,
        "n_episodes": len(episode_refs),
    }
