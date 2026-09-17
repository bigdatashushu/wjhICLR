"""M15 promote 原子切换测试（硬约束 12）。"""

import json

import pytest

from skill3d.schemas import CandidateRevision
from skill3d.skills.promote_atomic import (
    SnapshotValidationError,
    promote,
    read_active_snapshot,
    rollback,
)


def _cand(rid: str, root: str, parent: str | None = None) -> CandidateRevision:
    return CandidateRevision(
        revision_id=rid, root_candidate_id=root, parent_version=parent,
        candidate_type="skill", spec_content=f"template-of-{rid}", status="draft",
        induction_trace_refs=["t1"], evidence_lineage_ref="",
        created_by="gpt6_induction", created_at="2026-09-17T00:00:00Z")


def _active_id(store_dir) -> str:
    return json.loads((store_dir / "active_snapshot.json").read_text())["snapshot_id"]


def test_promote_switches_pointer_and_keeps_old(tmp_path):
    log = []
    snap1 = promote(tmp_path, _cand("rev-1", "cand-1"), promotion_log=log)
    assert _active_id(tmp_path) == snap1["snapshot_id"]
    assert "rev-1" in snap1["entries"]
    assert log[0]["snapshot_before"] == "genesis"

    snap2 = promote(tmp_path, _cand("rev-2", "cand-1", parent="rev-1"),
                    promotion_log=log)
    assert _active_id(tmp_path) == snap2["snapshot_id"]
    assert snap2["parent_snapshot_id"] == snap1["snapshot_id"]
    # 旧 snapshot 文件保留
    assert (tmp_path / f"snapshot_{snap1['snapshot_id']}.json").exists()


def test_rollback_restores_old_snapshot(tmp_path):
    log = []
    snap1 = promote(tmp_path, _cand("rev-1", "cand-1"), promotion_log=log)
    snap2 = promote(tmp_path, _cand("rev-2", "cand-1", parent="rev-1"),
                    promotion_log=log)
    rolled = rollback(tmp_path, snap1["snapshot_id"], promotion_log=log)
    assert _active_id(tmp_path) == snap1["snapshot_id"]
    assert "rev-2" not in rolled["entries"]


def test_validate_failure_does_not_switch(tmp_path):
    snap1 = promote(tmp_path, _cand("rev-1", "cand-1"))
    before = _active_id(tmp_path)
    with pytest.raises(SnapshotValidationError):
        # known_roots 不含 cand-2 → 引用不可解析 → 不切换
        promote(tmp_path, _cand("rev-2", "cand-2"), known_roots={"cand-1"})
    assert _active_id(tmp_path) == before == snap1["snapshot_id"]


def test_read_active_genesis_when_no_pointer(tmp_path):
    snap = read_active_snapshot(tmp_path)
    assert snap["snapshot_id"] == "genesis"
