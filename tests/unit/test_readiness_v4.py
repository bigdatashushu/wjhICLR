"""HC34 Experiment Readiness Gate 单测（§11.1；v5 能力名见 HC35/36/39）。

核心不变量：
- 四级是**单调布尔**（`real_poc_verified ⇒ connected ⇒ implemented`），违反即构造报错；
- 未达 `paper_eligible` 的能力**不得写主表**（`assert_paper_eligible` 抛错）；
- 状态变化落 `readiness_manifest.json`（含 evidence refs / commit / split hash / blockers）；
- 主表需 ≥3 seed（§7）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill3d.readiness.manifest import (
    PAPER_MIN_SEEDS,
    READINESS_MANIFEST_VERSION,
    ReadinessGateError,
    ReadinessManifest,
    baseline_manifest,
    capability_readiness,
    load_readiness_manifest,
    split_hash,
    write_readiness_manifest,
)
from skill3d.schemas.readiness import ExperimentReadiness


# ------------------------------------------------------------- 单调性 ----

@pytest.mark.parametrize("kwargs", [
    {"connected": True},
    {"real_poc_verified": True},
    {"paper_eligible": True},
    {"connected": True, "paper_eligible": True},
])
def test_readiness_is_monotonic(kwargs):
    """HC34：后一层为真必须蕴含所有前层为真（否则构造即报错）。"""
    with pytest.raises(Exception):
        ExperimentReadiness(capability="c", **kwargs)


def test_monotonic_full_chain_is_accepted():
    r = ExperimentReadiness(capability="c", implemented=True, connected=True,
                            real_poc_verified=True, paper_eligible=True)
    assert r.paper_eligible and r.as_row()["capability"] == "c"


def test_promote_to_enables_lower_levels():
    r = ExperimentReadiness(capability="c").promote_to(
        "real_poc_verified", evidence_refs=["receipt://poc-1"], note="真机跑通")
    assert r.implemented and r.connected and r.real_poc_verified
    assert not r.paper_eligible
    assert r.evidence_refs == ["receipt://poc-1"]
    assert "真机跑通" in r.notes
    with pytest.raises(ValueError):
        r.promote_to("unknown_level")


def test_blocked_by_records_blockers_without_duplicates():
    r = ExperimentReadiness(capability="c", implemented=True)
    r2 = r.blocked_by("缺 ARKitScenes GT 位姿").blocked_by("缺 ARKitScenes GT 位姿")
    assert r2.blockers == ["缺 ARKitScenes GT 位姿"]


# ---------------------------------------------------------- 门禁与 manifest ----

def test_assert_paper_eligible_blocks_unverified_capability():
    m = baseline_manifest()
    with pytest.raises(ReadinessGateError):
        m.assert_paper_eligible("scale_recovery")
    with pytest.raises(ReadinessGateError):
        m.assert_paper_eligible("never_registered")


def test_assert_paper_eligible_passes_only_when_all_four_true():
    m = ReadinessManifest()
    m.set(capability_readiness("demo", implemented=True, connected=True,
                               real_poc_verified=True, paper_eligible=True,
                               evidence_refs=["receipt://demo"]))
    m.assert_paper_eligible("demo")           # 不抛即通过
    assert m.paper_eligible() == ["demo"]


def test_g8_capability_is_gone_with_the_metric():
    """G8 指标已按附录 A 删除 → readiness 基线里不得再出现该能力项。"""
    m = baseline_manifest()
    assert m.get("g8_bbox_coverage") is None


def test_baseline_manifest_reflects_v5_reality():
    """§10.5 实况基线：当前没有任何能力达到 paper_eligible。"""
    m = baseline_manifest(code_commit="deadbeef")
    assert m.paper_eligible() == []
    blocked = {b["capability"] for b in m.blocked()}
    for cap in ("official_vggsfm_ba", "vggt_sparse_ba", "scale_recovery",
                "tool_contract_replay", "gpt6_offline_governance"):
        assert cap in blocked
    # 每一项都必须写明阻断原因（不许空 blockers）
    for b in m.blocked():
        assert b["blockers"], b["capability"]


def test_official_vggsfm_ba_is_rejected_not_connected():
    """v5 HC35：官方 BA 保留 implemented=true，但 connected/poc/paper 全 false。"""
    m = baseline_manifest()
    c = m.get("official_vggsfm_ba")
    assert c is not None
    assert c.implemented and not c.connected
    assert not c.real_poc_verified and not c.paper_eligible
    assert any("rejected_on_24g_oom" in b for b in c.blockers)


def test_vggt_sparse_ba_never_exceeds_implemented_without_l1():
    """v5 HC36：`vggt_sparse_ba` 未过 §10.1 L1 前，除 `implemented` 外全部为 false。

    实况：代码（pair graph/tracks/receipt + 前端 + PyCOLMAP 后端）已落码 →
    `implemented=true`；L1 单 episode 实测 `rejected_on_l1_gate` → 其余三项必须为 false、
    `connected=false`（生产开关恒关），且证据里写明止损理由。
    """
    m = baseline_manifest()
    c = m.get("vggt_sparse_ba")
    assert c is not None
    assert not any([c.connected, c.real_poc_verified, c.paper_eligible])
    assert any("rejected_on_l1_gate" in b for b in c.blockers)


def test_v5_schema_and_golden_capabilities_present():
    """v5 版本隔离的可审计落点必须在 readiness 里可见（HC39）。"""
    m = baseline_manifest()
    assert m.get("v5_schema_and_legacy_isolation") is not None
    golden = m.get("v5_golden")
    assert golden is not None and not golden.paper_eligible


def test_baseline_manifest_scale_recovery_is_connected_but_not_verified():
    """尺度多锚点：implemented/connected 可真，real_poc_verified 必须假（HC34）。"""
    m = baseline_manifest()
    c = m.get("scale_recovery")
    assert c is not None
    assert c.implemented and c.connected
    assert not c.real_poc_verified and not c.paper_eligible
    assert any("TODO_USER_INPUT" in b or "标定" in b for b in c.blockers)


def test_manifest_roundtrip_and_persistence(tmp_path):
    m = baseline_manifest(generated_at="2026-09-20T00:00:00Z", code_commit="abc123",
                          data_split_hash=split_hash(["s1", "s2"]),
                          scale_calibration_id="")
    out = write_readiness_manifest(m, tmp_path / "readiness.json")
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["schema_version"] == READINESS_MANIFEST_VERSION
    assert data["code_commit"] == "abc123"
    assert data["summary"]["paper_eligible"] == []
    assert len(data["summary"]["blocked"]) == len(m.capabilities)
    back = load_readiness_manifest(out)
    assert back is not None and len(back.capabilities) == len(m.capabilities)


def test_manifest_missing_file_returns_none(tmp_path):
    assert load_readiness_manifest(tmp_path / "nope.json") is None


def test_manifest_rejects_unknown_schema_version(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"schema_version": "v0"}), encoding="utf-8")
    with pytest.raises(ReadinessGateError):
        load_readiness_manifest(p)


def test_split_hash_is_order_insensitive_and_content_addressed():
    assert split_hash(["b", "a"]) == split_hash(["a", "b"])
    assert split_hash(["a"]) != split_hash(["a", "b"])
    assert len(split_hash(["a"])) == 64


def test_seed_requirement():
    """§7：主表需 ≥3 seed。"""
    m = ReadinessManifest()
    assert m.seed_requirement_ok(3) and not m.seed_requirement_ok(2)
    assert PAPER_MIN_SEEDS == 3


def test_cli_writes_manifest_and_gates(tmp_path, capsys):
    from skill3d.readiness.manifest import main

    out = tmp_path / "readiness.json"
    rc = main(["--out", str(out), "--code-commit", "c0ffee"])
    assert rc == 0 and out.is_file()
    assert "paper_eligible" in capsys.readouterr().out
    # --check 未达标 → 退出码 1（不得因"代码存在"就放行主表）
    rc = main(["--out", str(out), "--check", "official_vggsfm_ba"])
    assert rc == 1
