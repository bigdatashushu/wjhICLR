"""Snapshot freezing validates pins and audits before creating its destination."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import fullqa_custody as c
import fullqa_snapshot as snapshot


def _published_scene(tmp_path):
    root = tmp_path / "published"
    scene = "fixture-scene"
    key = f"precompute-fullqa-20261024-r01/{scene}/gpu_aggregate"
    scope = {"qa_ids": ["fixture-qa"], "split": "induction"}
    entry = {"scene_id": scene, "split": "induction",
             "publication_receipt_sha256": "a" * 64}
    catalog = {"identity_hash": c.IDENTITY, "updated_at": 1,
               "variants": {"base": {"fixture-qa": entry}}}
    raw = json.dumps(catalog, indent=2).encode()
    path = root / "deliverables/prepared_catalog.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    c.durable(root / "audits/summary.json",
              {"identity_hash": c.IDENTITY, "updated_at": 1, "errors": []})
    c.durable(root / "coordinator_jobs.json", {"jobs": [
        {"job_key": key, "identity": {"manifest": {"scenes": {scene: scope}}}}]})
    proof = root / "audits/scenes" / (c.digest(key.encode()) + ".json")
    c.durable(proof, {"status": "verified", "qa_ids": scope["qa_ids"],
                     "publication_receipt_sha256": "a" * 64})
    return root, raw, proof


def test_freeze_pins_checkout_provider_and_preserves_catalog_bytes(tmp_path):
    root, raw, _ = _published_scene(tmp_path)
    output = tmp_path / "snapshot"
    result = snapshot.freeze(root, output)
    plan = c.read(output / "extraction_plan.json")
    provider = Path(plan["provider_file"])
    assert provider == (Path(__file__).resolve().parents[2] / "src" / "skill3d" /
                        "evaluation" / "published_fitset_v11.py")
    assert plan["provider_sha256"] == c.digest(provider.read_bytes())
    assert plan["source_catalog_sha256"] == c.digest(raw)
    assert (output / "source_catalog.json").read_bytes() == raw
    assert result["questions"] == {"induction": 1}
    assert result["scenes"] == 1
    assert plan["model_requests"] == 0 and plan["label_access"] is False
    with pytest.raises(c.Refused, match="snapshot_destination_exists"):
        snapshot.freeze(root, output)


def test_missing_provider_leaves_no_partial_snapshot(tmp_path, monkeypatch):
    root, _, _ = _published_scene(tmp_path)
    output = tmp_path / "snapshot"
    monkeypatch.setattr(snapshot, "_PROVIDER_FILE", tmp_path / "missing-provider.py")
    with pytest.raises(FileNotFoundError):
        snapshot.freeze(root, output)
    assert not output.exists()


def test_scene_audit_mismatch_leaves_no_partial_snapshot(tmp_path):
    root, _, proof = _published_scene(tmp_path)
    output = tmp_path / "snapshot"
    value = c.read(proof)
    value["qa_ids"] = ["different-qa"]
    c.durable(proof, value)
    with pytest.raises(c.Refused, match="scene_audit_mismatch"):
        snapshot.freeze(root, output)
    assert not output.exists()
