"""Snapshot freezing and replay preserve immutable receipts after real checks."""
import json
from pathlib import Path
import shutil
import sys
import tarfile

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


@pytest.fixture
def extractable_snapshot(tmp_path, monkeypatch):
    from skill3d.adapters import episode_source
    from skill3d.online import runner
    from test_published_fitset_v11 import _published

    # Pinned loading replaces this module; restore it for other test suites.
    module_name = "skill3d.evaluation.published_fitset_v11"
    monkeypatch.setitem(sys.modules, module_name, sys.modules[module_name])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("snapshot replay must not run datasets, models or solver")

    monkeypatch.setattr(episode_source, "_load_from_sources", forbidden)
    monkeypatch.setattr(runner, "_bind_objects_best_effort", forbidden)
    monkeypatch.setattr(runner, "run_episode", forbidden)
    case = _published(tmp_path, n_questions=1)
    scene = case.frozen.manifest["scene_id"]
    key = f"precompute-fullqa-20261024-r01/{scene}/gpu_aggregate"
    scope = {"qa_ids": case.frozen.manifest["qa_ids"], "split": "induction"}
    root = tmp_path / "fullqa"
    c.durable(root / "deliverables/prepared_catalog.json",
              {"schema": "harness3d-published-binding-catalog/1",
               "identity_hash": c.IDENTITY, "updated_at": 1,
               "variants": {"base": case.catalog}})
    c.durable(root / "audits/summary.json",
              {"identity_hash": c.IDENTITY, "updated_at": 1, "errors": []})
    c.durable(root / "coordinator_jobs.json", {"jobs": [
        {"job_key": key, "identity": {"manifest": {"scenes": {scene: scope}}}}]})
    hashed = c.digest(key.encode())
    c.durable(root / "audits/scenes" / (hashed + ".json"),
              {"status": "verified", "qa_ids": scope["qa_ids"],
               "publication_receipt_sha256": case.catalog["qa-00"][
                   "publication_receipt_sha256"]})
    archive = root / "inputs/jobs" / (hashed + ".tar")
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as stream:
        for path in sorted(case.frozen.root.rglob("*")):
            stream.add(path, arcname=str(path.relative_to(case.frozen.root)),
                       recursive=False)
    provider = tmp_path / "pinned-provider.py"
    shutil.copyfile(snapshot._PROVIDER_FILE, provider)
    monkeypatch.setattr(snapshot, "_PROVIDER_FILE", provider)
    output = tmp_path / "snapshot"
    snapshot.freeze(root, output)
    return case, output


def _records(root):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
            for path in root.rglob("*") if path.is_file()}


def test_extract_rerun_revalidates_and_returns_original_receipt(
    extractable_snapshot, monkeypatch,
):
    from skill3d.cluster import scene_compute
    from skill3d.segmentation import prepared_binding_store as bindings
    from skill3d.storage.relocation import ValidatedRelocation

    case, output = extractable_snapshot
    clock = [100.0]
    monkeypatch.setattr(snapshot.time, "time", lambda: clock[0])
    calls = []
    load_scene = scene_compute._load_scene
    recompute = ValidatedRelocation.recompute_scene_input_identity
    load_binding = bindings.load_prepared_binding_reference

    def scene_spy(root, *, question_cap):
        calls.append(("scene", question_cap))
        return load_scene(root, question_cap=question_cap)

    def identity_spy(resolver, **kwargs):
        assert len(kwargs["pixels"]) == 32
        assert kwargs["episode"]["ground_truth"] == ""
        calls.append("identity")
        return recompute(resolver, **kwargs)

    def binding_spy(reference, **kwargs):
        assert reference == case.catalog["qa-00"]["reference"]
        calls.append("binding")
        return load_binding(reference, **kwargs)

    monkeypatch.setattr(scene_compute, "_load_scene", scene_spy)
    monkeypatch.setattr(ValidatedRelocation, "recompute_scene_input_identity", identity_spy)
    monkeypatch.setattr(bindings, "load_prepared_binding_reference", binding_spy)
    original = snapshot.extract(output)
    assert original["at"] == 100.0 and original["status"] == "verified"
    assert original["normal_loader_checked_scenes"] == 1
    assert original["model_requests"] == 0 and original["label_access"] is False
    assert original["solver_ran"] is False and original["scoring_ran"] is False
    records = _records(output) | _records(case.store.root)
    for timestamp in (200.0, 300.0):
        clock[0] = timestamp
        calls.clear()
        assert snapshot.extract(output) == original
        assert calls == [("scene", 1), "identity", "identity", "binding"]
        assert _records(output) | _records(case.store.root) == records
    assert c.read(output / "preflight.json") == original


@pytest.mark.parametrize("mutation, error, message", [
    ("catalog", c.Refused, "catalog_snapshot_changed"),
    ("provider", c.Refused, "source_pin_changed"),
    ("scene", c.Refused, "extracted_manifest_changed"),
    ("frame", ValueError, None),
    ("geometry", ValueError, None),
    ("binding", ValueError, None),
])
def test_extract_rerun_rejects_changed_inputs_without_replacing_preflight(
    extractable_snapshot, mutation, error, message,
):
    case, output = extractable_snapshot
    original = snapshot.extract(output)
    preflight = output / "preflight.json"
    record = _records(output)[preflight]
    plan = c.read(output / "extraction_plan.json")
    root = Path(plan["scenes"]["scene-one"]["output_root"])
    entry = case.catalog["qa-00"]
    paths = {
        "catalog": output / "source_catalog.json",
        "provider": Path(plan["provider_file"]),
        "scene": root / "scene.json",
        "frame": root / "data/frames/0000.png",
        "geometry": Path(entry["scene_artifact"]).parent / "depth_maps.npy",
        "binding": case.store.entry_path(entry["reference"]["key"]) / "payload/binding.json",
    }
    path = paths[mutation]
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(error, match=message):
        snapshot.extract(output)
    assert _records(output)[preflight] == record
    assert c.read(preflight) == original


@pytest.mark.parametrize("field, value", [
    ("status", "not_verified"),
    ("manifest_sha256", "0" * 64),
    ("normal_loader_checked_scenes", 0),
])
def test_extract_rerun_rejects_non_timestamp_preflight_changes(
    extractable_snapshot, monkeypatch, field, value,
):
    from skill3d.cluster import scene_compute

    _, output = extractable_snapshot
    original = snapshot.extract(output)
    preflight = output / "preflight.json"
    c.durable(preflight, {**original, field: value})
    record = _records(output)[preflight]
    load_scene = scene_compute._load_scene
    calls = []

    def scene_spy(root, *, question_cap):
        calls.append(root)
        return load_scene(root, question_cap=question_cap)

    monkeypatch.setattr(scene_compute, "_load_scene", scene_spy)
    with pytest.raises(c.Refused, match="immutable_record_changed"):
        snapshot.extract(output)
    assert len(calls) == 1
    assert _records(output)[preflight] == record


def test_extract_resumes_missing_preflight_after_persistence_failure(
    extractable_snapshot, monkeypatch,
):
    _, output = extractable_snapshot
    preflight = output / "preflight.json"
    durable = c.durable
    failed = False
    clock = [100.0]
    monkeypatch.setattr(snapshot.time, "time", lambda: clock[0])

    def interrupted(path, value, *, immutable=False):
        nonlocal failed
        if Path(path) == preflight and not failed:
            failed = True
            raise OSError("fixture preflight persistence interrupted")
        return durable(path, value, immutable=immutable)

    monkeypatch.setattr(c, "durable", interrupted)
    with pytest.raises(OSError, match="preflight persistence interrupted"):
        snapshot.extract(output)
    assert (output / "published_inputs.json").exists()
    assert list((output / "extracted").glob("*.json"))
    assert not preflight.exists()
    records = _records(output)
    clock[0] = 200.0
    result = snapshot.extract(output)
    assert result["status"] == "verified" and result["at"] == 200.0
    assert c.read(preflight) == result
    assert {p: row for p, row in _records(output).items() if p != preflight} == records
