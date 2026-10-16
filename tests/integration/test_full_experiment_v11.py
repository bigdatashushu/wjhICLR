"""Expanded campaign contracts, with real paired execution and no model/GPU."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "full_experiment_entry", ROOT / "scripts/run_full_experiment_v11.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
fixture_spec = importlib.util.spec_from_file_location(
    "full_pair_fixtures", Path(__file__).with_name("test_skill_ablation_v11.py"))
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)
panel = fixtures.panel
ab = fixtures.ab


def metadata():
    rows, anchors = [], []
    for task_index, task in enumerate(full.TASKS):
        for index in range(48):
            row = {
                "qa_id": f"{task_index}-{index:03}", "dataset": "scannet",
                "scene_name": f"scene-{index % 12:02}",
                "question_type": (task + "_" + ("easy", "medium", "hard")[index % 3]
                                  if task == "object_rel_direction" else task),
                "task": task, "question": "Which object is present?", "options": ["chair", "desk"],
                "ground_truth": "A", "quality": index, "file_exists": True,
            }
            rows.append(row)
            if index < (13 if task_index == 0 else 14):
                anchors.append({k: row[k] for k in
                                ("qa_id", "dataset", "scene_name", "question_type", "task")})
    original = {
        "panels": {"137": {"questions": anchors[:64]}, "138": {"questions": anchors[64:]}},
        "runs": [], "first_batch": {"run_ids": []},
    }
    for task in full.TASKS:
        for row in [r for r in anchors if r["task"] == task][:2]:
            for arm in ("B01", "B11"):
                run = {**row, "seed": 137, "arm": arm, "run_id": f"{row['qa_id']}-{arm}"}
                original["runs"].append(run)
                original["first_batch"]["run_ids"].append(run["run_id"])
    return rows, anchors, original


def test_expansion_preserves_111_anchors_labels_do_not_affect_sampling():
    rows, anchors, original = metadata()
    allowed = {r["scene_name"] for r in rows}
    chosen, receipts = full.select_panel(rows, original, allowed)
    assert len(chosen) == len(receipts) == 192
    assert all(sum(r["task"] == task for r in chosen) == 24 for task in full.TASKS)
    assert {r["qa_id"] for r in anchors} <= {r["qa_id"] for r in chosen}
    assert sum(r["anchor"] for r in receipts.values()) == 111
    altered = copy.deepcopy(rows)
    for row in altered:
        row["ground_truth"], row["quality"], row["file_exists"] = "Z", -1, False
    assert full.select_panel(altered[::-1], original, allowed) == (chosen, receipts)
    direction = [r for r in chosen if r["task"] == "object_rel_direction"]
    assert {level: sum(r["question_type"].endswith(level) for r in direction)
            for level in ("easy", "medium", "hard")} == {"easy": 8, "medium": 8, "hard": 8}


def test_scene_round_robin_prioritizes_uncovered_scenes():
    rows = [{"qa_id": f"{scene}-{i}", "scene_name": scene}
            for scene in ("covered", "new1", "new2") for i in range(3)]
    order = list(full.scene_cycle(rows, 137, "object_counting", {"covered"}))
    assert all(r["scene_name"] != "covered" for r in order[:6])
    assert len({r["scene_name"] for r in order[:2]}) == 2
    assert all(r["scene_name"] == "covered" for r in order[6:])


def test_identical_seed_panels_exact_768_unique_runs_first_16_pairs():
    rows, _, original = metadata()
    chosen, _ = full.select_panel(rows, original, {r["scene_name"] for r in rows})
    manifest = full.make_manifest(chosen, original, "test-full")
    assert manifest["planned_solver_count"] == len(manifest["runs"]) == 768
    assert len({r["run_id"] for r in manifest["runs"]}) == 768
    assert len(manifest["pair_order"]) == 384
    assert len(manifest["first_batch_pairs"]) == 16
    assert {r["qa_id"] for r in manifest["runs"] if r["seed"] == 137} == {
        r["qa_id"] for r in manifest["runs"] if r["seed"] == 138}
    assert all(r["max_tokens"] == 8192 for r in manifest["runs"])
    assert manifest["sha256"] == full.digest({k: v for k, v in manifest.items() if k != "sha256"})


def snapshot(busy=()):
    return {"gpus": [{"index": i, "uuid": f"GPU-{i}", "total_mib": 24564,
                      "used_mib": 22000 if i in busy else 6} for i in range(8)],
            "processes": [{"gpu_uuid": f"GPU-{i}", "pid": 123} for i in busy]}


def test_two_gpu_snapshots_exclude_others_and_assign_distinct_roles():
    roles = full.select_gpus(snapshot((0, 1, 2)), snapshot((0, 1, 4)))
    assert len({r["index"] for r in roles.values()}) == 2
    assert {r["index"] for r in roles.values()}.isdisjoint({0, 1, 2, 4})
    with pytest.raises(full.FullExperimentError, match="two unused GPUs"):
        full.select_gpus(snapshot(range(7)), snapshot())
    before = snapshot()
    before["gpus"][3]["total_mib"] = 16000
    assert full.select_gpus(before, snapshot())["geometry"]["index"] != 3


def test_preparation_gzip_cas_is_exact_and_corruption_refuses_restore(tmp_path):
    tree, store = tmp_path / "frozen", tmp_path / "blobs"
    tree.mkdir()
    np.save(tree / "points.npy", np.arange(1000, dtype=np.float32))
    (tree / "artifact.json").write_text(json.dumps({"point_map": str(tree / "points.npy")}))
    original = {p.name: p.read_bytes() for p in tree.iterdir()}
    receipt = full.seal_tree(tree, store, tmp_path / "sealed.json")
    assert not list(tree.iterdir())
    restored = full.restore_tree(receipt, store)
    assert {p.name: p.read_bytes() for p in restored.iterdir()} == original
    (tree / "points.npy").unlink()
    blob = full._blob(store, receipt["files"]["points.npy"]["sha256"])
    blob.write_bytes(b"bad gzip")
    with pytest.raises((full.FullExperimentError, OSError)):
        full.restore_tree(receipt, store)
    assert not (tree / "points.npy").exists()


def test_complete_schedule_auto_continues_all_384_pairs_and_resume_checks_refs(tmp_path):
    rows, _, original = metadata()
    chosen, _ = full.select_panel(rows, original, {r["scene_name"] for r in rows})
    manifest = full.make_manifest(chosen, original, "schedule")
    calls, gates = [], []

    def complete(seed, q, key):
        calls.append((seed, q))
        path = tmp_path / "pairs" / key
        path.mkdir(parents=True)
        full.write(path / "result.json", {"seed": seed, "qa_id": q})
        full.write(path / "completion.json", {
            "manifest_sha256": manifest["sha256"],
            "refs": {str((path / "result.json").relative_to(tmp_path)): full.file_hash(path / "result.json")}})

    def gate(first):
        assert len(calls) == 16
        gates.append(first)

    result = full.run_schedule(tmp_path, manifest, complete, gate)
    assert result["completed_solver_count"] == 768
    assert len(calls) == len(set(calls)) == 384 and len(gates) == 1
    full.run_schedule(tmp_path, manifest, complete, gate)
    assert len(calls) == 384 and len(gates) == 1
    first = manifest["pair_order"][0]
    (tmp_path / "pairs" / f"s{first[0]}-{first[1]}" / "result.json").write_text("{}")
    with pytest.raises(full.FullExperimentError, match="completed evidence changed"):
        full.run_schedule(tmp_path, manifest, complete, gate)


def test_unresolved_pair_is_never_implicitly_replayed(tmp_path):
    manifest = {"pair_order": [[137, "q"]], "first_batch_pairs": [[137, "q"]],
                "sha256": "manifest", "planned_solver_count": 2}
    path = tmp_path / "pairs/s137-q/full_requests"
    path.mkdir(parents=True)
    (path / "r1.json").write_text('{"status":"in_flight"}')
    with pytest.raises(full.FullExperimentError, match="unfinished pair"):
        full.run_schedule(tmp_path, manifest, lambda *args: pytest.fail("replayed"), lambda *args: None)


def test_real_executor_seals_before_solver_then_consumes_same_tree_without_rebinding(panel, tmp_path):
    item, artifact, bindings = panel
    item.episode.ground_truth = ""
    cfg = fixtures.OnlineRunConfig(mode="real", seed=137, max_solver_rounds=3,
                                  max_retries_per_operation=0)
    phases = []

    def client(phase, qa):
        phases.append(phase)
        return fixtures.FakeClient(['ReturnAnswer("A")\n'])

    source = tmp_path / "prepare"
    summary = ab.run_skill_ablation_v11(
        [item], artifact_paths={item.episode.qa_id: artifact}, output_dir=source,
        base_cfg=cfg, seed=137, prepare_only=True, client_factory=client)
    assert summary["status"] == "prepared" and phases == ["prepare"] and len(bindings) == 1
    assert not (source / "arms").exists()
    prior = json.loads((source / "manifest.json").read_text())
    frozen = Path(prior["prepared"][item.episode.qa_id]["directory"])
    receipt = full.seal_tree(frozen, tmp_path / "blobs", tmp_path / "sealed.json")
    full.restore_tree(receipt, tmp_path / "blobs")
    summary = ab.run_skill_ablation_v11(
        [item], artifact_paths={item.episode.qa_id: artifact}, output_dir=tmp_path / "solve",
        base_cfg=cfg, seed=137, client_factory=client, prepared_manifest=source / "manifest.json")
    assert summary["status"] == "completed"
    assert phases == ["prepare", "B01", "B11"] and len(bindings) == 1
    pair = json.loads((tmp_path / "solve/paired_results.jsonl").read_text())
    assert pair["comparable"]
    assert all(r["initial_tree_sha256"] == receipt["tree"]["sha256"] for r in pair["arms"].values())
    assert all(r["score"] is None for r in pair["arms"].values())
    with pytest.raises(ab.PairingError, match="config_sha256 changed"):
        ab.run_skill_ablation_v11(
            [item], artifact_paths={item.episode.qa_id: artifact}, output_dir=tmp_path / "changed",
            base_cfg=fixtures.OnlineRunConfig(mode="real", max_tokens=999), seed=137,
            client_factory=client, prepared_manifest=source / "manifest.json")
    assert len(bindings) == 1


def test_score_refuses_unfrozen_or_changed_answers(tmp_path, monkeypatch):
    manifest = {"sha256": "manifest", "planned_solver_count": 2}
    args = SimpleNamespace()
    monkeypatch.setattr(full, "verify_run", lambda _: (tmp_path, manifest))
    full.write(tmp_path / "answers_frozen.json", {
        "solver_count": 1, "manifest_sha256": "manifest", "refs": {}})
    with pytest.raises(full.FullExperimentError, match="not all frozen"):
        full.score(args)
    full.write(tmp_path / "answer.json", {"answer": "A"})
    full.write(tmp_path / "answers_frozen.json", {
        "solver_count": 2, "manifest_sha256": "manifest", "refs": {"answer.json": "wrong"}})
    with pytest.raises(full.FullExperimentError, match="frozen answer changed"):
        full.score(args)


def test_preparation_never_copies_legacy_inventory(panel, tmp_path):
    _, artifact, _ = panel
    legacy = artifact.parent / "scene_inventory_key.json"
    legacy.write_text('{"inventory_version":"m5-inventory-10-frozen-inputs"}')
    dest = tmp_path / "private-preparation"
    copied = ab._copy_base(artifact, dest)
    assert not (dest / legacy.name).exists()
    assert not (dest / "stale_m5_cache.json").exists()
    assert np.array_equal(np.load(copied.depth_maps),
                          np.load(json.loads(artifact.read_text())["depth_maps"]))


def test_runtime_envelope_keeps_services_out_of_strict_online_config(tmp_path):
    import yaml
    raw = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    raw["online_model"].update(max_tokens=8192, n_frames=32,
                               max_pixels=131072, max_model_len=32768)
    raw.update(max_solver_rounds=6, max_retries_per_operation=3, finalization_rounds=1,
               eval_visual_fallback=True)
    raw["experiment"] = {"full_services": {"detector_python": "python"}}
    path = tmp_path / "full.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg, effective = full.runtime_cfg(path, 138)
    assert cfg.seed == 138 and cfg.online_model == "qwen3.8-flash"
    assert effective["experiment"] == raw["experiment"]
    assert "experiment" not in yaml.safe_load((tmp_path / "full.online.yaml").read_text())


def test_legacy_geometry_migration_requires_original_pixel_and_array_evidence(panel, tmp_path):
    item, artifact, _ = panel
    from skill3d.schemas import ReconstructionArtifact
    art = ReconstructionArtifact.model_validate_json(artifact.read_bytes())
    art = art.model_copy(update={"reconstruction_input_identity": ""})
    artifact.write_text(art.model_dump_json())
    key = item.episode.frame_set.cache_identity()
    pixels = ab._pixels_identity(item.pixels)
    evidence = {
        "artifact": {"sha256": full.file_hash(artifact)},
        "cache_identity": key, "input_pixels": pixels,
        "source_frame_indices": art.source_frame_indices,
        "array_hashes": {name: {"sha256": full.file_hash(path)}
                         for name, path in ab._artifact_files(art, artifact).items()},
    }
    evidence_path = tmp_path / "evidence.json"
    full.write(evidence_path, evidence)
    confirmation_path = tmp_path / "confirmation.json"
    full.write(confirmation_path, {"evidence_ref": str(evidence_path),
                                   "evidence_sha256": full.file_hash(evidence_path)})
    entry = {"path": str(artifact), "sha256": full.file_hash(artifact),
             "quality_confirmation": str(confirmation_path)}
    migrated, provenance = full.verified_historical_geometry(
        entry, item, {"cache_identity": key, "pixels": pixels}, tmp_path)
    assert migrated.reconstruction_input_identity == key
    assert provenance["legacy_input_identity_migrated"] is True
    assert full.file_hash(artifact) == entry["sha256"]
    with pytest.raises(full.FullExperimentError, match="content/input evidence mismatch"):
        full.verified_historical_geometry(entry, item, {"cache_identity": "changed",
                                                       "pixels": pixels}, tmp_path)
    array_path = next(iter(ab._artifact_files(art, artifact).values()))
    array_path.write_bytes(b"changed bytes")
    with pytest.raises(full.FullExperimentError, match="content/input evidence mismatch"):
        full.verified_historical_geometry(
            entry, item, {"cache_identity": key, "pixels": pixels}, tmp_path)


def test_scene_cache_index_is_independent_of_new_question_id_and_checks_content(tmp_path):
    inputs, artifacts = {}, {}
    for q in ("old1", "old2"):
        payload = {"cache_identity": "same-frozen-rgb", "input_error": None, "qa_id": q}
        payload["sha256"] = ab._hash(payload)
        path = tmp_path / (q + ".json")
        full.write(path, payload)
        inputs[q] = {"path": str(path), "sha256": full.file_hash(path)}
        artifacts[q] = {"path": "/old/scene/artifact.json", "sha256": "same-artifact"}
    full.write(tmp_path / "input_manifest.json", inputs)
    full.write(tmp_path / "artifact_manifest.json", artifacts)
    index = full.historical_geometry_index(tmp_path)
    assert list(index) == ["same-frozen-rgb"]
    assert index["same-frozen-rgb"]["historical_qa_id"] == "old1"
    artifacts["old2"]["sha256"] = "different-artifact"
    full.write(tmp_path / "artifact_manifest.json", artifacts)
    with pytest.raises(full.FullExperimentError, match="multiple contents"):
        full.historical_geometry_index(tmp_path)


def test_full_entry_imports_vggt_from_its_own_tracked_checkout():
    pytest.importorskip("torch")
    from vggt.models import vggt
    assert Path(vggt.__file__).resolve().is_relative_to(ROOT / "third_party/vggt")
