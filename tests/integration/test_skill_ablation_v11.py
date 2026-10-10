"""Real-runner pairing contracts using synthetic geometry and injected clients."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from skill3d.adapters.episode_source import EpisodeItem, load_synthetic_items
from skill3d.evaluation import skill_ablation_v11 as ab
from skill3d.online.runner import OnlineRunConfig
from skill3d.reconstruction_gate.quality_metrics import compute_quality
from skill3d.schemas import (
    ConfidenceMap,
    InputErrorRecord,
    ReconstructionArtifact,
    SkillSpecV11,
)
from skill3d.segmentation import sam2_tracker

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def panel(tmp_path, monkeypatch):
    item = load_synthetic_items(
        "inner_validation", question_types=["object_rel_distance"],
        frame_size=(120, 160), n_frames=8)[0]
    g, fs = item.geometry, item.episode.frame_set
    names = [o.category_name for o in g.objects]
    item.episode.question = (
        f"Which is closest to the {names[1]}: the {names[0]} or the {names[2]}?")
    item.episode.options = [names[0], names[2]]
    from skill3d.tools.distance_primitives import robust_distance_between_pointsets
    distances = [robust_distance_between_pointsets(
        g.object_points[g.objects[1].obj_id], g.object_points[g.objects[i].obj_id]
    ).distance_normalized for i in (0, 2)]
    item.episode.ground_truth = "A" if distances[0] <= distances[1] else "B"
    source = tmp_path / "source"
    source.mkdir()
    refs = {}
    for key, arr in {"c2w_list": g.c2w, "intrinsics": g.intrinsics,
                     "depth_maps": g.depth_maps, "point_map": g.point_map,
                     "depth_conf": g.depth_conf}.items():
        p = source / f"{key}.npy"
        np.save(p, arr)
        refs[key] = str(p)
    quality = compute_quality(
        None, frames=item.pixels, depth_maps=g.depth_maps, c2w_list=g.c2w,
        intrinsics=g.intrinsics, point_map=g.point_map, depth_conf=g.depth_conf)
    art = ReconstructionArtifact(
        artifact_id="synthetic-v11", artifact_version="fixture-v1",
        scene_name=item.episode.scene_name, frame_ids=fs.frame_ids,
        source_frame_indices=fs.source_frame_indices, timestamps=fs.timestamps,
        frame_set_hash=fs.frame_set_hash, **refs, point_conf="",
        quality_status="computed", quality=quality,
        world_up=g.world_up, handedness=g.handedness, world_frame_status=g.world_frame_status,
        confidence=ConfidenceMap(per_point_confidence=""))
    artifact = source / "artifact.json"
    artifact.write_text(art.model_dump_json())
    (source / "stale_m5_cache.json").write_text('{"must_not_copy":true}')
    binding_calls = []

    def bind(frames, handle, question, *, out_dir, vlm_client, seed, **kwargs):
        root = Path(out_dir)
        assert not (root / "stale_m5_cache.json").exists()
        binding_calls.append(root)
        vlm_client.chat([{"role": "user", "content": "prepare inventory"}], seed=seed)
        objects = copy.deepcopy(g.objects)
        for obj in objects:
            path = root / f"{obj.obj_id}_points.npy"
            np.save(path, g.object_points[obj.obj_id])
            obj.pointcloud_world = str(path)
            obj.mask_per_frame = ""
            obj.centroid_ref = ""
        (root / "m5_inventory.json").write_text(
            json.dumps({"objects": [o.model_dump() for o in objects]}))
        return objects, {"track_ious": [0.95], "track_stable_ratio": 1.0}, ["synthetic M5"]

    monkeypatch.setattr(sam2_tracker, "bind_objects_for_scene", bind)
    return item, artifact, binding_calls


class FakeClient:
    def __init__(self, programs):
        self.programs = list(programs)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        reply = self.programs[min(len(self.calls) - 1, len(self.programs) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


def run(panel, tmp_path, *, programs=None, **kwargs):
    item, artifact, _ = panel
    clients = {}
    def factory(arm, qa):
        client = FakeClient((programs or {}).get(arm, ['ReturnAnswer("A")\n']))
        clients[(arm, qa)] = client
        return client
    root = tmp_path / "experiment"
    result = ab.run_skill_ablation_v11(
        [item], artifact_paths={item.episode.qa_id: artifact}, output_dir=root,
        base_cfg=OnlineRunConfig(mode="real", seed=0, max_solver_rounds=3,
                                 max_retries_per_operation=0),
        seed=137, client_factory=factory, **kwargs)
    pairs = [json.loads(line) for line in (root / "paired_results.jsonl").read_text().splitlines()]
    return result, pairs, clients, root


def test_real_runner_seed_skill_geometry_and_recomputed_summary(panel, tmp_path):
    item, artifact, binding_calls = panel
    ref = item.geometry.objects[1].obj_id
    a, b = item.episode.options
    program = (
        f'rank = relative_distance_rank("{ref}", ["{a}", "{b}"])\n'
        f'ReturnAnswer("A" if rank["closest_category"] == "{a}" else "B")\n')
    result, pairs, clients, root = run(
        panel, tmp_path, programs={"B01": [program], "B11": [program]})
    assert result["status"] == "completed"
    assert len(binding_calls) == 1
    assert result["groups"]["all"]["B11_minus_B01"]["Accuracy"] == 0.0
    assert result["groups"] == ab.summarize_pairs(pairs)["groups"]
    pair = pairs[0]
    assert pair["comparable"] is True
    off, on = (pair["arms"][arm] for arm in ab.ARMS)
    assert off["delivered_content_sha256"] == {}
    assert on["delivered_skill_versions"] == ["S03@1.1.0"]
    body = (root / "skills/S03/1.1.0/SKILL.md").read_text()
    assert body in clients[("B11", item.episode.qa_id)].calls[0][0][0]["content"][0]["text"]
    assert off["initial_tree_sha256"] == on["initial_tree_sha256"]
    assert off["first_common_request_sha256"] == on["first_common_request_sha256"]
    for c in clients.values():
        assert c.calls
        assert all(params["seed"] == 137 for _, params in c.calls)
    assert off["answer"] is not None, off["notes"]
    assert not off["program_error"], off["errors"]
    assert off["score"] == on["score"] == 1.0
    assert not result["formal_result_eligible"]  # fake model and unconfirmed defaults
    for path in (root / "arms").rglob("trace_record.jsonl"):
        rec = json.loads(path.read_text().splitlines()[0])
        assert rec["arm"] in ab.ARMS and rec["qa_id"] == item.episode.qa_id
    assert (artifact.parent / "stale_m5_cache.json").exists()


def test_campaign_adapter_revalidates_cached_panel_and_summary(panel, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from skill3d.evolution import adapters_v11 as adapters
    from skill3d.schemas import SkillCandidateV11

    item, artifact, calls = panel
    source = (ROOT / "skill_library/versions/S03/1.1.0/"
              "rank-object-distances/SKILL.md").read_text()
    parent = SkillSpecV11(skill_id="S03", version="1.1.0",
                         question_type="object_rel_distance", skill_md=source)
    spec = SkillSpecV11(skill_id="S03", version="1.2.0",
                       question_type=parent.question_type,
                       skill_md=source + "\n复核对象绑定后再提交。\n")
    candidate = SkillCandidateV11(
        candidate_id="candidate-test", parent_snapshot_id="parent-test",
        parent_skill_version="S03@1.1.0", full_skill_spec=spec,
        modification_reason="clarify", source_run_ref="test:revision")
    cfg = OnlineRunConfig(mode="real", seed=137, max_retries_per_operation=0)
    evaluator = adapters.V11PairedEvaluator(
        items=[item], artifact_paths={item.episode.qa_id: artifact}, base_cfg=cfg,
        output_root=tmp_path / "adapter", library_root=tmp_path / "library",
        client_factory=lambda arm, qa: FakeClient(['ReturnAnswer("A")\n']))
    monkeypatch.setattr(evaluator, "_parent_manifest_hash", lambda snapshot: "a" * 64)
    kwargs = dict(
        campaign_id="campaign", parent=parent, candidate=candidate,
        experience=SimpleNamespace(parent_snapshot_id="parent-test", cases=[]),
        question_type=parent.question_type, seed=137, model_id=cfg.vllm_model,
        model_config_sha256=adapters.model_config_identity_v11(cfg)["sha256"],
        quality_contract_sha256=ab.quality_contract()["sha256"],
        solver_config_sha256=adapters.solver_config_identity_v11(cfg, seed=137)["sha256"])
    first = evaluator(**kwargs)
    assert first.status == "completed" and not first.formal_result_eligible
    assert evaluator(**kwargs) == first
    assert len(calls) == 1
    kwargs["experience"].cases = [SimpleNamespace(scene_id=item.episode.scene_name)]
    with pytest.raises(adapters.V11AdapterError, match="scene overlap"):
        evaluator(**kwargs)
    kwargs["experience"].cases = []
    output = evaluator._output_dir("campaign")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    summary["groups"]["all"]["arms"]["B01"]["Accuracy"] = 0.123
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(adapters.V11AdapterError, match="recomputed"):
        evaluator(**kwargs)


def test_e11_fixed_parent_candidate_injection_is_auditable(panel, tmp_path):
    item, artifact, _ = panel
    source = (
        ROOT
        / "skill_library"
        / "versions"
        / "S03"
        / "1.1.0"
        / "rank-object-distances"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    parent = SkillSpecV11(
        skill_id="S03",
        version="1.1.0",
        question_type="object_rel_distance",
        skill_md=source,
    )
    candidate = SkillSpecV11(
        skill_id="S03",
        version="1.2.0",
        question_type="object_rel_distance",
        skill_md=source.replace(
            "重点核查会改变前两名的错误绑定",
            "优先核查会改变前两名的错误绑定",
            1,
        ),
    )
    clients = {}

    def factory(arm, qa_id):
        client = FakeClient(['ReturnAnswer("A")\n'])
        clients[(arm, qa_id)] = client
        return client

    root = tmp_path / "e11"
    result = ab.run_skill_pair_evaluation_v11(
        [item],
        artifact_paths={item.episode.qa_id: artifact},
        output_dir=root,
        base_cfg=OnlineRunConfig(
            mode="real",
            seed=0,
            max_solver_rounds=3,
            max_retries_per_operation=0,
        ),
        seed=137,
        parent=parent,
        candidate=candidate,
        parent_snapshot_ref="parent-snapshot",
        parent_manifest_sha256="a" * 64,
        candidate_snapshot_ref="candidate:fixed",
        candidate_manifest_sha256="b" * 64,
        client_factory=factory,
    )
    pair = json.loads(
        (root / "paired_results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert result["status"] == "completed"
    left, right = pair["arms"]["B01"], pair["arms"]["B11"]
    assert left["delivered_skill_versions"] == ["S03@1.1.0"]
    assert right["delivered_skill_versions"] == ["S03@1.2.0"]
    assert left["first_common_request_sha256"] == right["first_common_request_sha256"]
    assert parent.skill_md in clients[("B01", item.episode.qa_id)].calls[0][0][0][
        "content"
    ][0]["text"]
    assert candidate.skill_md in clients[("B11", item.episode.qa_id)].calls[0][0][0][
        "content"
    ][0]["text"]
    bindings = {}
    for arm in ab.ARMS:
        trace = next((root / "arms" / arm).rglob("episode_trace.jsonl"))
        record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        retrieval = record["retrieval_records"][0]
        bindings[arm] = retrieval["evaluation_binding"]
        assert retrieval["candidates"][0]["reason_code"] == "fixed_injection_selected"
    assert bindings["B01"]["arm"] == "parent"
    assert bindings["B11"]["arm"] == "candidate"


def test_state_observation_namespace_and_disk_are_isolated(panel, tmp_path, monkeypatch):
    from skill3d.tools.scene_handle import SceneHandle
    from skill3d.schemas import ObjectRecord
    from skill3d.sandbox.kernel import RestrictedNamespaceKernel

    original = SceneHandle.object_points
    init = RestrictedNamespaceKernel.__init__
    kernels = []
    def track_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        assert self.tool_results == []
        assert self.answer_slot.answer is None
        assert "secret" not in self._ns
        kernels.append(self)
    monkeypatch.setattr(RestrictedNamespaceKernel, "__init__", track_init)
    touched = []
    def get_points(self, object_id):
        arr = original(self, object_id)
        if not touched:
            touched.append(self)
            self._add_detected_objects([
                ObjectRecord(obj_id="only-B01", category_name="added",
                             grounding_status="tool_detection")])
            ref = Path(self.get_object(object_id).pointcloud_world)
            np.save(ref, arr + 100)
            arr[:] += 100
        return arr
    monkeypatch.setattr(SceneHandle, "object_points", get_points)
    item = panel[0]
    obj = item.geometry.objects[0].obj_id
    first = (f'x = robust_distance("camera", "{obj}")\n'
             'secret = 123\n'
             'im = inspect_frames([0])\n'
             'return YieldObservations([im], "look")\n')
    second = 'ReturnAnswer("A")\n'
    other = ('objects = list_objects()\n'
             'answer = "A"\n'
             'for o in objects:\n'
             '    if o["obj_id"] == "only-B01":\n'
             '        answer = "B"\n'
             'ReturnAnswer(answer)\n')
    result, pairs, _, root = run(panel, tmp_path, programs={"B01": [first, second], "B11": [other]})
    a, b = (pairs[0]["arms"][arm] for arm in ab.ARMS)
    assert result["status"] == "completed"
    assert a["answer"] == b["answer"] == "A"
    assert len(a["requests"]) == 2 and len(b["requests"]) == 1
    assert len(kernels) == 2
    assert kernels[0]._ns is not kernels[1]._ns
    assert kernels[0]._scene._state is not kernels[1]._scene._state
    assert kernels[0].answer_slot is not kernels[1].answer_slot
    assert kernels[0].tool_results is not kernels[1].tool_results
    assert kernels[0]._scene._ledger is not kernels[1]._scene._ledger
    assert a["image_ledger"] != b["image_ledger"]
    manifest = json.loads((root / "manifest.json").read_text())
    frozen = Path(manifest["prepared"][item.episode.qa_id]["directory"])
    assert ab.tree_identity(frozen)["sha256"] == manifest["prepared"][item.episode.qa_id]["sha256"]
    local = next((root / "arms/B11").glob("*/artifacts"))
    assert ab.tree_identity(local)["sha256"] == ab.tree_identity(frozen)["sha256"]


@pytest.mark.parametrize("failed_arm", ["prepare", "B01", "B11"])
def test_service_failure_is_incomplete_and_never_fabricates_score(panel, tmp_path, failed_arm):
    result, pairs, _, _ = run(panel, tmp_path, programs={failed_arm: [RuntimeError("service down")]})
    assert result["status"] == "incomplete"
    for arm in ab.ARMS if failed_arm == "prepare" else [failed_arm]:
        assert pairs[0]["arms"][arm]["score"] is None
        assert pairs[0]["arms"][arm]["service_errors"]
        if failed_arm != "prepare":
            assert pairs[0]["arms"][arm]["failure_code"] == "vllm_service_error"
            assert pairs[0]["arms"][arm]["terminal_failure"]["stage"] == "M8"
    assert pairs[0]["delta"] is None
    assert result["groups"]["all"]["B11_minus_B01"]["Accuracy"] is None


def test_program_failure_keeps_denominator_and_negative_delta_is_valid(panel, tmp_path):
    panel[0].episode.ground_truth = "A"
    result, pairs, _, _ = run(panel, tmp_path, programs={
        "B01": ['ReturnAnswer("A")\n'], "B11": ['x = 1 / 0\n']})
    assert result["status"] == "completed"
    assert pairs[0]["arms"]["B11"]["score"] == 0.0
    assert pairs[0]["arms"]["B11"]["program_error"]
    assert result["groups"]["all"]["arms"]["B11"]["n_scored"] == 1
    assert result["groups"]["all"]["B11_minus_B01"]["Accuracy"] == -1.0


def test_input_error_creates_zero_score_rows_for_both_arms(tmp_path):
    error = InputErrorRecord(
        qa_id="missing-input",
        scene_name="missing-scene",
        dataset="scannet",
        question_type="object_rel_distance",
        question="Which object is closest?",
        options=["chair", "table"],
        ground_truth="A",
        split="inner_validation",
        reason="video_missing",
        source_attempts=[{"source": "/missing.mp4", "result": "missing"}],
    )
    item = EpisodeItem(
        episode=error.as_episode(),
        pixels=[],
        source="vsi_bench",
        input_error=error,
    )
    clients = {}

    def factory(arm, qa):
        client = FakeClient([])
        clients[(arm, qa)] = client
        return client

    root = tmp_path / "input-error-experiment"
    result = ab.run_skill_ablation_v11(
        [item],
        artifact_paths={},
        output_dir=root,
        base_cfg=OnlineRunConfig(mode="real", seed=0),
        seed=137,
        expected_qa_ids=["missing-input"],
        client_factory=factory,
    )
    pair = json.loads((root / "paired_results.jsonl").read_text().splitlines()[0])

    assert result["status"] == "completed"
    assert result["n_input_error_pairs"] == 1
    assert result["formal_result_eligible"] is False
    assert result["groups"]["all"]["arms"]["B01"]["n_scored"] == 1
    assert result["groups"]["all"]["arms"]["B11"]["n_scored"] == 1
    assert result["groups"]["all"]["arms"]["B01"]["input_error_rate"] == 1.0
    assert result["groups"]["all"]["B11_minus_B01"]["input_error_rate"] == 0.0
    assert pair["comparable"]
    for arm in ab.ARMS:
        row = pair["arms"][arm]
        assert row["score"] == 0.0
        assert row["final_state"] == row["episode_status"] == "input_error"
        assert row["input_error_reason"] == "video_missing"
        assert clients[(arm, "missing-input")].calls == []


def test_configurable_numeric_task_uses_official_mra(panel, tmp_path):
    from skill3d.evaluation.mra import mra_single
    item = panel[0]
    item.episode.question_type = "object_counting"
    item.episode.question = "How many chairs are in the scene?"
    item.episode.options = None
    item.episode.ground_truth = "10"
    result, pairs, _, _ = run(
        panel, tmp_path, question_types=["object_counting"],
        programs={"B01": ["ReturnAnswer(8)\n"], "B11": ["ReturnAnswer(10)\n"]})
    group = result["groups"]["object_counting"]
    assert group["arms"]["B01"]["MRA"] == mra_single(8, 10)
    assert group["arms"]["B11"]["MRA"] == 1.0
    assert group["B11_minus_B01"]["MRA"] == pytest.approx(1 - mra_single(8, 10))
    assert result["groups"] == ab.summarize_pairs(pairs)["groups"]


def test_detection_service_fault_after_yield_stays_incomplete(panel, tmp_path, monkeypatch):
    from skill3d.segmentation import open_vocab_detector as ovd
    def broken(*args, **kwargs):
        broken.last_error = "detector service down"
        return []
    monkeypatch.setattr(ovd, "detect", broken)
    program = ('r = detect_objects([0], ["chair"])\n'
               'return YieldObservations([r], "inspect result")\n')
    result, pairs, _, _ = run(panel, tmp_path, programs={
        "B01": [program, 'ReturnAnswer("A")\n']})
    assert result["status"] == "incomplete"
    assert pairs[0]["arms"]["B01"]["score"] is None
    assert pairs[0]["arms"]["B01"]["service_errors"]


@pytest.mark.parametrize("recover", [True, False])
def test_detector_transport_attempts_reach_pair_and_online_trace(panel, tmp_path, monkeypatch, recover):
    from unittest.mock import Mock
    import requests
    from skill3d.segmentation import open_vocab_detector as ovd
    response = Mock(status_code=200)
    response.json.return_value = {"success": True, "detections": []}
    post = Mock(side_effect=([requests.Timeout("timeout"), response] if recover
                            else requests.Timeout("timeout")))
    monkeypatch.setenv("SKILL3D_DETECTOR_ENDPOINT", "http://detector")
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(ovd.time, "sleep", lambda _: None)
    program = ('r = detect_objects([0], ["chair"])\n'
               'return YieldObservations([r], "inspect result")\n')
    summary, pairs, _, root = run(panel, tmp_path, programs={
        "B01": [program, 'ReturnAnswer(AnswerPayload(value="A", unit="option", basis="visual_estimate"))\n']})
    row = pairs[0]["arms"]["B01"]
    assert len(row["detector_attempts"]) == (2 if recover else 3)
    assert summary["status"] == ("completed" if recover else "incomplete")
    assert (row["score"] is not None) == recover
    if recover:
        assert row["detector_attempts"][-1]["recovered"]
        assert not row["service_errors"]
        assert row["answer_basis"] == "visual_estimate"
        assert summary["groups"]["all"]["arms"]["B01"]["answer_basis_counts"]["visual_estimate"] == 1
    traces = [json.loads(line) for path in root.rglob("trace_record.jsonl")
              for line in path.read_text().splitlines()]
    assert any(t.get("detector_attempts") == row["detector_attempts"] for t in traces)


def test_synthesis_service_error_keeps_specific_terminal_reason(panel, tmp_path):
    _, pairs, _, _ = run(panel, tmp_path, programs={"B01": [ConnectionError("unavailable")]})
    row = pairs[0]["arms"]["B01"]
    assert row["failure_code"] == "vllm_service_error"
    assert row["terminal_failure"]["stage"] == "M8"
    assert row["terminal_failure"]["termination_reason"] == "synthesis_unavailable"
    assert row["score"] is None


def test_cli_runs_real_runner_with_injected_transport(panel, tmp_path, monkeypatch):
    import importlib.util
    # CLI sets a detector endpoint from YAML. Keep it disabled and restore the
    # environment after this in-process test, so later tests cannot contact it.
    monkeypatch.setenv("SKILL3D_DETECTOR_ENDPOINT", "")
    spec = importlib.util.spec_from_file_location(
        "v11_cli", ROOT / "scripts/run_skill_ablation_v11.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    item, artifact, _ = panel
    mapping = tmp_path / "artifacts.json"
    mapping.write_text(json.dumps({item.episode.qa_id: str(artifact)}))
    monkeypatch.setattr(cli, "load_jsonl_items", lambda *a, **k: [item])
    monkeypatch.setattr(ab.runner, "_make_vllm_client",
                        lambda cfg: FakeClient(['ReturnAnswer("A")\n']))
    config = ROOT / "configs/config.yaml"
    output = tmp_path / "cli-result"
    assert cli.main([
        "--source", "jsonl", "--episodes-jsonl", "injected.jsonl",
        "--artifacts-json", str(mapping), "--config", str(config),
        "--vllm-endpoint", "http://fake", "--seed", "137",
        "--output-dir", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["common_config"]["seed"] == 137
    assert manifest["common_config"]["prompt_template_version"] == "program_synth_v11_3"
    assert manifest["common_config"]["execution_protocol_version"] == (
        "solver-v11.4-eval-visual-fallback")
    assert manifest["common_config"]["tool_docs_version"] == "tool-docs-v11.2"
    assert manifest["snapshot"]["snapshot_id"] == "S0-v11-contract-repair"


@pytest.mark.parametrize("damage", ["missing_qa", "duplicate_qa", "frame_hash", "source",
                                  "missing_ref", "pixel_conflict"])
def test_invalid_inputs_fail_before_model_and_write_receipt(panel, tmp_path, damage):
    item, path, calls = panel
    items, expected = [item], [item.episode.qa_id]
    if damage == "missing_qa":
        expected.append("missing")
    elif damage == "duplicate_qa":
        items.append(item)
    elif damage == "pixel_conflict":
        other = copy.deepcopy(item)
        other.episode.qa_id = "same-artifact-different-pixels"
        other.pixels[0][0, 0, 0] ^= 1
        items.append(other)
        expected.append(other.episode.qa_id)
    else:
        data = json.loads(path.read_text())
        if damage == "frame_hash":
            data["frame_set_hash"] = "incorrect"
        elif damage == "source":
            data["scene_name"] = "wrong-scene"
        else:
            data["point_map"] = str(path.parent / "missing.npy")
        path.write_text(json.dumps(data))
    with pytest.raises(ab.PairingError):
        ab.run_skill_ablation_v11(
            items, expected_qa_ids=expected,
            artifact_paths={it.episode.qa_id: path for it in items}, output_dir=tmp_path / "bad",
            base_cfg=OnlineRunConfig(mode="real"), seed=137)
    assert not calls
    assert json.loads((tmp_path / "bad/failure.json").read_text())["status"] == "invalid"


def test_quality_confirmation_does_not_change_thresholds(panel, tmp_path):
    contract = ab.quality_contract()
    result, _, _, root = run(panel, tmp_path, quality_confirmation={
        "confirmed": True, "sha256": contract["sha256"], "evidence_ref": "synthetic-test-only",
        "artifact": {"sha256": ab._file_hash(panel[1])}})
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["quality_contract"] == contract
    assert manifest["quality_confirmation_status"] == "confirmed"
    assert result["formal_result_eligible"] is False


def test_old_quality_confirmation_cannot_reuse_pre_ranking_contract(panel, tmp_path):
    contract = ab.quality_contract()
    legacy = {k: v for k, v in contract.items() if k not in (
        "sha256", "evidence_source_sha256", "execution_protocol_version",
        "tool_docs_version", "prompt_template_version")}
    assert "tools/ranking_contract.py" in contract["evidence_source_sha256"]
    with pytest.raises(ab.PairingError, match="quality confirmation"):
        run(panel, tmp_path, quality_confirmation={
            "confirmed": True, "sha256": ab._hash(legacy), "evidence_ref": "old-engineering"})


def test_partial_rank_rejected_then_visual_answer_keeps_pair_denominator(panel, tmp_path):
    item = panel[0]
    ref = item.geometry.objects[1].obj_id
    candidates = list(item.episode.options)
    item.episode.options.append("unmeasured candidate")
    program = f"rank = relative_distance_rank({ref!r}, {candidates!r})\nReturnAnswer('A')\n"
    visual = "ReturnAnswer(AnswerPayload(value='A', unit='option', basis='visual_estimate'))\n"
    result, pairs, _, _ = run(panel, tmp_path, programs={
        "B01": [program, visual], "B11": [program, visual]})
    assert result["status"] == "completed"
    for arm in ab.ARMS:
        row = pairs[0]["arms"][arm]
        assert row["geometry_rejection"] and row["answer"] == "A"
        assert row["answer_basis"] == "visual_estimate"
        assert row["failure_code"] is None and row["terminal_failure"] == {}
        assert row["errors"][0]["downgraded_capabilities"] == {}
        assert result["groups"]["all"]["arms"][arm]["n_scored"] == 1
        assert result["groups"]["all"]["arms"][arm]["answer_basis_counts"]["visual_estimate"] == 1


def test_recovered_detector_attempts_reach_manifest_and_both_arm_traces(
        panel, tmp_path, monkeypatch):
    from unittest.mock import Mock
    import requests
    from skill3d.segmentation import open_vocab_detector as ovd
    original = sam2_tracker.bind_objects_for_scene
    def bind(frames, *args, **kwargs):
        ovd.detect(frames[0], "chair", endpoint="http://injected")
        return original(frames, *args, **kwargs)
    response = Mock(status_code=200)
    response.json.return_value = {"success": True, "detections": []}
    monkeypatch.setattr(requests, "post", Mock(side_effect=[
        requests.ConnectionError("transient"), response]))
    monkeypatch.setattr(ovd.time, "sleep", lambda _: None)
    monkeypatch.setattr(sam2_tracker, "bind_objects_for_scene", bind)
    result, pairs, _, root = run(panel, tmp_path)
    assert result["status"] == "completed"
    manifest = json.loads((root / "manifest.json").read_text())
    attempts = manifest["preparation_detector_attempts"][panel[0].episode.qa_id]
    assert len(attempts) == 2 and attempts[1]["recovered"]
    assert not manifest["preparation_errors"]
    for arm in ab.ARMS:
        row = pairs[0]["arms"][arm]
        assert row["detector_attempts"] == attempts
        assert not row["service_errors"] and row["score"] is not None
        trace_path = next((root / "arms" / arm).glob("*/trace/trace_record.jsonl"))
        trace = json.loads(trace_path.read_text().splitlines()[0])
        assert trace["detector_attempts"] == attempts


def test_malformed_program_counts_as_failure_not_service_outage(panel, tmp_path):
    result, pairs, _, _ = run(panel, tmp_path, programs={"B11": ["not a Python program"]})
    assert result["status"] == "completed"
    on = pairs[0]["arms"]["B11"]
    assert on["score"] == 0.0 and not on["legal_answer"]
    assert on["program_error"]
    assert not on["service_errors"]


def test_join_uses_qa_id_and_rejects_missing_duplicate_or_changed_inputs():
    def row(q):
        return dict(qa_id=q, input_sha256=q, config_sha256="same", task="object_rel_distance",
                    metric="Accuracy", delivery_ok=True, status="completed", score=1.0,
                    first_common_request_sha256="same")
    left = [row("x"), row("y")]
    right = [row("y"), row("x")]
    assert [r["qa_id"] for r in ab.pair_results(left, right, ["x", "y"])] == ["x", "y"]
    for bad in ([row("x")], [row("x"), row("x")],
                [row("x"), {**row("y"), "input_sha256": "changed"}]):
        with pytest.raises(ab.PairingError):
            ab.pair_results(left, bad, ["x", "y"])


def test_cli_help_and_quality_contract_inherit_pythonpath():
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", str(ROOT / "src"))}
    for flag in ("--help", "--print-quality-contract"):
        p = subprocess.run([sys.executable, "scripts/run_skill_ablation_v11.py", flag],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        if flag != "--help":
            assert json.loads(p.stdout) == ab.quality_contract()


@pytest.mark.parametrize("change", [
    {"scene_name": "another-scene"},
    {"frame_set_hash": ""},
    {"source_frame_indices": list(range(100, 108))},
    {"timestamps": [float(i + 100) for i in range(8)]},
])
def test_normal_solver_rejects_wrong_explicit_artifact_before_requests(panel, change):
    from skill3d.online.runner import run_episode

    item, path, calls = panel
    art = ReconstructionArtifact.model_validate_json(path.read_text())
    path.write_text(art.model_copy(update=change).model_dump_json())
    llm = FakeClient(["ReturnAnswer('A')"])
    with pytest.raises(ValueError, match="artifact .*mismatch"):
        run_episode(
            item.episode, item.pixels,
            OnlineRunConfig(mode="real", reuse_artifact=str(path)), llm=llm)
    assert llm.calls == []
    assert calls == []
