"""Meaningful offline tests for split isolation and complete executed tool traces."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sr import FakeClient, dataset, problem, prepared
import sr
import sr_trial
from sr_adapter import SRSolver
from sr_protocol import official_prompt


def test_split_is_deterministic_and_disjoint_without_host_answers(dataset):
    (dataset / "data/references.jsonl").unlink()
    first = sr_trial.split_manifest(dataset)
    assert first == sr_trial.split_manifest(dataset)
    assert len(first["train"]) == 43 and len(first["internal_test"]) == 100
    assert len(first["diagnostic20"]) == 20
    assert not set(first["train"]) & set(first["internal_test"])
    assert set(first["diagnostic20"]) <= set(first["internal_test"])
    assert first["reference_answers_read"] is False


def test_remaining_cohort_is_exact_complement_and_keeps_protocol(dataset, tmp_path):
    import pytest
    root = tmp_path / "remaining"
    root.mkdir()
    (root / "dataset").symlink_to(dataset)
    split = sr.write_once(root / "split.json", sr_trial.split_manifest(dataset))
    parent = sr.write_once(root / "parent_manifest.json", {
        "split_sha256": split["content_sha256"],
        "inspection": {"configuration": sr.configuration()[0],
                       "skill_sha256": sr.file_sha(sr.SKILL),
                       "sources": sr.source_identity()}})
    sr.write_once(root / "cohort.json", {
        "schema": "blink-sr-cohort/1", "selection": "remaining80", "questions": 80,
        "request_cap": 320, "concurrency": 8, "role": "internal_test_remaining",
        "parent_manifest_sha256": parent["content_sha256"]})
    remaining = sr_trial.selected_ids(root)
    assert len(remaining) == 80
    assert not set(remaining) & set(split["diagnostic20"])
    assert not set(remaining) & set(split["train"])
    assert set(remaining) | set(split["diagnostic20"]) == set(split["internal_test"])
    parent["inspection"]["configuration"]["model"]["model"] = "FAKE_ONLY_CHANGED"
    path = root / "parent_manifest.json"
    path.unlink()
    sr.write_once(path, {k: v for k, v in parent.items() if k != "content_sha256"})
    with pytest.raises(ValueError, match="protocol_changed"):
        sr_trial.selected_ids(root)


def test_remaining80_full_audit_uses_160_episodes_and_320_request_cap(dataset, tmp_path, monkeypatch):
    """Exercise the actual transport capture and host audit with fake responses."""
    from sr_adapter import FullResponseClient
    from skill3d.small_image.storage import save_prepared
    root = tmp_path / "remaining80"
    root.mkdir()
    (root / "dataset").symlink_to(dataset)
    split = sr.write_once(root / "split.json", sr_trial.split_manifest(dataset))
    parent = sr.write_once(root / "parent_manifest.json", {
        "split_sha256": split["content_sha256"],
        "inspection": {"configuration": sr.configuration()[0],
                       "skill_sha256": sr.file_sha(sr.SKILL), "sources": sr.source_identity()}})
    sr.write_once(root / "cohort.json", {
        "schema": "blink-sr-cohort/1", "selection": "remaining80", "questions": 80,
        "request_cap": 320, "concurrency": 8, "role": "internal_test_remaining",
        "parent_manifest_sha256": parent["content_sha256"]})
    _, items = sr.load_inputs(dataset, ("val",))
    selected = set(sr_trial.selected_ids(root))
    for p, _ in items:
        if p.sample_id in selected:
            save_prepared(prepared(p), root / "prepared" / p.input_sha256)
    sr_trial.freeze(root)
    manifest = sr.read(root / "manifest.json")
    assert len(manifest["jobs"]) == 160 and manifest["request_cap"] == 320
    monkeypatch.setenv("DASHSCOPE_API_KEY", "FAKE_ONLY_OFFLINE")

    class Response:
        def __init__(self, text):
            self.text = text
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def read(self):
            return json.dumps({
                "id": "FAKE_ONLY", "model": "qwen3.8-flash", "created": 1,
                "choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": self.text,
                    "reasoning_content": "FAKE_ONLY_REASONING"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()

    def opener(request, **_kwargs):
        payload = json.loads(request.data)
        text = payload["messages"][0]["content"][0]["text"]
        return Response(
            "def solve(ctx):\n"
            "    ReturnAnswer(AnswerPayload(value='A', unit='option', basis='visual_estimate'))"
            if "Task reasoning Skill:" in text else '{"answer":"A"}')

    monkeypatch.setattr(sr_trial, "FullResponseClient",
                        lambda **kwargs: FullResponseClient(**kwargs, opener=opener))
    for job in manifest["jobs"]:
        sr_trial.worker(root, job, manifest["inspection"]["configuration"]["model"],
                        cap=320, concurrency=8, role="internal_test_remaining")
    governor = sr.RequestGovernor(root / "budget", sr.RequestGovernorConfig(320, 8))
    sr.write_once(root / "completed.json", {
        "manifest_sha256": manifest["content_sha256"], "episodes": 160,
        "budget": governor.summary()})
    output = tmp_path / "audited"
    output.mkdir()
    report = sr_trial.audit_and_report(root, dataset, output)
    assert report["audit"]["checked_episodes"] == 160
    assert report["audit"]["checked_payloads"] == 160
    assert all(g["questions"] == 80 and g["requests"] == 80 for g in report["groups"].values())


def test_full_trace_preserves_executed_arguments_results_and_authorizations(tmp_path):
    p = problem(tmp_path)
    _, config, _ = sr.configuration()
    code = ("def solve(ctx):\n"
            "    r = ctx.tools.point_depth(view_index=0, x=7, y=6)\n"
            "    YieldObservations([r['result_id']])")
    solver = SRSolver(
        FakeClient([code, code, '{"answer":"A"}']), config=config,
        registry_builder=sr.scoped_registry, skill_text=sr.SKILL.read_text(),
        official_prompt=official_prompt(p))
    result = solver.solve(p, prepared=prepared(p), categories=("cat", "car"))
    assert result.status == "answered" and len(solver.execution_trace) == 3
    json.dumps(solver.execution_trace, allow_nan=False)
    first = solver.execution_trace[0]
    assert first["program_source"] and first["execution"]["terminated"] == "yield"
    assert first["tool_calls"][0]["args"] == {"view_index": 0, "x": 7, "y": 6}
    tool_result = first["tool_results"][0]
    assert tool_result["status"] == "ok" and tool_result["payload"]
    assert tool_result["authorization"] in first["authorization_receipts"]
    assert not solver.execution_trace[-1]["tool_calls"]


def test_candidate_budget_keeps_both_categories_and_records_all_rejections():
    from sr_native import BalancedDetector
    class Detector:
        identity = {"model": "FAKE_ONLY"}
        def __call__(self, rgb, categories):
            return [
                {"category": c, "confidence": confidence, "box_xyxy": [1, 1, 5, 5]}
                for c in categories for confidence in (.5, .9, .8, .7, .6)]
    import numpy as np
    detector = BalancedDetector(Detector())
    selected = detector(np.zeros((6, 6, 3), dtype=np.uint8), ("cat", "car"))
    assert [r["category"] for r in selected] == ["cat", "cat", "car", "car"]
    assert [r["confidence"] for r in selected] == [.9, .8, .9, .8]
    assert len(detector.trace[0]["raw_detections"]) == 10
    assert len(detector.trace[0]["discarded_indices"]) == 6


def test_full_transport_trace_keeps_thinking_json_and_excludes_credential(tmp_path, monkeypatch):
    from sr_adapter import FullRecordedClient, FullResponseClient
    monkeypatch.setenv("FAKE_ONLY_KEY_ENV", "FAKE_ONLY_AUTH_NEVER_LOG")
    body = {
        "id": "FAKE_ONLY_ID", "model": "qwen3.8-flash", "created": 1,
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": '{"answer":"A"}',
            "reasoning_content": "FAKE_ONLY_INTERNAL_REASONING"}}],
    }
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def read(self):
            return json.dumps(body).encode()
    wire = []
    def opener(request, **_kwargs):
        wire.append(json.loads(request.data))
        return Response()
    base = FullResponseClient(
        model="qwen3.8-flash", api_key_env="FAKE_ONLY_KEY_ENV",
        max_output_tokens=8192, opener=opener)
    governor = sr.RequestGovernor(tmp_path / "budget", sr.RequestGovernorConfig(1, 1))
    client = FullRecordedClient(base, governor, tmp_path, "FAKE_ONLY", "solver")
    p = problem(tmp_path)
    config = sr.configuration()[0]["solver"]
    result = sr.solve_one(
        {"arm": "direct", "problem": p.public_dict(), "solver": config,
         "official_prompt": official_prompt(p)}, client)
    assert result["answer"] == "A" and governor.summary()["completed_requests"] == 1
    assert client.request_trace[0]["payload"] == wire[0]
    assert client.request_trace[0]["response_json"] == body
    assert base.last_reasoning_content == "FAKE_ONLY_INTERNAL_REASONING"
    for path in (tmp_path / "requests").rglob("*.json"):
        assert "FAKE_ONLY_AUTH_NEVER_LOG" not in path.read_text()


def test_audit_accepts_only_the_canonical_image_marker(tmp_path):
    from skill3d.small_image.blink_protocol import original_messages
    import pytest
    p = problem(tmp_path)
    prompt = official_prompt(p) + sr.DIRECT_SUFFIX
    content = original_messages(p, prompt)[0]["content"]
    text, images = sr_trial.original_content(content)
    assert text == prompt and len(images) == 1
    with pytest.raises(ValueError, match="framing"):
        sr_trial.original_content(content + [{"type": "text", "text": "extra evidence"}])
    with pytest.raises(ValueError, match="framing"):
        sr_trial.original_content([content[0], {"type": "text", "text": "Image 2:"}, content[2]])


def test_actual_first_solver_prompt_matches_zero_request_preflight(tmp_path):
    p, skill = problem(tmp_path), sr.SKILL.read_text()
    value = prepared(p)
    _, config, _ = sr.configuration()
    registry, scene = sr.scoped_registry(value, p)
    names = set(registry.names_for_scope(
        scene.question_tool_scope, scene.available_artifacts,
        evidence_profile=scene.evidence_profile, question_type=scene.question_type,
        local_facts=scene.local_facts))
    assert names == sr.TOOLS
    expected = sr.build_prompt(
        p, value.frames, prompt=official_prompt(p), skill_text=skill,
        interfaces=sr._interfaces(registry, names), feedback=())
    client = FakeClient([
        "def solve(ctx):\n"
        "    ReturnAnswer(AnswerPayload(value='A', unit='option', basis='visual_estimate'))"])
    result = SRSolver(
        client, config=config, registry_builder=sr.scoped_registry,
        skill_text=skill, official_prompt=official_prompt(p)).solve(
            p, prepared=value, categories=("cat", "car"))
    actual = client.calls[0][0][0]["content"][0]["text"]
    assert result.status == "answered" and actual == expected
    assert "pair_point_distance" not in actual


def continuation_fixture(tmp_path):
    from sr_continuation import seed
    parent, root = tmp_path / "parent", tmp_path / "corrected"
    parent.mkdir()
    root.mkdir()
    jobs, corrected, seals, requests = [], [], [], []
    for i in range(20):
        for arm in ("direct", "skill_tools"):
            op = f"sr-{i:03d}-{arm}"
            job = {"operation_id": op, "spec": {"arm": arm}}
            jobs.append(job)
            corrected.append(job if arm == "direct" else {
                **job, "operation_id": f"sr-corrected-{i:03d}-{arm}"})
            payload = {"FAKE_ONLY": i}
            result = {"status": "answered", "answer": "A", "request_count": 1,
                      "reference_answer_visible": False, "attempted_tool_calls": 0,
                      "execution_trace": [], "request_trace": [{"payload": payload}]}
            episode = sr.write_once(parent / "episodes" / (op + ".json"), {
                "job_sha256": sr.sha_value(job), "result": result})
            requests.append({"status": "completed", "context": {"variant_id": op}})
            if arm == "direct":
                for suffix in ("intent", "payload", "full", "response"):
                    sr.write_once(parent / "requests" / op / f"001.{suffix}.json", payload)
                seals.append({"operation_id": op, "job_sha256": sr.sha_value(job),
                              "episode_sha256": episode["content_sha256"],
                              "payload_sha256": sr.sha_value(payload)})
    manifest = sr.write_once(parent / "manifest.json", {"jobs": jobs})
    sr.write_once(parent / "completed.json", {
        "manifest_sha256": manifest["content_sha256"], "episodes": 40,
        "budget": {"reserved_requests": 40, "error_requests": 0,
                   "unknown_requests": 0, "active_requests": 0}})
    (parent / "budget").mkdir()
    (parent / "budget/state.json").write_text(json.dumps({
        "policy": {"max_requests": 80, "max_inflight": 8, "requests_per_second": 0.0},
        "requests": requests}))
    seal_path = parent / "direct_verified.json"
    sr.write_once(seal_path, {"parent_manifest_sha256": manifest["content_sha256"],
                             "verified_episodes": 20, "model_requests": 20,
                             "reference_answers_read": False, "episodes": seals})
    return parent, root, corrected, seal_path, seed


def test_explicit_continuation_selects_only_new_tools_and_keeps_prior_budget(tmp_path):
    from sr_continuation import verify
    parent, root, jobs, seal, seed = continuation_fixture(tmp_path)
    continuation = seed(root, parent, seal, jobs)
    manifest = {"jobs": jobs, "continuation_sha256": continuation["content_sha256"]}
    selected, actual = sr_trial.initial_jobs(root, manifest, None)
    assert actual == continuation and len(selected) == 20
    assert all(j["spec"]["arm"] == "skill_tools" for j in selected)
    assert (root / "budget/state.json").read_bytes() == (parent / "budget/state.json").read_bytes()
    assert continuation["excluded_parent_tool_requests"] == 20
    assert len(list((root / "episodes").glob("*.json"))) == 20
    state = json.loads((root / "budget/state.json").read_text())
    state["requests"].append({"status": "completed",
                             "context": {"variant_id": selected[0]["operation_id"]}})
    (root / "budget/state.json").write_text(json.dumps(state))
    verify(root, manifest)
    import pytest
    with pytest.raises(ValueError, match="replay"):
        sr_trial.initial_jobs(root, manifest, None)
    state["requests"][0]["status"] = "error"
    (root / "budget/state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="prior_budget"):
        verify(root, manifest)


def test_continuation_rejects_unfinished_parent_and_changed_direct_receipt(tmp_path):
    import pytest
    from sr_continuation import verify
    parent, root, jobs, seal, seed = continuation_fixture(tmp_path)
    state = json.loads((parent / "budget/state.json").read_text())
    state["requests"][0]["status"] = "started"
    (parent / "budget/state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="not_complete"):
        seed(root, parent, seal, jobs)
    state["requests"][0]["status"] = "completed"
    (parent / "budget/state.json").write_text(json.dumps(state))
    continuation = seed(root, parent, seal, jobs)
    manifest = {"jobs": jobs, "continuation_sha256": continuation["content_sha256"]}
    path = root / "episodes" / "sr-000-direct.json"
    path.write_text(path.read_text().replace('"answer": "A"', '"answer": "B"'))
    with pytest.raises(ValueError, match="reused_receipt"):
        verify(root, manifest)
