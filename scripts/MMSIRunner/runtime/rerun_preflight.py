"""Revalidate all frozen MMSI inputs and interfaces without model requests."""
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "mmsi-365-test-multiagent-20261101"
for key in list(os.environ):
    if any(s in key.upper() for s in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
        os.environ.pop(key, None)
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, str(ROOT))
import pilot
from skill3d.small_image.runner import _validate_prepared, load_problem_frames, build_solver_prompt
from skill3d.small_image.storage import load_prepared
import audited_eval_transport


def run():
    started = time.monotonic()
    manifest = pilot.read(ROOT / "manifest.json")
    state = pilot.verify(manifest)
    assert not state["requests"] and not (ROOT / "intents").exists()
    prior = pilot.read(OLD / "preflight.json")
    records = {r["input_sha256"]: r for r in prior["all_test_records"]}
    assert len(records) == 365
    frozen = json.loads((ROOT / "frozen_skills.json").read_text())
    assert len(frozen) == 6
    unique = {j["input_sha256"]: j for j in manifest["jobs"]}
    assert len(unique) == 365 and len(manifest["jobs"]) == 730
    verified = []
    for index, job in enumerate(unique.values(), 1):
        problem = pilot.problem_for(job["problem"])
        assert problem.source_split == "test"
        original = records[job["input_sha256"]]
        folder = Path(job["prepared_root"]) / job["input_sha256"]
        assert pilot.sha(folder / "metadata.json") == original["metadata_sha256"]
        assert pilot.sha(folder / "arrays.npz") == original["arrays_sha256"]
        assert pilot.image_identities(problem) == job["images"]
        assert len(problem.image_paths) == original["n_images"] and 1 <= len(problem.image_paths) <= 6
        verified.append(job["input_sha256"])
        if index % 10 == 0 or index == 365:
            pilot.update(ROOT / "preflight_progress.json", {
                "status": "running", "stage": "all_original_inputs_and_cache_hashes",
                "completed": index, "expected": 365, "at": pilot.now(), "model_requests": 0})
    for job in manifest["jobs"]:
        if job["arm"] == "skill":
            assert job["skill"] == frozen[job["slot"]]["skill"]
            assert job["allowed_tools"] == [
                "pair_point_distance", "point_depth", "view_candidates", "view_depth_summary"]
        else:
            assert job["skill"] is None and job["allowed_tools"] == []
    rng = random.Random(137)
    loaded = []
    for slot in sorted(frozen):
        candidates = [j for j in manifest["jobs"] if j["slot"] == slot and j["arm"] == "skill"]
        for job in rng.sample(candidates, 5):
            problem = pilot.problem_for(job["problem"])
            prepared = load_prepared(Path(job["prepared_root"]) / job["input_sha256"], problem=problem)
            assert prepared.artifact_sha256 == job["artifact_sha256"]
            assert _validate_prepared(prepared, problem, load_problem_frames(problem)) is None
            registry, scene = pilot.registry_for(job["allowed_tools"])(prepared, problem)
            allowed = set(registry.names_for_scope(
                scene.question_tool_scope, scene.available_artifacts,
                evidence_profile=scene.evidence_profile, question_type=scene.question_type,
                local_facts=scene.local_facts))
            assert "ReadObservation" not in allowed and "candidate_depth_summary" not in allowed
            prompt = build_solver_prompt(
                problem, prepared.frames, registry, allowed, skill_text=job["skill"]["text"])
            assert "For CV3D" not in prompt and "POINT markers" not in prompt
            loaded.append(job["input_sha256"])
            del prepared, registry, scene
            gc.collect()
    smoke = subprocess.run([sys.executable, "-B", str(ROOT / "smoke.py")],
                           capture_output=True, timeout=90)
    assert smoke.returncode == 0
    synthetic = json.loads(smoke.stdout)
    assert synthetic["status"] == "passed" and synthetic["live_model_requests"] == 0
    pilot.write(ROOT / "smoke_revalidated.json", synthetic)
    # Validate the audited envelope locally; this transport never opens a socket.
    import io
    from contextlib import contextmanager
    import urllib.request
    captured = []

    @contextmanager
    def transport(request, timeout):
        captured.append(json.loads(request.data))
        yield io.BytesIO(b'{"local":"synthetic"}')

    adapter = audited_eval_transport.ProxyOpener(
        "a" * 64, "b" * 64, audited_eval_transport.SCHEMA,
        secret="synthetic-unit-transport-secret", transport=transport)
    request = urllib.request.Request("http://localhost/synthetic",
                                     data=json.dumps({"model": "qwen3.8-flash"}).encode())
    with adapter(request, 1) as response:
        assert response.read() == b'{"local":"synthetic"}'
    assert len(captured) == 1 and captured[0]["schema_version"] == audited_eval_transport.SCHEMA
    assert captured[0]["audit_identity"] == {"run_identity": "b" * 64, "operation_id": "a" * 64}
    health = audited_eval_transport.health()
    assert health["source_sha256"] == manifest["audited_gateway_source_sha256"]
    assert health["schema_version"] == manifest["jobs"][0]["proxy_schema"]
    assert not json.loads((ROOT / "requests/state.json").read_text())["requests"]
    pilot.write(ROOT / "preflight.json", {
        "status": "passed", "at": pilot.now(), "all_test_identity_verified": len(verified),
        "selected_full_content_verified": len(loaded), "six_frozen_skills_unchanged": True,
        "audited_transport_synthetic_passed": True, "gateway_source_verified": True,
        "old_preflight_sha256": pilot.sha(OLD / "preflight.json"),
        "model_requests": 0, "gpu_requests": 0, "reference_answers_read": False,
        "test_trajectories_used_for_revision": False, "seconds": time.monotonic() - started})
    pilot.write(ROOT / "ready.json", {
        "status": "ready", "at": pilot.now(), "manifest_sha256": manifest["content_sha256"],
        "questions": 365, "episodes": 730, "test_only": True, "model_requests": 0})
    pilot.update(ROOT / "preflight_progress.json", {
        "status": "complete", "completed": 365, "at": pilot.now(), "model_requests": 0})


if __name__ == "__main__":
    code = 0
    try:
        run()
    except BaseException as exc:
        code = 2
        pilot.update(ROOT / "preflight_failure.json", {
            "status": "failed", "at": pilot.now(), "model_requests": 0,
            "error_type": type(exc).__name__,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)]})
    pilot.update(ROOT / "preflight_exit.json", {"returncode": code, "at": pilot.now()})
    sys.exit(code)
