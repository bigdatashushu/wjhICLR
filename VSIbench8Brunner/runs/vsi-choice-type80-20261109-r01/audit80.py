#!/usr/bin/env python3
"""Audit the full new run on Titan without reference answers or inference."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "source_frozen/scripts"), str(ROOT / "source_frozen/src")]
from native80_terminal import verify
from skill3d.synthesis.eval_visual_fallback import build_visual_prompt


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, ensure_ascii=False) + "\n")


def images(messages):
    return [part for m in messages if isinstance(m.get("content"), list)
            for part in m["content"] if part.get("type") == "image_url"]


def main():
    from type80_adapter import verify_sources
    verify_sources()
    arm = ROOT / "harness"
    selected = {str(q["qa_id"]): q for q in read(ROOT / "sampling_manifest.json")["questions"]}
    assert len(selected) == 80
    scope = read(ROOT / "preparation_scope.json")
    launch = read(ROOT / "execute.pid.json")
    spec = read(ROOT / "experiment_spec.json")
    assert launch["execution_adapter_sha256"] == sha(ROOT / "type80_adapter.py")
    cutover = read(ROOT / "parallel_cutover_verified.json")
    assert cutover["issued_model_requests_repeated"] == 0
    assert cutover["solver_policy_changed"] is False
    assert cutover["source_frozen_modified"] is False
    for f in cutover["archive_files"]:
        assert sha(ROOT / f["file"]) == f["sha256"]
    preflight_launch = read(ROOT / "preflight.pid.json")
    assert sha(ROOT / "parallel_preflight80.py") == preflight_launch["parallel_preflight_source_sha256"]
    settings = read(arm / "parallel_execution.json")
    for key, value in cutover["execute_scheduling"].items():
        assert settings[key] == value
    preflight = read(arm / "preflight_verified.json")
    assert preflight["questions_verified"] == 80 and preflight["model_requests"] == preflight["gpu_requests"] == 0
    assert preflight["sampling_sha256"] == sha(ROOT / "sampling_manifest.json")
    identity = read(arm / "runtime_identity.json")
    for key in ("description", "reviewer_identity_sha256", "sam2_identity_sha256", "skill_snapshot"):
        assert identity[key] == spec[key]
    native = verify(ROOT, "harness", selected)
    index = read(arm / "trajectory_index.json")["questions"]
    rows, tokens = {}, Counter()
    described_count = binary_count = sam_count = 0
    for qa, q in selected.items():
        full = read(arm / index[qa]["file"])
        out = full["result"]["outcome"]
        requests = full["native_solver_requests"]
        fallback = out.get("eval_visual_fallback") or {}
        assert fallback.get("request_count", 0) in (0, 1)
        if fallback.get("request_count") == 1:
            messages = requests[-1]["record"]["request"]["messages"]
            round_ = fallback["image_round"]
            digest = hashlib.sha256(json.dumps(
                messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            assert round_["request_sha256"] == fallback["request_sha256"] == digest
            if requests[-1]["record"]["status"] == "returned":
                assert round_["delivered"] and round_["observed"]
            else:
                assert out["failure_code"] == "service_error"
            jpeg = [hashlib.sha256(base64.b64decode(
                im["image_url"]["url"].split(",", 1)[1], validate=True)).hexdigest()
                for im in images(messages)]
            assert jpeg == round_["jpeg_sha256"]
            if fallback.get("isolated_choice"):
                prompt = build_visual_prompt(q["question"], q.get("options") or [],
                                             full["result"]["question_type"], isolated_choice=True)
                assert out["is_mca"] and len(messages) == 1 and messages[0]["role"] == "user"
                assert len(messages[0]["content"]) == 33
                assert messages[0]["content"][0] == {"type": "text", "text": prompt}
                assert images(messages) == images(requests[0]["record"]["request"]["messages"])[-32:]
                assert round_["n_images"] == 32 and not round_["derived_image_ids"]
                assert not round_.get("scene_overview_image_ids")
                assert round_["image_ids"] == [f"frame-{i}" for i in range(32)]
                assert fallback["input_policy"] == "32_original_rgb_only"
                assert fallback["prompt_version"] == "eval-visual-choice-isolated-v2"
        binding = full["preparation"]["binding"]
        stats = binding["stats"]
        original_scope = scope["questions"][qa]
        ids = (stats.get("review_selection") or {}).get("selected_candidate_ids", [])
        assert ids == original_scope["review_candidate_ids"]
        assert not stats.get("actual_detector_requests")
        assert not full["native_parser_requests"]
        candidates = stats.get("candidate_receipts") or []
        observed = [c for c in candidates if c.get("semantic_visual")]
        assert [c["candidate_id"] for c in observed] == [
            c["candidate_id"] for c in candidates if c["candidate_id"] in ids]
        assert len(observed) == len(ids)
        for c in observed:
            visual = c["semantic_visual"]
            assert visual["status"] == "observed"
            desc = visual["target_description"]
            assert desc["status"] == "described"
            assert desc["raw_response"]["model"] == spec["description"]["model"]
        described_count += len(observed)
        binary_count += len(observed)
        sam_receipts = read(arm / "sam2_receipts" / (qa + ".json"))
        assert len(sam_receipts) <= original_scope["sam2_frame_batch_cap"]
        sam_count += len(sam_receipts)
        assert stats["generation"]["sha256"] == hashlib.sha256(json.dumps(
            stats["generation"]["identity"], sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode()).hexdigest()
        for req in requests:
            usage = (req["record"].get("raw_response") or {}).get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                tokens[key] += int(usage.get(key) or 0)
        rows[qa] = {
            "is_mca": out["is_mca"], "answer_basis": out.get("answer_basis"),
            "fallback": {key: fallback.get(key) for key in (
                "accepted", "request_count", "isolated_choice", "input_policy", "trigger",
                "prompt_version", "answer", "original_result")},
            "native_requests": len(requests), "description_requests": len(observed),
            "binary_requests": len(observed), "sam2_batches": len(sam_receipts),
        }
    budget = read(arm / "request_budget.json")
    assert budget["max_requests"] == scope["preparation_request_cap"]
    assert budget["used"] == len(budget["requests"]) == described_count + binary_count
    assert budget["used"] == budget["max_requests"]
    providers = Counter(r["provider"] for r in budget["requests"])
    assert providers == Counter(qwen3vl8b_description=described_count, intern_binary_review=binary_count)
    assert sam_count <= scope["sam2_frame_batch_cap"]
    policy = {
        "schema": "vsi-choice-balanced80-policy-audit/1", "questions_verified": 80,
        "isolated_requests_verified": sum(bool(r["fallback"].get("isolated_choice"))
                                         and r["fallback"].get("request_count") == 1 for r in rows.values()),
        "new_preparation_requests": budget["used"], "description_requests": described_count,
        "binary_requests": binary_count, "sam2_batches": sam_count,
        "new_parser_requests": 0, "new_detector_requests": 0, "new_reconstruction_requests": 0,
        "automatic_retry": False, "reference_answers_read": False,
        "reference_answer_visible": False, "token_usage": dict(tokens), "questions": rows,
        "valid_service_questions": native["valid_service_questions"],
        "terminal_service_failures": native["terminal_service_failures"],
    }
    write(ROOT / "choice_policy_verified.json", policy)
    receipt = {
        "schema": "vsi-type80-final-native-collection/1", "completed": 80, "at": time.time(),
        "sampling_sha256": sha(ROOT / "sampling_manifest.json"),
        "source_manifest_sha256": sha(ROOT / "source_manifest.json"),
        "adapter_sha256": sha(ROOT / "type80_adapter.py"),
        "preflight_sha256": sha(arm / "preflight_verified.json"),
        "scope_sha256": sha(ROOT / "preparation_scope.json"),
        "native_audit_sha256": sha(arm / "native_trajectory_verified.json"),
        "compact_score_inputs_sha256": sha(arm / "compact_score_inputs.json"),
        "policy_audit_sha256": sha(ROOT / "choice_policy_verified.json"),
        "execution_receipt_sha256": sha(ROOT / "execute.pid.json"),
        "parallel_cutover_sha256": sha(ROOT / "parallel_cutover_verified.json"),
        "parallel_preflight_source_sha256": sha(ROOT / "parallel_preflight80.py"),
        "audit_source_sha256": sha(Path(__file__)), "model_requests": native["model_requests"],
        "terminal_verifier_source_sha256": sha(ROOT / "native80_terminal.py"),
        "continuation_source_sha256": sha(ROOT / "continue_unissued80.py"),
        "continuation_plan_sha256": sha(ROOT / "continuation_unissued/plan.json"),
        "continuation_verified_sha256": sha(ROOT / "continuation_unissued/verified.json"),
        "valid_service_questions": native["valid_service_questions"],
        "terminal_service_failures": native["terminal_service_failures"],
        "reference_answers_read": False, "reference_answer_visible": False,
        "verification_added_model_requests": 0,
    }
    write(ROOT / "final_native_collection_verified.json", receipt)
    print(json.dumps({k: v for k, v in policy.items() if k != "questions"}))


if __name__ == "__main__":
    main()
