#!/usr/bin/env python3
"""Audit the actual isolated requests after complete native trajectory verification."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def images(messages):
    return [part for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"] if part.get("type") == "image_url"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root, arm = args.root, args.root / "harness"
    native = read(arm / "native_trajectory_verified.json")
    if native["questions_verified"] != 100:
        raise ValueError("policy audit requires complete native verification")
    spec = read(root / "experiment_spec.json")
    baseline = Path(spec["paired_baseline_server_root"]) / "harness"
    questions = {str(q["qa_id"]): q for q in read(root / "sampling_manifest.json")["questions"]}
    index = read(arm / "trajectory_index.json")["questions"]
    rows, tokens = {}, {}
    for qa, question in questions.items():
        full = read(arm / index[qa]["file"])
        outcome = full["result"]["outcome"]
        requests = full["native_solver_requests"]
        fallback = outcome.get("eval_visual_fallback") or {}
        if fallback.get("request_count", 0) not in (0, 1):
            raise ValueError("additional visual request cap exceeded")
        if fallback.get("request_count") == 1:
            payload = requests[-1]["record"]["request"]
            messages = payload["messages"]
            round_ = fallback["image_round"]
            digest = hashlib.sha256(json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            if (round_["request_sha256"] != digest or fallback["request_sha256"] != digest
                    or not round_["delivered"] or not round_["observed"]):
                raise ValueError("fallback request or observed ledger identity differs")
            jpg_hashes = [hashlib.sha256(base64.b64decode(
                image["image_url"]["url"].split(",", 1)[1], validate=True)).hexdigest()
                for image in images(messages)]
            if jpg_hashes != round_["jpeg_sha256"]:
                raise ValueError("actual JPEG bytes differ from the image ledger")
            if fallback.get("isolated_choice"):
                from skill3d.synthesis.eval_visual_fallback import build_visual_prompt
                initial = images(requests[0]["record"]["request"]["messages"])
                expected_prompt = build_visual_prompt(
                    question["question"], question.get("options") or [],
                    full["result"]["question_type"], isolated_choice=True)
                if (not outcome["is_mca"] or len(messages) != 1
                        or messages[0]["role"] != "user"
                        or len(messages[0]["content"]) != 33
                        or messages[0]["content"][0] != {"type": "text", "text": expected_prompt}
                        or images(messages) != initial[-32:]
                        or round_["n_images"] != 32 or round_["derived_image_ids"]
                        or round_.get("scene_overview_image_ids")
                        or round_["image_ids"] != [f"frame-{i}" for i in range(32)]
                        or fallback["input_policy"] != "32_original_rgb_only"
                        or fallback["prompt_version"] != "eval-visual-choice-isolated-v2"):
                    raise ValueError("isolated fallback included history or non-original pixels")
                if (fallback.get("trigger") == "accepted_visual_estimate"
                        and fallback["original_result"]["answer_basis"] != "visual_estimate"):
                    raise ValueError("accepted visual trigger provenance differs")
        previous = read(baseline / "outcomes" / f"{qa}.json")["outcome"]
        rows[qa] = {
            "is_mca": outcome["is_mca"], "answer_basis": outcome.get("answer_basis"),
            "old_answer_basis": previous.get("answer_basis"),
            "fallback": {key: fallback.get(key) for key in (
                "accepted", "request_count", "isolated_choice", "input_policy",
                "trigger", "prompt_version", "answer", "original_result")},
            "native_requests": len(requests),
            "program_errors": sum(bool(r["record"].get("cell", {}).get("error_code"))
                                  for r in full["program_executions"]),
        }
        for request in requests:
            usage = request["record"]["raw_response"].get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                tokens[key] = tokens.get(key, 0) + int(usage.get(key) or 0)
    budget = read(arm / "request_budget.json")
    if budget != {"max_requests": 0, "used": 0, "requests": []}:
        raise ValueError("cached-only experiment issued a preparation request")
    receipt = {
        "schema": "vsi-choice-isolated-policy-audit/1", "questions_verified": 100,
        "isolated_requests_verified": sum(
            bool(r["fallback"].get("isolated_choice"))
            and r["fallback"].get("request_count") == 1
            for r in rows.values()),
        "reference_answer_visible": False, "new_preparation_requests": 0,
        "automatic_retry": False, "token_usage": tokens, "questions": rows,
    }
    (root / "choice_policy_verified.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "questions"}), flush=True)


if __name__ == "__main__":
    main()
