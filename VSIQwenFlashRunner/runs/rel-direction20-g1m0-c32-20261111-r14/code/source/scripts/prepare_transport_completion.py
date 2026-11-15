#!/usr/bin/env python3
"""Freeze only transport-interrupted questions, preserving completed attempts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--perception-endpoint", required=True)
    parser.add_argument("--timeout-s", type=float, default=1200)
    parser.add_argument("--interruption-receipt", type=Path,
                        help="Known failed requests while unrelated original episodes still drain.")
    args = parser.parse_args()
    origin, root = args.origin_root.resolve(), args.run_root.resolve()
    assert not root.exists() and root.parent.is_dir()
    sys.path.insert(0, str(origin / "code/source/src"))
    from skill3d.infra.m5_identity import content_sha256
    import yaml

    old = read(origin / "diagnostic.json")
    assert content_sha256({k: v for k, v in old.items()
                           if k != "identity_sha256"}) == old["identity_sha256"]
    for name, sha in old["input_files"].items():
        assert hashlib.sha256((origin / "inputs" / name).read_bytes()).hexdigest() == sha
    for row in read(origin / "code/code_identity.json")["files"]:
        assert hashlib.sha256((origin / "code/source" / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    if args.interruption_receipt:
        snapshot = read(args.interruption_receipt)
        assert snapshot["origin_identity"] == old["identity_sha256"]
        errors = snapshot["failed_requests"]
        completed = set(snapshot["completed_variant_ids"])
        health = snapshot["gateway_upstream"]
        assert snapshot["direct_completed"] == old["n_questions"]
    else:
        assert read(origin / "exit.json")["status"] == "stopped"
        collection = read(origin / "collection_receipt.json")
        assert collection["diagnostic_identity"] == old["identity_sha256"]
        for name, identity in collection["files"].items():
            assert hashlib.sha256((origin / name).read_bytes()).hexdigest() == identity["sha256"]
        state = read(origin / "requests/state.json")
        assert all(row["status"] in ("completed", "error") for row in state["requests"])
        errors = [row for row in state["requests"] if row["status"] == "error"]
        completed = {read(p)["variant_id"] for p in (origin / "system/results").glob("*.json")}
        health = read(origin / "gateway_final_health.json")["upstream"]
        assert health["active"] == 0
        assert read(origin / "direct_receipt.json")["status"] == "complete"
        assert len(list((origin / "direct/responses").glob("*.json"))) == old["n_questions"]
    assert errors and all(row["exception_type"] == "DiagnosticOperationalStop" for row in errors)
    assert all(not row["context"]["variant_id"].startswith("baseline:") for row in errors)
    expected = {row["variant_id"] for row in old["variants"]}
    missing = {row["context"]["variant_id"] for row in errors}
    assert missing <= expected and not missing & completed
    if not args.interruption_receipt:
        assert missing == expected - completed
    assert len(errors) == len(missing), "each missing episode must have one recorded failed request"
    assert health["unknown"] == 0
    assert health["failed"] == len(errors)
    cfg = yaml.safe_load((origin / "inputs/online.yaml").read_text())
    previous_timeout = cfg["online_model"]["timeout_s"]
    assert previous_timeout < args.timeout_s <= 1800
    assert all(previous_timeout - 5 <= row["completed_at"] - row["started_at"]
               <= previous_timeout + 10 for row in errors), "not a wait-limit interruption"

    root.mkdir()
    shutil.copytree(origin / "inputs", root / "inputs")
    shutil.copytree(origin / "code", root / "code")
    for name in ("controller.py", "launcher.py"):
        shutil.copyfile(origin / name, root / name)
    code = read(root / "code/code_identity.json")
    code["source_root"] = str(Path(args.native_root) / "code/source")
    write(root / "code/code_identity.json", code)

    variants = [row for row in old["variants"] if row["variant_id"] in missing]
    qa_ids = {row["qa_id"] for row in variants}
    sample = [row for row in read(origin / "inputs/sample_questions.json")
              if row["qa_id"] in qa_ids]
    assert len(sample) == len(variants)
    write(root / "inputs/sample_questions.json", sample)
    panel = read(root / "inputs/panel.json")
    panel["system_variants"] = variants
    panel["direct_qwen_rows"] = [row for row in panel["direct_qwen_rows"] if row["qa_id"] in qa_ids]
    panel["experiment_id"] += "-transport-completion"
    slots = sorted({row["question_type"] + "/" + row["evidence_state"] for row in variants})
    for value in panel["panels"].values():
        value.update(system_input_variants=len(variants), skill_slots=slots,
                     skill_slot_count=len(slots))
    panel["proposal_sha256"] = content_sha256({k: v for k, v in panel.items()
                                             if k != "proposal_sha256"})
    write(root / "inputs/panel.json", panel)
    cfg["online_model"]["timeout_s"] = args.timeout_s
    (root / "inputs/online.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    gateway_cfg = read(origin / "gateway_config.json")
    assert gateway_cfg["online_model"]["timeout_s"] == previous_timeout
    gateway_cfg["online_model"]["timeout_s"] = args.timeout_s
    write(root / "completion_gateway_config.json", gateway_cfg)

    new = copy.deepcopy(old)
    new.pop("identity_sha256")
    new.pop("prior_sample_sha256", None)
    new.update(selection="Only original episodes interrupted at their request wait limit",
               n_questions=len(variants), n_scenes=len({row["scene_name"] for row in sample}),
               questions_per_type=len(variants), variants=variants, native_root=args.native_root)
    from collections import Counter
    new["task_counts"] = dict(Counter(row["question_type"] for row in variants))
    new["runtime"].update(
        episode_concurrency=len(variants), direct_workers=0,
        max_requests=7 * len(variants), scene_cache_size=len(variants),
        question_preflight="required_zero_model_calls",
        perception_endpoint=args.perception_endpoint, transport_timeout_s=args.timeout_s)
    new["comparison"].update(arms=["system"], system_request_cap=7 * len(variants),
                             direct_request_cap=0)
    new["transport_completion"] = {
        "origin_identity": old["identity_sha256"],
        "origin_root": str(origin), "origin_native_root": old["native_root"],
        "origin_failed_requests": errors,
        "failed_variant_ids": sorted(missing),
        "failed_request_ids": [row["request_id"] for row in errors],
        "origin_variants_not_repeated": sorted(expected - missing),
        "previous_timeout_s": previous_timeout, "new_timeout_s": args.timeout_s,
        "selection_used_reference_answers": False,
        "same_source_and_model_parameters_except_transport_wait": True,
        "automatic_model_retry": False,
    }
    if args.interruption_receipt:
        new["transport_completion"]["interruption_receipt_sha256"] = hashlib.sha256(
            args.interruption_receipt.read_bytes()).hexdigest()
    else:
        new["transport_completion"]["origin_collection_sha256"] = hashlib.sha256(
            (origin / "collection_receipt.json").read_bytes()).hexdigest()
    new["input_files"] = {str(p.relative_to(root / "inputs")): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "inputs").rglob("*")) if p.is_file()}
    new["identity_sha256"] = content_sha256(new)
    write(root / "diagnostic.json", new)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    print(json.dumps({"status": "frozen_transport_completion", "qa_ids": sorted(qa_ids),
                      "origin_identity": old["identity_sha256"],
                      "identity": new["identity_sha256"], "timeout_s": args.timeout_s}))


if __name__ == "__main__":
    main()
