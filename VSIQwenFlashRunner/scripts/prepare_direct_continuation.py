#!/usr/bin/env python3
"""Freeze a direct-only continuation after a verified input failure."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skill3d.infra.m5_identity import content_sha256
from skill3d.infra.vsi_runner_source import freeze_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-run", type=Path, required=True)
    parser.add_argument("--completed-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--native-root", required=True)
    args = parser.parse_args()
    origin, root = args.origin_run.resolve(), args.run_root.resolve()
    old = json.loads((origin / "diagnostic.json").read_text())
    completed = json.loads(args.completed_receipt.read_text())
    assert content_sha256({k: v for k, v in old.items() if k != "identity_sha256"}) == old["identity_sha256"]
    assert completed["status"] == "passed" and completed["automatic_model_retry"] is False
    assert completed["origin_identity"] == old["identity_sha256"]
    assert completed["origin_source_sha256"] == old["source_sha256"]
    done = set(completed["completed_qa_ids"])
    all_qa = {row["qa_id"] for row in old["variants"]}
    assert len(done) == completed["origin_model_requests"] and done < all_qa
    assert not root.exists()
    root.mkdir()
    shutil.copytree(origin / "inputs", root / "inputs")
    for name, expected in old["input_files"].items():
        assert hashlib.sha256((root / "inputs" / name).read_bytes()).hexdigest() == expected
    runner = Path(__file__).resolve().parents[1]
    receipt = freeze_source(runner, root / "code")
    receipt["source_root"] = str(Path(args.native_root) / "code/source")
    (root / "code/code_identity.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    shutil.copyfile(runner / "scripts/run_g1m1_diagnostic.py", root / "controller.py")
    continuation = {key: completed[key] for key in (
        "origin_native_root", "origin_identity", "origin_source_sha256",
        "completed_qa_ids", "origin_model_requests", "files")}
    continuation.update(
        remaining_qa_ids=sorted(all_qa - done), automatic_model_retry=False,
        reason="canonicalize_direction_subtype_before_rgb_question_identity_check",
        shared_request_governor=True, original_responses_preserved=True)
    manifest = {
        **{k: v for k, v in old.items() if k != "identity_sha256"},
        "native_root": args.native_root, "source_sha256": receipt["source_sha256"],
        "controller_sha256": hashlib.sha256((root / "controller.py").read_bytes()).hexdigest(),
        "direct_continuation": continuation,
    }
    manifest["identity_sha256"] = content_sha256(manifest)
    (root / "diagnostic.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "run_root": str(root),
                      "remaining_questions": len(all_qa - done),
                      "identity_sha256": manifest["identity_sha256"],
                      "source_sha256": receipt["source_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
