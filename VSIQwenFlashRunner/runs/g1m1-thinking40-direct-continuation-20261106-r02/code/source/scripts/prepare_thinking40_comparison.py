#!/usr/bin/env python3
"""Copy the previous forty inputs byte for byte and freeze the current runner."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skill3d.infra.vsi_runner_source import freeze_source
from skill3d.infra.m5_identity import content_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--native-root", required=True)
    args = parser.parse_args()
    previous, root = args.previous_run.resolve(), args.run_root.resolve()
    assert not root.exists()
    old = json.loads((previous / "diagnostic.json").read_text())
    assert content_sha256({k: v for k, v in old.items() if k != "identity_sha256"}) == old["identity_sha256"]
    assert old["n_questions"] == 40 and old["seed"] == 137
    root.mkdir()
    shutil.copytree(previous / "inputs", root / "inputs")
    for name, expected in old["input_files"].items():
        assert hashlib.sha256((root / "inputs" / name).read_bytes()).hexdigest() == expected
    runner = Path(__file__).resolve().parents[1]
    receipt = freeze_source(runner, root / "code")
    receipt["source_root"] = str(Path(args.native_root) / "code/source")
    (root / "code/code_identity.json").write_text(json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    shutil.copyfile(runner / "scripts/run_g1m1_diagnostic.py", root / "controller.py")
    manifest = {
        **{k: v for k, v in old.items() if k != "identity_sha256"},
        "native_root": args.native_root, "source_sha256": receipt["source_sha256"],
        "controller_sha256": hashlib.sha256((root / "controller.py").read_bytes()).hexdigest(),
        "previous_run_identity": old["identity_sha256"],
        "previous_source_sha256": old["source_sha256"],
        "comparison": {
            "arms": ["system", "direct"], "same_40_questions": True,
            "previous_input_bytes_preserved": True, "both_enable_thinking": True,
            "direct_protocol": "prepared-32", "direct_rgb_only": True,
            "system_request_cap": 280, "direct_request_cap": 40,
        },
        "runtime": {**old["runtime"], "enable_thinking": True, "max_requests": 320},
    }
    manifest["identity_sha256"] = content_sha256(manifest)
    (root / "diagnostic.json").write_text(json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"run_root": str(root), "n_questions": 40,
                      "enable_thinking": True, "source_sha256": receipt["source_sha256"],
                      "identity_sha256": manifest["identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
