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
    parser.add_argument("--transport-timeout-s", type=float)
    parser.add_argument("--perception-endpoint")
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
    input_files = dict(old["input_files"])
    if args.transport_timeout_s is not None:
        import yaml
        assert 180 < args.transport_timeout_s <= 1800
        path = root / "inputs/online.yaml"
        config = yaml.safe_load(path.read_text())
        config["online_model"]["timeout_s"] = args.transport_timeout_s
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        input_files["online.yaml"] = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = {**old["runtime"], "enable_thinking": True, "max_requests": 320}
    if args.perception_endpoint:
        from urllib.parse import urlsplit
        url = urlsplit(args.perception_endpoint)
        assert url.scheme == "http" and url.hostname == "10.10.0.198"
        assert url.port and not url.username and not url.password
        assert not url.query and not url.fragment and url.path in ("", "/")
        runtime["perception_endpoint"] = args.perception_endpoint.rstrip("/")
    runner = Path(__file__).resolve().parents[1]
    receipt = freeze_source(runner, root / "code")
    receipt["source_root"] = str(Path(args.native_root) / "code/source")
    (root / "code/code_identity.json").write_text(json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    shutil.copyfile(runner / "scripts/run_g1m1_diagnostic.py", root / "controller.py")
    manifest = {
        **{k: v for k, v in old.items() if k != "identity_sha256"},
        "native_root": args.native_root, "source_sha256": receipt["source_sha256"],
        "input_files": input_files,
        "controller_sha256": hashlib.sha256((root / "controller.py").read_bytes()).hexdigest(),
        "previous_run_identity": old["identity_sha256"],
        "previous_source_sha256": old["source_sha256"],
        "comparison": {
            "arms": ["system", "direct"], "same_40_questions": True,
            "previous_input_bytes_preserved": args.transport_timeout_s is None,
            "same_questions_and_rgb_bytes": True, "both_enable_thinking": True,
            "direct_protocol": "prepared-32", "direct_rgb_only": True,
            "system_request_cap": 280, "direct_request_cap": 40,
            "transport_timeout_s": args.transport_timeout_s,
        },
        "runtime": runtime,
    }
    manifest["identity_sha256"] = content_sha256(manifest)
    (root / "diagnostic.json").write_text(json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"run_root": str(root), "n_questions": 40,
                      "enable_thinking": True, "source_sha256": receipt["source_sha256"],
                      "identity_sha256": manifest["identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
