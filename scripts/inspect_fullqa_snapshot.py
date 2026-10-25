#!/usr/bin/env python3
"""Inspect a completed immutable input snapshot without any model request."""
import argparse
from collections import Counter
import json
from pathlib import Path

import fullqa_custody as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    proof = c.read(args.snapshot / "preflight.json")
    path = args.snapshot / "published_inputs.json"
    c.require(proof["status"] == "verified"
              and proof["manifest_sha256"] == c.digest(path.read_bytes())
              and proof["model_requests"] == 0, "snapshot_not_preflight_verified")
    manifest = c.read(path)
    c.require(manifest["source_catalog_sha256"] ==
              c.digest(Path(manifest["source_catalog_file"]).read_bytes()), "source_catalog_changed")
    counts = Counter(entry["split"] for entry in manifest["entries"].values())
    c.require(set(counts) <= {"induction", "inner_validation"}, "reserved_split_present")
    print(json.dumps({"status": "verified", "snapshot": str(args.snapshot),
                      "manifest_sha256": proof["manifest_sha256"],
                      "questions": dict(counts), "scenes": len(manifest["scene_scopes"]),
                      "normal_loader_checked_scenes": proof["normal_loader_checked_scenes"],
                      "model_requests": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
