"""Zero-model official reset/step diagnostic with complete public packets."""
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import sys

from protocol import ROOT, canonical, sha256, within_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work = within_root(args.work_dir)
    library = str(ROOT / "manipulation-runtime/coppeliasim")
    if os.environ.get("LD_LIBRARY_PATH") != library:
        from action_bridge import v2_environment
        env = v2_environment(work.parent)
        env["LD_LIBRARY_PATH"] = library
        os.execve(sys.executable, [sys.executable, "-B", __file__, *sys.argv[1:]], env)
    if not work.parent.is_dir():
        parser.error("work parent must exist")
    work.mkdir(mode=0o700, exist_ok=False)
    from manipulation_worker import OfficialManipulation, asset_catalog, probe, runtime
    runtime()
    from household_worker import native_log
    adapter = OfficialManipulation("base", work)
    packets, receipt = [], None
    try:
        with (work / "native.log").open("w") as log, native_log(log):
            receipt = probe()
            if not receipt["runtime_ready"]:
                raise RuntimeError("runtime_not_ready")
            packets.append(adapter.open({"subset": "base", "index": 0,
                                         "source_sha256": sha256(canonical(asset_catalog()))}))
            # Protocol diagnostic action, fixed before reset; no config or waypoint access.
            action = [50, 50, 30, 120, 60, 90, 1]
            packets.append(adapter.step(action, True))
            for packet in packets:
                adapter.validate_packet(packet)
    finally:
        adapter.close()
    images = []
    for packet in packets:
        for key, name in (("image_base64", "raw"), ("planner_image_base64", "planner")):
            path = work / (name + f"-step-{packet['step']:02d}.png")
            path.write_bytes(base64.b64decode(packet[key], validate=True))
            images.append(str(path.relative_to(work)))
    result = {"schema": "embodied-manipulation-reset-step/1", "status": "passed",
              "action": action, "runtime": receipt, "packets": packets, "images": images,
              "model_requests": 0, "simulator_launches": 1, "environment_steps": 1,
              "automatic_retry": False, "reference_answer_visible": False}
    (work / "result.json").write_bytes(canonical(result) + b"\n")
    print(canonical({k: result[k] for k in ("schema", "status", "model_requests",
                                          "simulator_launches", "environment_steps")}).decode())


if __name__ == "__main__":
    main()
