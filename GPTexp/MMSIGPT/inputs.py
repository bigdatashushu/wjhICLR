"""Label-free MMSI preparation using the existing immutable input bindings."""
from __future__ import annotations

from pathlib import Path


def make_specs():
    from ..adapters import COUNTS, ROOT, problem_for, public_spec, setup
    from ..traces import read

    setup("mmsi")
    from skill3d.small_image.blink_protocol import image_identities
    from skill3d.small_image.runner import _validate_prepared, load_problem_frames
    from skill3d.small_image.storage import load_prepared

    old = read(ROOT / "frozen/mmsi/manifest.json")
    specs = []
    for job in old["jobs"]:
        spec = public_spec(job)
        spec["arm"] = "direct" if spec["arm"] == "direct" else "harness"
        if spec["arm"] == "direct":
            for key in ("prepared_root", "artifact_sha256", "skill", "allowed_tools"):
                spec.pop(key, None)
        problem = problem_for(spec["problem"])
        if image_identities(problem) != job["images"]:
            raise ValueError("mmsi_image_order_changed")
        if spec["arm"] == "harness":
            prepared = load_prepared(
                Path(spec["prepared_root"]) / problem.input_sha256, problem=problem
            )
            if (
                prepared.artifact_sha256 != spec["artifact_sha256"]
                or _validate_prepared(prepared, problem, load_problem_frames(problem))
            ):
                raise ValueError("mmsi_prepared_changed")
        specs.append(spec)
    if len(specs) != COUNTS["mmsi"] * 2:
        raise ValueError("frozen_denominator_changed")
    return specs

