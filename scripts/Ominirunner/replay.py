"""Recover an existing public question contract without sampling or scoring."""
from pathlib import Path

from .common import Config, digest, read, selected_ids, sha
from .data import pair_identity, select
from .selection import validate as validate_selection

SCHEMA = "omni-frozen-question-replay/1"
FILES = ("frozen.json", "inputs.json", "preflight.json", "completed.json",
         "audit.json", "source_manifest.json")


def inspect(source, cache_root):
    source = Path(source).resolve()
    frozen = read(source / "frozen.json")
    inputs = read(source / "inputs.json")
    ids = selected_ids(frozen)
    identity = digest(frozen)
    if (frozen["config"] != Config().public()
            or sha(source / "inputs.json") != frozen["inputs_sha256"]
            or tuple(item["sample_id"] for item in inputs) != ids
            or inputs != select(Path(cache_root), ids)
            or read(source / "source_manifest.json") != frozen["source"]):
        raise ValueError("replayed_public_inputs_changed")
    if "selection" in frozen and validate_selection(cache_root, frozen["selection"]) != ids:
        raise ValueError("replayed_selection_changed")
    for name, expected in frozen["source"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha(source / "code" / path) != expected:
            raise ValueError("replayed_source_changed")
    completed, audit = read(source / "completed.json"), read(source / "audit.json")
    if (read(source / "preflight.json")["run_identity"] != identity
            or completed["status"] != "complete" or completed["run_identity"] != identity
            or completed["episodes"] != 2 * len(ids)
            or audit["status"] != "passed" or audit["run_identity"] != identity
            or [p["sample_id"] for p in frozen["pairs"]] != list(ids)
            or any(p["pair_identity"] != pair_identity(item)
                   for p, item in zip(frozen["pairs"], inputs))):
        raise ValueError("replayed_completed_contract_changed")
    return inputs, {
        "schema": SCHEMA, "source_root": str(source), "source_run_identity": identity,
        "files": {name: sha(source / name) for name in FILES},
        "sample_ids": list(ids), "selection": frozen.get("selection"),
        "reference_answers_read": False, "new_selection": False,
    }


def validate(receipt, cache_root):
    inputs, actual = inspect(receipt["source_root"], cache_root)
    if actual != receipt:
        raise ValueError("frozen_question_replay_changed")
    return inputs
