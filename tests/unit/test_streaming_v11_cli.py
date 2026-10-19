"""CLI contracts for the explicit 19-slot streaming plan."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "streaming_v11_cli_test",
    ROOT / "scripts/run_streaming_evolution_v11.py",
)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def _args(*extra):
    return CLI.build_parser().parse_args([
        "--stream-id",
        "test",
        "--run-root",
        "/tmp/test-v11-stream",
        *extra,
    ])


def test_default_targets_are_exactly_nineteen_reachable_slots():
    targets = CLI._target_slots(_args())
    assert len(targets) == 19
    assert len(set(targets)) == 19
    assert (
        "object_rel_distance",
        "g1m1",
    ) not in targets
    assert (
        "object_abs_distance",
        "g1m1",
    ) in targets


def test_single_slot_arguments_remain_supported():
    targets = CLI._target_slots(_args(
        "--question-types",
        "object_rel_distance",
        "--evidence-states",
        "g1m0",
    ))
    assert targets == [("object_rel_distance", "g1m0")]


def test_slot_manifest_rejects_unreachable_combinations(tmp_path):
    manifest = tmp_path / "slots.json"
    manifest.write_text(json.dumps({
        "slots": ["object_rel_distance/g1m1"],
    }), encoding="utf-8")
    with pytest.raises(
        CLI.V11StreamingCliError,
        match="unreachable",
    ):
        CLI._target_slots(_args("--slot-manifest", str(manifest)))
