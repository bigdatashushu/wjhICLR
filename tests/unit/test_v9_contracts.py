from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

# The repository's current SciPy/NumPy environment is incompatible at import
# time. Keep this compatibility shim local to the focused contract tests.
for _name, _value in {"long": np.int64, "ulong": np.uint64}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

from skill3d.online.runner import OnlineRunConfig, _invalidate_metric_gate
from skill3d.schemas import MetricEvidenceGateResult


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "skill_library"


def test_current_budget_fields_are_normalized_without_legacy_aliases():
    names = {field.name for field in fields(OnlineRunConfig)}
    assert {"max_agent_rounds", "reserve_final_rounds", "max_recovery"}.isdisjoint(names)
    explicit = OnlineRunConfig(
        max_solver_rounds=8, finalization_rounds=7, max_retries_per_operation=1)
    assert (explicit.max_solver_rounds, explicit.finalization_rounds,
            explicit.max_retries_per_operation) == (8, 7, 1)
    bounded = OnlineRunConfig(max_solver_rounds=1, finalization_rounds=99,
                              max_retries_per_operation=-3)
    assert (bounded.max_solver_rounds, bounded.finalization_rounds,
            bounded.max_retries_per_operation) == (1, 0, 0)


def test_metric_gate_invalidation_revalidates_partial_gate():
    gate = MetricEvidenceGateResult(
        gate_passed=True, gate_version="test", sub_results={"unrelated": True})
    invalidated = _invalidate_metric_gate(gate, "geometry_3d")
    assert invalidated is not None
    assert invalidated.gate_passed is False
    assert invalidated.sub_results["scene_route_full_3d"] is False
    assert invalidated.sub_results["unrelated"] is True
    assert "geometry_3d" in invalidated.invalidated_by
    assert "scene_route_full_3d" in invalidated.missing_subconditions
