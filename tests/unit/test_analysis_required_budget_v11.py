"""Zero-request budget visibility never expands the authorized shared allowance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(__file__).resolve().parents[2] / "scripts/run_analysis_successor_v11.py"
_spec = importlib.util.spec_from_file_location("analysis_budget_test", _path)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


@pytest.mark.parametrize("authorized", [56, 57])
def test_preflight_records_exact_slot_requirements_and_refuses_short_allowance(
        tmp_path, monkeypatch, authorized):
    slots = ["object_rel_direction/g0m0", "object_rel_distance/g0m0", "object_abs_distance/g0m0"]
    rows = [{"campaign_id": f"synthetic-{index}", "slot_key": slot,
             "batch_count": count, "total_request_cap": 2 * count + 3}
            for index, (slot, count) in enumerate(zip(slots, [9, 5, 10]))]
    client = SimpleNamespace(audit_identity=lambda **kwargs: {"model": "test-model", **kwargs})
    monkeypatch.setattr(module.tools, "protected", lambda *args: {})
    monkeypatch.setattr(module, "setup", lambda args: (None, {}, client, None, None, None, None))
    monkeypatch.setattr(module, "selections", lambda *args: rows)
    monkeypatch.setattr(module, "prepare_one", lambda *args: (args[-1], None))
    monkeypatch.setattr(module, "source_identity", lambda args: {"public": "source-hash"})
    args = SimpleNamespace(output=tmp_path, state_root=tmp_path, frozen_root=tmp_path,
                           reviser_request_cap=authorized, gate_request_cap=4096,
                           analysis_concurrency=4, selection="pilot")
    result = module.preflight(args)
    required = json.loads((tmp_path / "required_budget.json").read_text())
    assert required["required_total"] == 57
    assert list(required["required_revision_request_caps"][slot] for slot in slots) == [21, 13, 23]
    assert required["authorized_total"] == authorized
    assert required["gate_allowance_unchanged"] == 4096
    assert required["model_requests"] == result["model_requests"] == 0
    assert result["status"] == ("ready" if authorized == 57 else "budget_refused")
    assert (tmp_path / "preflight.json").exists() == (authorized == 57)
    assert not list(tmp_path.glob("**/request_intent_*.json"))
