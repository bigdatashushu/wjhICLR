"""panel v2 升级与 pilot 系统故障验收。"""

import copy
import json

import pytest

from skill3d.evaluation.fitset_panel_v11 import upgrade_fitset_panel
from skill3d.evaluation.fitset_v11 import (
    _variant_key, _write_create_once, audit_fitset_pilot, content_sha256,
    load_fitset_panel, select_panel_variants,
)
from test_fitset_v11 import _panel_file


def test_panel_upgrade_removes_unreachable_and_keeps_ready_inputs(tmp_path):
    path = _panel_file(tmp_path)
    panel = json.loads(path.read_text())
    panel["schema_version"] = "v11-same-question-system-vs-direct-qwen-proposal/1.0"
    panel["experiment_id"] = "fixture"
    invalid = copy.deepcopy(panel["system_variants"][0])
    invalid.update(qa_id="11", variant_id="11@g1m1",
                   question_type="object_counting", evidence_state="g1m1",
                   system_input_status="binding_preparation_required")
    panel["system_variants"].append(invalid)
    panel["direct_qwen_rows"].append({
        "qa_id": "11", "scene_id": invalid["scene_id"],
        "question_type": "object_counting", "split": invalid["split"],
        "system_variant_ids": ["11@g1m1"],
    })
    panel["panels"]["maximum_1166"] = {"system_input_variants": 3}
    panel.pop("proposal_sha256")
    panel["proposal_sha256"] = content_sha256(panel)
    path.write_text(json.dumps(panel))
    original = path.read_bytes()
    target = tmp_path / "v2.json"
    upgrade_fitset_panel(path, target)
    loaded = load_fitset_panel(target)
    assert path.read_bytes() == original
    assert len(loaded["system_variants"]) == 2
    assert loaded["skill_accumulation"]["skill_count"] == 19
    assert "maximum_1166" not in loaded["panels"]
    assert select_panel_variants(loaded, "ready_579")[0]["variant_id"] == "10@g1m0"
    with pytest.raises(Exception, match="新的目标"):
        upgrade_fitset_panel(path, target)


def test_pilot_rejects_sigalrm_failure_even_when_visual_fallback_answers(tmp_path):
    key = "qa@g0m0"
    body = {"variant_ids": [key]}
    manifest = {**body, "identity_sha256": content_sha256(body)}
    _write_create_once(tmp_path / "manifest.json", manifest)
    _write_create_once(tmp_path / "summary.json", {
        "manifest_identity_sha256": manifest["identity_sha256"],
        "completed_variants": 1, "pending_variants": 0,
    })
    _write_create_once(tmp_path / "results" / (_variant_key(key) + ".json"), {
        "variant_id": key, "evidence_state": "g0m0",
        "initial_skill_selection": {"evidence_state": "g0m0"},
        "answer_source": "direct_vlm_routed",
    })
    _write_create_once(tmp_path / "variants" / _variant_key(key) / "outcome.json", {
        "rounds": [{
            "executed": True, "error_code": "violation_runtime",
            "execution_diagnostics": {
                "error_type": "RuntimeError",
                "error_message": "cell timeout requires a main-thread SIGALRM runtime",
            },
        }],
    })
    report = audit_fitset_pilot(output_root=tmp_path, expected_variants=1)
    assert not report["expand_to_full"]
    assert report["infrastructure_error_counts"]
    assert report["visual_fallback_answers"] == 1
