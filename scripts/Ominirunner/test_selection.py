"""Offline checks for fresh sampling and variable paired trial denominators."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner import audit, report
from Ominirunner.common import Config, IDS, VERSION, digest, selected_ids, write
from Ominirunner.reuse import request_cap
from Ominirunner.selection import from_bindings, validate


def public_catalog():
    return {"questions": [{
        "benchmark": "Omni3D-Bench", "source_split": "test", "sample_id": str(i),
        "question": "How many chairs are there?", "image_keys": ["image-" + str(i)],
        "task_type": "int", "options": [], "input_sha256": "input-" + str(i),
    } for i in range(501)]}


def test_selection_excludes_previous_images_and_is_order_independent():
    document = public_catalog()
    document["questions"][100]["image_keys"] = ["image-" + IDS[0]]
    receipt = from_bindings(document)
    assert receipt["count"] == len(receipt["sample_ids"]) == 15
    assert len(set(receipt["sample_ids"])) == 15
    assert not set(receipt["sample_ids"]).intersection(IDS)
    assert "100" not in receipt["sample_ids"]
    assert receipt["eligible_questions"] == 480
    document["questions"].reverse()
    assert from_bindings(document) == receipt
    assert not receipt["selection_uses_outcomes"] and not receipt["direct_reuse"]
    assert request_cap({}, receipt["sample_ids"]) == 105
    assert request_cap({}) == 140
    assert request_cap(dict.fromkeys(IDS)) == 120


def test_twenty_new_questions_exclude_additional_old_questions_and_images():
    document = public_catalog()
    document["questions"][102]["image_keys"] = ["image-101"]
    receipt = from_bindings(document, 20, seed="omni-fresh20:20261104:137",
                            excluded_ids=[*IDS, "101"])
    assert receipt["schema"] == "omni-fresh-public-selection/2"
    assert len(receipt["sample_ids"]) == 20 and not {"101", "102"}.intersection(receipt["sample_ids"])
    assert "image-101" in receipt["excluded_image_keys"] and request_cap({}, receipt["sample_ids"]) == 140
    document["questions"].reverse()
    assert from_bindings(document, 20, seed=receipt["seed"],
                         excluded_ids=receipt["excluded_sample_ids"]) == receipt


def test_configurable_selection_revalidates_public_bindings_and_rejects_tampering(tmp_path, monkeypatch):
    from Ominirunner import selection
    document = public_catalog()
    write(tmp_path / "bindings.json", document)
    monkeypatch.setattr(selection, "sha", lambda p: selection.BINDINGS_SHA)
    receipt = from_bindings(document, 20, seed="omni-fresh20:20261104:137", excluded_ids=[*IDS, "101"])
    assert validate(tmp_path, receipt) == tuple(receipt["sample_ids"])
    changed = deepcopy(receipt)
    changed["sample_ids"].reverse()
    with pytest.raises(ValueError, match="frozen_selection_changed"):
        validate(tmp_path, changed)
    document["questions"][0]["answer"] = "not a public input"
    with pytest.raises(ValueError, match="selection_public_fields_changed"):
        from_bindings(document, 20)


def test_sampling_rejects_label_fields_and_changed_receipt(monkeypatch):
    document = public_catalog()
    receipt = from_bindings(document)
    from Ominirunner import selection
    monkeypatch.setattr(selection, "choose", lambda root: receipt)
    assert validate(Path("/unused"), receipt) == tuple(receipt["sample_ids"])
    changed = deepcopy(receipt)
    changed["sample_ids"].reverse()
    with pytest.raises(ValueError, match="frozen_selection_changed"):
        validate(Path("/unused"), changed)
    document["questions"][0]["answer"] = "never read by sampler"
    with pytest.raises(ValueError, match="selection_public_fields_changed"):
        from_bindings(document)
    with pytest.raises(ValueError, match="reuse_outside_selection"):
        request_cap({"unselected": {}}, receipt["sample_ids"])


@pytest.mark.parametrize("ids", [[], ["1", "1"], [1], ["x"], [True]])
def test_invalid_selected_ids(ids):
    with pytest.raises(ValueError, match="invalid_selected_questions"):
        selected_ids({"sample_ids": ids})


def test_audit_uses_thirty_episode_denominator(tmp_path, monkeypatch):
    ids = [str(i) for i in range(100, 115)]
    frozen = {"schema": VERSION, "config": Config().public(), "source": {},
              "max_requests": 105, "sample_ids": ids}
    write(tmp_path / "frozen.json", frozen)
    write(tmp_path / "inputs.json", [{"sample_id": i} for i in ids])
    monkeypatch.setattr(audit, "validate_source", lambda source: None)
    write(tmp_path / "completed.json", {"status": "complete", "episodes": 29})
    with pytest.raises(ValueError, match="trial_incomplete"):
        audit.run(tmp_path)
    write(tmp_path / "completed.json", {"status": "complete", "episodes": 30,
                                      "run_identity": "intentionally wrong"})
    with pytest.raises(ValueError, match="trial_input_identity_changed"):
        audit.run(tmp_path)


def test_fresh_report_has_correct_sampling_description(tmp_path):
    # Avoid historical IDs when exercising the empty subgroup case.
    ids = [str(i) for i in range(120, 135)]
    frozen = {"config": Config().public(), "max_episode_workers": 30,
              "selection": {"eligible_questions": 400, "seed": "fixed-seed",
                      "task_types": {"int": 15}, "excluded_sample_ids": []}}
    identity = digest(frozen)
    entries, scored = [], []
    table = {}
    for arm in ("direct", "skill"):
        table[arm] = {"metrics": {"int": {"n": 15, "value": 1}}, "legal": 15,
                      "requests": 15, "usage": {"total_tokens": 30, "prompt_tokens": 15,
                                              "completion_tokens": 15},
                      "sample_weighted_score": 1, "statuses": {"answered": 15},
                      "tool_calls": 0, "mean_episode_seconds": 1}
        for sample_id in ids:
            entries.append({"arm": arm, "status": "answered", "measurements": 0,
                            "valid_measurements": 0, "max_text_chars": 100,
                            "thinking_content_requests": 1, "round_statuses": {"answered": 1},
                            "surface_review_decisions": {}, "pending_surface_reviews": 0,
                            "answer_basis": "visual", "verified_arithmetic": False})
            scored.append({"sample_id": sample_id, "arm": arm, "score": 1,
                           "group": "int", "prediction": "1", "requests": 1, "tool_calls": 0})
            directory = tmp_path / "episodes" / (sample_id + "-" + arm)
            directory.mkdir(parents=True)
            write(directory / "result.json", {"rounds": [{"usage": {"completion_tokens": 1}}]})
    write(tmp_path / "frozen.json", frozen)
    write(tmp_path / "inputs.json", [{"sample_id": i} for i in ids])
    write(tmp_path / "scores.json", {"run_identity": identity, "pairs": 15,
                                   "table": table, "scored": scored})
    write(tmp_path / "audit.json", {
        "status": "passed", "run_identity": identity, "at": 0, "source_pins": 1,
        "charged_requests": 30, "episodes": entries,
        "budget_after": {"used": 30, "global_cap": 20000, "mirrored_request_copies_excluded": 0},
    })
    report.report(tmp_path)
    text = (tmp_path / "report.md").read_text()
    assert "15 题、30 个 episode" in text and "fixed-seed" in text
    assert "重新抽取的 15 题" in text and "## 原 12 题与新增 8 题" not in text
