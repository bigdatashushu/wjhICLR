"""Keep original missing geometry; never invent successful parser evidence."""
import copy
import importlib
from pathlib import Path
import sys

import pytest
from skill3d.segmentation.cached_presemantic import CachedPresemantic


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pilot = importlib.import_module("pilot40")


def source_binding():
    identity = {"question": {"qa_id": "synthetic"}, "frames": ["frozen"]}
    return pilot.runner.PreparedObjectBinding(
        stats={
            "object_parse": {"status": "unavailable", "objects": []},
            "generation": {"identity": identity,
                           "sha256": pilot.content_sha256(identity)},
            "candidate_receipts": [], "detection_receipts": [],
            "materialized": False,
        },
        notes=["original object evidence unavailable"],
    )


def test_preserves_source_missing_geometry_and_zero_requests():
    original = source_binding()
    preserved = copy.deepcopy(original)
    binding, receipt = pilot.reuse_unavailable_source_parser(original, {"key": "sealed"})
    assert binding == original == preserved
    assert binding is not original and binding.stats is not original.stats
    assert not binding.materialized and not binding.objects
    assert receipt["source_generation_sha256"] == original.stats["generation"]["sha256"]
    assert receipt["reference_answer_visible"] is False
    assert all(value == 0 for key, value in receipt.items() if key.startswith("new_"))


@pytest.mark.parametrize("field,value", [
    ("candidate_receipts", [{"candidate_id": "existing"}]),
    ("detection_receipts", [{"frame_slot": 0}]),
    ("review_selection", {"selected_candidate_ids": ["existing"]}),
    ("binding_error", "existing error"),
    ("detector_fault", True),
])
def test_refuses_conflicting_source_evidence(field, value):
    original = source_binding()
    original.stats[field] = value
    with pytest.raises(ValueError, match="contradicts"):
        pilot.reuse_unavailable_source_parser(original, {"key": "sealed"})


def test_refuses_materialized_source():
    original = source_binding()
    original.materialized = True
    with pytest.raises(ValueError, match="contradicts"):
        pilot.reuse_unavailable_source_parser(original, {"key": "sealed"})


def test_refuses_changed_generation_identity():
    original = source_binding()
    original.stats["generation"]["identity"]["frames"] = ["changed"]
    with pytest.raises(ValueError, match="digest differs"):
        pilot.reuse_unavailable_source_parser(original, {"key": "sealed"})


@pytest.mark.parametrize("status", ["parsed", "failed", None])
def test_leaves_other_source_statuses_on_existing_path(status):
    original = source_binding()
    original.stats["object_parse"]["status"] = status
    assert pilot.reuse_unavailable_source_parser(original, {"key": "sealed"}) is None


def test_unavailable_parser_replays_without_changing_status_or_source():
    original = source_binding()
    before = copy.deepcopy(original)
    cache = CachedPresemantic(original, {}, allow_unavailable_parser=True)
    objects, receipt = cache.parser(original.stats["generation"]["identity"]["question"])
    assert objects == []
    assert receipt["status"] == "unavailable"
    assert receipt["cache_hit"] and receipt["source_unavailable_preserved"]
    assert cache.receipt["new_parser_requests"] == 0
    assert original == before


def test_unavailable_parser_remains_opt_in_and_rejects_question_changes():
    original = source_binding()
    with pytest.raises(ValueError, match="not successful"):
        CachedPresemantic(original, {}).parser({})
    cache = CachedPresemantic(original, {}, allow_unavailable_parser=True)
    with pytest.raises(ValueError, match="question differs"):
        cache.parser({"question": "different"})


def test_unavailable_replay_rejects_published_candidates():
    original = source_binding()
    original.stats["candidate_receipts"] = [{"candidate_id": "published"}]
    with pytest.raises(ValueError, match="contradicts"):
        CachedPresemantic(original, {}, allow_unavailable_parser=True)
