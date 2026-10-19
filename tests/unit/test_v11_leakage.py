"""Candidate leakage checks distinguish generic methodology from sample answers."""

from skill3d.evolution.leakage import leakage_scan_text


def test_generic_ground_truth_warning_is_not_answer_leakage():
    text = (
        "The estimate is a partial-surface proxy and must not be described as "
        "official 3D ground truth."
    )
    assert leakage_scan_text(text) == []


def test_explicit_answer_assignments_are_rejected():
    for text in (
        "ground truth: B",
        "reference_answer = chair",
        "answer=A",
        "correct option: D",
    ):
        assert leakage_scan_text(text)


def test_sample_identity_remains_rejected():
    assert "sample_id:qa-secret-0042" in leakage_scan_text(
        "Handle qa-secret-0042 specially.",
        forbidden_sample_ids={"qa-secret-0042"},
    )
