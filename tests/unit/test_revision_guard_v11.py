"""Ordinary temporal prose passes; real identities and native modules remain protected."""
from types import FunctionType

from skill3d.evolution.leakage import leakage_scan_text as native_scan
from skill3d.evolution.revision_guard_v11 import leakage_scan_text, scoped_static_validator


def test_ordinary_verbs_are_not_sample_identifiers():
    for text in ("earliest sampled frame", "the questioned estimator"):
        assert "qa_id_pattern" in native_scan(text)
        assert leakage_scan_text(text) == []
        assert "qa_id_pattern" in native_scan(text)


def test_other_id_patterns_and_literal_forbidden_verbs_still_fail():
    for token in ("qa123", "qa_123", "qa-dead", "sample_0042", "sampled4",
                  "question-7", "questioned4", "episode_d", "SAMPLE_D"):
        assert "qa_id_pattern" in leakage_scan_text("Use sampled frames for " + token)
    for word in ("sampled", "questioned"):
        assert "sample_id:" + word in leakage_scan_text(word, {word})


def test_answer_and_literal_identity_checks_are_preserved():
    for text in ("ground truth: B", "reference_answer=chair", "answer=A",
                 "correct option: D"):
        original = [hit for hit in native_scan("Sampled frames. " + text)
                    if hit.startswith("answer_pattern:")]
        corrected = [hit for hit in leakage_scan_text("Sampled frames. " + text)
                     if hit.startswith("answer_pattern:")]
        assert original and corrected == original
    assert "sample_id:scene-private" in leakage_scan_text("scene-private", {"scene-private"})


def test_static_validator_uses_independent_namespace_and_preserves_keyword_defaults():
    def template(text, *, ids=None):
        return leakage_scan_text(text, forbidden_sample_ids=ids)

    namespace = {"leakage_scan_text": native_scan}
    native = FunctionType(template.__code__, namespace, "legacy")
    native.__kwdefaults__ = {"ids": None}
    scoped = scoped_static_validator(native)
    assert scoped.__globals__ is not native.__globals__
    assert native.__globals__["leakage_scan_text"] is native_scan
    assert "qa_id_pattern" in native("sampled frames")
    assert scoped("sampled frames") == []
    assert "sample_id:sampled" in scoped("sampled frames", ids={"sampled"})
    assert native.__kwdefaults__ == {"ids": None}


def test_editor_binds_guard_without_replacing_native_validator_globals():
    from skill3d.evolution.fitset_revision_v11 import _static_validate as native
    from skill3d.evolution.revision_edits_v11 import _static_validate as scoped

    assert native.__globals__["leakage_scan_text"] is native_scan
    assert scoped.__globals__["leakage_scan_text"] is leakage_scan_text
    assert native.__globals__ is not scoped.__globals__


def test_full_analysis_accepts_temporal_prose_with_original_case_binding():
    from skill3d.evolution.trajectory_analysis_v11 import ANALYSIS_PROTOCOL, validate_analysis

    value = {"protocol": ANALYSIS_PROTOCOL, "batch_sha256": "batch",
             "records": [{"case_index": 0,
                          "observed_facts": "Earliest sampled frame.",
                          "failure_hypothesis": "Sampling may hide earlier visibility.",
                          "supported_skill_change": "Preserve the upper bound.",
                          "uncertainty": "Within-slot order remains unknown."}]}
    result = validate_analysis(value, expected_cases={0: {1}}, batch_sha256="batch",
                               cases=(), forbidden_ids=set())
    assert result["records"][0]["round_indices"] == [1]
