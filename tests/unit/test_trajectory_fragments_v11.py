"""Full multi-field reconstruction and rejection of missing/tampered slices."""
from copy import deepcopy

import pytest

from skill3d.evolution.trajectory_fragments_v11 import (
    canonical, reassemble_fields, restore_fields, split_fields,
)


def sample():
    return {"case_index": 0, "case_part": True, "round_range": [0, 1],
            "complete_case": {"question_text": "authorized learning input", "rounds": [{
                "index": 3, "program_source": "keep_the_entire_program()",
                "results": [{"distance": i / 10, "allowed": False, "note": "a" * 400}
                            for i in range(12)],
                "execution_diagnostics": {"stdout": "quoted ' text\n中文 " * 300},
                "observation_reads": [{"data": [1, None, False], "note": "b" * 400}
                                      for _ in range(10)]}]}}


def test_several_large_fields_keep_program_and_all_recorded_values():
    row = sample()
    fits = lambda rows: len(canonical(rows)) < 3000
    assert not fits([row])
    pieces = split_fields(row, fits)
    assert len(pieces[0]["round_fragment"]["fields"]) >= 2
    assert all(fits([piece]) for piece in pieces)
    assert all(piece["complete_case"]["rounds"][0]["program_source"] ==
               row["complete_case"]["rounds"][0]["program_source"] for piece in pieces)
    assert restore_fields(pieces) == row
    before = {"native": "unfragmented"}
    assert reassemble_fields([before, *pieces, before], lambda rows: rows) == [before, row, before]


def test_missing_reordered_or_modified_fragments_never_pass_reconstruction():
    pieces = split_fields(sample(), lambda rows: len(canonical(rows)) < 3000)
    with pytest.raises(ValueError):
        restore_fields(pieces[:-1])
    with pytest.raises(ValueError):
        restore_fields(list(reversed(pieces)))
    changed = deepcopy(pieces)
    changed[0]["round_fragment"]["source_unit_sha256"] = "wrong"
    with pytest.raises(ValueError):
        restore_fields(changed)
    changed = deepcopy(pieces)
    changed[0]["complete_case"]["rounds"][0]["program_source"] = "changed_program()"
    with pytest.raises(ValueError):
        restore_fields(changed)


def test_oversized_program_refuses_instead_of_truncating():
    row = sample()
    row["complete_case"]["rounds"][0]["program_source"] = "x" * 10000
    with pytest.raises(ValueError):
        split_fields(row, lambda rows: len(canonical(rows)) < 3000)


def test_large_nested_string_splits_inside_diagnostics_and_retains_error_facts():
    row = sample()
    row["complete_case"]["rounds"][0]["execution_diagnostics"] = {
        "stdout": "unparsed output '中文'\n" * 1200,
        "error_type": "TypeError", "answer_untrusted": False, "error_line": None}
    pieces = split_fields(row, lambda rows: len(canonical(rows)) < 3000)
    assert restore_fields(pieces) == row
    assert any(field["path"][-1] == "stdout"
               for field in pieces[0]["round_fragment"]["fields"])
    assert all(piece["complete_case"]["rounds"][0]["execution_diagnostics"]["error_type"]
               == "TypeError" for piece in pieces)


def test_medium_array_items_can_be_fragments_without_repeating_all_small_fields():
    row = sample()
    row["complete_case"]["rounds"][0]["results"] = [
        {"payload": "a" * 9000}, *({"payload": "b" * 800} for _ in range(20))]
    pieces = split_fields(row, lambda rows: len(canonical(rows)) < 18000)
    assert restore_fields(pieces) == row
