"""Offline lossless evidence-table tests; no model or network calls."""
from copy import deepcopy
import json

import pytest

from MindcubeRunner.common import digest, encoded
from MindcubeRunner.observation_context import (
    SCHEMA, build_observation_context, restore_observations,
)


COORDINATES = "original image pixels: x right, y down; views remain independent"


def stability(*, reliable=True):
    return {
        "version": "mindcube-pixel-stability/1",
        "reliable": reliable,
        "reason_code": None if reliable else "edge_sampling",
        "jitter_pixels": 3,
        "probe_count": 9,
        "failed_probes": 0,
        "relative_depth_range": 0.037484645843506 if reliable else 0.231234567890123,
        "max_relative_depth_range": 0.10,
    }


def point(result_id="point-0", *, view_index=0, pixel_xy=None, depth=3.287484645843506):
    return {
        "result_id": result_id, "source_tool": "point_depth", "status": "ok",
        "data": {
            "view_index": view_index,
            "pixel_xy": [100.125, 200.875] if pixel_xy is None else pixel_xy,
            "depth": depth, "n_valid": 25, "unit": "per_view_relative",
            "definition": "local median of predicted camera-depth, radius 2 pixels",
            "coordinate_convention": COORDINATES, "status": "ok",
            "sample_stability": {"point": stability()},
        },
    }


def real_observations():
    pair = {
        "result_id": "pair-0", "source_tool": "pair_point_distance", "status": "ok",
        "data": {
            "view_index": 1, "a_xy": [11.125, 30.875], "b_xy": [210.25, 97],
            "distance": 1.2345678901234567, "unit": "per_view_relative",
            "definition": "Euclidean distance of local median camera points",
            "coordinate_convention": COORDINATES, "status": "ok",
            "sample_stability": {"a": stability(), "b": stability()},
        },
    }
    unreliable_point = point("point-edge")
    unreliable_point["data"].pop("depth")
    unreliable_point["data"].update(
        status="unreliable", reason_code="edge_sampling",
        reason="Nearby pixels vary sharply in depth. Resample an interior surface.",
        sample_stability={"point": stability(reliable=False)},
    )
    unreliable_pair = deepcopy(pair)
    unreliable_pair["result_id"] = "pair-edge"
    unreliable_pair["data"].pop("distance")
    unreliable_pair["data"].update(
        status="unreliable", reason_code="edge_sampling",
        reason="One endpoint crosses a depth boundary.",
        sample_stability={"a": stability(), "b": stability(reliable=False)},
        unreliable_endpoints=["b"],
    )
    candidates = {
        "result_id": "candidates-0", "source_tool": "view_candidates", "status": "ok",
        "data": {
            "candidates": [
                {"candidate_id": "region-2", "view_index": 1, "category": "chair",
                 "box_xyxy": [7.125, 2.5, 83.75, 109.875],
                 "category_confirmed": False, "mask_quality_assessed": False,
                 "geometry_usable": None, "review_status": "unreviewed"},
                {"candidate_id": "region-1", "view_index": 0, "category": "table",
                 "box_xyxy": [10, 20, 91, 111], "category_confirmed": True,
                 "review_status": "category_reviewed"},
            ],
            "count": 2, "coverage": "detections only; not exhaustive",
            "coordinate_convention": COORDINATES,
        },
    }
    summary = {
        "result_id": "summary-0", "source_tool": "view_depth_summary", "status": "ok",
        "data": {
            "views": [
                {"view_index": 0, "available": True, "n_valid": 30251,
                 "quantiles": [0.12345678901234568, 2.345678901234567, 6.789012345678901]},
                {"view_index": 1, "available": False},
                {"view_index": 2, "available": True, "n_valid": 0, "quantiles": []},
            ],
            "unit": "per_view_relative", "coordinate_convention": COORDINATES,
        },
    }
    return [
        point(), candidates, pair, unreliable_point, summary,
        point("point-1", pixel_xy=[22.75, 12.125], depth=2.718281828459045),
        unreliable_pair,
    ]


def assert_roundtrip(records, **kwargs):
    before = encoded(records)
    packet, receipt = build_observation_context(records, **kwargs)
    wire_packet = json.loads(encoded(packet))
    restored = restore_observations(wire_packet)
    assert restored == records
    assert encoded(restored) == before
    assert receipt["raw_sha256"] == digest(records) == digest(restored)
    assert receipt["compact_sha256"] == digest(packet) == digest(wire_packet)
    assert receipt["record_count"] == len(records)
    assert receipt["pending_count"] + receipt["history_count"] == len(records)
    assert encoded(records) == before
    return packet, receipt


def test_real_style_roundtrip_pending_history_and_nested_evidence():
    records = real_observations()
    packet, receipt = assert_roundtrip(
        records, review_ids=["pair-edge", "point-edge", "point-edge"],
        hint_ids=["summary-0", "candidates-0"],
    )
    assert packet["schema"] == SCHEMA
    assert set(packet) == {"schema", "pending", "history"}
    assert receipt["record_count"] == 7
    assert receipt["pending_count"] == 4
    assert receipt["history_count"] == 3
    assert receipt["pending_ids"] == ["candidates-0", "point-edge", "summary-0", "pair-edge"]
    assert receipt["history_ids"] == ["point-0", "pair-0", "point-1"]
    assert receipt["review_ids"] == ["point-edge", "pair-edge"]
    assert receipt["hint_ids"] == ["candidates-0", "summary-0"]
    for record in records:
        assert encoded(packet).count(encoded(record["result_id"])) == 1
    restored = restore_observations(packet)
    assert restored[3]["status"] == "ok"
    assert restored[3]["data"]["status"] == "unreliable"
    assert "depth" not in restored[3]["data"]
    assert "distance" not in restored[6]["data"]
    assert restored[6]["data"]["unreliable_endpoints"] == ["b"]
    assert [row["candidate_id"] for row in restored[1]["data"]["candidates"]] == [
        "region-2", "region-1"]


def test_tables_lift_only_identical_data_and_keep_global_indices():
    records = [
        point("a"),
        point("b", view_index=1),
        point("c", pixel_xy=[8.25, 9.5], depth=2.2222222222222223),
        point("d"),
    ]
    packet, _ = assert_roundtrip(records, review_ids=["d"])
    assert len(packet["history"]) == 2
    group = packet["history"][0]
    assert group["source_tool"] == "point_depth" and group["status"] == "ok"
    assert group["shared_data"] == {
        key: value for key, value in records[0]["data"].items()
        if key not in {"depth", "pixel_xy"}
    }
    assert group["data_keys"] == ["depth", "pixel_xy"]
    assert group["rows"] == [
        [0, "a", 3.287484645843506, [100.125, 200.875]],
        [2, "c", 2.2222222222222223, [8.25, 9.5]],
    ]
    assert packet["history"][1]["shared_data"]["view_index"] == 1
    assert packet["pending"][0]["rows"] == [[3, "d"]]


def test_absent_and_none_fields_have_different_schemas():
    absent, null, other = point("absent"), point("null"), point("other")
    null["data"]["reason_code"] = None
    other["data"]["reason_code"] = None
    packet, _ = assert_roundtrip([absent, null, other])
    assert len(packet["history"]) == 2
    assert "reason_code" not in packet["history"][0]["shared_data"]
    assert packet["history"][1]["shared_data"]["reason_code"] is None
    restored = restore_observations(packet)
    assert "reason_code" not in restored[0]["data"]
    assert restored[1]["data"]["reason_code"] is None


@pytest.mark.parametrize("values", [
    [True, 1, 1.0, None],
    [False, 0, 0.0, -0.0],
    [{"probe": [True]}, {"probe": [1]}, {"probe": [1.0]}, {"probe": [None]}],
])
def test_json_canonical_equality_preserves_unequal_types(values):
    records = [point(str(index)) for index in range(len(values))]
    for record, value in zip(records, values):
        record["data"]["extra"] = value
    packet, _ = assert_roundtrip(records)
    group = packet["history"][0]
    assert "extra" in group["data_keys"] and "extra" not in group["shared_data"]
    restored = restore_observations(packet)
    assert [encoded(record["data"]["extra"]) for record in restored] == [
        encoded(value) for value in values]


def test_nested_object_key_order_is_canonically_equal():
    records = [point("a"), point("b")]
    records[0]["data"]["extra"] = {"first": [1, True, None], "second": {"x": 1.0}}
    records[1]["data"]["extra"] = {"second": {"x": 1.0}, "first": [1, True, None]}
    packet, _ = assert_roundtrip(records)
    assert "extra" in packet["history"][0]["shared_data"]


def test_differing_nested_candidates_summaries_and_future_fields_are_preserved():
    original = real_observations()
    candidates, summary = deepcopy(original[1]), deepcopy(original[4])
    candidates["result_id"] = "candidates-1"
    candidates["data"]["candidates"].reverse()
    candidates["data"]["candidates"][0]["geometry_usable"] = False
    candidates["data"]["candidates"][0]["future_review"] = {
        "pixel_endpoints": [[1.125, 2.875], [19, 27.5]],
        "samples": [True, 1, 1.0, None, -0.0, 10**40],
    }
    summary["result_id"] = "summary-1"
    summary["data"]["views"][0]["quantiles"][1] = 2.345678901234568
    summary["data"]["views"][1]["n_valid"] = None
    records = [original[1], original[4], candidates, summary]
    packet, _ = assert_roundtrip(records)
    assert len(packet["history"]) == 2
    assert packet["history"][0]["data_keys"] == ["candidates"]
    assert packet["history"][1]["data_keys"] == ["views"]
    assert packet["history"][0]["rows"][1][0:2] == [2, "candidates-1"]
    assert packet["history"][1]["rows"][1][0:2] == [3, "summary-1"]


def test_grouping_distinguishes_outer_status_view_and_exact_schema():
    records = [point(str(index)) for index in range(8)]
    records[1]["status"] = "error"
    records[2]["data"]["view_index"] = 1
    records[3]["data"].pop("view_index")
    records[4]["data"]["view_index"] = None
    records[5]["data"]["view_index"] = True
    records[6]["data"]["view_index"] = 0.0
    records[7]["data"]["extra"] = None
    packet, _ = assert_roundtrip(records)
    assert len(packet["history"]) == 8


@pytest.mark.parametrize("kwargs, error", [
    ({"review_ids": ["missing"]}, "unknown_review_id"),
    ({"hint_ids": ["missing"]}, "unknown_hint_id"),
    ({"review_ids": ["candidates-0"]}, "invalid_review_observation_kind"),
    ({"review_ids": ["summary-0"]}, "invalid_review_observation_kind"),
    ({"hint_ids": ["point-0"]}, "invalid_hint_observation_kind"),
    ({"hint_ids": ["pair-edge"]}, "invalid_hint_observation_kind"),
    ({"review_ids": ["point-0"], "hint_ids": ["point-0"]}, "invalid_hint_observation_kind"),
    ({"review_ids": "point-0"}, "invalid_review_ids"),
    ({"hint_ids": [None]}, "invalid_hint_id"),
    ({"review_ids": None}, "invalid_review_ids"),
])
def test_pending_ids_are_known_and_kind_checked(kwargs, error):
    records = real_observations()
    before = encoded(records)
    with pytest.raises(ValueError, match=error):
        build_observation_context(records, **kwargs)
    assert encoded(records) == before


@pytest.mark.parametrize("change, error", [
    (lambda r: r.update(extra="not an observation field"), "exact_root_keys"),
    (lambda r: r.pop("status"), "exact_root_keys"),
    (lambda r: r.update(result_id=""), "invalid_result_id"),
    (lambda r: r.update(result_id="   "), "invalid_result_id"),
    (lambda r: r.update(result_id=1), "invalid_result_id"),
    (lambda r: r.update(source_tool="candidate_depth_summary"), "unknown_source_tool"),
    (lambda r: r.update(source_tool=[]), "unknown_source_tool"),
    (lambda r: r.update(status=None), "invalid_observation_status"),
    (lambda r: r.update(data=[]), "data_must_be_dict"),
])
def test_observation_contract_rejected(change, error):
    record = point()
    change(record)
    with pytest.raises(ValueError, match=error):
        build_observation_context([record])


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), float("-inf"), (1, 2), {1: "not a JSON key"}, {1, 2},
])
def test_only_finite_plain_json_is_accepted(bad):
    record = point()
    record["data"]["nested"] = {"value": [bad]}
    with pytest.raises(ValueError):
        build_observation_context([record])


def test_duplicate_ids_rejected_but_identical_data_is_not_deduplicated():
    with pytest.raises(ValueError, match="duplicate_result_id"):
        build_observation_context([point(), point()])
    packet, receipt = assert_roundtrip([point("a"), point("b"), point("c")])
    assert packet["history"][0]["rows"] == [[0, "a"], [1, "b"], [2, "c"]]
    assert receipt["record_count"] == 3


def test_empty_observations_and_single_empty_data_record():
    packet, receipt = assert_roundtrip([])
    assert packet == {"schema": SCHEMA, "pending": [], "history": []}
    assert receipt["record_count"] == receipt["pending_count"] == receipt["history_count"] == 0
    with pytest.raises(ValueError, match="unknown_review_id"):
        build_observation_context([], review_ids=["missing"])
    record = {"result_id": "empty", "source_tool": "view_candidates", "status": "ok", "data": {}}
    packet, _ = assert_roundtrip([record], hint_ids=["empty"])
    assert packet["history"] == []
    assert packet["pending"][0]["rows"] == [[0, "empty"]]


def test_deterministic_encoding_independent_of_key_and_pending_id_order():
    records = real_observations()
    packet, receipt = build_observation_context(
        records, review_ids=["point-0", "pair-0"], hint_ids=["candidates-0", "summary-0"])
    reordered = [
        {key: (dict(reversed(list(value.items()))) if key == "data" else value)
         for key, value in reversed(list(record.items()))} for record in records
    ]
    again, again_receipt = build_observation_context(
        iter(reordered), review_ids=iter(["pair-0", "point-0", "pair-0"]),
        hint_ids={"summary-0", "candidates-0"})
    assert encoded(packet) == encoded(again)
    assert encoded(receipt) == encoded(again_receipt)


@pytest.mark.parametrize("pending_mode", ["none", "sparse", "all"])
def test_more_than_1000_records_no_cap_truncation_rounding_or_deduplication(pending_mode):
    records = [
        point(f"point-{index}", view_index=index % 3,
              pixel_xy=[index + 0.125, index + 0.875],
              depth=3.287484645843506 + index / 10000)
        for index in range(1507)
    ]
    # Preserve records whose data are identical and records far beyond index 1000.
    records[1201]["data"] = deepcopy(records[1]["data"])
    review_ids = (
        [] if pending_mode == "none" else
        [record["result_id"] for record in records] if pending_mode == "all" else
        [records[index]["result_id"] for index in range(0, len(records), 7)]
    )
    packet, receipt = assert_roundtrip(records, review_ids=review_ids)
    assert receipt["record_count"] == 1507
    assert receipt["pending_count"] == len(review_ids)
    assert sum(len(group["rows"]) for part in ("pending", "history")
               for group in packet[part]) == 1507
    assert len(encoded(packet)) < len(encoded(records)) * 0.25
    assert restore_observations(packet)[-1] == records[-1]


def test_build_restore_and_returned_values_do_not_mutate_or_alias_inputs():
    records = [point("a"), point("b"), point("c", depth=9.876543210987654)]
    original = deepcopy(records)
    packet, _ = build_observation_context(records)
    packet_before = encoded(packet)
    restored = restore_observations(packet)
    restored[0]["data"]["sample_stability"]["point"]["probe_count"] = 99
    assert restored[1]["data"]["sample_stability"]["point"]["probe_count"] == 9
    restored[2]["data"]["pixel_xy"].append(123)
    assert encoded(packet) == packet_before
    assert records == original
    packet["history"][0]["shared_data"]["sample_stability"]["point"]["probe_count"] = 100
    assert records == original
    records[0]["data"]["sample_stability"]["point"]["probe_count"] = 101
    assert packet["history"][0]["shared_data"]["sample_stability"]["point"]["probe_count"] == 100


@pytest.mark.parametrize("change, error", [
    (lambda p: p.update(schema="unknown"), "invalid_observation_context_schema"),
    (lambda p: p.update(extra=True), "invalid_observation_context_schema"),
    (lambda p: p.update(pending={}), "groups_must_be_lists"),
    (lambda p: p["history"][0].update(extra=True), "invalid_observation_group"),
    (lambda p: p["history"][0].update(rows=[]), "invalid_observation_rows"),
    (lambda p: p["history"][0]["rows"][0].append("extra"), "invalid_observation_row"),
    (lambda p: p["history"][0]["rows"][0].__setitem__(0, True), "sequence_index"),
    (lambda p: p["history"][0]["rows"][0].__setitem__(0, -1), "sequence_index"),
    (lambda p: p["history"][0]["rows"][1].__setitem__(0, 0), "sequence_index"),
    (lambda p: p["history"][0]["rows"][1].__setitem__(0, 5), "noncontiguous"),
    (lambda p: p["history"][0]["rows"][1].__setitem__(1, "a"), "duplicate_result_id"),
    (lambda p: p["history"][0].update(data_keys=["depth", "depth"]), "data_columns"),
    (lambda p: p["history"][0].update(data_keys=["view_index"]), "data_columns"),
    (lambda p: p["history"][0]["shared_data"].update(depth=float("inf")), "invalid_json"),
])
def test_restore_rejects_malformed_tables(change, error):
    packet, _ = build_observation_context([point("a"), point("b")])
    change(packet)
    before = deepcopy(packet)
    with pytest.raises(ValueError, match=error):
        restore_observations(packet)
    assert packet == before
