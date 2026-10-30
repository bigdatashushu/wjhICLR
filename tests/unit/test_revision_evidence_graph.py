import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skill3d.evolution.evidence_graph_v11 import (
    canonical_text, decode_evidence_graph, encode_evidence_graph,
)
from skill3d.evolution.fitset_revision_v11 import (
    FitSetSlotReviser, RevisionOperationalError, load_active_parent_slots,
)
from skill3d.evolution.revision_view_v11 import (
    decision_bundle_view, pack_case_components, unpack_case_components,
    reassemble_round_components,
)
from skill3d.synthesis.request_audit import payload_bytes

LIBRARY = Path(__file__).resolve().parents[2] / "skill_library"


def test_full_programs_values_order_and_reserved_payload_shapes_roundtrip():
    row = {"program_source": "def solve(ctx,tools):\n    return -12.345678912345\n",
           "result": {"$record": "a real raw result field"}, "observed": [3, 1],
           "path": "/long/recorded/source/path/with/nested/files/frame00001.jpg",
           "values": [None, True, 0, -0.0, 1.23456789123456]}
    source = {"rounds": [copy.deepcopy(row) for _ in range(20)], "success": True}
    graph = encode_evidence_graph(source)
    assert decode_evidence_graph(graph) == source
    assert len(canonical_text(graph)) < len(canonical_text(source)) / 2
    graph["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source identity changed"):
        decode_evidence_graph(graph)


def reviser(tmp_path, client, **kwargs):
    return FitSetSlotReviser(
        output_root=tmp_path, panel={}, panel_name="synthetic", variants=[],
        accumulation_root=tmp_path, label_metadata={}, library_root=LIBRARY,
        client=client, **kwargs)


class NoChange:
    calls = 0
    def chat(self, *args, **kwargs):
        self.calls += 1
        return json.dumps({"action": "no_change", "modification_reason": "Synthetic check."})


def test_whole_request_budget_includes_utf8_json_escaping_and_fails_before_intent(tmp_path):
    client = NoChange()
    obj = reviser(tmp_path, client, input_max_bytes=4096)
    with pytest.raises(RevisionOperationalError, match="byte budget"):
        obj._request(slot=("object_counting", "g0m0"), attempt=1,
                     prompt="\"你好\n" * 1024)
    assert client.calls == 0
    assert not list(tmp_path.rglob("request_intent*"))


def test_batching_keeps_successes_and_complete_case_values(tmp_path):
    parent = load_active_parent_slots(LIBRARY)[0][("object_counting", "g0m0")]
    cases = [{"case_id": f"synthetic-{i}", "outcome": "success" if i % 2 else "failure",
              "rounds": [{"program_source": f"def solve(ctx, tools):\n    return {i}\n",
                         "observation_reads": [
                             {"data": {"value": str(i) + str(j) + ("xyz" * 1200)}}
                             for j in range(4)]}]}
             for i in range(12)]
    source = {"cases": cases, "source_split": "induction", "label_access": True}
    obj = reviser(tmp_path, None, input_max_bytes=110000, evidence_request_budget=17)
    batches, plan = obj._evidence_batches(
        parent=parent, feedback={}, experience=SimpleNamespace(
            model_dump=lambda **kwargs: source), diagnosis=None)
    assert plan["all_retained_values_reconstructed"] and plan["case_count"] == 12
    values = [decode_evidence_graph(json.loads(text.split("\n", 1)[1]))
              for text in batches]
    rebuilt = [row["complete_case"] for value in values for row in unpack_case_components(value)]
    assert rebuilt == decision_bundle_view(source)["cases"]


def test_one_unsplittable_program_is_explicit_error_before_any_model_call(tmp_path):
    parent = load_active_parent_slots(LIBRARY)[0][("object_counting", "g0m0")]
    obj = reviser(tmp_path, NoChange(), input_max_bytes=100000, evidence_request_budget=17)
    source = {"cases": [{"case_id": "synthetic", "program_source": "x" * 180000}]}
    with pytest.raises(RevisionOperationalError, match="component exceeds"):
        obj._evidence_batches(
            parent=parent, feedback={}, experience=SimpleNamespace(
                model_dump=lambda **kwargs: source), diagnosis=None)
    assert obj.client.calls == 0


def test_shared_case_header_and_external_read_are_self_contained():
    page = {"data": [{"obj_id": "chair-1", "confidence": 0.987654321}],
            "coverage_complete": False, "next_cursor": 1}
    header = {"case_id": "case-1", "question_text": "Synthetic question", "reference_answer": "A"}
    rounds = [
        {"program_source": "YieldObservations()", "observation_reads": [page]},
        {"program_source": "answer = 'A'",
         "delivered_observation_pages": [{"recorded_read_ref": [0, 0]}]},
        {"program_source": "ReturnAnswer(answer)"},
    ]
    rows = [{"case_index": 0, "case_id": "case-1", "source_case_sha256": "a" * 64,
             "view_case_sha256": "b" * 64, "complete_case": {**header, "rounds": [rounds[i]]},
             "round_range": [i, i + 1], "total_rounds": 3, "case_part": True}
            for i in (1, 2)]
    packed = pack_case_components(rows, [{**header, "rounds": rounds}])
    assert len(packed["case_contexts"]) == 1
    assert all("complete_case" not in row for row in packed["case_components"])
    assert packed["case_contexts"]["0"]["referenced_reads"] == {"0:0": page}
    assert unpack_case_components(decode_evidence_graph(encode_evidence_graph(packed))) == rows


def test_split_rounds_keep_programs_values_order_and_read_context_inside_budget(tmp_path):
    parent = load_active_parent_slots(LIBRARY)[0][("object_counting", "g0m0")]
    pages = [{"result_id": f"result-{i}", "data": {"value": f"{i}:" + "x" * 14000}}
             for i in range(6)]
    source = {"cases": [{
        "case_id": "synthetic-long-case", "question_text": "Synthetic question",
        "reference_answer": "A", "score": 0.0, "rounds": [{
            "index": i, "program_source": f"page = ReadObservation('result-{i}')",
            "observation_reads": [page],
            "feedback": ("Requested observation pages (raw data, original references):\n"
                         + json.dumps([pages[i - 1]]) if i else ""),
        } for i, page in enumerate(pages)],
    }]}
    before = copy.deepcopy(source)
    obj = reviser(tmp_path, NoChange(), evidence_request_budget=17)
    batches, plan = obj._evidence_batches(
        parent=parent, feedback={}, experience=SimpleNamespace(
            model_dump=lambda **kwargs: source), diagnosis=None)
    assert len(batches) > 1 and plan["all_retained_values_reconstructed"]
    rebuilt = []
    for batch in batches:
        value = decode_evidence_graph(json.loads(batch.split("\n", 1)[1]))
        rows = unpack_case_components(value)
        for row in rows:
            assert row["round_range"][0] == len(rebuilt)
            rebuilt.extend(row["complete_case"]["rounds"])
            for record in row["complete_case"]["rounds"]:
                for delivered in record.get("delivered_observation_pages", []):
                    r, offset = delivered["recorded_read_ref"]
                    assert value["case_contexts"]["0"]["referenced_reads"][f"{r}:{offset}"]["data"] == pages[r]["data"]
        prompt = obj._prompt(
            parent=parent, feedback={}, static_feedback=[], evidence_text=batch,
            batch_context={"working_skill_md": "\n" * obj.method_context_max_chars,
                           "coverage": "reserved" * 128})
        messages = [{"role": "system", "content":
                     "You are an offline Skill method reviser. Output strict JSON only."},
                    {"role": "user", "content": prompt}]
        assert sum(len(message["content"]) for message in messages) + 1024 <= obj.input_max_chars
        assert len(payload_bytes({"messages": messages, "max_tokens": obj.max_tokens, "seed": 138,
                                  "temperature": 0.0, "enable_thinking": False})) + 2048 <= obj.input_max_bytes
    assert rebuilt == decision_bundle_view(source)["cases"][0]["rounds"]
    assert source == before and obj.client.calls == 0


@pytest.mark.parametrize("field", ["results", "observation_reads"])
def test_oversized_tool_or_read_array_splits_losslessly_and_rejects_missing_slices(tmp_path, field):
    parent = load_active_parent_slots(LIBRARY)[0][("object_counting", "g0m0")]
    array = [{"obj_id": f"synthetic-{i}", "value": str(i) + "x" * 5000,
              "confidence": 0.9876543210987} for i in range(12)]
    record = ({"source_tool": "synthetic_tool", "payload": array} if field == "results"
              else {"result_id": "synthetic-result", "data": array})
    source = {"cases": [{
        "case_id": "synthetic-large-array", "question_text": "Synthetic question",
        "reference_answer": "A", "rounds": [{
            "index": 0, "program_source": "value = 1\nprint(value)",
            "error_code": "synthetic-error", "execution_diagnostics": {"error_line": 1},
            field: [record],
        }]}]}
    before = copy.deepcopy(source)
    obj = reviser(tmp_path, NoChange(), evidence_request_budget=17, input_max_chars=45000)
    batches, plan = obj._evidence_batches(
        parent=parent, feedback={}, experience=SimpleNamespace(
            model_dump=lambda **kwargs: source), diagnosis=None)
    rows = [row for batch in batches for row in unpack_case_components(
        decode_evidence_graph(json.loads(batch.split("\n", 1)[1])))]
    assert len(batches) > 1 and all(row.get("round_fragment") for row in rows)
    assert all(row["complete_case"]["rounds"][0]["program_source"] == "value = 1\nprint(value)"
               for row in rows)
    combined = reassemble_round_components(rows)
    assert len(combined) == 1
    assert combined[0]["complete_case"] == decision_bundle_view(source)["cases"][0]
    assert plan["all_retained_values_reconstructed"] and source == before
    with pytest.raises(ValueError, match="incomplete"):
        reassemble_round_components(rows[:-1])
    with pytest.raises(ValueError, match="first slice"):
        reassemble_round_components(rows[1:])
    changed = copy.deepcopy(rows)
    changed[1]["complete_case"]["rounds"][0]["program_source"] += "\nvalue = 2"
    with pytest.raises(ValueError, match="context changed"):
        reassemble_round_components(changed)
    assert obj.client.calls == 0
