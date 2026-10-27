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
from skill3d.evolution.revision_view_v11 import decision_bundle_view
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
    rebuilt = [row["complete_case"] for value in values for row in value["case_components"]]
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
