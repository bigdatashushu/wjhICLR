"""Analysis coverage, copy protection and isolated evidence projection."""
import json
import sys
from types import ModuleType

import pytest

from skill3d.evolution.trajectory_analysis_v11 import (
    ANALYSIS_PROTOCOL, compact_batch, project_printed_objects, validate_analysis,
)
from skill3d.evolution.revision_edits_v11 import EditProtocolError


def analysis():
    return {"protocol":ANALYSIS_PROTOCOL,"batch_sha256":"batch",
            "records":[{"case_index":0,"round_indices":[0,1],
                "observed_facts":"The program retried after a missing argument.",
                "failure_hypothesis":"An unchecked argument may explain the exception.",
                "supported_skill_change":"Check required arguments before calling a tool.",
                "uncertainty":"This does not establish a perception failure."}]}


def test_all_rounds_required_and_fabricated_references_rejected():
    kwargs=dict(expected_cases={0:{0,1}},batch_sha256="batch",cases=[],forbidden_ids=set())
    assert validate_analysis(analysis(),**kwargs)["records"][0]["round_indices"] == [0,1]
    for rounds in ([0],[0,1,2]):
        value=analysis();value["records"][0]["round_indices"]=rounds
        with pytest.raises(EditProtocolError,match="round_reference"):
            validate_analysis(value,**kwargs)
    value=analysis();value["records"][0].pop("round_indices")
    assert validate_analysis(value,**kwargs)["records"][0]["round_indices"] == [0,1]


def test_duplicate_cases_and_question_copy_never_become_analysis():
    kwargs=dict(expected_cases={0:{0,1}},batch_sha256="batch",cases=[],forbidden_ids=set())
    value=analysis();value["records"]*=2
    with pytest.raises(EditProtocolError,match="case_coverage"):
        validate_analysis(value,**kwargs)
    value=analysis()
    question="Which objects have the greatest distance from the doorway?"
    value["records"][0]["observed_facts"]=question
    with pytest.raises(EditProtocolError,match="sample_copy"):
        validate_analysis(value,**{**kwargs,"cases":[{"question_text":question}]})


def test_repair_codes_distinguish_size_answer_syntax_and_identifiers():
    kwargs=dict(expected_cases={0:{0,1}},batch_sha256="batch",cases=[],forbidden_ids=set())
    with pytest.raises(EditProtocolError,match="text_budget"):
        validate_analysis(analysis(),**kwargs,max_chars=10)
    value=analysis(); value["records"][0]["observed_facts"]="The code emitted answer = 3."
    with pytest.raises(EditProtocolError,match="answer_assignment"):
        validate_analysis(value,**kwargs)
    value=analysis(); value["records"][0]["observed_facts"]="The program repeated work on sample_abcdef."
    with pytest.raises(EditProtocolError,match="sample_identifier"):
        validate_analysis(value,**kwargs)


def test_projection_retains_values_code_and_authorization_semantics(monkeypatch):
    graph=ModuleType("skill3d.evolution.evidence_graph_v11")
    graph.canonical_text=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"))
    graph.decode_evidence_graph=lambda v:v["value"]
    graph.encode_evidence_graph=lambda v:{"value":v}
    monkeypatch.setitem(sys.modules,graph.__name__,graph)
    row={"index":0,"program_source":"answer = measured_value",
         "results":[{"payload":{"distance":2.7,"unit":"m"},"authorization":{"allowed":False}}],
         "observation_reads":[{"data":{"distance":2.7},"coverage_complete":False}],
         "source_ref":"/audit/file"}
    value={"source_bundle_sha256":"source","bundle_metadata":{"source_split":"validation",
           "label_access":True},"case_contexts":{"0":{"case_index":0,"source_case_sha256":"hash",
           "case_header":{"source_split":"validation","learning_authorization":{
               "authorized":True,"manifest_identity":{"path":"/host"}}}}},
           "case_components":[{"case_context_ref":"0","rounds":[row]}]}
    text="Graph instructions\n"+graph.canonical_text(graph.encode_evidence_graph(value))
    result,receipt=compact_batch(text)
    rebuilt=graph.decode_evidence_graph(json.loads(result.split("\n",1)[1]))
    kept=rebuilt["case_components"][0]["rounds"][0]
    assert kept["results"]==row["results"]
    assert kept["observation_reads"]==row["observation_reads"]
    assert kept["program_source"]==row["program_source"]
    assert rebuilt["case_contexts"]["0"]["case_header"]["source_split"]=="validation"
    assert rebuilt["case_contexts"]["0"]["case_header"]["learning_authorization"]=={"authorized":True}
    assert "source_ref" not in kept and receipt["all_retained_decision_values_preserved"]


def test_labeled_stdout_objects_drop_only_native_audit_fields(monkeypatch):
    native = ModuleType("skill3d.evolution.revision_view_v11")
    native._candidate_facts = lambda obj: {
        key: obj[key] for key in ("obj_id", "bbox") if key in obj}
    monkeypatch.setitem(sys.modules, native.__name__, native)
    obj = {"obj_id": "a", "bbox": [0, 0, 10, 10], "distance": 2.7,
           "allowed": False, "binding_provenance": {"audit": "x" * 1000},
           "mask_per_frame": {"0": "/storage/a"}}
    text = "objects " + repr([obj, obj]) + "\n" + repr(obj) + "\nunknown = None\n"
    value, receipt = project_printed_objects(text)
    assert receipt == {"projected_lines": 2, "projected_objects": 3}
    kept = value["lines"][0]["object_values"][0]
    assert kept == {"obj_id": "a", "bbox": [0, 0, 10, 10], "distance": 2.7,
                    "allowed": False}
    assert value["lines"][0]["prefix"] == "objects "
    assert value["lines"][0]["suffix"] == "\n"
    assert value["lines"][2] == {"verbatim": "unknown = None\n"}
    # An unknown printed value is neither normalized nor evaluated.
    ambiguous = "object = {'obj_id': 'a', 'value': calculate()}\n"
    assert project_printed_objects(ambiguous)[0] == ambiguous
