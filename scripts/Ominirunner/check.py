"""Offline checks of the changed scalar, context, scope and review contracts."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import csv
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from skill3d.sandbox.ast_guard import ast_guard
from .common import ADDED_IDS, Config, IDS, PRIOR_IDS
from .native import NativeKernel, parse_json, parse_program, scalar
from .protocol import (OUTPUT, REVIEW_OUTPUT, EVIDENCE_OUTPUT, RAW_FORMAT,
                       crop_sheets, messages, overlay, packed, review_summary)
from .review import program_reviews, validate as validate_reviews
from .evidence import (ABS_TOL, REL_TOL, EvidenceError, dependency_ids, expression_value,
                       feedback as evidence_feedback, validate as validate_evidence)
from .score import ENGINE_SHA, THRESHOLDS, evaluate
from .tools import ALLOWED, build_registry, empty_counts
from .worker import solve


def fixture():
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:40, :50]
    depth = 1 + xx/50 + yy/40
    points = np.stack([xx*.01, yy*.01, depth], axis=-1)
    output = BytesIO()
    Image.fromarray(frame).save(output, format="JPEG")
    candidates = [
        {"candidate_id": name, "category": name, "view_index": 0,
         "box_xyxy": [1, 2, 20, 25], "category_confirmed": True}
        for name in ("chair", "cup", "lamp")]
    cache = SimpleNamespace(frames=(frame,), candidates=candidates,
                            geometries={0: {"depth": depth, "valid_mask": depth > 0,
                                            "points": points}},
                            provenance={"question": "SECRET_UNION_QUESTION"})
    item = {"question": "How many chairs are visible?", "sample_id": "15",
            "benchmark": "Omni3D-Bench", "source_split": "benchmark",
            "task_type": "int", "image_keys": ["image"], "input_sha256": "q",
            "options": [], "image_sha256": "unused"}
    return item, output.getvalue(), frame, cache


def score_parity(engine):
    from .common import sha
    if sha(engine) != ENGINE_SHA:
        raise ValueError("official_engine_changed")
    tree = ast.parse(Path(engine).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "write_summarized_results")
    namespace = {"csv": csv}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<official_summary>", "exec"), namespace)
    examples = [
        ("int", "3", "3.0"), ("int", "3", "3"), ("int", "3", "3e0"),
        ("str", "yes", "YES"), ("str", "no", "no "),
        ("str", "coffee table", "Coffee Table"), ("str", "coffee table", " coffee table"),
        ("float", "1", "5e-1"), ("float", "1", "0.95"), ("float", "2", "bad"),
    ] + [("float", "1", str(1+t)) for t in THRESHOLDS]
    with TemporaryDirectory() as d:
        path = Path(d)
        with (path / "input.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["type", "gt", "pred"])
            writer.writerows(examples)
        namespace["write_summarized_results"](None, str(path/"input.csv"), str(path/"output.txt"))
        official = {line.split(": ")[0]: float(line.split(": ")[1])
                    for line in (path/"output.txt").read_text().splitlines()[1:]}
    groups = {}
    for kind, gt, pred in examples:
        group, value, _ = evaluate(kind, gt, pred)
        groups.setdefault(group, []).append(value)
    mapping = {"yes_no": "Yes/No Accuracy", "other_text": "Multiple Choice Accuracy",
               "int": "Numeric (count) Accuracy", "float": "Numeric (other) MRA"}
    for group, values in groups.items():
        assert abs(sum(values)/len(values)-official[mapping[group]]) < 1e-12


def fenced(body):
    return "```python\ndef solve(ctx):\n" + "\n".join(
        "    " + line for line in body.splitlines()) + "\n```"


def main(engine=None):
    item, rgb, frame, cache = fixture()
    question = SimpleNamespace(**{k: item[k] for k in ("question", "benchmark", "input_sha256")})
    registry, scene, counters = build_registry(cache, question, Config())
    assert [c["category"] for c in scene.candidates] == ["chair"]
    kernel = NativeKernel(registry, scene, frames=[frame])
    assert ast_guard('def solve(ctx):\n    ReturnAnswer(3.0)', allowed_tools=ALLOWED).ok
    cell = kernel.run_cell('def solve(ctx):\n    ReturnAnswer(3.0)')
    assert cell.answer == "3.0"
    kernel.reset_user_namespace()
    assert kernel.answer_slot.evidence is None
    cell = kernel.run_cell('def solve(ctx):\n    r=ctx.tools.point_depth(view_index=0,x=10,y=10)\n    ReturnAnswer(3)')
    assert len(kernel.cell_results) == 1 and cell.answer == "3"
    assert parse_json('{"answer":3.0}') == 3.0 and scalar(3.0) == "3.0"
    for invalid in ('{"answer":true}', '{"answer":[3]}', '{"answer":1,"answer":2}'):
        try:
            parse_json(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid scalar accepted")
    for arm in ("direct", "skill"):
        for index in range(Config().request_limit(arm)):
            context, receipt = messages(item, rgb, frame, cache, arm=arm,
                                        config=Config(), round_index=index)
            assert receipt["image_count"] == (1 if arm == "direct" else 2)
            raw = json.dumps(context)
            assert "SECRET_UNION_QUESTION" not in raw and '"answer_type"' not in raw
            assert sum(p.get("text", "").count(item["question"])
                       for p in context[0]["content"]) == 1
            assert receipt["text_cap"] is None or receipt["text_chars"] <= receipt["text_cap"]
            assert receipt["text_over_threshold"] == (receipt["text_chars"] > 10000)
            assert receipt["phase"] == ("final" if arm == "direct" or index == 5 else "program")
            assert receipt["tool_limits"] == {"tool_calls": None, "measurements": None}
            assert "EPISODE QUOTA" not in raw and "Excess calls are blocked" not in raw
            if arm == "skill":
                assert "no count limits." in raw and RAW_FORMAT in context[0]["content"][1]["text"]
                text = context[0]["content"][1]["text"]
                assert EVIDENCE_OUTPUT in text  # full protocol remains visible from round one
                if receipt["phase"] == "final":
                    assert "def solve(ctx)" not in text and "ReturnAnswer(" not in text
                    assert "FIRST literal assignment" not in text and "tools closed" in text
                else:
                    assert 'final JSON {"answer"' not in text and "FIRST literal assignment" in text
                    assert "answer = 2.0 * h / w" in text
                    assert "ReturnAnswer(answer, evidence=evidence)" in text
                    assert "Do not hand-compute or round a numeric answer literal." in text
                assert "chord is not full size and MUST omit geometry" in text
                assert "rel_tol=1e-9, abs_tol=1e-9" in text
    # The longest verified question still fits, with the complete examples.
    longest = {**item, "question": "Q"*206}
    assert messages(longest, rgb, frame, cache, arm="skill", config=Config())[1]["text_chars"] <= 10000
    import re
    for example in re.findall(r"```python\n[\s\S]*?```", OUTPUT):
        assert ast_guard(parse_program(example), allowed_tools=ALLOWED).ok
    assert len(set(IDS)) == len(IDS) == 20 and IDS == PRIOR_IDS + ADDED_IDS
    assert Config().trial_request_cap == 140
    from .budget import MIRRORS, mirror_receipt
    ledger = next(iter(MIRRORS))
    row = {"request_id": "local", "category": "solver",
           "context": {"campaign_id": MIRRORS[ledger], "slot_key": "offline"},
           "usage": {"total_tokens": 2}, "status": "completed"}
    shared = {**row, "request_id": "shared"}
    assert mirror_receipt(ledger, [row], [shared])["matched_duplicate_count"] == 1
    assert mirror_receipt(ledger, [row, row], [shared])["matched_duplicate_count"] == 1
    assert mirror_receipt(ledger, [row], [{**shared, "usage": {}}])["matched_duplicate_count"] == 0
    assert mirror_receipt(ledger, [row], [{**shared, "context": {}}])["matched_duplicate_count"] == 0
    for kind, gt, pred, expected in (
            ("int", "3", "3.0", 0), ("int", "3", "3", 1),
            ("str", "no", "NO", 1), ("str", "no", "no ", 0),
            ("float", "1", "1.5", 0), ("float", "1", "1", 1)):
        assert evaluate(kind, gt, pred)[1] == expected
    # Exercise actual solve loop with no filesystem cache or live transport.
    from . import worker
    original_load = worker.load
    worker.load = lambda *args, **kwargs: (rgb, frame, cache)
    class Budget:
        @contextmanager
        def request(self):
            yield {"usage": {}}
    class Client:
        def __init__(self, outputs=(), auto_review=True, auto_evidence=True):
            self.count = 0
            self.outputs = list(outputs)
            self.auto_review = auto_review
            self.auto_evidence = auto_evidence
            self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            self.last_response_identity = {"model": Config().model, "model_reported": True}
            self.last_generation = {}
            self.last_reasoning_content = None
            self.last_service_error = {}
        def chat(self, context, **kwargs):
            assert kwargs["enable_thinking"] is True
            self.count += 1
            output = self.outputs[self.count-1]
            text = "\n".join(p["text"] for p in context[0]["content"] if p["type"] == "text")
            facts = (json.loads(text.split("\nTOOL FACTS ", 1)[1].split("\n", 1)[0])
                     if "\nTOOL FACTS " in text else [])
            if callable(output):
                output = output(facts)
            if self.auto_review and facts:
                reviews = [
                    {"result_id": r["result_id"], "decision": "usable"}
                    for r in facts if r.get("review") == "unreviewed"]
                if output.startswith("```python\n") and "measurement_reviews" not in output:
                    output = output.replace("def solve(ctx):\n",
                        "def solve(ctx):\n    measurement_reviews = " + repr(reviews) + "\n", 1)
                elif output.startswith('{"answer"'):
                    value = json.loads(output)
                    value["measurement_reviews"] = reviews
                    output = json.dumps(value)
            if self.auto_evidence and any(r["source_tool"] in ("point_depth", "pair_point_distance")
                                          for r in facts):
                evidence = {"kind": "visual_estimate", "basis": "Synthetic RGB estimate"}
                if output.startswith("```python\n") and "evidence=" not in output:
                    output = re.sub(r"ReturnAnswer\(([^()\n]*)\)",
                                    lambda m: "ReturnAnswer(" + m[1] + ", evidence=" + repr(evidence) + ")",
                                    output)
                elif output.startswith('{"answer"'):
                    value = json.loads(output)
                    value.setdefault("evidence", evidence)
                    output = json.dumps(value)
            return output

    checked = []

    # Nonlinear, mixed-surface data distinguish original pixels from any statistic.
    from copy import deepcopy
    raw_cache = deepcopy(cache)
    g = raw_cache.geometries[0]
    g["depth"][8:13, 8:13] = 10.0
    g["depth"][10, 10] = 2.5
    g["depth"][20, 20] = 4.25
    g["points"][8:13, 8:13] = [100, 100, 10]
    g["points"][10, 10] = [1, 2, 2.5]
    g["points"][20, 20] = [4, 6, 4.25]
    g["valid_mask"][8, 8] = False
    raw_registry, raw_scene, _ = build_registry(raw_cache, question, Config())
    raw_kernel = NativeKernel(raw_registry, raw_scene, frames=[frame])

    def raw_call(body):
        raw_kernel.reset_user_namespace()
        cell = raw_kernel.run_cell("def solve(ctx):\n    " + body + "\n    YieldObservations([])")
        assert not cell.error_code
        receipt = raw_kernel.cell_results[-1]
        return {"result_id": receipt.result_id, "source_tool": receipt.tool,
                "data": json.loads(receipt.value)}

    mixed = raw_call("ctx.tools.point_depth(view_index=0,x=10,y=10)")
    assert mixed["data"]["depth"] == 2.5 and mixed["data"]["n_valid"] == 24
    samples = mixed["data"]["depth_samples"]
    assert samples["origin_xy"] == [8, 8]
    assert samples["values"][0][0] is None and samples["values"][2] == [10, 10, 2.5, 10, 10]
    assert "At object edges" in raw_registry.get("point_depth").spec.description
    assert "edge:" not in REVIEW_OUTPUT and "avoid" not in RAW_FORMAT
    measured_pair = raw_call("ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20])")
    assert measured_pair["data"]["distance"] == float(np.linalg.norm(np.array([1, 2, 2.5])-np.array([4, 6, 4.25])))
    assert measured_pair["data"]["a_depth_samples"] == samples
    corner = raw_call("ctx.tools.point_depth(view_index=0,x=49.7,y=39.7)")
    assert corner["data"]["pixel_xy"] == [49, 39]
    assert corner["data"]["depth_samples"]["origin_xy"] == [47, 37]
    assert len(corner["data"]["depth_samples"]["values"]) == 3
    marked = overlay(rgb, [corner])[1]["markers"][0]
    assert marked["pixel_xy"] == marked["patch_xy"] == [49, 39]
    g["valid_mask"][8:13, 8:13] = False
    g["valid_mask"][10, 11] = True  # one valid neighbor is still delivered; center null
    single_registry, single_scene, _ = build_registry(raw_cache, question, Config())
    raw_kernel = NativeKernel(single_registry, single_scene, frames=[frame])
    sparse = raw_call("ctx.tools.point_depth(view_index=0,x=10,y=10)")
    assert sparse["data"]["depth"] is None and sparse["data"]["n_valid"] == 1
    nullable_pair = raw_call("ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20])")
    assert nullable_pair["data"]["distance"] is None
    judgment = {"result_id": sparse["result_id"], "decision": "usable"}
    selected = {"kind": "calculation", "expression": "z", "bindings": {
        "z": {"result_id": sparse["result_id"], "quantity": "camera_z", "sample_xy": [11, 10]}}}
    assert validate_evidence(10.0, selected, [sparse], {sparse["result_id"]: judgment},
                             [sparse["result_id"]])["calculated_value"] == 10.0
    for coordinate, code in (([8, 8], "calculation_sample_invalid"),
                             ([20, 20], "calculation_sample_not_delivered"),
                             ([11.0, 10], "calculation_sample_xy_invalid"),
                             (None, "calculation_sample_xy_invalid")):
        invalid = deepcopy(selected)
        invalid["bindings"]["z"]["sample_xy"] = coordinate
        try:
            validate_evidence(10.0, invalid, [sparse], {sparse["result_id"]: judgment}, [sparse["result_id"]])
        except ValueError as exc:
            assert str(exc) == code
        else:
            raise AssertionError("invalid pixel binding accepted")
    checked.append("mixed surfaces preserve each raw depth and raw 3D endpoint distance")
    checked.append("edge and one-valid-pixel patches remain usable with exact pixel coordinates")
    checked.append("explicit calculation sample selects exact delivered value and rejects invalid cells")

    for example in re.findall(r"```python\n[\s\S]*?```", OUTPUT):
        if "ReturnAnswer(" in example:
            kernel.reset_user_namespace()
            cell = kernel.run_cell(parse_program(example))
            assert not cell.error_code
            if kernel.answer_slot.evidence["kind"] == "calculation":
                example_values = {"FULL_HEIGHT_ID": 3.141592653589793,
                                  "FULL_WIDTH_ID": 1.2345678901234567}
                example_facts = [{"result_id": key, "source_tool": "pair_point_distance",
                                  "data": {"distance": value}}
                                 for key, value in example_values.items()]
                example_reviews = {key: {"result_id": key, "decision": "usable"}
                                   for key in example_values}
                checked_example = validate_evidence(
                    kernel.answer_slot.raw_value, kernel.answer_slot.evidence,
                    example_facts, example_reviews, list(example_values))
                assert checked_example["calculated_value"] == 2.0*example_values["FULL_HEIGHT_ID"]/example_values["FULL_WIDTH_ID"]
            else:
                assert cell.answer == "1.25"
                assert validate_evidence(kernel.answer_slot.raw_value, kernel.answer_slot.evidence,
                                         [mixed], {}, [])["kind"] == "visual_estimate"
    checked.append("submission and Python calculation examples execute with their declared evidence")

    archive = [{**mixed, "result_id": f"p{n}"} for n in range(17)]
    anchor_review = {"p0": {"result_id": "p0", "decision": "usable"}}
    chosen, rows, _ = packed(archive, anchor_review, cache, 5000,
                             review_ids=["p13", "p14", "p15", "p16"])
    ids = [r["result_id"] for r in chosen]
    assert ids[:5] == ["p13", "p14", "p15", "p16", "p0"] and len(ids) <= 8
    chosen, _, _ = packed(archive, anchor_review, cache, 5000,
                          review_ids=["p13", "p14"], dependency_ids=["p0", "p9"])
    assert [r["result_id"] for r in chosen][:4] == ["p0", "p9", "p13", "p14"]
    assert dependency_ids({"kind": "calculation", "bindings": {"z": {"result_id": "p9"}}},
                          archive, ["p0"]) == ["p9", "p0"]
    checked.append("usable scale and fresh samples precede old backlog; declared dependencies are pinned")

    def episode(name, outputs, verify, arm="skill", auto_review=True, auto_evidence=True):
        from .audit import replay_contexts
        with TemporaryDirectory() as temp:
            path = Path(temp)
            job = {"arm": arm, "item": item, "output_directory": str(path),
                   "stop_file": str(path/"STOP"), "run_identity": "offline",
                   "initial_receipt": messages(item, rgb, frame, cache, arm=arm,
                                               config=Config())[1]}
            client = Client(outputs, auto_review=auto_review, auto_evidence=auto_evidence)
            result = solve(job, client, Budget())
            assert client.count == result["request_count"] <= Config().request_limit(arm)
            verify(result, path)
            reconstructed = dict(replay_contexts(item, result, rgb, frame, cache))
            for index, context in reconstructed.items():
                assert json.loads((path/f"{index:02d}.messages.json").read_text()) == context
            checked.append(name)
            return result

    try:
        def reviewed(result, path):
            assert result["request_count"] == 2 and result["answer"] == "3"
            assert result["rounds"][0]["status"] == "answer_evidence_required"
            assert result["rounds"][1]["request"]["reviewed_measurement_ids"]
        episode("next-request numeric review", [
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\nReturnAnswer(3)'),
            fenced('ReturnAnswer(3)')], reviewed)

        def distance_review(result, path):
            row = result["rounds"][0]
            receipt = row["tool_receipts"][0]
            data = json.loads(receipt["value"])
            points = cache.geometries[0]["points"]
            a, b = points[10, 10], points[20, 20]
            assert data["a_xy"] == [10, 10] and data["b_xy"] == [20, 20]
            assert abs(data["distance"] - float(np.linalg.norm(a-b))) < 1e-12
            final = result["rounds"][1]["request"]
            assert final["reviewed_measurement_ids"] == [receipt["result_id"]]
            assert len(final["depth_display"]["overlay"]["markers"]) == 2
            assert result["tool_counts"]["measurements"] == 1 and result["answer"] == "3"
        episode("point distance parameters values and markers", [
            fenced('r=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20])\n'
                   'YieldObservations([r["result_id"]])'), fenced('ReturnAnswer(3)')
        ], distance_review)

        def sixth(result, path):
            assert result["request_count"] == 6 and result["answer"] == "3.0"
            assert [r["phase"] for r in result["rounds"]] == ["program"]*5+["final"]
            assert [r["request"]["requests_remaining"] for r in result["rounds"]] == [6,5,4,3,2,1]
            assert result["rounds"][-1]["request"]["tools_enabled"] is False
            assert result["rounds"][-1]["request"]["reviewed_measurement_ids"]
        episode("five programs plus sixth JSON", [
            fenced('YieldObservations([], reason="Continue")')]*4 + [
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                   'YieldObservations([r["result_id"]], reason="Review depth")'),
            '{"answer":3.0}'], sixth)

        def sixth_closed(result, path):
            assert result["status"] == "invalid_answer" and result["tool_counts"]["calls"] == 0
            assert result["rounds"][-1]["request"]["tools_enabled"] is False
        episode("sixth program cannot execute", [
            fenced('YieldObservations([])')]*5 + [
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\nReturnAnswer(3)')
        ], sixth_closed)

        def corrected(result, path):
            assert result["answer"] == "3" and result["request_count"] == 3
            assert result["rounds"][0]["status"] == "repairable_namespace_error"
            assert result["rounds"][0]["executed"] is False
            assert result["rounds"][1]["tool_counts_before"] == empty_counts()
            assert "control_namespace_error" in (path/"01.messages.json").read_text()
        episode("global controls corrected without execution", [
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                   'ctx.tools.YieldObservations([r["result_id"]])'),
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                   'YieldObservations([r["result_id"]])'),
            fenced('ReturnAnswer(3)')], corrected)

        def formats(result, path):
            assert result["answer"] == "3" and result["request_count"] == 4
            assert all(r["status"] == "program_parse_error" for r in result["rounds"][:3])
            assert result["tool_counts"] == empty_counts()
        episode("XML prose and incomplete fences rejected", [
            '<tool_result>{"answer":42}</tool_result>',
            "Here is code:\n" + fenced('ReturnAnswer(42)'),
            "```python\ndef solve(ctx):\n    ReturnAnswer(42)",
            fenced('ReturnAnswer(3)')], formats)

        def dangerous(result, path):
            assert result["status"] == "program_rejected" and result["request_count"] == 1
            assert result["tool_counts"] == empty_counts()
        episode("security errors remain fatal", [
            fenced('import os\nctx.tools.ReturnAnswer(3)')], dangerous)

        def measured(n):
            return (f'for i in range({n}):\n'
                    '    r=ctx.tools.point_depth(view_index=0,x=10+i,y=10)\n'
                    'YieldObservations([r["result_id"]], reason="Review samples")')

        def unlimited_measurements(result, path):
            assert result["answer"] == "3" and result["request_count"] == 3
            counts = result["tool_counts"]
            assert counts["measurements"] == counts["calls"] == counts["attempted"] == 11
            assert counts["attempted_measurements"] == 11 and counts["rejected"] == 0
            assert len(result["rounds"][0]["tool_receipts"]) == 7
            row = result["rounds"][1]
            assert row["status"] == "yielded" and row["cell"]["answer_untrusted"] is False
            assert len(row["tool_receipts"]) == 4
            assert all(not r.get("quota_denials") for r in result["rounds"])
            review = result["rounds"][2]["request"]
            assert review["phase"] == "program" and review["text_cap"] is None
            assert review["tool_limits"] == {"tool_calls": None, "measurements": None}
            assert set(review["reviewed_measurement_ids"]) == set(row["new_measurement_ids"])
            assert review["reviewed_measurement_ids"][0] == row["new_measurement_ids"][-1]
            all_receipts = sum([r.get("tool_receipts", []) for r in result["rounds"]], [])
            assert len(all_receipts) == 11
            text = (path/"02.messages.json").read_text()
            markers = review["depth_display"]["overlay"]["markers"]
            assert len(markers) == 11
            assert len(review["rgb_crop_sheets"]) <= 2
            for receipt in all_receipts:
                assert receipt["status"] == "ok" and not receipt["error"]
                data = json.loads(receipt["value"])
                x, y = data["pixel_xy"]
                expected = float(cache.geometries[0]["depth"][int(y), int(x)])
                assert data["depth"] == expected
                assert data["n_valid"] == 25
                if receipt["result_id"] in review["visible_result_ids"]:
                    assert str(data["depth"]) in text
        episode("eleven measurements execute with bounded context delivery", [
            fenced(measured(7)), fenced(measured(4)), fenced('ReturnAnswer(3)')
        ], unlimited_measurements)

        def call_cap(result, path):
            counts = result["tool_counts"]
            assert counts["calls"] == counts["attempted"] == 9 and counts["measurements"] == 6
            assert counts["rejected"] == 0
            receipts = result["rounds"][0]["tool_receipts"]
            assert len(receipts) == 9 and all(r["status"] == "ok" for r in receipts)
            assert result["rounds"][1]["phase"] == "program" and result["answer"] == "3"
        episode("ninth tool call executes without forcing final JSON", [
            fenced('ctx.tools.view_candidates(view_index=0,category="chair")\n'
                   'ctx.tools.view_depth_summary()\n'
                   'for i in range(6):\n'
                   '    ctx.tools.point_depth(view_index=0,x=10+i,y=10)\n'
                   'ctx.tools.view_depth_summary()\nYieldObservations([])'),
            fenced('ReturnAnswer(3)')], call_cap)

        def exact_cap(result, path):
            assert result["tool_counts"]["calls"] == 8
            assert result["tool_counts"]["rejected"] == 0
            assert result["rounds"][1]["phase"] == "program" and result["answer"] == "3"
        episode("former eight-call threshold stays in program phase", [
            fenced('ctx.tools.view_candidates(view_index=0,category="chair")\n'
                   'ctx.tools.view_depth_summary()\n' + measured(6)),
            fenced('ReturnAnswer(3)')], exact_cap)

        def admitted_failure(result, path):
            assert result["status"] == "run_error" and result["request_count"] == 1
            assert result["tool_counts"]["calls"] == result["tool_counts"]["measurements"] == 1
            assert result["tool_counts"]["rejected"] == 0
            assert result["rounds"][0]["cell"]["answer_untrusted"] is True
            assert (path/"00.execution.json").exists()
        episode("admitted domain failure charged and fatal", [
            fenced('ctx.tools.point_depth(view_index=0,x=-1,y=10)\nReturnAnswer(3)')
        ], admitted_failure)
        def signature_failure(result, path):
            assert result["status"] == "run_error" and result["request_count"] == 1
            assert result["tool_counts"] == empty_counts()
            assert len(result["rounds"][0]["tool_receipts"]) == 1
            assert result["rounds"][0]["cell"]["error_code"] == "tool_contract"
        episode("invalid signature fails before registry admission", [
            fenced('ctx.tools.point_depth(view_index=0,x=10)\nReturnAnswer(3)')
        ], signature_failure)
        valid_mask = cache.geometries[0]["valid_mask"].copy()
        try:
            cache.geometries[0]["valid_mask"][8:13, 8:13] = False
            episode("confidence failure remains fatal", [
                fenced('ctx.tools.point_depth(view_index=0,x=10,y=10)\nReturnAnswer(3)')
            ], admitted_failure)
        finally:
            cache.geometries[0]["valid_mask"][:] = valid_mask

        def runtime_repair(result, path):
            assert result["answer"] == "3" and result["request_count"] == 2
            assert result["rounds"][0]["status"] == "repairable_program_error"
            assert result["rounds"][1]["request"]["reviewed_measurement_ids"]
        episode("successful evidence survives repairable runtime error", [
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                   'x=1/0\nReturnAnswer(3)'), fenced('ReturnAnswer(3)')], runtime_repair)

        def direct(result, path):
            assert result["request_count"] == 1 and result["answer"] == "3"
            assert result["tool_counts"] == empty_counts()
            assert result["rounds"][0]["request"]["image_count"] == 1
            assert "STATIC SKILL" not in (path/"00.messages.json").read_text()
        episode("original RGB direct baseline", ['{"answer":3}'], direct, arm="direct")

        def absent(result, path):
            assert result["request_count"] == 6 and result["status"] == "invalid_answer"
            assert len(result["pending_measurement_ids"]) == 1
            assert all(r["request"]["pending_measurement_ids"]
                       for r in result["rounds"][1:])
        episode("delivery alone cannot certify a surface review", [
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\nReturnAnswer(3)')] + [
            fenced('ReturnAnswer(3)')]*4 + ['{"answer":3}'], absent,
            auto_review=False, auto_evidence=False)

        def rejected(result, path):
            assert result["status"] == "answered" and result["basis"] == "visual_estimate"
            assert result["measurement_reviews"][0]["decision"] == "resample"
            assert len(result["rounds"][0]["tool_receipts"]) == 1
        class RejectClient(Client):
            def chat(self, context, **kwargs):
                return super().chat(context, **kwargs).replace("'usable'", "'resample'")
        old_client = Client
        Client = RejectClient
        try:
            episode("rejected sample remains in raw evidence", [
                fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\nYieldObservations([])'),
                fenced('ReturnAnswer(3)')], rejected)
        finally:
            Client = old_client

        observation = {"result_id": "known-id", "source_tool": "point_depth",
                       "data": {"pixel_xy": [10, 10], "depth": 1.45}}
        review = {"result_id": "known-id", "decision": "usable"}
        program = "def solve(ctx):\n    measurement_reviews = " + repr([review]) + "\n    ReturnAnswer(3)"
        assert validate_reviews(program_reviews(program), [observation]) == ({"known-id": review}, [])
        for bad in ([{**review, "result_id": "unknown"}], [{**review, "reason": "unneeded"}],
                    [{**review, "decision": "certified"}], [review, review]):
            accepted, errors = validate_reviews(bad, [observation])
            assert errors
            assert accepted == ({"known-id": review} if len(bad) == 2 else {})
        annotated, info = overlay(rgb, [observation])
        depth_annotated, depth_info = overlay(annotated, [observation])
        assert info["markers"] == depth_info["markers"]
        sheets = crop_sheets(rgb, info["markers"], ["known-id"])
        marker = sheets[0][1]["markers"][0]
        assert marker["sheet_pixel_xy"] == [26, 34]
        assert marker["source_box_xyxy"] == [0, 0, 50, 40]
        assert len(sheets) == 1 and Image.open(BytesIO(sheets[0][0])).size == (896, 224)
        checked.append("literal review validation and RGB crop coordinate identity")

        summary = {"result_id": "summary-id", "source_tool": "view_depth_summary", "data": {}}
        second = {**observation, "result_id": "second-id"}
        batch = [review, {**review, "result_id": "second-id", "decision": "certified"},
                 {**review, "result_id": "summary-id"}, review]
        accepted, errors = validate_reviews(batch, [observation, second, summary])
        assert accepted == {"known-id": review}
        assert [e["code"] for e in errors] == [
            "review_decision_invalid", "review_not_measurement", "review_duplicate"]
        assert errors[0]["index"] == 1 and errors[0]["field"] == "decision"
        assert validate_reviews([review], [observation], [])[1][0]["code"] == "review_not_delivered"
        assert len(REVIEW_OUTPUT) < 550 and '"reason":' not in REVIEW_OUTPUT
        checked.append("minimal reason-free reviews and per-record decision/type/duplicate feedback")

        def calculation(facts, *, decision="usable", tamper=0, quantity="camera_z",
                        unknown=False, geometry=None, final=False):
            measurements = [r for r in facts if r["source_tool"] in ("point_depth", "pair_point_distance")]
            r = next((r for r in measurements if r.get("review") == "unreviewed"), measurements[0])
            pair = r["source_tool"] == "pair_point_distance"
            judgment = {"result_id": r["result_id"], "decision": decision}
            evidence = {"kind": "calculation", "expression": "2*z",
                        "bindings": {"z": {"result_id": "unknown" if unknown else r["result_id"],
                                          "quantity": quantity}}}
            if geometry is not None:
                evidence["bindings"]["z"]["geometry"] = geometry
            value = 2*r["data"]["distance" if pair else "depth"] + tamper
            if final:
                return json.dumps({"answer": value, "measurement_reviews": [judgment], "evidence": evidence})
            return fenced("measurement_reviews = " + repr([judgment]) + "\nReturnAnswer(" +
                          repr(value) + ", evidence=" + repr(evidence) + ")")

        def verified_calc(result, path):
            from copy import deepcopy
            from .audit import replay_contexts
            assert result["status"] == "answered" and result["basis"] == "calculation"
            assert result["answer_evidence"]["verified_arithmetic"] is True
            assert result["answer_evidence"]["calculated_value"] == 2*1.45
            altered = deepcopy(result)
            altered["answer_evidence"]["calculated_value"] += .5
            try:
                list(replay_contexts(item, altered, rgb, frame, cache))
            except ValueError:
                pass
            else:
                raise AssertionError("tampered evidence receipt accepted by audit")
        sample = fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\nYieldObservations([])')
        episode("usable measurement arithmetic verified", [sample, calculation], verified_calc)
        episode("sixth JSON arithmetic verified", [fenced('YieldObservations([])')]*4 + [
            sample, lambda facts: calculation(facts, final=True)], verified_calc)

        def exact_program(facts):
            row = next(r for r in facts if r["source_tool"] == "point_depth")
            evidence = {"kind": "calculation", "expression": "2.2*z/0.7",
                        "bindings": {"z": {"result_id": row["result_id"], "quantity": "camera_z"}}}
            return fenced("z = " + repr(row["data"]["depth"]) +
                          "\nanswer = 2.2 * z / 0.7\nReturnAnswer(answer, evidence=" + repr(evidence) + ")")

        def exact_verified(result, path):
            assert result["status"] == "answered" and result["request_count"] == 2
            assert result["answer_evidence"]["calculated_value"] == 2.2*1.45/0.7
            assert result["raw_answer"] == result["answer_evidence"]["calculated_value"]
            assert "answer = 2.2 * z / 0.7" in result["rounds"][1]["program"]
        episode("Python computes full-precision answer from original measurements",
                [sample, exact_program], exact_verified)

        def rounded_program(facts):
            row = next(r for r in facts if r["source_tool"] == "point_depth")
            evidence = {"kind": "calculation", "expression": "2.2*z/0.7",
                        "bindings": {"z": {"result_id": row["result_id"], "quantity": "camera_z"}}}
            value = round(2.2*row["data"]["depth"]/0.7, 4)
            return fenced("ReturnAnswer(" + repr(value) + ", evidence=" + repr(evidence) + ")")

        def detailed_repair(result, path):
            from copy import deepcopy
            from .audit import replay_contexts
            error = result["rounds"][1]["answer_feedback_errors"][0]
            assert error["code"] == "answer_calculation_mismatch" and error["field"] == "answer"
            assert error["calculated"] == 2.2*1.45/0.7
            assert error["submitted"] == round(error["calculated"], 4)
            assert error["abs_error"] == abs(error["calculated"]-error["submitted"])
            assert error["allowed_error"] == max(ABS_TOL, REL_TOL*max(
                abs(error["calculated"]), abs(error["submitted"])))
            assert error["rel_tol"] == error["abs_tol"] == 1e-9
            assert "Compute answer in Python" in error["repair"]
            context = json.loads((path/"02.messages.json").read_text())
            text = context[0]["content"][1]["text"]
            delivered = json.loads(text.split("\nERRORS ", 1)[1].split("\n", 1)[0])
            assert delivered == [error]
            assert result["status"] == "answered" and result["request_count"] == 3
            stored = json.loads((path/"result.json").read_text())
            for field, replacement in (("calculated", 99.0), ("allowed_error", 1.0),
                                       ("field", "evidence.expression")):
                changed = deepcopy(stored)
                changed["rounds"][1]["answer_feedback_errors"][0][field] = replacement
                try:
                    list(replay_contexts(item, changed, rgb, frame, cache))
                except ValueError:
                    pass
                else:
                    raise AssertionError("tampered arithmetic feedback accepted: " + field)
        episode("rounding feedback carries original recomputation and tolerances; Python repair succeeds",
                [sample, rounded_program, exact_program], detailed_repair)

        # Retain strict comparison: an ordinary rounded value is not auto-accepted.
        for exact, rounded in ((2.3782871654120665, 2.38),
                               (6.144193760437812, 6.1442),
                               (5.505072571592052, 5.5051)):
            precision_observation = {"result_id": "precision", "source_tool": "point_depth",
                                     "data": {"depth": exact}}
            precision_declared = {"kind": "calculation", "expression": "z", "bindings": {
                "z": {"result_id": "precision", "quantity": "camera_z"}}}
            precision_reviews = {"precision": {"result_id": "precision", "decision": "usable"}}
            try:
                validate_evidence(rounded, precision_declared, [precision_observation],
                                  precision_reviews, ["precision"])
            except EvidenceError as exc:
                error = evidence_feedback(exc)[0]
                assert error["calculated"] == exact and error["submitted"] == rounded
                assert not math.isclose(exact, rounded, rel_tol=REL_TOL, abs_tol=ABS_TOL)
                json.dumps(error, allow_nan=False)
            else:
                raise AssertionError("rounded answer silently normalized")
            assert validate_evidence(exact, precision_declared, [precision_observation],
                                     precision_reviews, ["precision"])[
                "verified_arithmetic"] is True
        checked.append("precision regressions remain strict and report JSON-safe computed values")

        def reverse_binding_final(facts):
            measured = [r for r in facts if r["source_tool"] == "point_depth"]
            first = next(r for r in measured if r["data"]["pixel_xy"] == [10, 10])
            second = next(r for r in measured if r["data"]["pixel_xy"] == [15, 10])
            evidence = {"kind": "calculation", "expression": "z_second+a_first", "bindings": {
                "z_second": {"result_id": second["result_id"], "quantity": "camera_z"},
                "a_first": {"result_id": first["result_id"], "quantity": "camera_z"}}}
            return json.dumps({
                "answer": first["data"]["depth"] + second["data"]["depth"],
                "measurement_reviews": [{"result_id": r["result_id"], "decision": "usable"}
                                        for r in measured],
                "evidence": evidence})

        def persisted_binding_order(result, path):
            from copy import deepcopy
            from .audit import replay_contexts
            stored = json.loads((path/"result.json").read_text())
            assert stored["status"] == "answered" and stored["basis"] == "calculation"
            last = stored["rounds"][-1]
            assert list(last["evidence_declaration"]["bindings"]) == ["a_first", "z_second"]
            measurements = stored["rounds"][4]["tool_receipts"]
            assert last["calculation_dependency_ids"] == [
                measurements[1]["result_id"], measurements[0]["result_id"]]
            contexts = dict(replay_contexts(item, stored, rgb, frame, cache))
            assert contexts[5] == json.loads((path/"05.messages.json").read_text())
            altered = deepcopy(stored)
            altered["rounds"][-1]["calculation_dependency_ids"].reverse()
            try:
                list(replay_contexts(item, altered, rgb, frame, cache))
            except ValueError:
                pass
            else:
                raise AssertionError("tampered final binding priority accepted")

        episode("persisted final JSON preserves original binding priority and rejects order tampering",
                [fenced('YieldObservations([])')]*4 + [
                    fenced('ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                           'ctx.tools.point_depth(view_index=0,x=15,y=10)\nYieldObservations([])'),
                    reverse_binding_final], persisted_binding_order,
                auto_review=False, auto_evidence=False)

        def pixel_calculation(facts):
            row = next(r for r in facts if r["source_tool"] == "point_depth")
            evidence = {"kind": "calculation", "expression": "z", "bindings": {
                "z": {"result_id": row["result_id"], "quantity": "camera_z", "sample_xy": [11, 10]}}}
            value = cache.geometries[0]["depth"][10, 11]
            return fenced("measurement_reviews = " + repr([{"result_id": row["result_id"], "decision": "usable"}]) +
                          "\nReturnAnswer(" + repr(float(value)) + ", evidence=" + repr(evidence) + ")")

        def pixel_verified(result, path):
            assert result["status"] == "answered" and result["answer_evidence"]["calculated_value"] == 1.47
            assert result["answer_evidence"]["bindings"]["z"]["sample_xy"] == [11, 10]
        episode("selected raw pixel binds through worker and replay", [sample, pixel_calculation], pixel_verified)

        def anchor_then_backlog(facts):
            row = next(r for r in facts if r["source_tool"] == "point_depth")
            evidence = {"kind": "calculation", "expression": "z", "bindings": {
                "z": {"result_id": row["result_id"], "quantity": "camera_z"}}}
            return fenced("measurement_reviews = " + repr([{"result_id": row["result_id"], "decision": "usable"}]) +
                          "\nfor i in range(16):\n    ctx.tools.point_depth(view_index=0,x=15,y=10)" +
                          "\nReturnAnswer(0, evidence=" + repr(evidence) + ")")

        def anchor_retry(facts):
            row = next(r for r in facts if r.get("review") == "usable")
            evidence = {"kind": "calculation", "expression": "z", "bindings": {
                "z": {"result_id": row["result_id"], "quantity": "camera_z"}}}
            return fenced("ReturnAnswer(" + repr(row["data"]["depth"]) + ", evidence=" + repr(evidence) + ")")

        def dependencies_retained(result, path):
            from .audit import replay_contexts
            scale = result["rounds"][0]["new_measurement_ids"][0]
            assert result["rounds"][1]["answer_feedback_errors"][0]["code"] == "answer_calculation_mismatch"
            last = result["rounds"][2]
            assert last["request"]["visible_measurement_ids"][0] == scale
            assert last["request"]["reviewed_measurement_ids"]
            assert last["request"]["context_packing"]["omitted_count"] == 0
            assert result["calculation_dependency_ids"] == [scale] and result["basis"] == "calculation"
            for target in ("round", "request", "episode"):
                tampered = deepcopy(result)
                if target == "round":
                    tampered["rounds"][1]["calculation_dependency_ids"] = []
                elif target == "request":
                    tampered["rounds"][2]["request"]["context_packing"]["calculation_dependency_ids"] = []
                else:
                    tampered["calculation_dependency_ids"] = []
                try:
                    list(replay_contexts(item, tampered, rgb, frame, cache))
                except ValueError:
                    pass
                else:
                    raise AssertionError("tampered calculation dependency accepted: " + target)
        episode("calculation dependency survives sixteen fresh samples and audit rejects state tampering",
                [sample, anchor_then_backlog, anchor_retry], dependencies_retained,
                auto_review=False, auto_evidence=False)

        def blocked(code):
            def verify(result, path):
                row = result["rounds"][1]
                assert row["status"] == "answer_evidence_required"
                assert row["answer_feedback_errors"][0]["code"] == code
                assert result["status"] == "answered" and result["basis"] == "visual_estimate"
            return verify
        for name, kwargs, code in (
                ("resample cannot support calculation", {"decision": "resample"}, "calculation_measurement_not_usable"),
                ("uncertain cannot support calculation", {"decision": "uncertain"}, "calculation_measurement_not_usable"),
                ("unknown reference cannot support calculation", {"unknown": True}, "calculation_measurement_unknown"),
                ("tampered numeric answer rejected", {"tamper": .1}, "answer_calculation_mismatch"),
                ("camera Z cannot declare height", {"quantity": "height"}, "point_is_camera_z")):
            episode(name, [sample, lambda f, k=kwargs: calculation(f, **k), fenced('ReturnAnswer(3)')],
                    blocked(code))

        def fresh_blocked(result, path):
            assert result["rounds"][0]["answer_feedback_errors"][0]["code"] == "calculation_measurement_not_delivered"
            assert result["request_count"] == 2 and result["basis"] == "visual_estimate"
        episode("same-program fresh measurement cannot certify answer", [
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                   'ReturnAnswer(r["data"]["depth"], evidence={"kind":"calculation","expression":"z",'
                   '"bindings":{"z":{"result_id":r["result_id"],"quantity":"camera_z"}}})'),
            fenced('ReturnAnswer(3)')], fresh_blocked)

        pair_sample = fenced('r=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20])\nYieldObservations([])')
        geometry = {"coverage": "partial", "axis": "object_vertical",
                    "anchors": ["top", "base"], "objects": ["chair", "chair"]}

        def specific_binding_feedback(result, path):
            error = result["rounds"][1]["answer_feedback_errors"][0]
            assert error["code"] == "calculation_binding_fields"
            assert error["field"] == "evidence.bindings.z.geometry"
            assert error["quantity"] == "chord" and "chord forbids geometry" in error["repair"]
            assert result["status"] == "answered"
        episode("chord geometry conflict names the exact field and legal alternatives", [
            pair_sample, lambda f: calculation(f, quantity="chord",
                                               geometry={**geometry, "coverage": "complete"}),
            fenced('ReturnAnswer(3)')], specific_binding_feedback)

        def reversed_invalid_final(facts):
            row = next(r for r in facts if r["source_tool"] == "pair_point_distance")
            binding = {"result_id": row["result_id"], "quantity": "chord",
                       "geometry": {**geometry, "coverage": "complete"}}
            return json.dumps({
                "answer": 2*row["data"]["distance"],
                "measurement_reviews": [{"result_id": row["result_id"], "decision": "usable"}],
                "evidence": {"kind": "calculation", "expression": "z+a",
                             "bindings": {"z": binding, "a": binding}}})

        def stable_final_feedback(result, path):
            stored = json.loads((path/"result.json").read_text())
            assert stored["status"] == "invalid_answer"
            error = stored["rounds"][-1]["answer_feedback_errors"][0]
            assert error["field"] == "evidence.bindings.a.geometry"
            assert error["code"] == "calculation_binding_fields"
            assert stored["rounds"][-1]["calculation_dependency_ids"] == [
                stored["rounds"][4]["new_measurement_ids"][0]]
        episode("final multi-binding diagnostics survive sorted JSON and replay", [
            fenced('YieldObservations([])')]*4 + [pair_sample, reversed_invalid_final],
                stable_final_feedback)

        def text_feedback(result, path):
            error = result["rounds"][1]["answer_feedback_errors"][0]
            assert error["code"] == "calculation_numeric_required" and error["field"] == "answer"
            assert error["actual_type"] == "str" and "category/text" in error["repair"]
        episode("text answer with numeric calculation receives a type-specific repair", [
            sample,
            lambda f: calculation(f).replace("ReturnAnswer(2.9,", "ReturnAnswer('tv',"),
            fenced('ReturnAnswer(3)')], text_feedback)

        episode("partial extent cannot support full height", [
            pair_sample, lambda f: calculation(f, quantity="height", geometry=geometry),
            fenced('ReturnAnswer(3)')], blocked("complete_dimension_not_declared"))
        for replacement in ({"coverage": "unknown"}, {"coverage": "complete", "axis": "segment"},
                            {"coverage": "complete", "anchors": ["interior", "base"]},
                            {"coverage": "complete", "objects": ["chair", "table"]},
                            {"coverage": "complete", "objects": ["", ""]},
                            {"coverage": "complete", "anchors": [["top"], "base"]}):
            episode("missing coverage axis or extremum cannot support height", [
                pair_sample, lambda f, g={**geometry, **replacement}: calculation(f, quantity="height", geometry=g),
                fenced('ReturnAnswer(3)')], blocked("complete_dimension_not_declared"))

        def complete_calc(result, path):
            assert result["status"] == "answered" and result["answer_evidence"]["verified_arithmetic"]
            assert result["answer_evidence"]["semantic_verification"] == "same_model_declaration"
            assert all(set(r) == {"result_id", "decision"} for r in result["measurement_reviews"])
            assert result["answer_evidence"]["bindings"]["z"]["geometry"]["coverage"] == "complete"
        episode("consistent complete-height declaration accepted", [
            pair_sample, lambda f: calculation(f, quantity="height", geometry={**geometry, "coverage": "complete"})
        ], complete_calc)
        episode("full-height binding requires geometry despite usable point review", [
            pair_sample, lambda f: calculation(f, quantity="height"),
            fenced('ReturnAnswer(3)')], blocked("complete_dimension_not_declared"))
        for quantity, axis, anchors in (("width", "object_horizontal", ["left", "right"]),
                                       ("length", "object_length", ["front", "back"])):
            episode("complete " + quantity + " geometry belongs to calculation binding", [
                pair_sample, lambda f, q=quantity, a=axis, s=anchors: calculation(
                    f, quantity=q, geometry={"coverage": "complete", "axis": a,
                                            "anchors": s, "objects": ["chair", "chair"]})
            ], complete_calc)

        def judgment_program(facts, decision, action="YieldObservations([])"):
            record = {"result_id": facts[0]["result_id"], "decision": decision}
            return fenced("measurement_reviews = " + repr([record]) + "\n" + action)

        def saved_calculation(facts):
            row = next(r for r in facts if r["source_tool"] == "point_depth")
            evidence = {"kind": "calculation", "expression": "z", "bindings": {
                "z": {"result_id": row["result_id"], "quantity": "camera_z"}}}
            return fenced("ReturnAnswer(" + repr(row["data"]["depth"]) +
                          ", evidence=" + repr(evidence) + ")")

        def replacement_state(result, path):
            from copy import deepcopy
            from .audit import replay_contexts
            first = result["rounds"][0]["new_measurement_ids"][0]
            replacement = result["rounds"][4]["new_measurement_ids"][0]
            usable = [{"result_id": first, "decision": "usable"}]
            rejected = [{"result_id": first, "decision": "resample"}]
            assert result["rounds"][1]["measurement_review_state"] == usable
            assert result["rounds"][2]["status"] == "repairable_program_error"
            assert result["rounds"][2]["measurement_review_state"] == rejected
            assert result["rounds"][3]["measurement_reviews"] == []
            assert result["rounds"][3]["measurement_review_state"] == rejected
            assert result["rounds"][3]["answer_feedback_errors"][0]["code"] == "calculation_measurement_not_usable"
            for index in (3, 4, 5):
                context = json.loads((path/f"{index:02d}.messages.json").read_text())
                text = "\n".join(p["text"] for p in context[0]["content"] if p["type"] == "text")
                facts = json.loads(text.split("\nTOOL FACTS ", 1)[1].split("\n", 1)[0])
                assert next(r for r in facts if r["result_id"] == first)["review"] == "resample"
                assert "\nOMITTED REVIEWS " not in text
            assert result["measurement_reviews"] == [
                *rejected, {"result_id": replacement, "decision": "usable"}]
            assert result["answer_evidence"]["result_ids"] == [replacement]
            assert json.loads((path/"05.execution.json").read_text())["measurement_review_state"] == result["measurement_reviews"]
            for target in ("round", "episode", "request"):
                altered = deepcopy(result)
                if target == "round":
                    altered["rounds"][3]["measurement_review_state"][0]["decision"] = "usable"
                elif target == "episode":
                    altered["measurement_reviews"][0]["decision"] = "usable"
                else:
                    altered["rounds"][3]["request"]["surface_review_state"]["judgments"]["M1"] = "usable"
                try:
                    list(replay_contexts(item, altered, rgb, frame, cache))
                except ValueError:
                    pass
                else:
                    raise AssertionError("tampered saved review accepted: " + target)
        episode("downgraded review survives runtime error and blocks old value until replacement", [
            sample,
            lambda f: judgment_program(f, "usable"),
            lambda f: judgment_program(f, "resample", "x=1/0\nYieldObservations([])"),
            saved_calculation,
            fenced('r=ctx.tools.point_depth(view_index=0,x=15,y=10)\nYieldObservations([r["result_id"]])'),
            lambda f: calculation(f, final=True)
        ], replacement_state, auto_review=False, auto_evidence=False)

        def parse_state(result, path):
            assert result["rounds"][2]["status"] == "program_parse_error"
            assert result["rounds"][2]["measurement_review_state"] == result["rounds"][1]["measurement_review_state"]
            assert result["rounds"][3]["request"]["surface_review_state"]["judgments"] == {}
            assert result["rounds"][3]["request"]["surface_review_counts"]["uncertain"] == 1
            assert result["measurement_reviews"][0]["decision"] == "uncertain"
        episode("uncertain state survives later malformed output", [
            sample, lambda f: judgment_program(f, "uncertain"),
            '<tool_result>{"answer":3}</tool_result>',
            fenced('ReturnAnswer(3, evidence={"kind":"visual_estimate","basis":"RGB"})')
        ], parse_state, auto_review=False, auto_evidence=False)

        def rejected_final(result, path):
            assert result["status"] == "invalid_answer" and result["request_count"] == 6
            assert result["measurement_reviews"][0]["decision"] == "resample"
            assert result["pending_measurement_ids"] == []
            final = json.loads((path/"05.execution.json").read_text())
            assert final["measurement_review_state"] == result["measurement_reviews"]
            assert final["answer_feedback_errors"][0]["code"] == "calculation_measurement_not_usable"
        episode("final rejection records point judgment despite invalid calculation", [
            fenced('YieldObservations([])')]*4 + [
            sample, lambda f: calculation(f, decision="resample", final=True)
        ], rejected_final, auto_review=False, auto_evidence=False)

        saved = {str(n): {"result_id": str(n), "decision": "resample" if n == 0 else "usable"}
                 for n in range(40)}
        summary = review_summary(saved, {str(n): f"M{n+1}" for n in range(40)})
        assert summary["judgments"]["M1"] == "resample"
        assert summary["judgments"] == {"M1": "resample"} and summary["omitted_blocked"] == 0
        assert review_summary(saved, {str(n): f"M{n+1}" for n in range(40)}, ["0"])["judgments"] == {}
        blocked_saved = {k: {**v, "decision": "uncertain"} for k, v in saved.items()}
        bounded = review_summary(blocked_saved, {str(n): f"M{n+1}" for n in range(40)})
        assert len(bounded["judgments"]) == 32 and bounded["omitted_blocked"] == 8
        assert len(json.dumps(summary, separators=(",", ":"))) < 1000
        checked.append("review summary excludes visible decisions and usable history, bounds omitted blocked rows")

        for attack in ("__import__('os')", "z.__class__", "z[0]", "2**999999999",
                       "(lambda: z)()", "sum([z])", "'string'", "True", "z/0",
                       "z+1e101", "1", "z if True else 0"):
            try:
                expression_value(attack, {"z": 1.45})
            except ValueError:
                pass
            else:
                raise AssertionError("unsafe expression accepted: " + attack)
        assert expression_value("-(z-2)/2", {"z": 1.45}) == -(1.45-2)/2
        checked.append("bounded AST arithmetic rejects calls attributes powers unused names and nonfinite values")

        def many(result, path):
            assert result["tool_counts"]["calls"] == result["tool_counts"]["measurements"] == 80
            assert len(result["rounds"][0]["tool_receipts"]) == 80
            assert result["request_count"] == 6 and result["status"] == "answered"
            for row in result["rounds"]:
                receipt = row["request"]
                assert receipt["image_count"] <= 5
                if receipt["phase"] == "final":
                    assert receipt["text_chars"] <= 10000
                    assert len(receipt["visible_measurement_ids"]) <= 8
                elif row["round"] > 0:
                    assert receipt["text_chars"] > 10000 and receipt["text_over_threshold"]
                    assert len(receipt["visible_measurement_ids"]) == 80
                assert len(receipt["rgb_crop_sheets"]) <= 2
            assert result["context_text"]["over_threshold_rounds"] == [1, 2, 3, 4]
            assert result["pending_measurement_ids"] == []  # all samples were delivered for review
            assert result["rounds"][-1]["request"]["context_packing"]["omitted_count"] > 0
        episode("eighty calls all execute and final JSON survives bounded backlog", [
            fenced('for i in range(80):\n'
                   '    ctx.tools.point_depth(view_index=0,x=10,y=10)\nYieldObservations([])')
        ] + [fenced('YieldObservations([])')]*4 + ['{"answer":3}'], many)

        def replace_backlog(facts):
            return judgment_program(facts, "resample",
                'r=ctx.tools.point_depth(view_index=0,x=15,y=10)\n'
                'YieldObservations([r["result_id"]])')

        def prioritized(result, path):
            assert result["status"] == "answered" and result["basis"] == "calculation"
            fresh = result["rounds"][1]["new_measurement_ids"][0]
            receipt = result["rounds"][2]["request"]
            assert receipt["visible_measurement_ids"][0] == fresh
            assert receipt["reviewed_measurement_ids"][0] == fresh
            assert receipt["context_packing"]["omitted_count"] == 0
            assert len(receipt["visible_measurement_ids"]) == len(set(receipt["visible_measurement_ids"])) == 81
            assert result["answer_evidence"]["result_ids"] == [fresh]
            assert receipt["surface_review_counts"]["resample"] == 1
        episode("replacement delivered first with full eighty-record program backlog", [
            fenced('for i in range(80):\n'
                   '    ctx.tools.point_depth(view_index=0,x=10,y=10)\nYieldObservations([])'),
            replace_backlog, calculation
        ], prioritized, auto_review=False, auto_evidence=False)

        def partial_program(facts, *, use_second=False):
            measures = [r for r in facts if r["source_tool"] == "point_depth"]
            judgments = [{"result_id": r["result_id"],
                          "decision": "usable" if n == 0 else "certified"}
                         for n, r in enumerate(measures)]
            selected = measures[int(use_second)]
            evidence = {"kind": "calculation", "expression": "z", "bindings": {
                "z": {"result_id": selected["result_id"], "quantity": "camera_z"}}}
            return fenced("measurement_reviews = " + repr(judgments) +
                          "\nReturnAnswer(" + repr(selected["data"]["depth"]) +
                          ", evidence=" + repr(evidence) + ")")
        def partial_verified(result, path):
            assert result["status"] == "answered" and result["basis"] == "calculation"
            assert len(result["measurement_reviews"]) == 1 and len(result["pending_measurement_ids"]) == 1
            assert result["rounds"][1]["review_feedback_errors"][0]["code"] == "review_decision_invalid"
        episode("good calculation survives unrelated rejected review", [
            fenced('ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                   'ctx.tools.point_depth(view_index=0,x=15,y=10)\nYieldObservations([])'),
            partial_program], partial_verified)

        def combined_feedback(result, path):
            row = result["rounds"][1]
            assert row["status"] == "answer_evidence_required"
            assert [e["code"] for e in row["answer_feedback_errors"]] == [
                "calculation_measurement_not_usable", "review_decision_invalid"]
            assert result["status"] == "answered" and result["basis"] == "visual_estimate"
            text = (path/"02.messages.json").read_text()
            assert "calculation_measurement_not_usable" in text and "review_decision_invalid" in text
        episode("failed calculation preserves combined answer and review feedback", [
            fenced('ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                   'ctx.tools.point_depth(view_index=0,x=15,y=10)\nYieldObservations([])'),
            lambda f: partial_program(f, use_second=True), fenced('ReturnAnswer(3)')
        ], combined_feedback)

        # Audits must detect changes to declarations or the accepted arithmetic receipt.
        declaration = {"kind": "calculation", "expression": "z", "bindings": {
            "z": {"result_id": "known-id", "quantity": "camera_z"}}}
        assert validate_evidence(1.45, declaration, [observation], {"known-id": review},
                                 ["known-id"])["verified_arithmetic"]
        checked.append("declared evidence enforces delivered usable sources with native scalar identity")

        class Failure(Client):
            def chat(self, *args, **kwargs):
                self.count += 1
                raise TimeoutError("synthetic")
        with TemporaryDirectory() as temp:
            path = Path(temp)
            failed = Failure()
            job = {"arm": "skill", "item": item, "output_directory": str(path),
                   "stop_file": str(path/"STOP"), "run_identity": "offline",
                   "initial_receipt": messages(item, rgb, frame, cache, arm="skill",
                                               config=Config())[1]}
            result = solve(job, failed, Budget())
            assert failed.count == 1 and result["status"] == "service_error"
            checked.append("service failure stops without retry")
        from .common import digest, write
        from .traces import export
        with TemporaryDirectory() as temp:
            path = Path(temp)
            frozen = {"config": Config().public()}
            write(path/"frozen.json", frozen)
            write(path/"audit.json", {"status": "passed", "run_identity": digest(frozen),
                                     "charged_requests": 2})
            write(path/"inputs.json", [item])
            for arm in ("skill", "direct"):
                directory = path/"episodes"/(item["sample_id"]+"-"+arm)
                directory.mkdir(parents=True)
                text = "x"*12001 if arm == "skill" else "original RGB answer"
                payload = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                write(directory/"00.messages.json", payload)
                write(directory/"result.json", {"rounds": [{"round": 0,
                    "output": "synthetic output", "thinking": "synthetic thinking",
                    "request": {"messages_sha256": digest(payload), "text_chars": len(text)}}]})
            exported = export(path)
            assert exported["over_threshold_requests"] == [{
                "sample_id": item["sample_id"], "arm": "skill", "round": 0,
                "text_chars": 12001, "over_threshold_chars": 2001}]
            stored = json.loads((path/"trajectories.json").read_text())
            for arm in ("skill", "direct"):
                round_row = stored["cases"][0]["arms"][arm]["result"]["rounds"][0]
                assert round_row["output"] == "synthetic output" and round_row["thinking"] == "synthetic thinking"
            checked.append("JSON trajectory preserves output thinking and exact oversized-request index")
    finally:
        worker.load = original_load
    if engine:
        score_parity(engine)
    print("passed: scalar scoring, question scope, bounded images and", len(checked),
          "offline episode checks; live model requests=0")
    for name in checked:
        print("  " + name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
