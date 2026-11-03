"""Offline checks of the changed scalar, context, scope and review contracts."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import csv
from io import BytesIO
import json
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
from .protocol import OUTPUT, messages
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
            assert receipt["text_chars"] <= receipt["text_cap"]
            assert receipt["phase"] == ("final" if arm == "direct" or index == 5 else "program")
    # The longest verified question still fits, with the complete examples.
    longest = {**item, "question": "Q"*206}
    assert messages(longest, rgb, frame, cache, arm="skill", config=Config())[1]["text_chars"] <= 4000
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
        def __init__(self, outputs=()):
            self.count = 0
            self.outputs = list(outputs)
            self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            self.last_response_identity = {"model": Config().model, "model_reported": True}
            self.last_generation = {}
            self.last_reasoning_content = None
            self.last_service_error = {}
        def chat(self, context, **kwargs):
            assert kwargs["enable_thinking"] is True
            self.count += 1
            return self.outputs[self.count-1]

    checked = []

    def episode(name, outputs, verify, arm="skill"):
        from .audit import replay_contexts
        with TemporaryDirectory() as temp:
            path = Path(temp)
            job = {"arm": arm, "item": item, "output_directory": str(path),
                   "stop_file": str(path/"STOP"), "run_identity": "offline",
                   "initial_receipt": messages(item, rgb, frame, cache, arm=arm,
                                               config=Config())[1]}
            client = Client(outputs)
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
            assert result["rounds"][0]["status"] == "measurement_review_required"
            assert result["rounds"][1]["request"]["reviewed_measurement_ids"]
        episode("next-request numeric review", [
            fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\nReturnAnswer(3)'),
            fenced('ReturnAnswer(3)')], reviewed)

        def distance_review(result, path):
            row = result["rounds"][0]
            receipt = row["tool_receipts"][0]
            data = json.loads(receipt["value"])
            points = cache.geometries[0]["points"]
            a = np.median(points[8:13, 8:13].reshape(-1, 3), axis=0)
            b = np.median(points[18:23, 18:23].reshape(-1, 3), axis=0)
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

        def exhausted(result, path):
            assert result["answer"] == "3" and result["request_count"] == 3
            counts = result["tool_counts"]
            assert counts["measurements"] == counts["calls"] == 6
            assert counts["attempted"] == 7 and counts["rejected"] == 1
            row = result["rounds"][1]
            assert row["status"] == "quota_exhausted" and row["cell"]["answer_untrusted"] is False
            assert len(row["tool_receipts"]) == 3 and len(row["quota_denials"]) == 1
            assert "result_id" not in row["quota_denials"][0]
            final = result["rounds"][2]["request"]
            assert final["phase"] == "final" and final["text_cap"] == 4500
            assert final["remaining_quota"] == {"tool_calls": 2, "measurements": 0}
            assert final["reviewed_measurement_ids"] == row["new_measurement_ids"]
            all_receipts = sum([r.get("tool_receipts", []) for r in result["rounds"]], [])
            text = (path/"02.messages.json").read_text()
            markers = final["depth_display"]["overlay"]["markers"]
            assert len(markers) == 6
            for receipt in all_receipts:
                assert receipt["result_id"] in text
                data = json.loads(receipt["value"])
                x, y = data["pixel_xy"]
                expected = float(np.median(cache.geometries[0]["depth"][int(y)-2:int(y)+3,
                                                                       int(x)-2:int(x)+3]))
                assert data["depth"] == expected
                assert str(data["depth"]) in text and data["n_valid"] == 25
        episode("seventh measurement blocked and six receipts delivered", [
            fenced(measured(3)),
            fenced('try:\n' + "\n".join("    " + line for line in measured(4).splitlines()) +
                   '\nexcept Exception:\n    ReturnAnswer(42)'),
            '{"answer":3}'], exhausted)

        def call_cap(result, path):
            counts = result["tool_counts"]
            assert counts["calls"] == 8 and counts["measurements"] == 6
            assert counts["attempted"] == 9 and counts["rejected"] == 1
            assert len(result["rounds"][0]["tool_receipts"]) == 8
            assert result["rounds"][0]["quota_denials"][0]["reason"] == "tool_calls_exhausted"
            assert result["rounds"][1]["phase"] == "final" and result["answer"] == "3"
        episode("ninth tool call blocked", [
            fenced('ctx.tools.view_candidates(view_index=0,category="chair")\n'
                   'ctx.tools.view_depth_summary()\n'
                   'for i in range(6):\n'
                   '    ctx.tools.point_depth(view_index=0,x=10+i,y=10)\n'
                   'ctx.tools.view_depth_summary()\nYieldObservations([])'),
            '{"answer":3}'], call_cap)

        def exact_cap(result, path):
            assert result["tool_counts"]["calls"] == 8
            assert result["tool_counts"]["rejected"] == 0
            assert result["rounds"][1]["phase"] == "final" and result["answer"] == "3"
        episode("fully consumed quota enters final without rejection", [
            fenced('ctx.tools.view_candidates(view_index=0,category="chair")\n'
                   'ctx.tools.view_depth_summary()\n' + measured(6)),
            '{"answer":3}'], exact_cap)

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
