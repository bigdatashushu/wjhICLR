"""Offline interface regressions using synthetic images and a fake client only.

Run from the workspace root: python -B -m GPTexp.MMSIGPT.selfcheck
"""
from __future__ import annotations

import ast
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


def _suite():
    from .. import adapters
    source = adapters.setup("mmsi")
    if not (source / "src").is_dir():
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    import numpy as np
    from PIL import Image
    from skill3d.sandbox.ast_guard import ast_guard
    from skill3d.small_image import runner as shared
    from skill3d.small_image.blink_protocol import original_messages
    from skill3d.small_image.datasets import SmallImageProblem
    from skill3d.small_image.precompute import _prepared, validate_prepared_content
    from skill3d.tools.contract import DomainValueError

    from . import runner
    from .program import assemble_program, canonical_source
    from .prompts import build_solver_prompt, direct_prompt
    from .public_input import INTERFACE_VERSION, PROMPT_VERSION, public_input
    from .runtime import MMSIKernel
    from .solver import MMSISolver
    from .tools import TOOL_VERSION, registry_for

    answer = "ReturnAnswer(AnswerPayload(value='A', unit='option', basis='visual_estimate'))"
    allowed = {"view_candidates", "view_depth_summary", "point_depth", "pair_point_distance"}

    class FakeClient:
        model = "synthetic-model"
        last_usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}

        def __init__(self, *responses):
            self.responses = iter(responses)
            self.calls = []

        def build_payload(self, messages, **parameters):
            return {"messages": messages, **parameters}

        def chat(self, messages, **parameters):
            self.calls.append((messages, parameters))
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return response

    class Checks(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory(prefix="mmsigpt-interface-")
            self.addCleanup(self.temp.cleanup)
            images = []
            for index in range(2):
                path = Path(self.temp.name) / f"synthetic-{index}.png"
                Image.new("RGB", (12, 10), color=(index * 40, 20, 30)).save(path)
                images.append(str(path))
            self.problem = SmallImageProblem(
                benchmark="mmsi", sample_id="synthetic-only", source_split="test",
                question="Synthetic interface fixture.",
                options=("A. synthetic left", "B. synthetic right"),
                image_paths=tuple(images), task_type="Positional Relationship (Cam.-Obj.)")
            self.frames = shared.load_problem_frames(self.problem)
            yy, xx = np.mgrid[:10, :12]
            self.points = np.stack((xx, yy, np.full_like(xx, 2)), axis=-1).astype(float)

        def prepared(self, *, points=True, partial=False, sparse=False, malformed=False):
            depths = {index: np.full((10, 12), 2.0) for index in range(2)}
            point_maps = {}
            if points:
                data = self.points.copy()
                if sparse:
                    data[:] = np.nan
                    data[3:5, 3:6] = self.points[3:5, 3:6]
                if malformed:
                    data = data[:5, :5]
                point_maps = {0: data}
                if not partial:
                    point_maps[1] = data.copy()
            result = _prepared(
                problem=self.problem.public_dict(), frames=self.frames,
                candidates=(), depths=depths, points=point_maps, intrinsics={},
                geometry_masks={index: np.ones((10, 12), dtype=bool) for index in depths},
                geometries={index: {"relative_only": True} for index in depths},
                categories=(), provenance={"fixture": "synthetic-only"},
                capabilities={"relative_depth": True, "relative_points": bool(points)},
                cache_key="0" * 64)
            validate_prepared_content(result)
            return result

        def kernel(self, prepared=None):
            registry, scene = registry_for(allowed)(prepared or self.prepared(), self.problem)
            return MMSIKernel(registry, scene, frames=list(self.frames), problem=self.problem)

        def solve(self, *responses, prepared=None, **config):
            client = FakeClient(*responses)
            solver = MMSISolver(client, registry_builder=registry_for(allowed),
                                config=shared.SmallImageRunnerConfig(**config))
            result = solver.solve(self.problem, prepared=prepared or self.prepared(), categories=())
            return result, solver, client

        def test_eleven_reported_interface_failure_shapes(self):
            sources = [
                f"def solve(ctx):\n    rows = getattr(ctx, 'options', None)\n    assert rows[0]['label'] == 'A'\n    {answer}",
                f"def solve(ctx):\n    rows = getattr(ctx, 'options')\n    assert rows[1]['text'] == 'synthetic right'\n    {answer}",
                "from harness3d import ReturnAnswer, AnswerPayload\n" + answer,
                f"def solve(ctx):\n    from harness3d import ReturnAnswer, AnswerPayload\n    {answer}",
                "from harness3d import (\n    ReturnAnswer,\n    AnswerPayload,\n)\n" + answer,
                "from harness3d import AnswerPayload\nfrom harness3d import ReturnAnswer\n" + answer,
                f"def solve(ctx):\n    assert ctx.input['question'] == ctx.question\n    {answer}",
                f"def solve(ctx):\n    assert ctx.input['options'][0]['label'] == 'A'\n    {answer}",
                f"def solve(ctx):\n    assert ctx.input['views'][1]['view_index'] == 1\n    {answer}",
                f"def solve(ctx):\n    assert ctx.options == ctx.input['options']\n    {answer}",
                answer.replace("visual_estimate'", "visual_estimate: synthetic comment'"),
            ]
            original_kernel = shared.RestrictedNamespaceKernel
            original_assemble = shared.assemble_program
            original_prompt = shared.build_solver_prompt
            for source in sources:
                with self.subTest(shape=sources.index(source)):
                    result, solver, client = self.solve(source, capture_responses=True)
                    self.assertEqual((result.status, result.answer), ("answered", "A"))
                    self.assertEqual(result.request_count, 1)
                    self.assertEqual(len(client.calls), 1)
                    self.assertEqual(solver.execution_trace[0]["execution"]["error_code"], None)
                    assembled = solver.interface_adaptations[0]
                    executed = solver.execution_trace[0]["execution"]["program_sha256"]
                    self.assertEqual(assembled["canonical_program_sha256"], executed)
                    self.assertEqual(result.prompt_version, PROMPT_VERSION)
                    self.assertEqual(result.tool_version, TOOL_VERSION)
                    receipt = replace(result, receipt_sha256="").to_dict()
                    self.assertEqual(result.receipt_sha256, hashlib.sha256(json.dumps(
                        receipt, sort_keys=True, ensure_ascii=True, allow_nan=False,
                        separators=(",", ":")).encode("ascii")).hexdigest())
            self.assertIs(shared.RestrictedNamespaceKernel, original_kernel)
            self.assertIs(shared.assemble_program, original_assemble)
            self.assertIs(shared.build_solver_prompt, original_prompt)

        def test_canonical_identity_unicode_multiline_and_no_repair(self):
            source = "标记 = '左'; rows = getattr(ctx, 'options', None)\n" + answer
            canonical, kinds = canonical_source(source)
            self.assertIn("标记 = '左'; rows = ctx.options", canonical)
            self.assertEqual(kinds, ["literal_public_context_lookup"])
            source = "from harness3d import (\nReturnAnswer, AnswerPayload\n); " + answer
            canonical, _ = canonical_source(source)
            self.assertEqual(canonical.count("\n"), source.count("\n"))
            ast.parse(canonical)
            canonical, kinds = canonical_source(answer)
            self.assertEqual(canonical, answer)
            self.assertEqual(kinds, [])
            program = assemble_program(source)
            self.assertEqual(program.program_id,
                             hashlib.sha256(program.program_source.encode()).hexdigest()[:16])

        def test_reflection_import_variants_and_host_access_still_rejected(self):
            unsafe = [
                "getattr(ctx, '_scene', None)", "getattr(ctx, '__class__', None)",
                "getattr(ctx, 'options', {})", "getattr(ctx, field, None)",
                "getattr(ctx.scene, 'summary', None)", "getattr(ctx, 'options', default=None)",
                "from harness3d import ReturnAnswer as submit",
                "from harness3d import ReturnAnswer, open",
                "from harness3d import *", "from .harness3d import ReturnAnswer",
                "from os import getenv", "import numpy as np\nnp.fromfile('fixture')",
                "import numpy as np\nnp.ctypeslib",
                "ctx.input['options'][0].__class__", "ctx.tools.candidate_depth_summary('x')",
            ]
            for source in unsafe:
                with self.subTest(source=source):
                    program = assemble_program(source + "\n" + answer)
                    self.assertFalse(ast_guard(program.program_source, allowed_tools=allowed).ok)

        def test_context_is_public_and_read_only(self):
            expected = public_input(self.problem, self.frames)
            self.assertEqual(set(expected), {"benchmark", "task_type", "question", "options", "views"})
            registry, _scene = registry_for(allowed)(self.prepared(), self.problem)
            prompt = build_solver_prompt(self.problem, self.frames, registry, allowed)
            self.assertIn("Input: " + json.dumps(expected, ensure_ascii=True) + "\n", prompt)
            attempts = [
                "ctx.question = 'changed'", "ctx.input['question'] = 'changed'",
                "ctx.options[0]['label'] = 'B'", "ctx.input['views'][0]['width'] = 1",
                "ctx.frames[0][0, 0, 0] = 1",
            ]
            for mutation in attempts:
                with self.subTest(mutation=mutation):
                    kernel = self.kernel()
                    cell = kernel.run_cell(f"def solve(ctx):\n    {mutation}\n    {answer}")
                    self.assertEqual(cell.error_code, "violation_runtime")
                    self.assertIsNone(cell.answer)
                    self.assertEqual(kernel._public_input["question"], self.problem.question)
            cell = self.kernel().run_cell(
                f"def solve():\n    {answer}")
            self.assertEqual(cell.answer, "A")

        def test_exact_tool_exposure_reset_and_disabled_namespace(self):
            kernel = self.kernel()
            self.assertNotIn("candidate_depth_summary", kernel._ns)
            self.assertNotIn("ReadObservation", kernel._ns)
            self.assertFalse(hasattr(kernel._ns["tools"], "candidate_depth_summary"))
            cell = kernel.run_cell("r = point_depth(view_index=0, x=4, y=4)\n"
                                   "YieldObservations([r['result_id']])")
            self.assertEqual(cell.terminated, "yield")
            self.assertEqual(len(kernel.tool_calls), 1)
            first_id = kernel.tool_results[0].result_id
            kernel.reset_user_namespace()
            self.assertNotIn("r", kernel._ns)
            self.assertEqual(kernel.tool_results[0].result_id, first_id)
            self.assertNotIn("candidate_depth_summary", kernel._ns)
            kernel.set_tools_enabled(False)
            kernel.reset_user_namespace()
            self.assertFalse(any(name in kernel._ns for name in allowed))
            self.assertFalse(hasattr(kernel._ns["tools"], "point_depth"))
            self.assertEqual(len(kernel.tool_calls), 1)

        def test_depth_count_does_not_count_optional_points(self):
            for prepared in (self.prepared(sparse=True), self.prepared(points=False),
                             self.prepared(malformed=True)):
                kernel = self.kernel(prepared)
                cell = kernel.run_cell("r = point_depth(view_index=0, x=4, y=4)\n"
                                       "assert r['data']['n_valid'] == 25\n"
                                       "assert r['data']['depth'] == 2\n" + answer)
                self.assertIsNone(cell.error_code)
                self.assertEqual(cell.answer, "A")

        def test_missing_points_catalog_and_view_authorization(self):
            for prepared in (self.prepared(points=False), self.prepared(malformed=True)):
                kernel = self.kernel(prepared)
                self.assertNotIn("pair_point_distance", kernel._ns)
                self.assertIn("point_depth", kernel._ns)
            kernel = self.kernel(self.prepared(partial=True))
            self.assertIn("pair_point_distance", kernel._ns)
            registry, scene = kernel._registry, kernel._scene
            decision = registry.authorize(
                "pair_point_distance", registry.get("pair_point_distance").spec,
                scope=scene.question_tool_scope, available_artifacts=scene.available_artifacts,
                args={"view_index": 1})
            self.assertFalse(decision.allowed)
            self.assertIn("finite_predicted_points_for_view", decision.missing_refs)
            self.assertEqual(registry.decision_version, TOOL_VERSION)
            with self.assertRaises(DomainValueError):
                scene.sample(2, [4, 4])

        def test_real_tool_roundtrip_feedback_and_pair_distance(self):
            source = ("def solve(ctx):"
                      "\n    r = ctx.tools.pair_point_distance(view_index=0, a_xy=[3,4], b_xy=[7,4])"
                      "\n    assert r['status'] == 'ok'"
                      "\n    assert r['data']['distance'] == 4"
                      "\n    YieldObservations([r['result_id']])")
            result, solver, client = self.solve(source, answer)
            self.assertEqual((result.status, result.answer, result.basis), ("answered", "A", "mixed"))
            self.assertEqual((result.request_count, result.attempted_tool_calls,
                              result.succeeded_tool_calls), (2, 1, 1))
            self.assertEqual(len(solver.execution_trace), 2)
            self.assertEqual(solver.execution_trace[0]["authorization_receipts"][0]["allowed"], True)
            self.assertEqual(solver.execution_trace[0]["authorization_receipts"][0]["decision_version"],
                             TOOL_VERSION)
            self.assertEqual(solver.execution_trace[0]["tool_results"][0]["evidence_version"],
                             TOOL_VERSION)
            self.assertIn("pair_point_distance", json.dumps(client.calls[1][0]))

        def test_invalid_basis_and_tool_failure_do_not_fall_back(self):
            for basis in ("mixed: note", "tool_derived: note", "unknown"):
                result, solver, client = self.solve(answer.replace("visual_estimate", basis))
                self.assertEqual(result.status, "run_error")
                self.assertEqual(len(client.calls), 1)
            source = ("def solve(ctx):\n"
                      "    try:\n"
                      "        point_depth(view_index=0, x=-1, y=4)\n"
                      "    except Exception:\n"
                      "        pass\n    " + answer)
            result, solver, client = self.solve(source, answer)
            self.assertEqual(result.status, "run_error")
            self.assertEqual(len(client.calls), 1)
            self.assertTrue(solver.execution_trace[0]["execution"]["answer_untrusted"])

        def test_service_error_has_one_attempt_and_no_retry(self):
            result, _solver, client = self.solve(RuntimeError("synthetic service failure"), answer)
            self.assertEqual(result.status, "service_error")
            self.assertEqual(result.request_count, 1)
            self.assertEqual(len(client.calls), 1)

        def test_finalization_is_answer_only_with_both_protocols(self):
            for explicit in (False, True):
                result, _solver, client = self.solve(
                    "YieldObservations([])", '{"answer":"B"}',
                    max_solve_rounds=1, explicit_final_answer=explicit)
                self.assertEqual((result.status, result.answer, result.request_count), ("answered", "B", 2))
                final_prompt = json.dumps(client.calls[-1][0])
                self.assertIn("Emit only that JSON object", final_prompt)
                self.assertNotIn("optional", final_prompt)
                self.assertNotIn("Finalization:", final_prompt)
                self.assertEqual(result.finalization_requests, 1)

        def test_runner_direct_is_unchanged_and_harness_receipts(self):
            direct = {"arm": "direct", "problem": self.problem.public_dict(), "solver": {}}
            client = FakeClient('{"answer":"A"}')
            with patch("skill3d.small_image.storage.load_prepared",
                       side_effect=AssertionError("direct_loaded_geometry")):
                result = runner.solve(direct, client, Path(self.temp.name), "synthetic")
            self.assertEqual((result["status"], result["answer"]), ("answered", "A"))
            self.assertEqual(result["attempted_tool_calls"], 0)
            self.assertEqual(client.calls, [(
                original_messages(self.problem, direct_prompt(self.problem)),
                {"max_tokens": 2048, "seed": 137, "temperature": 0, "enable_thinking": False})])
            prepared = self.prepared()
            harness = {**direct, "arm": "harness", "prepared_root": self.temp.name,
                       "artifact_sha256": prepared.artifact_sha256,
                       "allowed_tools": sorted(allowed), "skill": {"text": "Return only an answer."}}
            with patch("skill3d.small_image.storage.load_prepared", return_value=prepared):
                result = runner.solve(harness, FakeClient(
                    "from harness3d import ReturnAnswer, AnswerPayload\n" + answer),
                    Path(self.temp.name), "synthetic")
                with self.assertRaisesRegex(ValueError, "prepared_identity_changed"):
                    runner.solve({**harness, "artifact_sha256": "1" * 64},
                                 FakeClient(answer), Path(self.temp.name), "synthetic")
            self.assertEqual(result["status"], "answered")
            self.assertEqual(result["interface_version"], INTERFACE_VERSION)
            self.assertEqual(result["interface_adaptation_count"], 1)
            self.assertEqual(len(result["execution_trace"]), 1)

        def test_changed_prepared_is_rejected_before_any_request(self):
            prepared = self.prepared()
            prepared.geometries[0]["depth"] = np.full((10, 12), 99.0)
            result, _solver, client = self.solve(answer, prepared=prepared)
            self.assertEqual(result.status, "prepared_identity_mismatch")
            self.assertEqual(client.calls, [])

        def test_public_evidence_capture_and_missing_never_change_answer(self):
            summary = {
                "observations": [{"view_index": 0, "fact": "Synthetic object beside a wall."}],
                "reference_frame": "Synthetic camera forward.",
                "relation": "Synthetic left.", "selected_option": "A",
                "tool_rationale": "Synthetic visual fixture requires no measurement.",
            }
            source = "evidence_audit = " + repr(summary) + "\n" + answer
            result, solver, client = self.solve(source)
            audit = solver.execution_trace[0]["evidence_audit"]
            self.assertEqual((result.status, result.answer), ("answered", "A"))
            self.assertEqual((audit["status"], audit["answer_consistent"]), ("provided", True))
            self.assertFalse(audit["facts_verified"])
            self.assertEqual(len(client.calls), 1)
            result, solver, client = self.solve(answer)
            self.assertEqual(solver.execution_trace[0]["evidence_audit"]["status"], "missing")
            self.assertEqual((result.answer, len(client.calls)), ("A", 1))
            kernel = self.kernel()
            kernel.run_cell("evidence_audit = " + repr(summary) + "\nYieldObservations([])")
            kernel.reset_user_namespace()
            kernel.run_cell(answer)
            self.assertEqual(kernel.execution_trace[-1]["evidence_audit"]["status"], "missing")

        def test_malformed_or_inconsistent_evidence_is_flagged_without_retry(self):
            from .evidence import capture
            base = {
                "observations": [{"view_index": 0, "fact": "Synthetic visible point."}],
                "reference_frame": "Synthetic camera.", "relation": "Synthetic right.",
                "selected_option": "B", "tool_rationale": "No measurement.",
            }
            result, solver, client = self.solve("evidence_audit = " + repr(base) + "\n" + answer)
            self.assertEqual(result.answer, "A")
            self.assertEqual(len(client.calls), 1)
            self.assertIn("selected_option_answer_mismatch",
                          solver.execution_trace[0]["evidence_audit"]["issues"])
            bad = [
                object(), {1: "unsupported"},
                {**base, "observations": [{"view_index": True, "fact": "bad index"}]},
                {**base, "observations": [{"view_index": 2, "fact": "outside views"}]},
                {**base, "observations": [object()]},
                {**base, "observations": base["observations"] * 9},
                {**base, "relation": "x" * 401},
            ]
            for value in bad:
                audit = capture(value, n_views=2, labels=("A", "B"), answer="A")
                self.assertEqual(audit["status"], "invalid")

        def test_diagnostic_selection_covers_every_type_without_dispatched_cases(self):
            from .diagnostic import select_pairs, validate_diagnostic
            pairs = [
                {"dataset": "mmsi", "sample_id": f"synthetic-{task}-{i}",
                 "pair_identity": f"synthetic-pair-{task}-{i}",
                 "arms": [{"arm": arm, "problem": {"task_type": f"synthetic-type-{task}"}}
                          for arm in ("direct", "harness")]}
                for task in range(6) for i in range(8)
            ]
            excluded = [row["sample_id"] for row in pairs if row["sample_id"].endswith("-0")]
            selected = select_pairs(pairs, excluded=excluded)
            self.assertEqual(len(selected), 30)
            self.assertEqual(len({p["arms"][0]["problem"]["task_type"] for p in selected[:6]}), 6)
            self.assertFalse(set(excluded) & {p["sample_id"] for p in selected})
            self.assertEqual(selected, select_pairs(list(reversed(pairs)), excluded=excluded))
            manifest = {
                "diagnostic_only": True, "reference_answer_visible": False, "automatic_retry": False,
                "host_scoring": "disabled", "questions": 6, "arms": 12,
                "counts": {"mmsi": 6}, "max_requests": 30, "config": {"max_requests": 30},
                "pairs": selected[:6],
            }
            validate_diagnostic(manifest)
            with self.assertRaises(ValueError):
                validate_diagnostic({**manifest, "host_scoring": "enabled"})
            with self.assertRaises(ValueError):
                select_pairs(pairs, per_type=6)

        def test_diagnostic_completed_execution_exports_without_loading_references(self):
            from .. import run as scheduler
            from ..traces import write
            directory = Path(self.temp.name) / "diagnostic-export"
            directory.mkdir()
            pair = {"dataset": "mmsi", "pair_identity": "synthetic-pair",
                    "arms": [{"arm": arm} for arm in ("direct", "harness")]}
            manifest = {"run_identity": "synthetic-run", "pairs": [pair], "arms": 2,
                        "config": {"model": "synthetic", "max_workers": 2,
                                   "apis": [{"id": "synthetic-api", "enabled": True,
                                             "validated": True, "timeout_seconds": 10}]}}
            write(directory / "manifest.json", manifest)
            for arm in pair["arms"]:
                root = directory / "episodes/mmsi/synthetic-pair" / arm["arm"]
                root.mkdir(parents=True)
                write(root / "episode.json", {"run_identity": "synthetic-run",
                                             "pair_identity": "synthetic-pair"})
            with patch.object(scheduler, "verify_manifest"), patch.object(scheduler, "ApiPool") as pool:
                pool.return_value.paused.return_value = False
                pool.return_value.summary.return_value = {"synthetic": True}
                with patch.object(scheduler.subprocess, "run") as postprocess:
                    scheduler.execute(directory, score_results=False)
            self.assertEqual(postprocess.call_args.args[0][3], "GPTexp.export")
            self.assertEqual(postprocess.call_count, 1)

    return unittest.defaultTestLoader.loadTestsFromTestCase(Checks)


def main():
    def blocked(*_args, **_kwargs):
        raise AssertionError("network_disabled_for_offline_check")

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True))
        for name in ("create_connection",):
            stack.enter_context(patch.object(socket, name, blocked))
        for name in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, name, blocked))
        result = unittest.TextTestRunner(verbosity=2).run(_suite())
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"checks": result.testsRun, "live_model_requests": 0,
                      "real_inputs_read": False, "reference_answers_read": False}))


if __name__ == "__main__":
    main()
