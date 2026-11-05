"""Offline tests for guidance only; no worker, tools or model dependencies."""
import ast
from copy import deepcopy
from itertools import product
from pathlib import Path
import unittest
import re

from MindcubeRunner.context_stage import format_feedback, guidance, stage_actions


class StageActionsTests(unittest.TestCase):
    def test_program_action_combinations(self):
        for repair, review, hints in product((False, True), repeat=3):
            with self.subTest(repair=repair, review=review, hints=hints):
                expected = [action for enabled, action in (
                    (repair, "repair"),
                    (review, "review_measurements"),
                    (hints, "inspect_hints"),
                ) if enabled] or ["plan_or_measure"]
                self.assertEqual(stage_actions(
                    final=False, errors=[{"code": "runtime_error"}] if repair else [],
                    review_ids=["measurement"] if review else [],
                    hint_ids=["hint"] if hints else [],
                ), expected)

    def test_final_overrides_all_program_actions(self):
        for repair, review, hints in product((False, True), repeat=3):
            with self.subTest(repair=repair, review=review, hints=hints):
                self.assertEqual(stage_actions(
                    final=True, errors=[{"code": "runtime_error"}] if repair else [],
                    review_ids=["measurement"] if review else [],
                    hint_ids=["hint"] if hints else [],
                ), ["answer_json"])

    def test_inputs_and_returned_lists_are_independent(self):
        inputs = {"errors": [{"code": "runtime_error"}],
                  "review_ids": ["measurement"], "hint_ids": ["hint"]}
        original = deepcopy(inputs)
        first = stage_actions(final=False, **inputs)
        first.clear()
        self.assertEqual(stage_actions(final=False, **inputs),
                         ["repair", "review_measurements", "inspect_hints"])
        stage_actions(final=True, **inputs)
        self.assertEqual(inputs, original)


class GuidanceTests(unittest.TestCase):
    def test_program_combinations_include_each_action_and_one_format(self):
        headings = {"repair": "REPAIR", "review_measurements": "REVIEW MEASUREMENTS",
                    "inspect_hints": "INSPECT HINTS", "plan_or_measure": "PLAN OR MEASURE"}
        for repair, review, hints in product((False, True), repeat=3):
            actions = stage_actions(
                final=False, errors=["error"] if repair else [],
                review_ids=["measurement"] if review else [],
                hint_ids=["hint"] if hints else [],
            )
            with self.subTest(actions=actions):
                text = guidance(actions, final=False)
                self.assertEqual(text.count("OUTPUT FORMAT"), 1)
                for action, heading in headings.items():
                    self.assertEqual(heading in text.splitlines(), action in actions)
                self.assertNotIn('{"answer":', text)
                self.assertNotIn("100", text)
                for forbidden in ("quota", "call limit", "measurement limit", "slots remaining"):
                    self.assertNotIn(forbidden, text)

    def test_program_format_requires_yield_and_prompt_evidence(self):
        text = guidance(["plan_or_measure"], final=False)
        for required in ("one closed python fence", "only def solve(ctx):",
                         "direct Python function call", "YieldObservations", "unlimited",
                         "MUST yield all new result_ids", "unreliable notices",
                         "Do not answer after measuring", "Variables reset",
                         "evidence remains in the prompt", "no ReadObservation"):
            self.assertIn(required, text)
        self.assertNotIn("ctx.tools.ReturnAnswer", text)
        self.assertNotIn("ctx.tools.YieldObservations", text)
        self.assertNotIn("provisional", text)

    def test_program_examples_have_valid_entries_and_bare_controls(self):
        text = guidance(["repair"], final=False)
        sources = re.findall(r"```python\n([\s\S]*?)\n```", text)
        controls = set()
        for source in sources:
            with self.subTest(source=source):
                tree = ast.parse(source)
                self.assertEqual(len(tree.body), 1)
                entry = tree.body[0]
                self.assertEqual(entry.name, "solve")
                self.assertEqual([arg.arg for arg in entry.args.args], ["ctx"])
                calls = [node for node in ast.walk(entry)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id in {"ReturnAnswer", "YieldObservations"}]
                self.assertEqual(len(calls), 1)
                controls.add(calls[0].func.id)
                for node in ast.walk(entry):
                    if isinstance(node, ast.Attribute):
                        self.assertNotIn(node.attr, {"ReturnAnswer", "YieldObservations"})
        self.assertEqual(controls, {"ReturnAnswer", "YieldObservations"})
        self.assertNotIn("GLOBAL ReturnAnswer", text)

    def test_repair_preserves_review_and_hint_instructions(self):
        text = guidance(["repair", "review_measurements", "inspect_hints"], final=False)
        for required in ("REPAIR", "REVIEW MEASUREMENTS", "INSPECT HINTS",
                         "unreliable_endpoints", "box center"):
            self.assertIn(required, text)
        self.assertNotIn("PLAN OR MEASURE", text)

    def test_pending_evidence_does_not_repeat_initial_query_example(self):
        for actions in (["inspect_hints"], ["review_measurements"],
                        ["repair", "inspect_hints", "review_measurements"]):
            text = guidance(actions, final=False)
            self.assertNotIn("```", text)
            self.assertIn("YieldObservations", text)
            self.assertIn("ReturnAnswer", text)
        self.assertIn("ctx.tools.view_candidates(", guidance(["plan_or_measure"], final=False))

    def test_final_is_closed_even_with_stale_program_actions(self):
        for actions in (["answer_json"], ["repair", "review_measurements",
                                        "inspect_hints", "plan_or_measure"], []):
            with self.subTest(actions=actions):
                text = guidance(actions, final=True)
                self.assertEqual(text.count("OUTPUT FORMAT"), 1)
                self.assertEqual(text.count('{"answer":"A"}'), 1)
                for required in ("Tools are closed", "current legal option",
                                 "A is an example", "replace A",
                                 "data.status", "sample_stability", "supplied evidence"):
                    self.assertIn(required, text)
                for forbidden in ("```", "python", "def solve", "ctx.tools",
                                  "ReturnAnswer", "YieldObservations", "ReadObservation",
                                  "resample", "resampling", "query", "MUST yield"):
                    self.assertNotIn(forbidden, text)

    def test_guidance_does_not_mutate_or_duplicate_actions(self):
        actions = ["repair", "review_measurements", "repair"]
        original = actions.copy()
        text = guidance(actions, final=False)
        self.assertEqual(text.splitlines().count("REPAIR"), 1)
        guidance(actions, final=True)
        self.assertEqual(actions, original)
        self.assertEqual(text, guidance(tuple(actions), final=False))


class FeedbackTests(unittest.TestCase):
    def test_final_preserves_diagnostics_and_replaces_stale_program_advice(self):
        errors = [
            {"code": "control_namespace_error", "type": "AttributeError", "line": 4,
             "repair": "Use global ReturnAnswer or YieldObservations; ctx.tools has evidence tools only.",
             "message": "unexposed runtime detail"},
            {"code": "python_fence_required", "type": None, "line": None,
             "repair": "Output one closed python fence with only def solve(ctx):."},
            {"code": "runtime_error", "type": "NameError", "line": 7,
             "repair": "Correct the Python error; valid answers are current option letters."},
        ]
        original = deepcopy(errors)
        result = format_feedback(errors, final=True)
        self.assertEqual(errors, original)
        self.assertEqual(len(result), len(errors))
        for source, clean in zip(errors, result):
            self.assertIsNot(source, clean)
            self.assertEqual({key: clean[key] for key in ("code", "type", "line")},
                             {key: source[key] for key in ("code", "type", "line")})
            self.assertEqual(clean["repair"], "Follow the final output format above.")
            self.assertEqual(set(clean), {"code", "type", "line", "repair"})
            for forbidden in ("python", "Python", "solve", "ctx.tools", "ReturnAnswer",
                              "YieldObservations", "```", '{"answer":'):
                self.assertNotIn(forbidden, clean["repair"])

    def test_program_preserves_useful_repairs_and_is_a_copy(self):
        errors = [{"code": "runtime_error", "type": "NameError", "line": 2,
                   "repair": "Correct the Python error at line 2.",
                   "traceback": ["private detail"]}]
        original = deepcopy(errors)
        result = format_feedback(errors, final=False)
        self.assertEqual(result, [{key: errors[0][key]
                                  for key in ("code", "type", "line", "repair")}])
        result[0]["repair"] = "changed by caller"
        self.assertEqual(errors, original)

    def test_diagnostic_only_feedback_gets_phase_correct_repair(self):
        errors = [{"code": "invalid_observation_reference", "line": 3}]
        program = format_feedback(errors, final=False)
        final = format_feedback(errors, final=True)
        self.assertEqual(program[0]["code"], errors[0]["code"])
        self.assertEqual(final[0]["code"], errors[0]["code"])
        self.assertIn("program output format above", program[0]["repair"])
        self.assertIn("final output format above", final[0]["repair"])
        self.assertEqual(errors, [{"code": "invalid_observation_reference", "line": 3}])
        self.assertEqual(format_feedback((), final=True), [])
        self.assertEqual(format_feedback((), final=False), [])

    def test_only_sanitized_scalar_fields_survive(self):
        errors = [{"code": ["not a diagnostic"], "type": {"unsafe": "object"},
                   "line": True, "column": True, "source": {"unsafe": "object"},
                   "source_line": ["unsafe"], "syntax_message": ["unsafe"],
                   "previous_response": ["unsafe"],
                   "repair": ["not repair text"], "extra": "drop"},
                  {"code": "runtime_error", "type": "NameError", "line": "traceback text"},
                  "not a mapping"]
        original = deepcopy(errors)
        for final in (False, True):
            with self.subTest(final=final):
                result = format_feedback(errors, final=final)
                self.assertEqual(len(result), 2)
                self.assertEqual(set(result[0]), {"repair"})
                self.assertEqual(set(result[1]), {"code", "type", "repair"})
        self.assertEqual(errors, original)

    def test_previous_program_and_syntax_details_are_program_phase_only(self):
        source = 'def solve(ctx):\n    GLOBAL ReturnAnswer("B")'
        errors = [{"code": "python_syntax_error", "type": "SyntaxError", "line": 2,
                   "column": 12, "source": source, "source_utf8_sha256": "source-hash",
                   "source_line": source.splitlines()[1], "syntax_message": "invalid syntax",
                   "previous_response": "```python\n" + source + "\n```"}]
        original = deepcopy(errors)
        program = format_feedback(errors, final=False)[0]
        for field in errors[0]:
            self.assertEqual(program[field], errors[0][field])
        final = format_feedback(errors, final=True)[0]
        self.assertEqual(set(final), {"code", "type", "line", "column", "repair"})
        self.assertEqual(final["repair"], "Follow the final output format above.")
        self.assertEqual(errors, original)


class StaticSkillTests(unittest.TestCase):
    def test_global_rules_preserve_science_and_compact_result_meaning(self):
        text = (Path(__file__).parents[1] / "static_skill.txt").read_text(encoding="utf-8")
        for required in ("GLOBAL RULES", "OVERALL CONDITIONAL SEQUENCE",
                         "independent camera frame and relative scale",
                         "not verified identities, object counts or mask quality",
                         "interior surfaces", "background", "local camera Z",
                         "local median camera points", 'Outer status="ok"',
                         'data.status="unreliable"', "sample_stability",
                         "shared_data, data_keys and rows",
                         "[sequence_index, result_id, *values]",
                         "The first two row items are metadata",
                         "current phase determines admissible actions"):
            self.assertIn(required, text)
        for forbidden in ("```", "def solve", "ReturnAnswer", "YieldObservations",
                          '{"answer":', "(100,100)", "slots", "quota"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
