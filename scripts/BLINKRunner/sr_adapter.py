"""SR integration and complete execution capture, confined to worker processes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from skill3d.small_image.datasets import SmallImageProblem, _input_sha256
from skill3d.small_image import runner
from skill3d.small_image import blink_protocol
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.synthesis.program_assembler import SynthesisError
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.evaluation.blink_experiment import RecordedClient, write_once
from sr_protocol import (
    VERSION, validate_problem, validate_official_prompt, build_prompt,
    explicit_sr_answer,
)

REPAIRABLE_ERRORS = frozenset({
    "KeyError", "TypeError", "IndexError", "ValueError", "NameError",
    "UnboundLocalError", "AttributeError", "ZeroDivisionError",
})


class SRProblem(SmallImageProblem):
    """Keep genuine val provenance and the shared content-hash contract."""
    def __post_init__(self):
        validate_problem(self)
        for name in ("sample_id", "question", "task_type"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("invalid_sr_public_text")
        if not isinstance(self.options, tuple) or not isinstance(self.image_paths, tuple):
            raise ValueError("invalid_sr_public_sequences")
        for path in self.image_paths:
            if not isinstance(path, str) or not Path(path).is_absolute() or not Path(path).is_file():
                raise ValueError("invalid_sr_original_path")
        object.__setattr__(self, "input_sha256", _input_sha256(self._input_dict()))


def solver_config(value):
    if value.get("prompt_profile") != VERSION:
        raise ValueError("invalid_sr_profile")
    return runner.SmallImageRunnerConfig(**{**value, "prompt_profile": runner.PROMPT_VERSION})


def jsonable(value):
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


@contextmanager
def integration(prompt, traces, submissions=None):
    """Scoped hooks are restored even on failure; the shared kernel stays intact."""
    previous = (runner._problem_config, runner.build_solver_prompt,
                runner.RestrictedNamespaceKernel, runner.assemble_program,
                blink_protocol.explicit_answer)
    submissions = submissions if submissions is not None else []
    repair_feedback = []
    state = {"finalizing": False, "submission": {}}

    def config_for(problem, config):
        validate_problem(problem)
        validate_official_prompt(problem, prompt)
        return config

    def prompt_for(problem, frames, registry, allowed_tools, *, feedback=(),
                   skill_text="", finalizing=False, official_prompt=None, **_kwargs):
        state["finalizing"] = finalizing
        return build_prompt(
            problem, frames, prompt=official_prompt, skill_text=skill_text,
            interfaces=runner._interfaces(registry, allowed_tools), feedback=feedback,
            finalizing=finalizing, repair_feedback=repair_feedback)

    def answer_for(text, options):
        if tuple(options) != ("yes", "no"):
            raise ValueError("unexpected_sr_options")
        answer, method = explicit_sr_answer(text)
        state["submission"] = {"submission_format": method, "host_constructed": True}
        return answer, method

    def assemble_for(text, *args, **kwargs):
        if state["finalizing"]:
            return previous[3](text, *args, **kwargs)
        state["submission"] = {"submission_format": "model_python", "host_constructed": False}
        raw = text.strip()
        # JSON-looking submissions are parsed only as data, never as Python/prose.
        if raw.startswith(("{", "[", "```json")):
            try:
                answer, method = explicit_sr_answer(text)
            except ValueError as exc:
                raise SynthesisError("Invalid SR JSON submission") from exc
            state["submission"] = {"submission_format": method, "host_constructed": True}
            source = (
                "ReturnAnswer(AnswerPayload("
                f"value={answer!r}, unit='option', basis='visual_estimate'))"
            )
            program = previous[3](source, *args, **kwargs)
        else:
            program = previous[3](text, *args, **kwargs)
        submissions.append({"model_response": text, **state["submission"]})
        return program

    class CapturedKernel(RestrictedNamespaceKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ns["ReturnAnswer"] = self._return_answer

        def _valid_result_ids(self):
            return [r.result_id for r in self.cell_results
                    if r.status == "ok" and r.error is None and not r.invalidated_by
                    and r.result_id not in self.unresolved_result_ids]

        def _trusted_cell(self):
            return (
                not self.answer_untrusted and not self.contract_violations
                and all(r.status == "ok" and r.error is None and not r.invalidated_by
                        and r.result_id not in self.unresolved_result_ids
                        and not (isinstance(r.payload, dict) and (
                            r.payload.get("status") == "fault"
                            or r.payload.get("service_healthy") is False))
                        for r in self.cell_results)
                and all(row.get("allowed") is True for row in
                        self.authorization_receipts[self._cell_authorization_mark:])
            )

        def _return_answer(self, value):
            if self.cell_calls:
                self._proposed_answer = jsonable(value)
                self._host_readback = True
                if not self._trusted_cell():
                    self.answer_untrusted = True
                self.yield_slot(
                    self._valid_result_ids(),
                    "SR host returned tool evidence for model readback before answer submission.")
            self.answer_slot(value)

        def run_cell(self, code, **kwargs):
            self._host_readback = False
            self._proposed_answer = None
            kwargs.setdefault("yield_on_fallthrough", True)
            cell = super().run_cell(code, **kwargs)
            if self._host_readback:
                cell = replace(cell, host_observation_yield=True)
            record = {
                "cell_index": len(traces) + 1, "program_source": code,
                "execution": cell, "tool_calls": self.cell_calls,
                "tool_results": self.cell_results,
                "authorization_receipts": self.authorization_receipts[
                    self._cell_authorization_mark:],
                "observation_reads": self.observation_reads[self._cell_read_mark:],
                "contract_violations": self.contract_violations[self._cell_violation_mark:],
                "answer_payload": self.answer_slot.payload,
                **state["submission"],
            }
            if self._host_readback:
                record["proposed_answer_not_submitted"] = self._proposed_answer
            effective = cell
            if (cell.error_code == "violation_runtime"
                    and cell.error_type in REPAIRABLE_ERRORS
                    and not cell.answer_untrusted and not self.answer_slot.given
                    and self._trusted_cell()):
                note = {
                    "cell_index": len(traces) + 1, "code": cell.error_code,
                    "type": cell.error_type, "line": cell.error_line,
                    "hint": (
                        "Use documented r['data'] fields: point_depth uses 'depth'; "
                        "view_candidates uses 'candidates' with 'box_xyxy'; "
                        "view_depth_summary uses 'views'. Use retained observations."
                    ),
                }
                repair_feedback.append(note)
                record["bounded_repair_requested"] = note
                effective = replace(
                    cell, error=None, error_code=None, answer=None,
                    terminated="yield", yielded_result_ids=self._valid_result_ids(),
                    yielded_reason="SR bounded repair of a trusted Python error.",
                    host_observation_yield=True, traceback_text="", error_type="",
                    error_message="", error_line=None, error_source_line="",
                )
            record["effective_execution"] = effective
            traces.append(jsonable(record))
            return effective

    runner._problem_config = config_for
    runner.build_solver_prompt = prompt_for
    runner.RestrictedNamespaceKernel = CapturedKernel
    runner.assemble_program = assemble_for
    blink_protocol.explicit_answer = answer_for
    try:
        yield
    finally:
        (runner._problem_config, runner.build_solver_prompt,
         runner.RestrictedNamespaceKernel, runner.assemble_program,
         blink_protocol.explicit_answer) = previous


class SRSolver(runner.SmallImageSolver):
    def solve(self, problem, **kwargs):
        self.execution_trace = []
        submissions = []
        with integration(self.official_prompt, self.execution_trace, submissions):
            result = super().solve(problem, **kwargs)
        # Annotate interpretation separately; keep actual provider text byte-for-byte.
        for row in result.rounds:
            if row["phase"] == "solve":
                match = next((item for item in submissions
                              if item["model_response"] == row.get("model_response")), None)
                if match:
                    row.update({key: value for key, value in match.items()
                                if key != "model_response"})
            elif row.get("answer_parse_method", "").startswith("sr_strict_json"):
                row.update(submission_format=row["answer_parse_method"], host_constructed=True)
        result = replace(result, prompt_version=VERSION, receipt_sha256="")
        digest = hashlib.sha256(json.dumps(
            result.to_dict(), sort_keys=True, ensure_ascii=True, allow_nan=False,
            separators=(",", ":")).encode("ascii")).hexdigest()
        return replace(result, receipt_sha256=digest)


class FullResponseClient(BailianClient):
    """Capture successful provider JSON without headers, credential or error body."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_opener = self._opener
        owner = self

        class Response:
            def __init__(self, response):
                self.response = response
            def __enter__(self):
                self.response.__enter__()
                return self
            def __exit__(self, *args):
                return self.response.__exit__(*args)
            def read(self):
                raw = self.response.read()
                credential = os.environ.get(owner.api_key_env, "")
                if credential and credential.encode() in raw:
                    raise ValueError("response_contains_credentials")
                owner.last_response_json = json.loads(raw)
                return raw

        def opener(request, **options):
            return Response(base_opener(request, **options))
        self._opener = opener
        self.last_response_json = None

    def chat(self, messages, **kwargs):
        self.last_response_json = None
        return super().chat(messages, **kwargs)


class FullRecordedClient(RecordedClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.request_trace = []

    def chat(self, messages, **kwargs):
        payload = copy.deepcopy(self.client.build_payload(messages, **kwargs))
        credential = os.environ.get(getattr(self.client, "api_key_env", ""), "")
        if credential and credential in json.dumps(payload):
            raise ValueError("request_contains_credentials")
        row = {"request_index": self.index + 1, "payload": payload,
               "reference_answer_visible": False}
        write_once(self.root / f"{self.index + 1:03d}.payload.json", row)
        try:
            text = super().chat(messages, **kwargs)
        finally:
            row["response_json"] = getattr(self.client, "last_response_json", None)
            row["service_error"] = getattr(self.client, "last_service_error", {})
            self.request_trace.append(row)
            write_once(self.root / f"{self.index:03d}.full.json", row)
        return text
