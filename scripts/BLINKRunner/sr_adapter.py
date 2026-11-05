"""SR integration and complete execution capture, confined to worker processes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from skill3d.small_image.datasets import SmallImageProblem, _input_sha256
from skill3d.small_image import runner
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.evaluation.blink_experiment import RecordedClient, write_once
from sr_protocol import VERSION, validate_problem, validate_official_prompt, build_prompt


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
def integration(prompt, traces):
    """Scoped hooks are restored even on failure; the shared kernel stays intact."""
    previous = (runner._problem_config, runner.build_solver_prompt,
                runner.RestrictedNamespaceKernel)

    def config_for(problem, config):
        validate_problem(problem)
        validate_official_prompt(problem, prompt)
        return config

    def prompt_for(problem, frames, registry, allowed_tools, *, feedback=(),
                   skill_text="", finalizing=False, official_prompt=None, **_kwargs):
        return build_prompt(
            problem, frames, prompt=official_prompt, skill_text=skill_text,
            interfaces=runner._interfaces(registry, allowed_tools), feedback=feedback,
            finalizing=finalizing)

    class CapturedKernel(RestrictedNamespaceKernel):
        def run_cell(self, code, **kwargs):
            cell = super().run_cell(code, **kwargs)
            traces.append(jsonable({
                "cell_index": len(traces) + 1, "program_source": code,
                "execution": cell, "tool_calls": self.cell_calls,
                "tool_results": self.cell_results,
                "authorization_receipts": self.authorization_receipts[
                    self._cell_authorization_mark:],
                "observation_reads": self.observation_reads[self._cell_read_mark:],
                "contract_violations": self.contract_violations[self._cell_violation_mark:],
                "answer_payload": self.answer_slot.payload,
            }))
            return cell

    runner._problem_config = config_for
    runner.build_solver_prompt = prompt_for
    runner.RestrictedNamespaceKernel = CapturedKernel
    try:
        yield
    finally:
        (runner._problem_config, runner.build_solver_prompt,
         runner.RestrictedNamespaceKernel) = previous


class SRSolver(runner.SmallImageSolver):
    def solve(self, problem, **kwargs):
        self.execution_trace = []
        with integration(self.official_prompt, self.execution_trace):
            result = super().solve(problem, **kwargs)
        from dataclasses import replace
        return replace(result, prompt_version=VERSION)


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
