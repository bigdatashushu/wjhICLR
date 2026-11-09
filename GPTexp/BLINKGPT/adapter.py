from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys

from .common import ROOT, digest, parse_official

CPU_COMPAT = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat"


def setup(dataset):
    source = ROOT / "vendor" / dataset
    paths = [str(source / "src"), str(source / "scripts/BLINKRunner")]
    if Path(CPU_COMPAT).is_dir():
        paths.append(CPU_COMPAT)
    sys.path[:0] = paths


def problem_for(public, dataset):
    setup(dataset)
    if dataset == "sr":
        from sr_adapter import SRProblem as Problem
    else:
        from skill3d.small_image.datasets import SmallImageProblem as Problem
    problem = Problem(**{k: tuple(v) if k in {"options", "image_paths"} else v
                         for k, v in public.items() if k != "input_sha256"})
    if problem.input_sha256 != public["input_sha256"]:
        raise ValueError("original_input_identity_changed")
    return problem


def official_messages(row):
    content = [{"type": "text", "text": row["benchmark_prompt"]}]
    for i, image in enumerate(row["images"], 1):
        raw = Path(image["path"]).read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != image["sha256"]:
            raise ValueError("official_image_changed")
        content.extend([{"type": "text", "text": f"Image {i}:"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{image['mime_type']};base64,"
                                   + base64.b64encode(raw).decode()}}])
    return [{"role": "user", "content": content}]


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
def captured_kernel(module, cells):
    original = module.RestrictedNamespaceKernel

    class Captured(original):
        def run_cell(self, source, *args, **kwargs):
            result = super().run_cell(source, *args, **kwargs)
            cells.append(jsonable({
                "program_source": source, "execution": result,
                "tool_calls": self.cell_calls, "tool_results": self.cell_results,
                "authorization_receipts": self.authorization_receipts[
                    self._cell_authorization_mark:],
                "observation_reads": self.observation_reads[self._cell_read_mark:],
                "contract_violations": self.contract_violations[self._cell_violation_mark:]}))
            return result

    module.RestrictedNamespaceKernel = Captured
    try:
        yield
    finally:
        module.RestrictedNamespaceKernel = original


def initial(job, client):
    spec, dataset = job["spec"], job["dataset"]
    if job["arm"] == "official":
        return official_messages(spec["row"]), []
    problem = problem_for(spec["problem"], dataset)
    from skill3d.small_image.blink_protocol import original_messages, image_identities
    from skill3d.small_image import runner
    from skill3d.small_image.storage import load_prepared
    from skill3d.small_image.runner import load_problem_frames, _validate_prepared
    actual = image_identities(problem)
    if actual != job["images"]:
        raise ValueError("paired_image_identity_changed")
    if job["arm"] == "direct":
        if set(spec) != {"problem", "solver", "official_prompt"}:
            raise ValueError("direct_received_extra_evidence")
        if dataset == "mv":
            from skill3d.evaluation.blink_experiment import DIRECT_SUFFIX
        else:
            from sr_protocol import DIRECT_SUFFIX
        return original_messages(problem, spec["official_prompt"] + DIRECT_SUFFIX), []
    prepared = load_prepared(Path(spec["prepared_root"]) / problem.input_sha256, problem=problem)
    if (prepared.artifact_sha256 != spec["artifact_sha256"]
            or _validate_prepared(prepared, problem, load_problem_frames(problem))):
        raise ValueError("prepared_artifact_changed")
    if dataset == "mv":
        from skill3d.evaluation.blink_experiment import scoped_registry
        registry, scene = scoped_registry(prepared, problem)
        allowed = registry.names_for_scope(
            scene.question_tool_scope, scene.available_artifacts,
            evidence_profile=scene.evidence_profile, question_type=scene.question_type,
            local_facts=scene.local_facts)
        prompt = runner.build_solver_prompt(
            problem, prepared.frames, registry, set(allowed), feedback=(),
            skill_text=spec["skill_text"])
    else:
        from sr import scoped_registry
        from sr_protocol import build_prompt, TOOLS
        if not prepared.has_relative_depth:
            raise ValueError("sr_relative_depth_missing")
        registry, scene = scoped_registry(prepared, problem)
        allowed = set(registry.names_for_scope(
            scene.question_tool_scope, scene.available_artifacts,
            evidence_profile=scene.evidence_profile, question_type=scene.question_type,
            local_facts=scene.local_facts)) & TOOLS
        prompt = build_prompt(problem, prepared.frames, prompt=spec["official_prompt"],
                              skill_text=spec["skill_text"],
                              interfaces=runner._interfaces(registry, allowed), feedback=())
    return original_messages(problem, prompt), sorted(allowed)


def solve(job, client):
    spec, dataset, arm = job["spec"], job["dataset"], job["arm"]
    if arm == "official":
        text = client.chat(official_messages(spec["row"]), max_tokens=8192,
                           temperature=0, seed=137, enable_thinking=True)
        answer, method = parse_official(text, spec["row"]["options"])
        return {"status": "answered" if answer else "invalid_answer", "answer": answer,
                "answer_parse_method": method, "request_count": 1,
                "model_response": text, "response_identity": client.last_response_identity,
                "attempted_tool_calls": 0, "succeeded_tool_calls": 0}
    problem = problem_for(spec["problem"], dataset)
    from skill3d.small_image import runner
    from skill3d.small_image.blink_protocol import explicit_answer, response_record
    if arm == "direct":
        messages, _ = initial(job, client)
        text = client.chat(messages, max_tokens=8192, temperature=0, seed=137, enable_thinking=True)
        try:
            if dataset == "sr":
                from sr_protocol import explicit_sr_answer
                answer, method = explicit_sr_answer(text)
            else:
                answer, method = explicit_answer(text, problem.options)
        except (ValueError, TypeError):
            answer, method = None, "invalid"
        return {"status": "answered" if answer else "invalid_answer", "answer": answer,
                "answer_parse_method": method, "request_count": 1,
                "rounds": [{"phase": "direct", **response_record(client, text)}],
                "attempted_tool_calls": 0, "succeeded_tool_calls": 0}
    if dataset == "sr":
        from sr import solve_one
        return solve_one({"arm": "skill_tools", **spec}, client)
    from skill3d.small_image.storage import load_prepared
    from skill3d.evaluation.blink_experiment import scoped_registry
    prepared = load_prepared(Path(spec["prepared_root"]) / problem.input_sha256, problem=problem)
    if prepared.artifact_sha256 != spec["artifact_sha256"]:
        raise ValueError("prepared_identity_changed")
    cells = []
    with captured_kernel(runner, cells):
        result = runner.SmallImageSolver(
            client, config=runner.SmallImageRunnerConfig(**spec["solver"]),
            skill_text=spec["skill_text"], registry_builder=scoped_registry).solve(
                problem, prepared=prepared, categories=prepared.categories).to_dict()
    result["execution_trace"] = cells
    return result
