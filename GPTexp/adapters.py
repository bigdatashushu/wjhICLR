"""Thin dataset dispatch; MMSI GPT logic is owned by the MMSIGPT package."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys

from .traces import digest, file_digest, read

ROOT = Path(__file__).resolve().parent
BASE = "/home/wangy/wangjh/harness3d/four-datasets-20261031"
COUNTS = {"blink-multiview": 93, "cv3d": 840, "mmsi": 365, "mindcube-tiny": 700, "blink-sr": 100}
SUBDIRS = {"blink-multiview": "code/source", "cv3d": "source_frozen",
           "mmsi": "source_frozen", "mindcube-tiny": "source", "blink-sr": "source"}
CPU_COMPAT = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat"


def setup(dataset):
    source = ROOT / "frozen" / dataset / SUBDIRS[dataset]
    paths = [str(source / "src"), str(source / "scripts"), str(source / "tests")]
    if Path(CPU_COMPAT).is_dir():
        paths.append(CPU_COMPAT)
    sys.path[:0] = paths
    if dataset == "blink-sr":
        sys.path.insert(0, str(source / "scripts/BLINKRunner"))
    return source


def problem_for(public):
    from skill3d.small_image.datasets import SmallImageProblem
    if public["task_type"] == "Spatial_Relation":
        from sr_adapter import SRProblem
        cls = SRProblem
    else:
        cls = SmallImageProblem
    value = cls(**{k: tuple(v) if k in ("options", "image_paths") else v
                   for k, v in public.items() if k != "input_sha256"})
    if value.input_sha256 != public["input_sha256"]:
        raise ValueError("original_input_identity_changed")
    return value


def solver_config(value):
    return {**value, "expected_model": "gpt-6.1-sol"}


def public_spec(spec):
    result = deepcopy(spec)
    for key in ("model", "run_root", "operation_id", "category", "max_requests", "request_concurrency",
                "shared_policy", "test_policy", "proxy_schema", "run_identity", "ordinal", "slot",
                "benchmark", "input_sha256", "images"):
        result.pop(key, None)
    if "solver" in result:
        result["solver"] = solver_config(result["solver"])
    return result


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
def capture_kernel(module, cells):
    original = module.RestrictedNamespaceKernel

    class Captured(original):
        def run_cell(self, source, *args, **kwargs):
            cell = super().run_cell(source, *args, **kwargs)
            cells.append(jsonable({
                "program_source": source, "execution": cell,
                "tool_calls": self.cell_calls, "tool_results": self.cell_results,
                "authorization_receipts": self.authorization_receipts[
                    getattr(self, "_cell_authorization_mark", 0):],
                "observation_reads": self.observation_reads[getattr(self, "_cell_read_mark", 0):],
                "contract_violations": self.contract_violations[
                    getattr(self, "_cell_violation_mark", 0):],
                "answer_payload": self.answer_slot.payload}))
            return cell

    module.RestrictedNamespaceKernel = Captured
    try:
        yield
    finally:
        module.RestrictedNamespaceKernel = original


def mind_config(value):
    from MindcubeRunner.common import Config

    class GPTConfig(Config):
        def __post_init__(self):
            actual = self.model
            if actual != "gpt-6.1-sol":
                raise ValueError("unexpected_gpt_model")
            object.__setattr__(self, "model", "qwen3.8-flash")
            try:
                super().__post_init__()
            finally:
                object.__setattr__(self, "model", actual)

    return GPTConfig(**{**value, "model": "gpt-6.1-sol", "timeout_s": 36000})


def make_specs(dataset):
    """Zero-request validation of every image, prepared artifact and initial prompt."""
    if dataset == "mmsi":
        from .MMSIGPT.inputs import make_specs as make_mmsi_specs
        return make_mmsi_specs()
    source = setup(dataset)
    origin = ROOT / "frozen" / dataset
    from skill3d.small_image.blink_protocol import image_identities, original_messages, sha_value
    from skill3d.small_image.storage import load_prepared
    from skill3d.small_image.runner import _validate_prepared, load_problem_frames
    specs = []
    if dataset == "blink-multiview":
        old = read(origin / "manifest.json")
        from skill3d.evaluation.blink_experiment import benchmark_prompt, DIRECT_SUFFIX
        for job in old["jobs"]:
            spec = public_spec(job["worker"])
            spec["arm"] = "direct" if spec.pop("mode") == "direct" else "harness"
            problem = problem_for(spec["problem"])
            if image_identities(problem) != job["identity"]["images"]:
                raise ValueError("blink_image_order_changed")
            if spec["arm"] == "harness":
                prepared = load_prepared(Path(spec["prepared_root"]) / problem.input_sha256, problem=problem)
                if (prepared.artifact_sha256 != spec["artifact_sha256"]
                        or _validate_prepared(prepared, problem, load_problem_frames(problem))):
                    raise ValueError("blink_prepared_changed")
            else:
                original_messages(problem, benchmark_prompt(problem) + DIRECT_SUFFIX)
            specs.append(spec)
    elif dataset == "cv3d":
        from CV3D.data import PublicInput
        from CV3Drunner.config import RunnerConfig
        from CV3Drunner.protocol import FrozenSkills
        from CV3Drunner.program import prepare_request
        old = read(origin / "manifest.json")
        # This file was published with public fields only in the original run.
        import json
        annotations = [json.loads(line) for line in
                       Path(old["config"]["official_annotations"]).read_text().splitlines() if line.strip()]
        prompts = {(row["task"], row["question"], tuple(row["choices"])): row["prompt"]
                   for row in annotations}
        config = RunnerConfig(**solver_config(old["runner"]))
        frozen = FrozenSkills(source / "scripts/CV3Drunner/skills/frozen_skills.json")
        for sample_id in old["test_ids"]:
            public = old["request_identities"]["direct"][sample_id]["problem"]
            problem = problem_for(public)
            prompt = prompts[(problem.task_type, problem.question, problem.options)]
            item = PublicInput(problem, prompt)
            directory = str(Path(old["config"]["prepared_root"]) / problem.input_sha256)
            for arm, legacy in (("direct", "direct"), ("harness", "tools")):
                prepared, messages, _, _, identity = prepare_request(
                    item, directory if arm == "harness" else None,
                    arm=legacy, frozen=frozen, config=config)
                previous = old["request_identities"][legacy][sample_id]
                if identity["initial_messages_sha256"] != previous["initial_messages_sha256"]:
                    raise ValueError("cv3d_initial_prompt_or_images_changed")
                spec = {"arm": arm, "problem": public, "official_prompt": prompt,
                        "solver": asdict(config), "initial_messages_sha256": identity["initial_messages_sha256"]}
                if arm == "harness":
                    if prepared.artifact_sha256 != previous["artifact_sha256"]:
                        raise ValueError("cv3d_prepared_changed")
                    spec["prepared_directory"] = directory
                    spec["artifact_sha256"] = prepared.artifact_sha256
                specs.append(spec)
    elif dataset == "mindcube-tiny":
        from MindcubeRunner.data import load, select
        from MindcubeRunner.protocol import messages
        old = read(origin / "run.config.json")
        config = mind_config(old["runner"])
        items = select(old["cache_root"], old["selection"], pins=old["input_pins"])
        for item in items:
            for arm, legacy in (("direct", "direct"), ("harness", "skill")):
                cache = load(item, prepared=arm == "harness")
                _, initial = messages(item, cache, arm=legacy, config=config)
                public_item = deepcopy(item)
                if arm == "direct":
                    for view in public_item["views"]:
                        view.pop("prepared_directory", None)
                specs.append({"arm": arm, "item": public_item, "config": config.public(),
                              "initial_receipt": initial})
    elif dataset == "blink-sr":
        from sr_data import load_inputs
        from sr import prepared_for
        from sr_protocol import DIRECT_SUFFIX
        old = read(origin / "manifest.json")
        split = read(origin / "split.json")
        _, items = load_inputs(BASE + "/blink-sr-paired20-20261104-r03/dataset", ("val",))
        by_id = {p.sample_id: (p, prompt) for p, prompt in items}
        caches = {job["spec"]["problem"]["sample_id"]: (
            job["spec"]["prepared_root"], job["spec"]["artifact_sha256"],
            job["spec"]["problem"]["input_sha256"])
            for job in old["jobs"] if job["spec"]["arm"] == "skill_tools"}
        for name in ("precompute-0.json", "precompute-1.json"):
            for row in read(ROOT / "inputs/sr80" / name)["inputs"]:
                caches[row["sample_id"]] = (
                    str(ROOT / "inputs/sr80/prepared"), row["artifact_sha256"], row["input_sha256"])
        for sample_id in split["internal_test"]:
            canonical, prompt = by_id[sample_id]
            directory, artifact, original_input = caches[sample_id]
            cached = load_prepared(Path(directory) / original_input)
            if cached.artifact_sha256 != artifact:
                raise ValueError("sr_prepared_changed")
            # Input hashes include absolute paths. Keep each cache's original
            # binding and verify it against the official input byte for byte.
            problem = problem_for(dict(cached.problem))
            if (any(problem.public_dict()[key] != canonical.public_dict()[key]
                    for key in canonical.public_dict() if key not in ("image_paths", "input_sha256"))
                    or problem.input_sha256 != original_input
                    or image_identities(problem) != image_identities(canonical)):
                raise ValueError("sr_cached_public_input_changed")
            prepared = prepared_for(problem, directory)
            if prepared.artifact_sha256 != artifact:
                raise ValueError("sr_prepared_changed")
            for arm in ("direct", "harness"):
                spec = {"arm": arm, "problem": problem.public_dict(), "official_prompt": prompt,
                        "solver": solver_config(old["inspection"]["configuration"]["solver"])}
                if arm == "harness":
                    spec.update(prepared_root=directory, artifact_sha256=artifact,
                                skill_text=(source / "scripts/BLINKRunner/sr_skill.txt").read_text())
                else:
                    original_messages(problem, prompt + DIRECT_SUFFIX)
                specs.append(spec)
    else:
        raise ValueError("unknown_dataset")
    if len(specs) != COUNTS[dataset] * 2:
        raise ValueError("frozen_denominator_changed")
    return specs


def solve(spec, dataset, client, output, run_identity):
    """Run on the fresh subprocess main thread; no reference file is supplied."""
    if dataset == "mmsi":
        from .MMSIGPT.runner import solve as solve_mmsi
        return solve_mmsi(spec, client, output, run_identity)
    if dataset == "mindcube-tiny":
        from MindcubeRunner import worker
        class Budget:
            @contextmanager
            def request(self):
                yield {"usage": {}}

        worker.Config = lambda **value: mind_config(value)
        job = {**spec, "arm": "direct" if spec["arm"] == "direct" else "skill",
               "output_directory": str(output), "run_identity": run_identity,
               "stop_file": str(Path(output) / "stop.json"), "source": {}}
        return worker.solve(job, client, Budget())
    problem = problem_for(spec["problem"])
    if dataset == "blink-sr":
        from sr import solve_one
        legacy = {**spec, "arm": "direct" if spec["arm"] == "direct" else "skill_tools"}
        return solve_one(legacy, client)
    if dataset == "cv3d":
        from CV3D.data import PublicInput, build_direct_messages, verified_prepared
        from CV3D.inference import parse_final_answer
        from CV3Drunner.config import RunnerConfig
        from CV3Drunner.protocol import FrozenSkills
        from CV3Drunner.tools import build_registry
        from CV3Drunner import runner
        public = PublicInput(problem, spec["official_prompt"])
        if spec["arm"] == "direct":
            text = client.chat(build_direct_messages(public), max_tokens=8192, seed=137,
                               temperature=0, enable_thinking=True)
            answer = parse_final_answer(text, problem.options)
            if client.last_generation.get("status") in {"max_output_tokens", "content_filter"}:
                answer = None
            return {"status": "answered" if answer else "invalid_answer", "answer": answer,
                    "request_count": 1, "attempted_tool_calls": 0, "succeeded_tool_calls": 0}
        prepared = verified_prepared(public, spec["prepared_directory"])
        if prepared.artifact_sha256 != spec["artifact_sha256"]:
            raise ValueError("cv3d_prepared_changed")
        source = ROOT / "frozen/cv3d/source_frozen"
        frozen = FrozenSkills(source / "scripts/CV3Drunner/skills/frozen_skills.json")
        cells = []
        with capture_kernel(runner, cells):
            result = runner.CV3DSolver(
                client, config=RunnerConfig(**spec["solver"]),
                skill_text=frozen.texts[problem.task_type], official_prompt=spec["official_prompt"],
                registry_builder=build_registry).solve(problem, prepared=prepared).to_dict()
        result["execution_trace"] = cells
        return result
    from skill3d.small_image.blink_protocol import explicit_answer, original_messages, response_record
    from skill3d.small_image import runner
    from skill3d.small_image.storage import load_prepared
    config = runner.SmallImageRunnerConfig(**spec["solver"])
    if spec["arm"] == "direct":
        from skill3d.evaluation.blink_experiment import benchmark_prompt, DIRECT_SUFFIX
        prompt = benchmark_prompt(problem) + DIRECT_SUFFIX
        text = client.chat(original_messages(problem, prompt), max_tokens=config.max_tokens,
                           seed=config.seed, temperature=0, enable_thinking=config.enable_thinking)
        try:
            answer, method = explicit_answer(text, problem.options)
        except (ValueError, TypeError):
            answer, method = None, None
        return {"status": "answered" if answer else "invalid_answer", "answer": answer,
                "answer_parse_method": method, "request_count": 1,
                "rounds": [{"phase": "direct", **response_record(client, text)}],
                "attempted_tool_calls": 0, "succeeded_tool_calls": 0}
    prepared = load_prepared(Path(spec["prepared_root"]) / problem.input_sha256, problem=problem)
    if prepared.artifact_sha256 != spec["artifact_sha256"]:
        raise ValueError("prepared_identity_changed")
    from skill3d.evaluation.blink_experiment import scoped_registry
    registry = scoped_registry
    text = spec["skill_text"]
    cells = []
    with capture_kernel(runner, cells):
        result = runner.SmallImageSolver(
            client, config=config, skill_text=text, registry_builder=registry).solve(
                problem, prepared=prepared, categories=prepared.categories).to_dict()
    result["execution_trace"] = cells
    return result
