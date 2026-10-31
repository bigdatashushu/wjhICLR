"""BLINK-only campaign: frozen inputs, isolated solving, training Gate and scoring.

No clients, credentials, references or request governors are opened on import.
The command-line owner freezes code before invoking this module in a new run.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

from skill3d.small_image.blink_protocol import (
    ANSWER_PARSER, DIRECT_SUFFIX, VERSION, benchmark_prompt, explicit_answer,
    image_identities, original_messages, response_record, sha_value, strict_json,
)
from skill3d.small_image.datasets import SmallImageProblem, _read_manifest, _sha256, load_manifest
from skill3d.small_image.runner import (
    SmallImageRunnerConfig, SmallImageSolver, _validate_prepared, load_problem_frames,
    validate_request_config,
)
from skill3d.small_image.storage import load_prepared
from skill3d.small_image.tools import build_registry
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_governor import GovernedClient, RequestGovernor, RequestGovernorConfig

SCHEMA = "harness3d-blink-campaign/1"
TOOLS = ["pair_point_distance", "point_depth", "view_candidates", "view_depth_summary"]
STAGES = ["direct", "no_skill", "initial", "training", "freeze_skill", "final", "score"]
OPERATIONAL = {
    "service_error", "input_error", "precompute_error", "precompute_unavailable",
    "prepared_identity_mismatch", "prepared_input_mismatch", "precompute_contract_error",
    "request_exhausted",
}


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_identity(path):
    st = Path(path).stat()
    return {"device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns}


def seal(value):
    return {**value, "content_sha256": sha_value(value)}


def checked(value):
    if value.get("content_sha256") != sha_value(
            {k: v for k, v in value.items() if k != "content_sha256"}):
        raise ValueError("record_content_changed")
    return value


def read(path):
    return checked(strict_json(Path(path).read_text(encoding="utf-8")))


def write_once(path, value):
    path = Path(path)
    record = seal(value)
    if path.exists():
        if read(path) != record:
            raise ValueError("immutable_record_conflict")
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Publish without replacing an existing result, even on resume/race.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def profile_at(path, data_root=None):
    profile = strict_json(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema") != SCHEMA:
        raise ValueError("unsupported_blink_profile")
    if set(profile) != {"schema", "data_root", "dataset_root", "inputs", "model", "solver", "runtime"}:
        raise ValueError("unexpected_profile_fields")
    # Preserve server absolute paths during Mac-side dry-run. resolve() rewrites
    # /home and /tmp through macOS mount aliases even when the path is remote.
    root = os.path.abspath(os.path.expanduser(str(data_root or profile["data_root"])))
    profile["data_root"] = root
    profile["dataset_root"] = os.path.abspath(
        os.path.expanduser(str(profile["dataset_root"])))
    profile["inputs"] = {k: v.replace("{data_root}", root) for k, v in profile["inputs"].items()}
    if set(profile["inputs"]) != {"public_manifest", "prepared_root", "host_answers", "seed_skills"}:
        raise ValueError("unexpected_input_fields")
    config = SmallImageRunnerConfig(**profile["solver"])
    client = BailianClient(**profile["model"])
    validate_request_config(client, config)
    if (client.model != "qwen3.8-flash" or client.temperature != 0
            or config.max_tokens != 8192 or client.max_output_tokens != 8192
            or config.seed != 137 or config.expected_model != client.model
            or not all((config.enable_thinking, config.original_image_bytes,
                        config.explicit_final_answer, config.capture_responses))):
        raise ValueError("profile_must_use_frozen_blink_request_protocol")
    runtime = profile["runtime"]
    expected = {"request_concurrency", "max_requests", "revision_interval", "gate_size",
                "gate_seeds", "revision_max_tokens", "expected_train", "expected_test"}
    if set(runtime) != expected:
        raise ValueError("unexpected_runtime_fields")
    if (runtime["revision_interval"] != 12 or runtime["gate_size"] != 8
            or runtime["gate_seeds"] != [137, 138] or runtime["revision_max_tokens"] != 2048
            or runtime["expected_train"] != 40 or runtime["expected_test"] != 93
            or type(runtime["request_concurrency"]) is not int
            or not 1 <= runtime["request_concurrency"] <= 4
            or type(runtime["max_requests"]) is not int or runtime["max_requests"] < 0):
        raise ValueError("invalid_blink_runtime")
    return profile


def runtime_identity():
    import cv2
    return {
        "python": sys.version, "executable": sys.executable,
        "opencv": cv2.__version__,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "scipy", "Pillow", "pydantic")},
    }


def select_inputs(profile):
    whole = _read_manifest(profile["inputs"]["public_manifest"])
    rows = [r for r in whole["problems"] if r["benchmark"] == "blink"]
    body = {"schema": whole["schema"], "problems": rows}
    problems = load_manifest({**body, "manifest_sha256": _sha256(body)}, profile["dataset_root"])
    groups = {split: [p for p in problems if p.source_split == split] for split in ("train", "test")}
    if [len(groups[s]) for s in ("train", "test")] != [
            profile["runtime"]["expected_train"], profile["runtime"]["expected_test"]]:
        raise ValueError("blink_denominator_mismatch")
    if len({p.sample_id for p in problems}) != len(problems):
        raise ValueError("overlapping_blink_roles")
    for p in problems:
        if (p.task_type != "Multi-view_Reasoning" or len(p.image_paths) != 2
                or p.options != ("left", "right")
                or "clockwise (left)" not in p.question
                or "counter-clockwise (right)" not in p.question
                or "static scene" not in p.question):
            raise ValueError("blink_task_definition_mismatch")
    return problems


def preflight(profile):
    """Validate every public input/cache, without opening GT or issuing requests."""
    problems = select_inputs(profile)
    seed_artifact = strict_json(Path(profile["inputs"]["seed_skills"]).read_text())
    seeds = [s for s in seed_artifact["skills"].values() if s["benchmark"] == "blink"]
    if len(seeds) != 1:
        raise ValueError("expected_one_blink_seed")
    seed = seeds[0]
    # Original campaign hashes ASCII canonical JSON, retain its source identity.
    if (seed["content_sha256"] != _sha256({k: v for k, v in seed.items() if k != "content_sha256"})
            or seed["allowed_tools"] != TOOLS or seed["task_type"] != "Multi-view_Reasoning"):
        raise ValueError("seed_identity_mismatch")
    caches = {}
    for problem in problems:
        directory = Path(profile["inputs"]["prepared_root"]) / problem.input_sha256
        prepared = load_prepared(directory, problem=problem)
        frames = load_problem_frames(problem)
        if _validate_prepared(prepared, problem, frames):
            raise ValueError("prepared_cache_preflight_failed")
        original_messages(problem, benchmark_prompt(problem))
        caches[problem.sample_id] = {
            "input_sha256": problem.input_sha256, "artifact_sha256": prepared.artifact_sha256,
            "metadata_sha256": file_sha(directory / "metadata.json"),
            "images": image_identities(problem),
        }
    result = {
        "schema": SCHEMA, "protocol": VERSION, "parser": ANSWER_PARSER,
        "inputs": {key: (file_identity(value) if key == "host_answers" else file_sha(value))
                   for key, value in profile["inputs"].items() if key != "prepared_root"},
        "problems": [p.public_dict() for p in problems], "cache_identities": caches,
        "seed": seed, "model_requests": 0, "reference_answers_read": False,
        "stages": STAGES, "runtime_identity": runtime_identity(),
    }
    return seal(result)


def scoped_registry(prepared, problem):
    registry, scene = build_registry(prepared, problem)
    registry._model_tools = frozenset(TOOLS)
    return registry, scene


class RecordedClient:
    """Persist per-request intent + response in addition to aggregate budget state."""
    def __init__(self, client, governor, root, operation_id, category):
        self.client = client
        self.governed = GovernedClient(client, governor, category=category,
                                      context={"variant_id": operation_id})
        self.root, self.index = Path(root) / "requests" / operation_id, 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, messages, **kwargs):
        self.index += 1
        stem = self.root / f"{self.index:03d}"
        intent, response = stem.with_suffix(".intent.json"), stem.with_suffix(".response.json")
        if intent.exists() or response.exists():
            raise RuntimeError("request_replay_forbidden")
        write_once(intent, {"payload_sha256": sha_value(self.client.build_payload(messages, **kwargs)),
                            "parameters": kwargs, "started_at": time.time()})
        try:
            text = self.governed.chat(messages, **kwargs)
        except BaseException as exc:
            write_once(response, {"status": "error", "error_type": type(exc).__name__,
                                  **response_record(self.client, None)})
            raise
        record = response_record(self.client, text)
        write_once(response, {"status": "received", "finished_at": time.time(), **record})
        actual = self.client.last_response_identity
        if actual.get("model") != self.client.model or actual.get("model_reported") is not True:
            raise RuntimeError("response_model_mismatch")
        return text


def solve_worker(spec):
    """Spawn boundary: receives public input, selected Skill, and request policy only."""
    problem = SmallImageProblem(**{k: tuple(v) if k in ("options", "image_paths") else v
                                   for k, v in spec["problem"].items() if k != "input_sha256"})
    if problem.input_sha256 != spec["problem"]["input_sha256"]:
        raise ValueError("worker_input_changed")
    base = BailianClient(**spec["model"])
    governor = RequestGovernor(Path(spec["run_root"]) / "budget", RequestGovernorConfig(
        spec["max_requests"], spec["request_concurrency"]))
    client = RecordedClient(base, governor, spec["run_root"], spec["operation_id"], spec["category"])
    config = SmallImageRunnerConfig(**spec["solver"])
    if spec["mode"] == "direct":
        messages = original_messages(problem, benchmark_prompt(problem) + DIRECT_SUFFIX)
        text = client.chat(messages, max_tokens=config.max_tokens, seed=config.seed,
                           temperature=0, enable_thinking=config.enable_thinking)
        row = {"phase": "direct", "request_parameters": {
            "max_tokens": config.max_tokens, "seed": config.seed,
            "temperature": 0, "enable_thinking": config.enable_thinking},
               **response_record(base, text)}
        try:
            answer, method = explicit_answer(text, problem.options)
            row.update(status="answered", answer_parse_method=method)
        except ValueError:
            answer = None
            row["status"] = "invalid_answer"
        result = {"benchmark": "blink", "sample_id": problem.sample_id,
                  "source_split": problem.source_split, "input_sha256": problem.input_sha256,
                  "status": row["status"], "answer": answer, "rounds": [row], "request_count": 1,
                  "attempted_tool_calls": 0, "succeeded_tool_calls": 0,
                  "denied_authorizations": 0}
    else:
        prepared = load_prepared(Path(spec["prepared_root"]) / problem.input_sha256, problem=problem)
        if prepared.artifact_sha256 != spec["artifact_sha256"]:
            raise ValueError("worker_prepared_changed")
        result = SmallImageSolver(client, config=config, skill_text=spec["skill_text"],
                                  registry_builder=scoped_registry).solve(
            problem, prepared=prepared, categories=prepared.categories).to_dict()
    return {"status": "stopped" if result["status"] in OPERATIONAL else "complete",
            "result": result}


def metrics(receipts, references):
    if len({r["result"]["sample_id"] for r in receipts}) != len(receipts):
        raise ValueError("duplicate_scored_samples")
    correct = legal = errors = denied = runtime_errors = 0
    for receipt in receipts:
        row = receipt["result"]
        valid = row["status"] == "answered" and row["answer"] in ("A", "B")
        correct += bool(valid and row["answer"] == references[row["sample_id"]])
        legal += valid
        errors += any(r["status"] in {
            "run_error", "program_rejected", "program_parse_error", "program_missing_action"}
            for r in row["rounds"])
        runtime_errors += row["status"] == "run_error"
        denied += row["denied_authorizations"]
    return {"questions": len(receipts), "correct": correct, "legal": legal,
            "program_errors": errors, "runtime_errors": runtime_errors,
            "denied_authorizations": denied, "accuracy": correct / len(receipts)}


def gate_decision(pairs, references):
    scores = {}
    backend_fingerprint_observed = True
    for seed in (137, 138):
        selected = [(a, b) for a, b in pairs if a["seed"] == seed]
        if len(selected) != 8 or len({a["result"]["sample_id"] for a, _ in selected}) != 8:
            raise ValueError("incomplete_gate_denominator")
        for a, b in selected:
            for field in ("input_sha256", "artifact_sha256", "seed", "solver", "images", "model"):
                if a[field] != b[field]:
                    raise ValueError("unpaired_gate_identity")
            for receipt in (a, b):
                if receipt["status"] != "complete" or receipt["result"]["source_split"] != "train":
                    raise ValueError("invalid_gate_role_or_status")
                for row in receipt["result"]["rounds"]:
                    actual = row.get("response_identity", {})
                    if actual.get("model") != "qwen3.8-flash" or actual.get("model_reported") is not True:
                        raise ValueError("missing_gate_response_identity")
                    if row.get("request_parameters", {}).get("seed") != seed:
                        raise ValueError("actual_gate_seed_mismatch")
                    if row["phase"] in ("solve", "finalization") and row.get(
                            "delivered_skill_sha256") != receipt["skill_sha256"]:
                        raise ValueError("actual_gate_skill_mismatch")
            fingerprints = [{r["response_identity"].get("backend_fingerprint")
                             for r in receipt["result"]["rounds"]} for receipt in (a, b)]
            if fingerprints[0] != fingerprints[1] or any(len(f) != 1 for f in fingerprints):
                raise ValueError("gate_backend_identity_mismatch")
            backend_fingerprint_observed &= None not in fingerprints[0] and "" not in fingerprints[0]
        parent, candidate = metrics([a for a, _ in selected], references), metrics([b for _, b in selected], references)
        passed = candidate["correct"] > parent["correct"] and all(
            candidate[k] <= parent[k] for k in ("program_errors", "runtime_errors", "denied_authorizations"))
        passed = passed and candidate["legal"] >= parent["legal"]
        scores[str(seed)] = {"parent": parent, "candidate": candidate, "passed": passed}
    return {"promoted": all(s["passed"] for s in scores.values()),
            "seeds": scores, "pairs": 16, "gate_source_split": "train",
            "backend_fingerprint_observed": backend_fingerprint_observed}


class Campaign:
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), checked(manifest)
        self.profile = manifest["profile"]
        self.inputs = checked(manifest["preflight"])
        self.problems = [SmallImageProblem(**{
            k: tuple(v) if k in ("options", "image_paths") else v
            for k, v in r.items() if k != "input_sha256"}) for r in self.inputs["problems"]]
        if [p.public_dict() for p in self.problems] != self.inputs["problems"]:
            raise ValueError("campaign_public_input_changed")
        self.train = [p for p in self.problems if p.source_split == "train"]
        self.tests = [p for p in self.problems if p.source_split == "test"]
        runtime = self.profile["runtime"]
        self.governor = RequestGovernor(self.root/"budget", RequestGovernorConfig(
            runtime["max_requests"], runtime["request_concurrency"]))

    def verify(self):
        if runtime_identity() != self.inputs["runtime_identity"]:
            raise ValueError("runtime_changed")
        if source_files(self.root/"code/source") != self.manifest["source_files"]:
            raise ValueError("frozen_source_changed")
        for key, expected in self.inputs["inputs"].items():
            current = (file_identity(self.profile["inputs"][key]) if key == "host_answers"
                       else file_sha(self.profile["inputs"][key]))
            if current != expected:
                raise ValueError("campaign_input_file_changed")
        usage = self.governor.summary()
        if any(usage[k] for k in ("unknown_requests", "active_requests", "error_requests")):
            raise RuntimeError("unresolved_or_failed_prior_requests")

    def references(self, role):
        if role not in ("train", "test"):
            raise ValueError("invalid_reference_role")
        path = Path(self.profile["inputs"]["host_answers"])
        if (file_identity(path) != self.inputs["inputs"]["host_answers"]
                or file_sha(path) != self.manifest["reference_file_sha256"]):
            raise ValueError("reference_identity_changed")
        values = strict_json(path.read_text())
        expected = {p.sample_id for p in (self.train if role == "train" else self.tests)}
        rows = [r for r in values if r["benchmark"] == "blink" and r["source_split"] == role]
        if (len(rows) != len(expected) or {r["sample_id"] for r in rows} != expected
                or any(r["reference_answer"] not in ("A", "B") for r in rows)):
            raise ValueError("reference_denominator_mismatch")
        return {r["sample_id"]: r["reference_answer"] for r in rows}

    def operation(self, problem, text, stage, seed=137, mode="system"):
        self.verify()
        config = {**self.profile["solver"], "seed": seed}
        cache = self.inputs["cache_identities"][problem.sample_id]
        identity = {"stage": stage, "input_sha256": problem.input_sha256,
                    "skill_sha256": sha_value(text), "seed": seed, "solver": config,
                    "images": cache["images"], "artifact_sha256": cache["artifact_sha256"],
                    "model": self.profile["model"], "campaign": self.manifest["content_sha256"],
                    "mode": mode, "allowed_tools": TOOLS if mode == "system" else []}
        op_id = sha_value(identity)
        destination = self.root/"results"/stage/(op_id+".json")
        intent = self.root/"intents"/(op_id+".json")
        if destination.exists():
            result = read(destination)
            if result["identity"] != identity or result["status"] != "complete":
                raise ValueError("saved_operation_changed_or_failed")
            if any(file_sha(self.root/p) != h for p, h in result["request_records"].items()):
                raise ValueError("saved_request_record_changed")
            return result
        if intent.exists():
            raise RuntimeError("unfinished_operation_no_automatic_replay")
        if replace(problem).input_sha256 != problem.input_sha256 or image_identities(problem) != cache["images"]:
            raise ValueError("operation_images_changed")
        write_once(intent, {"identity": identity, "started_at": time.time()})
        spec = {"problem": problem.public_dict(), "skill_text": text,
                "model": self.profile["model"], "solver": config,
                "run_root": str(self.root), "prepared_root": self.profile["inputs"]["prepared_root"],
                "artifact_sha256": cache["artifact_sha256"], "operation_id": op_id,
                "category": "gate" if stage.startswith("gate-") else "solver", "mode": mode,
                "max_requests": self.profile["runtime"]["max_requests"],
                "request_concurrency": self.profile["runtime"]["request_concurrency"]}
        # A spawn worker never inherits host label dictionaries or runs SIGALRM in a thread.
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
                result = pool.submit(solve_worker, spec).result()
        except BaseException as exc:
            write_once(destination, {**identity, "identity": identity, "status": "stopped",
                                     "error_type": type(exc).__name__})
            raise
        request_records = {str(p.relative_to(self.root)): file_sha(p)
                           for p in sorted((self.root/"requests"/op_id).glob("*.json"))}
        saved = write_once(destination, {**identity, "identity": identity, **result,
                                         "request_records": request_records})
        if saved["status"] != "complete":
            raise RuntimeError("operational_failure_no_automatic_retry")
        print(json.dumps({"stage": stage, "sample_id": problem.sample_id,
                          "status": result["result"]["status"],
                          "request_count": result["result"]["request_count"]}), flush=True)
        return saved

    def revision(self, parent, pending, references, attempt):
        path = self.root/"skills"/f"attempt-{attempt:03d}"/"revision.json"
        training_identity = [r["content_sha256"] for r in pending]
        if path.exists():
            saved = read(path)
            if (saved["parent_sha256"] != sha_value(parent)
                    or saved["training_results"] != training_identity):
                raise ValueError("revision_parent_changed")
            if any(file_sha(self.root/p) != h for p, h in saved["request_records"].items()):
                raise ValueError("revision_request_changed")
            return saved
        intent = path.with_name("revision.intent.json")
        if intent.exists():
            raise RuntimeError("unfinished_revision_no_replay")
        outcome = metrics(pending, references)
        outcome["result_statuses"] = dict(Counter(r["result"]["status"] for r in pending))
        request = {"task_type": "Multi-view_Reasoning", "current_skill": parent,
                   "allowed_tools": TOOLS, "aggregate_training_feedback": outcome}
        write_once(intent, {"parent_sha256": sha_value(parent), "feedback_sha256": sha_value(outcome),
                            "training_results": training_identity})
        client = RecordedClient(BailianClient(**self.profile["model"]), self.governor,
                                self.root, f"revision-{attempt:03d}", "reviser")
        text = client.chat([
            {"role": "system", "content": (
                "Improve the reusable spatial reasoning Skill from aggregate training feedback. "
                "Keep its per-view evidence boundaries and Python solve(ctx)/ReturnAnswer contract. "
                "Do not copy sample identifiers, questions or answers. Do not change model, tools, "
                'or budget. Return JSON {"skill_text":"complete replacement"} or {"no_change":true}. '
                "Keep the replacement between 100 and 12000 characters.")},
            {"role": "user", "content": json.dumps(request)}],
            max_tokens=2048, seed=137, temperature=0, enable_thinking=False,
            response_format={"type": "json_object"})
        try:
            parsed = strict_json(text)
            if parsed == {"no_change": True}:
                candidate, status = parent, "no_change"
            elif (isinstance(parsed, dict) and set(parsed) == {"skill_text"}
                  and isinstance(parsed["skill_text"], str)
                  and 100 <= len(parsed["skill_text"]) <= 12000):
                candidate, status = parsed["skill_text"], "candidate"
                if any(p.sample_id in candidate or p.question in candidate for p in self.problems):
                    raise ValueError("sample_text_in_reusable_skill")
                if candidate == parent:
                    status = "no_change"
            else:
                raise ValueError("invalid_revision")
        except (ValueError, TypeError):
            candidate, status = parent, "invalid_revision"
        return write_once(path, {"parent_sha256": sha_value(parent), "candidate": candidate,
                                 "status": status, "aggregate_feedback": outcome,
                                 "training_results": training_identity,
                                 "request_records": {
                                     str(p.relative_to(self.root)): file_sha(p)
                                     for p in sorted((self.root/"requests"/f"revision-{attempt:03d}").glob("*.json"))},
                                 "individual_samples_visible": False, "label_access": True,
                                 "label_access_scope": "aggregate_training_scores",
                                 **response_record(client, text)})

    def train_skill(self, seed_text):
        references = self.references("train")
        parent, pending, since, attempt = seed_text, [], 0, 0
        for problem in self.train:
            receipt = self.operation(problem, parent, "training")
            pending.append(receipt)
            since += 1
            if since != 12:
                continue
            attempt += 1
            revision = self.revision(parent, pending, references, attempt)
            candidate = revision["candidate"]
            if revision["status"] == "candidate":
                selected = random.Random(137+attempt).sample(self.train, 8)
                pairs = []
                for seed in (137, 138):
                    for p in selected:
                        a = self.operation(p, parent, f"gate-{attempt:03d}-parent", seed)
                        b = self.operation(p, candidate, f"gate-{attempt:03d}-candidate", seed)
                        pairs.append((a, b))
                decision = gate_decision(pairs, references)
            else:
                selected, decision = [], {"promoted": False, "reason": revision["status"]}
            write_once(self.root/"skills"/f"attempt-{attempt:03d}"/"decision.json", {
                **decision, "parent_sha256": sha_value(parent),
                "candidate_sha256": sha_value(candidate),
                "selected_train_ids": [p.sample_id for p in selected]})
            if decision["promoted"]:
                parent, pending = candidate, []
            since = 0
        return write_once(self.root/"skills/frozen.json", {
            "text": parent, "training_questions": len(self.train), "attempts": attempt,
            "unrevised_tail_questions": since})["text"]

    def run(self):
        self.verify()
        initial = self.inputs["seed"]["text"]
        groups = {}
        for stage, text, mode in (("direct", "", "direct"), ("no_skill", "", "system"),
                                  ("initial", initial, "system")):
            groups[stage] = [self.operation(p, text, stage, mode=mode) for p in self.tests]
        final_text = self.train_skill(initial)
        groups["final"] = [self.operation(p, final_text, "final") for p in self.tests]
        # Test references are opened only after every method and prediction is frozen.
        references = self.references("test")
        for p, a, b, n, direct in zip(self.tests, groups["initial"], groups["final"],
                                    groups["no_skill"], groups["direct"]):
            for r in (a, b, n, direct):
                if r["input_sha256"] != p.input_sha256 or r["images"] != a["images"]:
                    raise ValueError("final_comparison_input_mismatch")
        scores = {name: metrics(rows, references) for name, rows in groups.items()}
        costs = {}
        for name, rows in groups.items():
            rounds = [q for r in rows for q in r["result"]["rounds"]]
            costs[name] = {"requests": sum(r["result"]["request_count"] for r in rows),
                           "usage": {key: sum(q.get("usage", {}).get(key, 0) for q in rounds)
                                     for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
        self.verify()
        return write_once(self.root/"completion.json", {
            "status": "complete", "schema": SCHEMA, "campaign_sha256": self.manifest["content_sha256"],
            "scores": scores, "costs": costs, "frozen_skill_sha256": sha_value(final_text),
            "initial_final_accuracy_delta": scores["final"]["accuracy"]-scores["initial"]["accuracy"],
            "request_usage": self.governor.summary(), "test_used_for_selection": False,
            "result_files": {str(p.relative_to(self.root)): file_sha(p)
                             for directory in ("results", "skills", "requests", "intents")
                             for p in sorted((self.root/directory).rglob("*.json"))}})


def source_files(repo):
    paths = sorted((repo/"src/skill3d").rglob("*.py"))
    paths.extend([
        repo/"scripts/run_blink_experiment.py",
        repo/"configs/experiments/blink.json",
    ])
    return {str(p.relative_to(repo)): file_sha(p) for p in paths}


def create_run(repo, root, profile, inspection):
    if root.exists():
        raise ValueError("new_run_requires_new_directory")
    if profile["runtime"]["max_requests"] < 1:
        raise ValueError("explicit_request_budget_required")
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if status.strip():
        raise ValueError("formal_source_must_be_committed_and_clean")
    if file_identity(profile["inputs"]["host_answers"]) != inspection["inputs"]["host_answers"]:
        raise ValueError("reference_changed_after_preflight")
    reference_sha = file_sha(profile["inputs"]["host_answers"])
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    sources = source_files(repo)
    # Copy only package/entry source; data, secrets and unrelated BLINK runs are excluded.
    for relative in sources:
        target = root/"code/source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/relative, target)
        if file_sha(target) != sources[relative]:
            raise ValueError("source_changed_during_freeze")
    manifest = write_once(root/"manifest.json", {
        "schema": SCHEMA, "profile": profile, "preflight": inspection,
        "source_files": sources, "git_commit": commit, "reference_file_sha256": reference_sha,
    })
    if source_files(repo) != sources:
        raise ValueError("source_changed_during_freeze")
    return manifest
