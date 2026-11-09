#!/usr/bin/env python3
"""Isolated, one-request Direct comparator using verified official Qwen frames."""
from __future__ import annotations

import argparse
import ast
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

from skill3d.online.config import load_config, online_model_from, resolve_runtime
from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas import CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.eval_visual_fallback import (
    EVAL_VISUAL_PROMPT_VERSION, build_visual_request, parse_visual_answer,
)
from export_pilot_trajectories import export_question, write_json
from panel_contract import validate_panel

INPUT_ID = "a66888843994de92d4435882c81b7a9e6123f746aa9edc327d73c3ab9da2b471"
TOOL_SHA = "f3710ced8ffe735da57d08cf5236cf9d326ff646ed1bdad2e62593a7801adc33"
ROOT = Path(__file__).resolve().parents[1]
SECRET_NAMES = ("DASHSCOPE_API_KEY", "QWEN3VL8B_API_KEY", "VSI_SAM2_TOKEN")


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest()


def frame_validator(path):
    """Load only the pure validation function; never import service/auth setup."""
    source = Path(path).read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "validate_frames")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "base64": base64, "hashlib": hashlib, "io": io, "digest": digest,
        "INPUT_ID": INPUT_ID, "TOOL_SHA": TOOL_SHA,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["validate_frames"]


def preflight(args):
    sampling = read(args.sampling)
    rows = validate_panel(sampling, args.expected_questions)
    public = read(args.official_root / "public_inputs.json")
    if (public["schema"] != "qwen-official32-public-inputs/1"
            or public["identity_sha256"] != INPUT_ID
            or digest({k: v for k, v in public.items() if k != "identity_sha256"}) != INPUT_ID
            or public["reference_answer_visible"] is not False
            or public["real_labels_available"] is not False):
        raise ValueError("official public-input identity mismatch")
    expected_keys = {
        "qa_id", "scene_id", "dataset", "question_type", "question", "options",
        "source_split", "experiment_split", "video_path", "public_question_sha256"}
    by_id = {}
    for row in public["questions"]:
        if (set(row) != expected_keys
                or digest({k: v for k, v in row.items()
                           if k != "public_question_sha256"}) != row["public_question_sha256"]):
            raise ValueError("official public question identity mismatch")
        qa = str(row["qa_id"])
        if qa in by_id:
            raise ValueError("duplicate official question")
        by_id[qa] = row
    if len(by_id) != 700:
        raise ValueError("official public-input scope changed")
    by_scene = {}
    for original in by_id.values():
        identity = {k: original[k] for k in ("dataset", "scene_id", "video_path")}
        scene = (original["dataset"], original["scene_id"])
        if scene in by_scene and by_scene[scene] != identity:
            raise ValueError("official scene video identity is inconsistent")
        by_scene[scene] = identity
    source_by_id = None
    if args.source_questions:
        if file_hash(args.source_questions) != sampling["source_question_sha256"]:
            raise ValueError("authoritative public question file changed")
        source = [json.loads(line) for line in args.source_questions.read_text().splitlines()
                  if line.strip()]
        if any(r.get(k) not in (None, "") for r in source for k in (
                "ground_truth", "reference_answer", "answer", "label")):
            raise ValueError("authoritative public source contains labels")
        source_by_id = {str(r["qa_id"]): r for r in source}
        if len(source_by_id) != len(source):
            raise ValueError("authoritative source has duplicate question identifiers")
    model = online_model_from(resolve_runtime(load_config(ROOT / "runner.yaml"), seed=137))
    if (model.model != "qwen3.8-flash" or model.temperature != 0.
            or model.max_tokens != 8192 or not model.enable_thinking
            or model.timeout_s != 900 or model.n_frames != 32):
        raise ValueError("Direct model settings differ from the current Harness arm")
    client = BailianClient(
        model=model.model, model_revision=model.model_revision, base_url=model.base_url,
        api_key_env=model.api_key_env, timeout_s=model.timeout_s,
        temperature=model.temperature, max_output_tokens=model.max_tokens)
    validate = frame_validator(args.validator)
    scenes, selected = {}, {}
    for row in rows:
        qa = str(row["qa_id"])
        if source_by_id is None:
            original = by_id[qa]
        else:
            if source_by_id[qa] != row:
                raise ValueError("sampled public question differs from authoritative source")
            scene = by_scene[(row["dataset"], row["scene_name"])]
            original = {
                **{k: row[k] for k in ("qa_id", "dataset", "question_type", "question",
                                      "source_split", "experiment_split")},
                "scene_id": row["scene_name"], "options": row.get("options") or [],
                "video_path": scene["video_path"],
            }
            original["public_question_sha256"] = digest(original)
        if (any(row[k] != original[k] for k in (
                "dataset", "question_type", "question", "source_split", "experiment_split"))
                or (row.get("options") or []) != (original["options"] or [])
                or row["scene_name"] != original["scene_id"]):
            raise ValueError("Direct/Harness public question identity mismatch")
        key = digest([original["dataset"], original["scene_id"]])[:24]
        if key not in scenes:
            frame_path = args.official_root / "frames" / f"{key}.json"
            receipt_path = args.official_root / "receipts" / f"{key}.json"
            identity = {k: original[k] for k in ("dataset", "scene_id", "video_path")}
            frames, receipt = read(frame_path), read(receipt_path)
            artifact_hash = validate(frames, identity)
            if (receipt["status"] != "complete" or receipt["n_images"] != 32
                    or receipt["reference_answer_visible"] is not False
                    or receipt["input_identity_sha256"] != INPUT_ID
                    or receipt["scene_key"] != key or receipt["scene"] != identity
                    or receipt["source_video_sha256"] != frames["source_video_sha256"]
                    or receipt["file_sha256"] != file_hash(frame_path)
                    or receipt["frame_artifact_sha256"] != artifact_hash):
                raise ValueError("official frame publication receipt mismatch")
            scenes[key] = {"images": frames["images"], "receipt": {
                "scene_key": key, "frame_artifact_sha256": artifact_hash,
                "frame_file_sha256": receipt["file_sha256"],
                "publication_receipt_sha256": file_hash(receipt_path),
                "source_video_sha256": frames["source_video_sha256"],
                "jpeg_sha256": frames["jpeg_sha256"],
                "source_frame_indices": frames["source_frame_indices"],
            }}
        task = canonical_task(row["question_type"])
        prompt, messages, messages_hash = build_visual_request(
            question=row["question"], options=row.get("options") or [],
            question_type=task, encoded_frames=scenes[key]["images"], max_images=32)
        payload = client.build_payload(
            messages, max_tokens=model.max_tokens, seed=137, temperature=0.,
            enable_thinking=True)
        selected[qa] = {
            "question": row, "task": task, "messages": messages,
            "preflight": {
                "qa_id": qa, "public_question_sha256": original["public_question_sha256"],
                "messages_sha256": messages_hash, "payload_sha256": digest(payload),
                "prompt_version": EVAL_VISUAL_PROMPT_VERSION, "prompt": prompt,
                **scenes[key]["receipt"],
            },
        }
    manifest = {
        "schema": "vsi40-official32-direct/1", "role": "Qwen Direct versus current Harness",
        "selected_qa_ids": list(selected), "sampling_sha256": file_hash(args.sampling),
        "public_input_identity_sha256": INPUT_ID,
        "official_frame_validator_sha256": file_hash(args.validator),
        "runner_source_sha256": file_hash(__file__),
        "prompt_source_sha256": file_hash(ROOT / "src/skill3d/synthesis/eval_visual_fallback.py"),
        "client_source_sha256": file_hash(ROOT / "src/skill3d/synthesis/bailian_client.py"),
        "config_sha256": file_hash(ROOT / "runner.yaml"),
        "model": model.model, "model_revision": model.model_revision,
        "endpoint_sha256": client.endpoint_hash, "seed": 137, "temperature": 0.,
        "max_tokens": 8192, "enable_thinking": True, "timeout_s": 900,
        "workers": args.workers, "model_requests_cap": len(rows),
        "per_question_request_cap": 1, "automatic_retry": False,
        "n_images": 32, "tool_calls": 0, "skill_delivery": False,
        "prepared_geometry_loaded": False, "overview_images": 0,
        "reference_answer_visible": False, "real_labels_available": False,
        "questions": {qa: value["preflight"] for qa, value in selected.items()},
    }
    if args.source_questions:
        manifest["source_question_sha256"] = file_hash(args.source_questions)
    manifest["identity_sha256"] = digest(manifest)
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=True)
    target = root / "runtime_identity.json"
    if target.exists() and read(target) != manifest:
        raise ValueError("Direct frozen preflight changed")
    if not target.exists():
        write_json(root / "sampling_manifest.json", sampling)
        write_json(target, manifest)
        write_json(root / "preflight_verified.json", {
            "at": time.time(), "manifest_identity_sha256": manifest["identity_sha256"],
            "questions_verified": len(rows), "scenes_verified": len(scenes),
            "model_requests": 0, "gpu_requests": 0, "reference_answers_read": False,
        })
    print(json.dumps({
        "phase": "preflight", "questions": len(rows), "scenes": len(scenes),
        "model_requests": 0, "manifest_sha256": manifest["identity_sha256"],
    }), flush=True)
    return selected, model, manifest


class DirectClient(BailianClient):
    def __init__(self, model, root, qa, expected):
        super().__init__(
            model=model.model, model_revision=model.model_revision, base_url=model.base_url,
            api_key_env=model.api_key_env, timeout_s=model.timeout_s,
            temperature=model.temperature, max_output_tokens=model.max_tokens)
        self.root, self.qa, self.expected = root, qa, expected
        self.native_opener, self._opener = self._opener, self.record
        self.issued = False

    def record(self, request, timeout):
        if self.issued:
            raise RuntimeError("Direct per-question request cap exceeded")
        self.issued = True
        payload = json.loads(request.data)
        if digest(payload) != self.expected:
            raise ValueError("Direct request changed after zero-request preflight")
        path = self.root / "solver_requests" / self.qa / "001.json"
        if path.exists():
            raise ValueError("Direct request already issued; automatic retry is forbidden")
        receipt = {
            "request": payload, "request_sha256": self.expected, "request_index": 1,
            "reference_answer_visible": False, "automatic_retry": False,
            "status": "started", "started_at": time.time(), "transport_timeout_s": timeout,
        }
        write_json(path, receipt)
        try:
            with self.native_opener(request, timeout=timeout) as response:
                raw = response.read()
                receipt["response_http_status"] = getattr(response, "status", 200)
            if any(os.environ.get(name, "").encode() in raw
                   for name in SECRET_NAMES if os.environ.get(name)):
                raise ValueError("provider response contains a credential")
            body = json.loads(raw)
            if not isinstance(body, dict) or not body.get("model"):
                raise ValueError("provider response lacks model identity")
            receipt.update(status="returned", raw_response=body,
                           raw_response_text=raw.decode("utf-8"),
                           raw_response_sha256=hashlib.sha256(raw).hexdigest())
        except Exception as exc:
            receipt.update(status="unavailable", error_type=type(exc).__name__)
            if hasattr(exc, "code"):
                receipt["http_status"] = int(exc.code)
            raise
        finally:
            receipt["elapsed_s"] = time.time() - receipt["started_at"]
            receipt["network_elapsed_s"] = receipt["elapsed_s"]
            write_json(path, receipt)
        return io.BytesIO(raw)


def execute(root, selected, model, manifest):
    stopped, guard = threading.Event(), threading.Lock()
    total = len(selected)
    budget = {
        "schema": "vsi40-direct-request-budget/1", "max_requests": total,
        "used": 0, "qa_ids": [], "automatic_retry": False,
    }
    state = {
        "status": "running", "selected": total, "completed": 0, "legal": 0,
        "operational_errors": 0, "active_requests": 0, "started_at": time.time(),
        "reference_answer_visible": False, "automatic_retry": False,
    }
    models = set()

    def publish():
        state["updated_at"] = time.time()
        write_json(root / "live_status.json", state)
        write_json(root / "request_budget.json", budget)

    def one(qa):
        with guard:
            if stopped.is_set():
                return
            budget["used"] += 1
            budget["qa_ids"].append(qa)
            if budget["used"] > total or len(set(budget["qa_ids"])) != budget["used"]:
                raise ValueError("Direct request budget violated")
            state["active_requests"] += 1
            publish()
        prepared = selected[qa]
        row, task = prepared["question"], prepared["task"]
        client = DirectClient(model, root, qa, prepared["preflight"]["payload_sha256"])
        failure = None
        result = {
            "qa_id": qa, "question_type": row["question_type"],
            "source_split": row["source_split"], "reference_answer_visible": False,
            "automatic_retry": False, "direct_manifest_sha256": manifest["identity_sha256"],
            "status": "finished",
        }
        outcome = {
            "answer": None, "predicted": None, "legal_answer": False,
            "answer_source": "qwen_direct_official32",
            "is_mca": CANONICAL_UNIT_BY_QUESTION_TYPE[task] == "option",
            "agent_rounds": 1, "rounds": [], "tool_calls": 0,
        }
        try:
            text = client.chat(
                prepared["messages"], max_tokens=8192, seed=137,
                temperature=0., enable_thinking=True)
            actual = client.last_response_identity["model"]
            with guard:
                models.add(actual)
                if len(models) != 1 or actual != model.model:
                    stopped.set()
                    raise ValueError("Direct response model identity changed")
            result.update(response_text=text, usage=client.last_usage,
                          response_model=actual, generation=client.last_generation)
            try:
                answer, replay = parse_visual_answer(
                    text, question_type=task, options=row.get("options") or [])
            except (ValueError, TypeError, AttributeError):
                result["answer_error"] = "invalid_answer"
            else:
                outcome.update(answer=answer.value, predicted=answer.value, legal_answer=True)
                result["derivation_replay"] = replay
        except Exception as exc:
            failure = {"stage": "direct_request", "error_type": type(exc).__name__,
                       "reference_answer_visible": False, "automatic_retry": False}
            result.update(status="failed", error=failure)
            stopped.set()
        result["outcome"] = outcome
        write_json(root / "outcomes" / f"{qa}.json", result)
        export_question(root, qa, status=result["status"], error=failure)
        with guard:
            state["active_requests"] -= 1
            state["completed"] += 1
            state["legal"] += int(outcome["legal_answer"])
            state["operational_errors"] += int(failure is not None)
            state["response_models"] = sorted(models)
            publish()
        print(json.dumps({"qa_id": qa, "legal": outcome["legal_answer"],
                          "completed": state["completed"], "status": result["status"]}), flush=True)

    with (root / ".direct-execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list((root / "solver_requests").glob("*/*.json")) or (root / "request_budget.json").exists():
            raise ValueError("Direct execution already started; automatic rerun is forbidden")
        publish()
        iterator = iter(selected)
        with ThreadPoolExecutor(max_workers=manifest["workers"]) as pool:
            active = {pool.submit(one, qa) for qa in
                      [next(iterator, None) for _ in range(manifest["workers"])] if qa}
            while active:
                done, active = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        future.result()
                    except Exception as exc:
                        stopped.set()
                        write_json(root / "scheduler_error.json", {
                            "error_type": type(exc).__name__, "automatic_retry": False})
                if not stopped.is_set():
                    for _ in done:
                        qa = next(iterator, None)
                        if qa:
                            active.add(pool.submit(one, qa))
        state["status"] = "stopped_on_operational_error" if stopped.is_set() else "complete"
        state["elapsed_s"] = time.time() - state["started_at"]
        publish()
    return 1 if stopped.is_set() else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--expected-questions", type=int, default=40)
    parser.add_argument("--source-questions", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between one and 32")
    selected, model, manifest = preflight(args)
    if args.execute:
        return execute(args.run_dir, selected, model, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
