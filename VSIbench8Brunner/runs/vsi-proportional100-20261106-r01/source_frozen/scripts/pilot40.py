#!/usr/bin/env python3
"""Fixed 40-question pilot with cached detections and complete native traces."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path
import time
import traceback
import urllib.request

import numpy as np

from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.online import runner
from skill3d.online.config import load_config
from skill3d.online.scene_overview import SceneOverviewStore
from skill3d.routing.task_classifier import canonical_task
from skill3d.segmentation.cached_presemantic import CachedPresemantic
from skill3d.segmentation.remote_sam2 import RemoteSAM2
from skill3d.segmentation.binding_validation import validate_binding
from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts
from skill3d.skills.registry import load_active_skills
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.trace.store import TraceStore
from export_pilot_trajectories import export_question
from panel_contract import validate_panel

ROOT = Path(__file__).resolve().parents[1]
CURRENT = {}
CPU_SLOT = None
API_SLOT = None
CPU_OWNED = False


@contextmanager
def cpu_admission():
    global CPU_OWNED
    if CPU_SLOT is None:
        yield
        return
    CPU_SLOT.acquire()
    CPU_OWNED = True
    try:
        yield
    finally:
        CPU_OWNED = False
        CPU_SLOT.release()


@contextmanager
def network_admission():
    """An episode waiting on Qwen does not occupy a CPU admission slot."""
    global CPU_OWNED
    released = CPU_SLOT is not None and CPU_OWNED
    if released:
        CPU_OWNED = False
        CPU_SLOT.release()
    try:
        if API_SLOT is not None:
            API_SLOT.acquire()
        try:
            yield
        finally:
            if API_SLOT is not None:
                API_SLOT.release()
    finally:
        if released:
            CPU_SLOT.acquire()
            CPU_OWNED = True


def serial(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(json_value(value), ensure_ascii=False,
                                indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(child) for child in value]
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "data": json_value(value.tolist())}
    if isinstance(value, float) and not math.isfinite(value):
        return {"$nonfinite": "nan" if math.isnan(value) else
                "positive_infinity" if value > 0 else "negative_infinity"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json_value(serial(value))


def install_execution_recorder(root):
    original = runner._execute_program
    counters = {}

    def record(*args, **kwargs):
        episode = args[0] if args else kwargs["episode"]
        qa_id = str(episode.qa_id)
        counters[qa_id] = counters.get(qa_id, 0) + 1
        path = root / "program_executions" / qa_id / f"{counters[qa_id]:03d}.json"
        receipt = {
            "qa_id": qa_id, "execution_index": counters[qa_id],
            "program_source": args[1].program_source if len(args) > 1
                              else kwargs["program"].program_source,
            "reference_answer_visible": False, "status": "started",
            "started_at": time.time(),
        }
        write(path, receipt)
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            receipt.update(status="failed", error_type=type(exc).__name__,
                           stack=[{"file": frame.filename, "line": frame.lineno,
                                   "function": frame.name}
                                  for frame in traceback.extract_tb(exc.__traceback__)])
            write(path, receipt)
            export_question(root, qa_id, status="failed", error={
                "stage": "program_execution", "error_type": type(exc).__name__,
                "stack": receipt["stack"]})
            raise
        kernel, cell, trace = result
        receipt.update({
            "status": "returned", "elapsed_s": time.time() - receipt["started_at"],
            "program_trace": trace,
            "authorization_receipts": list(kernel.authorization_receipts[
                kernel._cell_authorization_mark:]),
            "observation_reads": list(kernel.observation_reads[kernel._cell_read_mark:]),
            "episode_ledger": {
                "tool_calls": kernel.tool_calls, "tool_results": kernel.tool_results,
                "authorization_receipts": kernel.authorization_receipts,
                "observation_reads": kernel.observation_reads,
                "unresolved_result_ids": sorted(kernel.unresolved_result_ids),
            },
            "cell": {
                key: getattr(cell, key, None) for key in (
                    "stdout", "stdout_tail", "error_code", "error_type", "error_message",
                    "traceback_text", "yielded_result_ids", "yielded_reason")},
        })
        write(path, receipt)
        export_question(root, qa_id, incremental=True)
        return result

    runner._execute_program = record


class RecordingClient(BailianClient):
    def __init__(self, cfg, root):
        super().__init__(
            model=cfg.online_model, base_url=cfg.online_base_url,
            api_key_env=cfg.online_api_key_env,
            timeout_s=cfg.online_timeout_s,
            temperature=cfg.temperature, max_output_tokens=cfg.max_tokens,
            model_revision=cfg.online_model_revision)
        self.root, self.requests = Path(root), []
        self.run_root, self.qa_id = self.root.parents[1], self.root.name
        self.native_opener = self._opener
        self._opener = self.record

    def record(self, request, timeout):
        if len(self.requests) >= 12:
            raise RuntimeError("pilot episode model request cap exceeded")
        payload = json.loads(request.data)
        index = len(self.requests) + 1
        receipt = {
            "request": payload, "request_sha256": content_sha256(payload),
            "request_index": index, "reference_answer_visible": False,
            "automatic_retry": False, "status": "queued",
            "transport_timeout_s": self.timeout_s, "queued_at": time.time(),
        }
        self.requests.append(receipt)
        target = self.root / f"{index:03d}.json"
        write(target, receipt)
        export_question(self.run_root, self.qa_id, incremental=True)
        try:
            with network_admission():
                receipt.update(status="started", started_at=time.time())
                write(target, receipt)
                with self.native_opener(request, timeout=timeout) as response:
                    raw = response.read()
                    receipt["response_http_status"] = getattr(response, "status", 200)
                receipt["network_elapsed_s"] = time.time() - receipt["started_at"]
            secret = os.environ.get(self.api_key_env, "")
            if secret and secret.encode() in raw:
                raise ValueError("provider response contains credentials")
            import hashlib
            receipt.update(raw_response=json.loads(raw), status="returned",
                           raw_response_text=raw.decode("utf-8"),
                           raw_response_sha256=hashlib.sha256(raw).hexdigest())
            receipt["elapsed_s"] = time.time() - receipt.get("started_at", receipt["queued_at"])
            write(target, receipt)
        except Exception as exc:
            receipt.update(status="unavailable", error_type=type(exc).__name__)
            receipt["elapsed_s"] = time.time() - receipt.get("started_at", receipt["queued_at"])
            from urllib.error import HTTPError
            if isinstance(exc, HTTPError):
                receipt["http_status"] = int(exc.code)
                error_body = exc.read()
                secret = os.environ.get(self.api_key_env, "")
                if secret and secret.encode() in error_body:
                    receipt["error_response_omitted"] = "credential_detected"
                else:
                    receipt["raw_error_response_text"] = error_body.decode("utf-8", errors="replace")
            write(target, receipt)
            export_question(self.run_root, self.qa_id, status="failed")
            raise
        export_question(self.run_root, self.qa_id, incremental=True)
        return io.BytesIO(raw)


def base_config():
    spec = importlib.util.spec_from_file_location("pilot_fitset", ROOT / "scripts/run_fitset_v11.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module._base_config(load_config(ROOT / "runner.yaml"))
    cfg.skills, warnings, snapshot = load_active_skills(ROOT / "skill_library/snapshots")
    if warnings or not cfg.skills or not cfg.online_enable_thinking:
        raise RuntimeError("pilot Skill or thinking configuration unavailable")
    cfg.active_snapshot_ref = snapshot
    return cfg, snapshot


def variant(question, entry):
    return {
        "qa_id": str(question["qa_id"]), "scene_id": entry["scene_id"],
        "split": entry["split"], "source_split": entry["split"],
        "artifact_ref": entry["reference"]["scene_artifact_ref"],
        "question_type": canonical_task(question["question_type"]),
    }


def save_binding(root, qa_id, binding):
    write(root / "bindings" / f"{qa_id}.json", {
        "objects": binding.objects, "stats": binding.stats, "notes": binding.notes,
        "materialized": binding.materialized,
        "prepared_measurements": binding.prepared_measurements,
    })


def load_binding(root, qa_id):
    from skill3d.schemas.reconstruction import ObjectRecord
    value = json.loads((root / "bindings" / f"{qa_id}.json").read_text())
    value["objects"] = [ObjectRecord.model_validate(row) for row in value["objects"]]
    return runner.PreparedObjectBinding(**value)


def reuse_unavailable_source_parser(binding, reference):
    """Preserve sealed missing object evidence without issuing a new parser call."""
    parser = binding.stats.get("object_parse") or {}
    if parser.get("status") != "unavailable":
        return None
    if (binding.materialized or binding.objects or binding.prepared_measurements
            or parser.get("objects") or binding.stats.get("candidate_receipts")
            or binding.stats.get("detection_receipts")
            or (binding.stats.get("review_selection") or {}).get("selected_candidate_ids")
            or binding.stats.get("binding_error") or binding.stats.get("detector_fault")):
        raise ValueError("source parser unavailable contradicts published object evidence")
    generation = binding.stats["generation"]
    if generation["sha256"] != content_sha256(generation["identity"]):
        raise ValueError("source unavailable generation digest differs")
    return copy.deepcopy(binding), {
        "schema": "pilot-source-parser-unavailable-reuse/1",
        "source_reference_sha256": content_sha256(reference),
        "source_generation_sha256": generation["sha256"],
        "source_parser_status": "unavailable",
        "materialized": False, "n_objects": 0, "n_candidates": 0,
        "new_parser_requests": 0, "new_detector_requests": 0,
        "new_sam2_requests": 0, "new_description_requests": 0,
        "new_binary_review_requests": 0, "reference_answer_visible": False,
    }


def initialize(args):
    root = args.run_dir.absolute()
    root.mkdir(parents=True, exist_ok=True)
    CURRENT["root"] = root
    sampling = json.loads(args.sampling.read_text())
    validate_panel(sampling, args.expected_questions)
    cfg, snapshot = base_config()
    provider = PublishedFitSetInputProvider(args.published_inputs, max_scenes=1)
    overview = SceneOverviewStore(args.published_inputs.parent /
                                 "scene_overviews_20261031/manifest.json")
    predictor = RemoteSAM2(os.environ["VSI_SAM2_ENDPOINT"])
    review = cfg.semantic_review_config
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    with opener(review["endpoint"] + "/health", timeout=30) as response:
        health = json.load(response)
    if health["identity_sha256"] != review["expected_identity_sha256"]:
        raise ValueError("original reviewer identity changed")
    description = review["target_description"]
    with opener(urllib.request.Request(
            description["base_url"] + "/models",
            headers={"Authorization": "Bearer " + os.environ["QWEN3VL8B_API_KEY"]}),
            timeout=30) as response:
        models = json.load(response)
    if description["model"] not in {model["id"] for model in models["data"]}:
        raise ValueError("8B descriptor service is unavailable")
    if not os.environ.get(cfg.online_api_key_env):
        raise ValueError("main Qwen credential is unavailable")
    with opener(urllib.request.Request(
            cfg.online_base_url + "/models",
            headers={"Authorization": "Bearer " + os.environ[cfg.online_api_key_env]}),
            timeout=30) as response:
        raw = response.read()
    if os.environ[cfg.online_api_key_env].encode() in raw:
        raise ValueError("main Qwen auth response contains credentials")
    identity = {
        "schema": "vsi8b-pilot-runtime/1", "sampling_sha256": file_identity(args.sampling)["sha256"],
        "published_inputs_sha256": file_identity(args.published_inputs)["sha256"],
        "solver_model": cfg.online_model, "solver_enable_thinking": True,
        "description": description, "reviewer_identity_sha256": health["identity_sha256"],
        "sam2_identity_sha256": predictor.identity_sha256, "skill_snapshot": snapshot,
        "n_skills": len(cfg.skills), "reference_answer_visible": False,
        "description_review_concurrency": 2,
        "automatic_retry": False, "selected_questions": sampling["n_questions"],
        "new_detector_requests": 0,
        "new_reconstruction_requests": 0,
    }
    existing_identity = root / "runtime_identity.json"
    if existing_identity.exists():
        if json.loads(existing_identity.read_text()) != identity:
            raise ValueError("pilot runtime identity changed")
    else:
        write(existing_identity, identity)
    print(json.dumps({"status": "preflight_ready", **identity}), flush=True)
    return root, sampling, cfg, provider, overview, predictor


def prepare_scene(root, cfg, provider, predictor, questions, scene_id, qa_ids):
    receipt_path = root / "scene_audits" / f"{scene_id}.json"
    if receipt_path.exists():
        return qa_ids
    original_bindings, bindings = {}, {}
    for qa_id in provider.scene_scopes[scene_id]["qa_ids"]:
        entry = provider._source_entries[qa_id]
        scene = provider._scene(entry)
        item = next(row for row in scene.items if str(row.episode.qa_id) == qa_id)
        v = variant({"qa_id": qa_id, "question_type": item.episode.question_type}, entry)
        transport = provider._transport_entry(v, item, entry)
        from skill3d.segmentation.prepared_binding_store import load_prepared_binding_reference
        original_bindings[qa_id] = load_prepared_binding_reference(
            entry["reference"], scene_input_identity=transport["identity"],
            migration_receipt=transport["migration"],
            scene_artifact=str(transport["resolver"].artifact_path),
            migration_destination_root=str(transport["destination_root"]))
    for qa_id in sorted(qa_ids):
        CURRENT["qa_id"] = qa_id
        path = root / "bindings" / f"{qa_id}.json"
        if path.exists():
            bindings[qa_id] = load_binding(root, qa_id)
            continue
        entry = provider.entries[qa_id]
        v = variant(questions[qa_id], entry)
        item = provider(v)
        transport = provider.prepare_published(v, item)
        old = transport["prepared"]
        source_unavailable = reuse_unavailable_source_parser(old, entry["reference"])
        replay = CachedPresemantic(
            old, entry["reference"],
            allow_unavailable_parser=source_unavailable is not None)
        preparation_cfg = replace(copy.deepcopy(cfg), seed=int(
            entry["reference"]["request_identity"]["seed"]))
        llm = RecordingClient(preparation_cfg, root / "parser_requests" / qa_id)
        art = runner._prepare_m4_quality(
            transport["artifact"], item.pixels, persist_quality=False)
        before = len(predictor.receipts)
        values = runner._bind_objects_best_effort(
            art, item.pixels, None, item.episode, preparation_cfg, llm,
            m5_out_dir=str(root / "preparation" / qa_id),
            predictor=predictor, cached_presemantic=replay)
        binding = runner.PreparedObjectBinding(*values)
        save_binding(root, qa_id, binding)
        write(root / "sam2_receipts" / f"{qa_id}.json", predictor.receipts[before:])
        if binding.stats.get("binding_error") or binding.stats.get("detector_fault"):
            raise RuntimeError("candidate regeneration failed; preserved failure receipt")
        if llm.requests or binding.stats.get("actual_detector_requests"):
            raise RuntimeError("cached stage unexpectedly requested a model/detector")
        rows = binding.stats.get("candidate_receipts", [])
        failed = [row for row in rows if row.get("semantic_visual") and
                  row["semantic_visual"]["status"] != "observed"]
        if failed:
            raise RuntimeError("descriptor/reviewer failed; preserved native response")
        if source_unavailable is not None:
            if (binding.materialized or binding.objects or predictor.receipts[before:]
                    or (binding.stats.get("object_parse") or {}).get("status") != "unavailable"
                    or binding.stats.get("candidate_receipts")
                    or binding.stats.get("detection_receipts")):
                raise RuntimeError("missing source evidence was not preserved")
            receipt = {
                **source_unavailable[1],
                "current_generation_sha256": binding.stats["generation"]["sha256"],
                "current_binding_version": binding.stats["version"],
                "migration": "standard_generation_with_frozen_unavailable_parser",
            }
            write(root / "preparation" / qa_id / "source_unavailable_reuse.json", receipt)
            print(json.dumps({"status": "source_parser_unavailable_preserved",
                              "qa_id": qa_id, **receipt}), flush=True)
        bindings[qa_id] = binding
        print(json.dumps({
            "status": "question_prepared", "qa_id": qa_id,
            "reviewed": binding.stats.get("review_selection", {}).get("n_selected", 0),
            "objects": len(binding.objects)}), flush=True)
    audit_inputs = [(qa, bindings.get(qa, original_bindings[qa]).objects)
                    for qa in sorted(original_bindings)]
    audited, audits = audit_semantic_mask_conflicts(audit_inputs, scene_identity=scene_id)
    for qa_id, binding in bindings.items():
        binding.objects = [obj.model_copy(update={"binding_validation": validate_binding(
            obj, generation_sha256=binding.stats["generation"]["sha256"])})
            for obj in audited[qa_id]]
        binding.stats["semantic_conflict_audits"] = audits
        save_binding(root, qa_id, binding)
    write(receipt_path, {"scene_id": scene_id, "full_original_audit_qa_ids":
                        sorted(original_bindings), "refreshed_qa_ids": sorted(qa_ids),
                        "audits": audits, "reference_answer_visible": False})
    return qa_ids


def solve_question(root, cfg, provider, overview, question, qa_id, trace_store=None):
    CURRENT["qa_id"] = qa_id
    result_path = root / "outcomes" / f"{qa_id}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    entry = provider.entries[qa_id]
    if not (root / "scene_audits" / f"{entry['scene_id']}.json").exists():
        raise ValueError("scene audit is incomplete")
    # Recovery requires an explicit archived attempt; never resend an existing call.
    if list((root / "solver_requests" / qa_id).glob("*.json")):
        raise RuntimeError("unfinished solver attempt retained; automatic retry disabled")
    v = variant(question, entry)
    item = provider(v)
    transport = provider.prepare_published(v, item)
    local_cfg = copy.deepcopy(cfg)
    local_cfg.reuse_artifact = str(transport["artifact_path"])
    local_cfg.prepared_binding_seed = int(entry["reference"]["request_identity"]["seed"])
    local_cfg.scene_overview = overview.descriptor(entry["scene_id"], entry=entry)
    local_cfg.max_images = 33
    client = RecordingClient(local_cfg, root / "solver_requests" / qa_id)
    binding = load_binding(root, qa_id)
    trace_store = trace_store or TraceStore(root / "solver_trace" / qa_id)
    outcome = runner.run_episode(
        item.episode, item.pixels, local_cfg,
        trace_store=trace_store.bind(qa_id=qa_id, run_id=root.name), llm=client,
        prepared_binding=binding, artifact_relocation=transport["artifact_relocation"],
        original_item=item)
    result = {"qa_id": qa_id, "question_type": v["question_type"],
              "source_split": entry["split"], "reference_answer_visible": False,
              "transport_receipt": transport["transport_receipt"],
              "outcome": outcome, "native_requests": client.requests}
    write(result_path, result)
    export_question(root, qa_id)
    if any(row["status"] != "returned" for row in client.requests):
        raise RuntimeError("main solver operational failure; no automatic retry")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--published-inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=40)
    parser.add_argument("--preparation-request-cap", type=int, default=3884)
    parser.add_argument("--phase", choices=("preflight", "all", "solve"), default="all")
    args = parser.parse_args()
    root, sampling, cfg, provider, overview, predictor = initialize(args)
    if args.phase == "preflight":
        return
    from skill3d.infra.vsi_preparation_budget import ENV
    budget = root / "request_budget.json"
    if not budget.exists():
        write(budget, {"max_requests": args.preparation_request_cap, "used": 0, "requests": []})
    os.environ[ENV] = str(budget)

    questions = {str(row["qa_id"]): row for row in sampling["questions"]}
    scenes = {}
    for qa_id in questions:
        scenes.setdefault(provider.entries[qa_id]["scene_id"], []).append(qa_id)
    trace_store = TraceStore(root / "solver_trace")
    install_execution_recorder(root)
    started = time.time()
    completed = 0
    for scene_id, qa_ids in sorted(scenes.items()):
        receipt_path = root / "scene_audits" / f"{scene_id}.json"
        if not receipt_path.exists() and args.phase != "solve":
            prepare_scene(root, cfg, provider, predictor, questions, scene_id, qa_ids)
        if not receipt_path.exists():
            raise ValueError("scene audit is incomplete")
        for qa_id in sorted(qa_ids):
            CURRENT["qa_id"] = qa_id
            result_path = root / "outcomes" / f"{qa_id}.json"
            if result_path.exists():
                completed += 1
                continue
            result = solve_question(root, cfg, provider, overview, questions[qa_id], qa_id,
                                    trace_store=trace_store)
            outcome = result["outcome"]
            completed += 1
            write(root / "live_status.json", {
                "status": "running", "completed": completed, "total": sampling["n_questions"],
                "last_qa_id": qa_id, "elapsed_s": time.time() - started})
            print(json.dumps({"status": "question_solved", "qa_id": qa_id,
                              "completed": completed, "answer": outcome.answer,
                              "legal_answer": outcome.legal_answer}), flush=True)
    write(root / "live_status.json", {"status": "complete", "completed": completed,
                                      "total": sampling["n_questions"],
                                      "elapsed_s": time.time() - started})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error = {
            "error_type": type(exc).__name__,
            "stack": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                      for frame in traceback.extract_tb(exc.__traceback__)],
            "reference_answer_visible": False,
        }
        if CURRENT.get("root") and CURRENT.get("qa_id"):
            write(CURRENT["root"] / "host_errors" / f"{CURRENT['qa_id']}.json", error)
            export_question(CURRENT["root"], CURRENT["qa_id"], status="failed", error=error)
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__,
                          "error": "See the per-question JSON error record"}), flush=True)
        raise
