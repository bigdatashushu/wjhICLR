"""Frozen 32-frame direct-Qwen requests for the v11 fit-set comparison."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable, Iterable, Sequence

from skill3d.adapters import vsibench_loader as vl
from skill3d.adapters.frame_set import with_pixel_identity
from skill3d.routing.task_classifier import canonical_task
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.eval_visual_fallback import (
    EVAL_VISUAL_PROMPT_VERSION,
    N_FRAMES,
    build_visual_request,
    parse_visual_answer,
)
from skill3d.synthesis.request_context import request_phase

DIRECT_INDEX_SCHEMA = "v11-direct-qwen-request-index/1.0"
DIRECT_REQUEST_SCHEMA = "v11-direct-qwen-request/1.0"
DIRECT_RESPONSE_SCHEMA = "v11-direct-qwen-response/1.0"
DIRECT_RUN_SCHEMA = "v11-direct-qwen-run/1.0"
EXPECTED_MODEL = "qwen3.8-flash"
EXPECTED_PROVIDER = "bailian"
DIRECT_PHASE = "direct_qwen_visual"


class DirectQwenError(ValueError):
    """The direct-Qwen panel or frozen input contract is invalid."""


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def content_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_identity(path: str | Path) -> dict:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(source.resolve()),
        "size": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def write_json_atomic(
    path: str | Path,
    value,
    *,
    sync_directory: bool = True,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    if not sync_directory:
        return
    sync_parent_directory(destination)


def sync_parent_directory(path: str | Path) -> None:
    destination = Path(path)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectQwenError(message)


def load_request_index(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(value.get("schema_version") == DIRECT_INDEX_SCHEMA,
             "unsupported direct-Qwen request index schema")
    contract = value.get("contract")
    rows = value.get("rows")
    _require(isinstance(contract, dict), "direct-Qwen contract must be a mapping")
    _require(isinstance(rows, list) and rows, "direct-Qwen rows must be non-empty")
    expected = {
        "model_provider": EXPECTED_PROVIDER,
        "model": EXPECTED_MODEL,
        "prompt_version": EVAL_VISUAL_PROMPT_VERSION,
        "seed": 137,
        "temperature": 0.0,
        "enable_thinking": False,
        "skills_visible": False,
        "tools_visible": False,
        "geometry_visible": False,
        "derived_images_visible": False,
        "reference_answer_visible": False,
    }
    for key, wanted in expected.items():
        _require(contract.get(key) == wanted,
                 f"direct-Qwen contract {key} must be {wanted!r}")
    seen: set[str] = set()
    normalized = []
    for raw in rows:
        _require(isinstance(raw, dict), "direct-Qwen row must be a mapping")
        qa_id = str(raw.get("qa_id") or "")
        _require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+", qa_id)),
                 f"invalid direct-Qwen qa_id: {qa_id!r}")
        _require(qa_id not in seen, f"duplicate direct-Qwen qa_id: {qa_id}")
        seen.add(qa_id)
        variants = raw.get("system_variant_ids")
        _require(isinstance(variants, list) and variants,
                 f"{qa_id}: system_variant_ids must be non-empty")
        normalized.append({
            "qa_id": qa_id,
            "scene_id": str(raw.get("scene_id") or ""),
            "question_type": canonical_task(str(raw.get("question_type") or "")),
            "split": str(raw.get("split") or ""),
            "system_variant_ids": [str(item) for item in variants],
        })
    return {**value, "rows": normalized}


def load_safe_metadata(path: str | Path) -> dict[str, dict]:
    """Project metadata to online-safe fields and discard reference answers."""
    rows: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw_id = raw.get("qa_id")
        if raw_id is None or raw_id == "":
            raw_id = raw.get("id")
        qa_id = "" if raw_id is None else str(raw_id)
        _require(qa_id and qa_id not in rows,
                 f"duplicate or empty metadata qa_id: {qa_id!r}")
        options = raw.get("options")
        _require(options is None or isinstance(options, list),
                 f"{qa_id}: metadata options must be a list or null")
        rows[qa_id] = {
            "qa_id": qa_id,
            "id": qa_id,
            "dataset": str(raw.get("dataset") or ""),
            "scene_name": str(raw.get("scene_name") or ""),
            "question_type": str(raw.get("question_type") or ""),
            "question": str(raw.get("question") or ""),
            "options": [str(item) for item in (options or [])],
        }
    return rows


def validate_metadata_row(index_row: dict, metadata_row: dict) -> None:
    qa_id = index_row["qa_id"]
    _require(metadata_row["qa_id"] == qa_id, f"{qa_id}: metadata identity changed")
    _require(metadata_row["scene_name"] == index_row["scene_id"],
             f"{qa_id}: scene identity changed")
    _require(canonical_task(metadata_row["question_type"])
             == index_row["question_type"],
             f"{qa_id}: question type changed")
    _require(bool(metadata_row["dataset"]) and bool(metadata_row["question"]),
             f"{qa_id}: incomplete safe metadata")


def load_input_receipt(scene_id: str, roots: Sequence[str | Path]) -> dict:
    matches = []
    for root in roots:
        path = Path(root) / f"{scene_id}.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            matches.append((path, value))
    _require(bool(matches), f"{scene_id}: frozen input receipt not found")
    fields = ("scene_id", "frame_set_hash", "pixel_hashes", "source_frame_indices")
    baseline = {key: matches[0][1].get(key) for key in fields}
    for path, value in matches[1:]:
        _require({key: value.get(key) for key in fields} == baseline,
                 f"{scene_id}: conflicting frozen input receipt at {path}")
    return {
        **baseline,
        "receipt_paths": [str(path.resolve()) for path, _ in matches],
    }


@dataclass(frozen=True)
class PreparedDirectRequest:
    qa_id: str
    scene_id: str
    question_type: str
    options: tuple[str, ...]
    messages: list[dict]
    preflight: dict


def prepare_direct_request(
    *,
    index_row: dict,
    metadata_row: dict,
    loaded: dict,
    input_receipt: dict,
    contract: dict,
    endpoint_sha256: str,
    max_tokens: int,
) -> PreparedDirectRequest:
    validate_metadata_row(index_row, metadata_row)
    qa_id = index_row["qa_id"]
    pixels = list(loaded.get("pixels") or ())
    planned = [int(item) for item in (loaded.get("planned") or ())]
    missing = [int(item) for item in (loaded.get("missing") or ())]
    _require(len(planned) == N_FRAMES and len(pixels) == N_FRAMES and not missing,
             f"{qa_id}: direct Qwen requires exactly 32 readable frozen frames")
    source = loaded.get("source")
    _require(source is not None, f"{qa_id}: video source is missing")
    frame_set = vl.build_frame_set(
        int(loaded["total"]), n_frames=N_FRAMES, fps=float(loaded["fps"]),
        episode_id=qa_id, dataset_id=metadata_row["dataset"],
        video_id=vl.video_id_for(metadata_row, source),
        scene_name=index_row["scene_id"], readable_frame_ids=planned,
    )
    frame_set = with_pixel_identity(frame_set, pixels)
    _require(frame_set.frame_set_hash == input_receipt.get("frame_set_hash"),
             f"{qa_id}: frozen frame_set_hash changed")
    _require(frame_set.source_frame_indices
             == input_receipt.get("source_frame_indices"),
             f"{qa_id}: frozen source frame indices changed")
    _require(frame_set.pixel_sha256 == input_receipt.get("pixel_hashes"),
             f"{qa_id}: frozen frame pixels changed")
    options = tuple(metadata_row["options"])
    prompt, messages, messages_sha256 = build_visual_request(
        question=metadata_row["question"], options=options,
        question_type=index_row["question_type"], frames=pixels,
        max_images=N_FRAMES,
    )
    request_identity = {
        "schema_version": DIRECT_REQUEST_SCHEMA,
        "provider": EXPECTED_PROVIDER,
        "model": EXPECTED_MODEL,
        "model_revision": str(contract.get("model_revision") or ""),
        "endpoint_sha256": str(endpoint_sha256),
        "prompt_version": EVAL_VISUAL_PROMPT_VERSION,
        "messages_sha256": messages_sha256,
        "seed": int(contract["seed"]),
        "temperature": float(contract["temperature"]),
        "max_tokens": int(max_tokens),
        "enable_thinking": False,
    }
    preflight = {
        "schema_version": DIRECT_REQUEST_SCHEMA,
        "qa_id": qa_id,
        "scene_id": index_row["scene_id"],
        "split": index_row["split"],
        "question_type": index_row["question_type"],
        "system_variant_ids": list(index_row["system_variant_ids"]),
        "prompt": prompt,
        "n_images": N_FRAMES,
        "frame_set_hash": frame_set.frame_set_hash,
        "source_frame_indices": list(frame_set.source_frame_indices),
        "pixel_sha256": list(frame_set.pixel_sha256),
        "input_receipt_paths": list(input_receipt["receipt_paths"]),
        "video_source": str(source.path),
        "messages_sha256": messages_sha256,
        "request_identity": request_identity,
        "request_sha256": content_sha256(request_identity),
        "reference_answer_visible": False,
    }
    return PreparedDirectRequest(
        qa_id=qa_id,
        scene_id=index_row["scene_id"],
        question_type=index_row["question_type"],
        options=options,
        messages=messages,
        preflight=preflight,
    )


def execute_direct_request(
    prepared: PreparedDirectRequest,
    *,
    client_factory: Callable[[], object],
) -> dict:
    identity = prepared.preflight["request_identity"]
    started_at = time.time()
    started = time.perf_counter()
    result = {
        "schema_version": DIRECT_RESPONSE_SCHEMA,
        "qa_id": prepared.qa_id,
        "scene_id": prepared.scene_id,
        "question_type": prepared.question_type,
        "request_sha256": prepared.preflight["request_sha256"],
        "messages_sha256": prepared.preflight["messages_sha256"],
        "requested_model": identity["model"],
        "started_at": started_at,
        "accepted": False,
        "api_response_received": False,
        "service_error": False,
    }
    try:
        client = client_factory()
        with request_phase(DIRECT_PHASE):
            text = client.chat(
                prepared.messages,
                max_tokens=int(identity["max_tokens"]),
                seed=int(identity["seed"]),
                temperature=float(identity["temperature"]),
                enable_thinking=False,
            )
    except Exception as exc:  # A direct-panel question is never retried.
        result.update(
            status="service_error" if isinstance(exc, ServiceUnavailable)
            else "client_error",
            service_error=isinstance(exc, ServiceUnavailable),
            error=f"{type(exc).__name__}: {exc}",
        )
    else:
        result.update(
            api_response_received=True,
            response_text=text,
            usage=dict(getattr(client, "last_usage", None) or {}),
            generation=dict(getattr(client, "last_generation", None) or {}),
            response_identity=dict(
                getattr(client, "last_response_identity", None) or {}),
        )
        try:
            answer, replay = parse_visual_answer(
                text, question_type=prepared.question_type,
                options=prepared.options)
        except (ValueError, TypeError, AttributeError) as exc:
            result.update(status="invalid_answer",
                          error=f"invalid_answer: {exc}")
        else:
            result.update(
                status="accepted",
                accepted=True,
                answer=answer.model_dump(mode="json"),
                derivation_replay=replay,
            )
    result["elapsed_s"] = time.perf_counter() - started
    result["completed_at"] = time.time()
    return result


def execute_batch(
    prepared: Sequence[PreparedDirectRequest],
    *,
    workers: int,
    client_factory: Callable[[], object],
    response_dir: str | Path,
) -> tuple[list[dict], dict]:
    _require(type(workers) is int and workers > 0, "workers must be positive")
    _require(bool(prepared), "direct-Qwen batch must be non-empty")
    root = Path(response_dir)
    started_at = time.time()
    started = time.perf_counter()

    def run_one(item: PreparedDirectRequest) -> dict:
        result = execute_direct_request(item, client_factory=client_factory)
        write_json_atomic(
            root / f"{item.qa_id}.json", result, sync_directory=False)
        return result

    results = []
    with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="direct-qwen") as executor:
        futures = {executor.submit(run_one, item): item.qa_id for item in prepared}
        for future in as_completed(futures):
            results.append(future.result())
    sync_parent_directory(root / "responses.json")
    results.sort(key=lambda row: row["qa_id"])
    elapsed = time.perf_counter() - started
    latencies = sorted(float(row["elapsed_s"]) for row in results)
    api_started_at = min(float(row["started_at"]) for row in results)
    api_completed_at = max(float(row["completed_at"]) for row in results)
    api_wall_elapsed = api_completed_at - api_started_at

    def percentile(fraction: float) -> float:
        index = max(0, min(len(latencies) - 1,
                           math.ceil(fraction * len(latencies)) - 1))
        return latencies[index]

    counts: dict[str, int] = {}
    for row in results:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    batch = {
        "schema_version": DIRECT_RUN_SCHEMA,
        "workers": workers,
        "n_requests": len(results),
        "started_at": started_at,
        "completed_at": time.time(),
        "wall_elapsed_s": elapsed,
        "throughput_requests_per_s": len(results) / elapsed if elapsed else 0.0,
        "api_wall_elapsed_s": api_wall_elapsed,
        "api_throughput_requests_per_s": (
            len(results) / api_wall_elapsed if api_wall_elapsed else 0.0),
        "receipt_overhead_s": max(0.0, elapsed - api_wall_elapsed),
        "latency_p50_s": percentile(0.50),
        "latency_p95_s": percentile(0.95),
        "status_counts": counts,
        "api_response_count": sum(
            bool(row.get("api_response_received")) for row in results),
        "service_error_count": sum(
            bool(row.get("service_error")) for row in results),
        "http_429_count": sum(
            "HTTP 429" in str(row.get("error") or "") for row in results),
        "http_5xx_count": sum(
            bool(re.search(r"HTTP 5\d\d", str(row.get("error") or "")))
            for row in results),
        "qa_ids": [row["qa_id"] for row in results],
    }
    return results, batch


def response_models(rows: Iterable[dict]) -> set[str]:
    return {
        str(row.get("response_identity", {}).get("model") or "")
        for row in rows if row.get("api_response_received")
    } - {""}
