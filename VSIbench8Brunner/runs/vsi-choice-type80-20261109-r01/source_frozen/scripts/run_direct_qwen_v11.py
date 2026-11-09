#!/usr/bin/env python3
"""Run resumable frozen 32-frame qwen3.8-flash direct-answer requests."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill3d.adapters.episode_source import _load_from_sources
from skill3d.evaluation.direct_qwen_v11 import (
    DIRECT_RUN_SCHEMA,
    EXPECTED_MODEL,
    EXPECTED_PROVIDER,
    DirectQwenError,
    execute_batch,
    file_identity,
    import_reused_responses,
    load_input_receipt,
    load_request_index,
    load_safe_metadata,
    prepare_direct_request,
    response_models,
    write_json_atomic,
)
from skill3d.online.config import load_config, online_model_from, paths_from
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.eval_visual_fallback import encode_visual_frames


def _parse_positive_csv(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise DirectQwenError("concurrency levels must be integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise DirectQwenError("concurrency levels must be positive")
    return parsed


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _new_batch_path(root: Path) -> Path:
    existing = sorted((root / "batches").glob("*.json"))
    next_index = 1 + max(
        (int(path.stem) for path in existing if path.stem.isdigit()),
        default=0,
    )
    return root / "batches" / f"{next_index:04d}.json"


def _manifest(
    *,
    args,
    request_index: dict,
    selected_rows: list[dict],
    model,
    video_root: str,
    fallback_roots: list[str],
) -> dict:
    value = {
        "schema_version": DIRECT_RUN_SCHEMA,
        "role": "fit-set direct visual arm; qwen3.8-flash only",
        "request_index": file_identity(args.request_index),
        "metadata": file_identity(args.metadata),
        "config": file_identity(args.config),
        "contract": request_index["contract"],
        "selected_qa_ids": [row["qa_id"] for row in selected_rows],
        "video_root": str(Path(video_root).resolve()),
        "video_fallback_roots": [
            str(Path(path).resolve()) for path in fallback_roots],
        "input_receipt_roots": [
            str(Path(path).resolve()) for path in args.input_receipt_root],
        "reuse_run_roots": [
            str(Path(path).resolve()) for path in args.reuse_run_root],
        "request_model": model.model,
        "request_model_revision": model.model_revision,
        "endpoint_sha256": BailianClient(
            model=model.model,
            model_revision=model.model_revision,
            base_url=model.base_url,
            api_key_env=model.api_key_env,
            timeout_s=model.timeout_s,
            temperature=model.temperature,
            max_output_tokens=model.max_tokens,
        ).endpoint_hash,
        "seed": request_index["contract"]["seed"],
        "temperature": request_index["contract"]["temperature"],
        "max_tokens": model.max_tokens,
        "enable_thinking": False,
        "reference_answer_visible": False,
    }
    from skill3d.evaluation.direct_qwen_v11 import content_sha256

    value["identity_sha256"] = content_sha256(value)
    return value


def _load_existing_responses(root: Path, manifest: dict) -> dict[str, dict]:
    selected = set(manifest["selected_qa_ids"])
    responses = {}
    for path in sorted((root / "responses").glob("*.json")):
        value = _read_json(path)
        qa_id = str(value.get("qa_id") or "")
        if qa_id not in selected:
            raise DirectQwenError(f"response outside frozen panel: {path}")
        if qa_id in responses:
            raise DirectQwenError(f"duplicate direct-Qwen response: {qa_id}")
        preflight_path = root / "preflight" / f"{qa_id}.json"
        if not preflight_path.is_file():
            raise DirectQwenError(f"{qa_id}: response has no preflight receipt")
        preflight = _read_json(preflight_path)
        if value.get("request_sha256") != preflight.get("request_sha256"):
            raise DirectQwenError(f"{qa_id}: response request identity changed")
        responses[qa_id] = value
    return responses


class _SceneInputCache:
    def __init__(
        self,
        *,
        max_scenes: int,
        input_receipt_roots: list[str],
        video_root: str,
        fallback_roots: list[str],
    ) -> None:
        self.max_scenes = max_scenes
        self.input_receipt_roots = input_receipt_roots
        self.video_root = video_root
        self.fallback_roots = fallback_roots
        self._values: OrderedDict[tuple[str, str], dict] = OrderedDict()

    def get(self, index_row: dict, metadata_row: dict) -> dict:
        key = (metadata_row["dataset"], index_row["scene_id"])
        if key in self._values:
            value = self._values.pop(key)
            self._values[key] = value
            return value
        loaded, attempts = _load_from_sources(
            metadata_row, self.video_root, self.fallback_roots, 32)
        if loaded is None:
            raise DirectQwenError(
                f"{index_row['qa_id']}: frozen video input unavailable: {attempts}")
        value = {
            "loaded": loaded,
            "input_receipt": load_input_receipt(
                index_row["scene_id"], self.input_receipt_roots),
            "encoded_frames": encode_visual_frames(
                loaded["pixels"], max_images=32),
        }
        self._values[key] = value
        while len(self._values) > self.max_scenes:
            self._values.popitem(last=False)
        return value


def _prepare(
    *,
    index_row: dict,
    metadata_by_id: dict[str, dict],
    request_index: dict,
    scene_cache: _SceneInputCache,
    endpoint_sha256: str,
    max_tokens: int,
    output_root: Path,
):
    qa_id = index_row["qa_id"]
    metadata_row = metadata_by_id.get(qa_id)
    if metadata_row is None:
        raise DirectQwenError(f"{qa_id}: metadata row is missing")
    scene = scene_cache.get(index_row, metadata_row)
    prepared = prepare_direct_request(
        index_row=index_row,
        metadata_row=metadata_row,
        loaded=scene["loaded"],
        input_receipt=scene["input_receipt"],
        contract=request_index["contract"],
        endpoint_sha256=endpoint_sha256,
        max_tokens=max_tokens,
        encoded_frames=scene["encoded_frames"],
    )
    path = output_root / "preflight" / f"{qa_id}.json"
    if path.is_file():
        existing = _read_json(path)
        if existing.get("request_sha256") != prepared.preflight["request_sha256"]:
            raise DirectQwenError(f"{qa_id}: preflight request identity changed")
    else:
        write_json_atomic(path, prepared.preflight)
    return prepared


def _summary(root: Path, manifest: dict, *, status: str) -> dict:
    responses = _load_existing_responses(root, manifest)
    preflight_count = len(list((root / "preflight").glob("*.json")))
    reuse_path = root / "reuse_receipt.json"
    reused_count = (
        int(_read_json(reuse_path).get("imported_count", 0))
        if reuse_path.is_file() else 0)
    counts: dict[str, int] = {}
    for value in responses.values():
        key = str(value.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    batches = [_read_json(path) for path in sorted((root / "batches").glob("*.json"))]
    models = sorted(response_models(responses.values()))
    value = {
        "schema_version": DIRECT_RUN_SCHEMA,
        "status": status,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "selected_requests": len(manifest["selected_qa_ids"]),
        "preflight_requests": preflight_count,
        "reused_requests": reused_count,
        "completed_requests": len(responses),
        "pending_requests": len(manifest["selected_qa_ids"]) - len(responses),
        "status_counts": counts,
        "response_models": models,
        "model_identity_consistent": len(models) <= 1,
        "batches": batches,
        "updated_at": time.time(),
    }
    write_json_atomic(root / "summary.json", value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--request-index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--video-root")
    parser.add_argument("--video-fallback-root", action="append", default=[])
    parser.add_argument("--input-receipt-root", action="append", required=True)
    parser.add_argument("--reuse-run-root", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scene-cache-size", type=int, default=2)
    parser.add_argument(
        "--concurrency-levels", default="",
        help="comma-separated one-wave pilot, for example 1,2,4,8",
    )
    parser.add_argument(
        "--max-requests", type=int, default=0,
        help="maximum pending requests to process in this invocation",
    )
    parser.add_argument("--qa-ids", default="",
                        help="optional comma-separated frozen panel subset")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.workers <= 0 or args.scene_cache_size <= 0 or args.max_requests < 0:
        parser.error(
            "--workers/--scene-cache-size must be positive and "
            "--max-requests non-negative")
    try:
        levels = _parse_positive_csv(args.concurrency_levels)
        raw_config = load_config(args.config)
        model = online_model_from(raw_config)
        paths = paths_from(raw_config)
        if (not model.enabled or model.provider != EXPECTED_PROVIDER
                or model.model != EXPECTED_MODEL):
            raise DirectQwenError(
                "direct comparison requires enabled bailian/qwen3.8-flash")
        request_index = load_request_index(args.request_index)
        metadata_by_id = load_safe_metadata(args.metadata)
        wanted = {
            item for item in args.qa_ids.split(",") if item
        }
        if wanted:
            known = {row["qa_id"] for row in request_index["rows"]}
            unknown = wanted - known
            if unknown:
                raise DirectQwenError(
                    f"qa_ids outside request index: {sorted(unknown)}")
        selected_rows = [
            row for row in request_index["rows"]
            if not wanted or row["qa_id"] in wanted
        ]
        if not selected_rows:
            raise DirectQwenError("selected direct-Qwen panel is empty")
        video_root = args.video_root or paths.raw_videos
        fallback_roots = (
            list(args.video_fallback_root)
            if args.video_fallback_root else list(paths.raw_video_fallbacks)
        )
        root = Path(args.output_dir).resolve()
        manifest = _manifest(
            args=args, request_index=request_index,
            selected_rows=selected_rows, model=model,
            video_root=video_root, fallback_roots=fallback_roots,
        )
        manifest_path = root / "manifest.json"
        if root.exists():
            if not args.resume:
                raise DirectQwenError(
                    "--output-dir already exists; use --resume for the same run")
            if not manifest_path.is_file():
                raise DirectQwenError("existing output has no manifest")
            if _read_json(manifest_path) != manifest:
                raise DirectQwenError("direct-Qwen run manifest changed during resume")
        else:
            root.mkdir(parents=True)
            write_json_atomic(manifest_path, manifest)
        reuse_receipt_path = root / "reuse_receipt.json"
        if args.reuse_run_root and not reuse_receipt_path.is_file():
            import_reused_responses(
                output_root=root,
                selected_qa_ids=[row["qa_id"] for row in selected_rows],
                reuse_roots=args.reuse_run_root,
                target_manifest=manifest,
            )
        existing = _load_existing_responses(root, manifest)
        pending = [
            row for row in selected_rows if row["qa_id"] not in existing]
        if args.max_requests:
            pending = pending[:args.max_requests]
        if levels:
            needed = sum(levels)
            if len(pending) < needed:
                raise DirectQwenError(
                    f"concurrency pilot needs {needed} pending requests, "
                    f"only {len(pending)} remain")
            pending = pending[:needed]
            grouped = []
            offset = 0
            for level in levels:
                grouped.append((level, pending[offset:offset + level]))
                offset += level
        else:
            pending.sort(key=lambda row: (row["scene_id"], row["qa_id"]))
            grouped = [
                (args.workers, pending[offset:offset + args.workers])
                for offset in range(0, len(pending), args.workers)
            ]

        client_template = BailianClient(
            model=model.model,
            model_revision=model.model_revision,
            base_url=model.base_url,
            api_key_env=model.api_key_env,
            timeout_s=model.timeout_s,
            temperature=model.temperature,
            max_output_tokens=model.max_tokens,
        )
        if not args.preflight_only and not client_template.configured:
            raise DirectQwenError(f"{model.api_key_env} is not configured")

        def client_factory():
            return BailianClient(
                model=model.model,
                model_revision=model.model_revision,
                base_url=model.base_url,
                api_key_env=model.api_key_env,
                timeout_s=model.timeout_s,
                temperature=model.temperature,
                max_output_tokens=model.max_tokens,
            )

        scene_cache = _SceneInputCache(
            max_scenes=args.scene_cache_size,
            input_receipt_roots=args.input_receipt_root,
            video_root=video_root,
            fallback_roots=fallback_roots,
        )
        stopped_status = ""
        for workers, rows in grouped:
            prepared = [
                _prepare(
                    index_row=row,
                    metadata_by_id=metadata_by_id,
                    request_index=request_index,
                    scene_cache=scene_cache,
                    endpoint_sha256=client_template.endpoint_hash,
                    max_tokens=model.max_tokens,
                    output_root=root,
                )
                for row in rows
            ]
            if args.preflight_only:
                continue
            results, batch = execute_batch(
                prepared, workers=workers, client_factory=client_factory,
                response_dir=root / "responses")
            prior_models = response_models(existing.values())
            current_models = response_models(results)
            all_models = prior_models | current_models
            batch["response_models"] = sorted(current_models)
            batch["model_identity_consistent"] = len(all_models) <= 1
            write_json_atomic(_new_batch_path(root), batch)
            existing.update({row["qa_id"]: row for row in results})
            if len(all_models) > 1:
                _summary(root, manifest, status="blocked_model_identity_changed")
                raise DirectQwenError(
                    f"API response model identity changed: {sorted(all_models)}")
            if levels and (
                    batch["api_response_count"] != batch["n_requests"]
                    or batch["http_429_count"] or batch["http_5xx_count"]):
                stopped_status = "pilot_stopped_on_service_error"
                break

        status = stopped_status or ("preflight_complete" if args.preflight_only else (
            "complete" if len(existing) == len(selected_rows) else "partial")
        )
        summary = _summary(root, manifest, status=status)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except DirectQwenError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
