#!/usr/bin/env python3
"""Fixed-panel metric preparation and frozen B01/B11 development validation.

Labels are absent from these inputs. Scores in the ablation engine are placeholders;
an independent process must join labels only after all solver output is frozen.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation import skill_ablation_v11 as ab
from skill3d.reconstruction import metric_evidence as me
from skill3d.reconstruction.metric_fusion import make_moge2_model
from skill3d.reconstruction_gate.scene_state import build_scene_state
from skill3d.schemas import ReconstructionArtifact, VSIBenchEpisode


def read(path):
    return json.loads(Path(path).read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def panel(path):
    value = read(path)
    require(value["split"] == "inner_validation" and value["language"] == "en"
            and value["arms"] == ["B01", "B11"]
            and value["frozen_skill"] == ab.SNAPSHOT_ID
            and value["allow_evolution"] is False
            and value["allow_final_test"] is False, "unsupported validation protocol")
    rows = value["questions"]
    require(rows and len({r["qa_id"] for r in rows}) == len(rows), "duplicate/empty panel")
    require(value["first_scene_qa_id"] == rows[0]["qa_id"], "first scene is not panel head")
    require(all(r["task"] in {"object_abs_distance", "object_size_estimation",
                             "room_size_estimation"} for r in rows), "non-metric task")
    return value


def load_input(archive, row):
    entry = read(archive / "input_manifest.json")[row["qa_id"]]
    path = Path(entry["path"])
    require(me.file_sha256(path) == entry["sha256"], "historical input file changed")
    value = read(path)
    require(ab._hash({k: v for k, v in value.items() if k != "sha256"}) == value["sha256"],
            "historical input payload changed")
    require(value["input_error"] is None, "frozen input failed; keep question, do not replace it")
    episode = VSIBenchEpisode.model_validate(value["episode"])
    require(episode.ground_truth == "" and episode.qa_id == row["qa_id"]
            and episode.scene_name == row["scene_name"]
            and episode.split == "inner_validation"
            and ab.canonical_task(episode.question_type) == row["task"],
            "label or panel identity mismatch")
    for ref in value["frame_files"]:
        require(me.file_sha256(ref) == value["frame_file_hashes"][Path(ref).name],
                "frozen frame file changed")
    pixels = [np.load(ref, allow_pickle=False) for ref in value["frame_files"]]
    require(ab._pixels_identity(pixels) == value["pixels"], "frozen pixels changed")
    require(me.source_identity(pixels, episode.frame_set) == value["cache_identity"],
            "frozen FrameSet changed")
    return EpisodeItem(episode=episode, pixels=pixels, source=value["source"],
                       video_path=value["video_path"]), value


def historical_geometry(archive, item, value):
    entry = read(archive / "artifact_manifest.json")[item.episode.qa_id]
    path = Path(entry["path"])
    require(me.file_sha256(path) == entry["sha256"], "historical artifact changed")
    confirmation = read(entry["quality_confirmation"])
    evidence_path = Path(confirmation["evidence_ref"])
    require(me.file_sha256(evidence_path) == confirmation["evidence_sha256"],
            "historical geometry evidence changed")
    evidence = read(evidence_path)
    identity = ab.input_identity(item, path)
    require(identity["artifact_json_sha256"] == evidence["artifact"]["sha256"]
            and identity["files"] == evidence["array_hashes"]
            and identity["pixels"] == evidence["input_pixels"] == value["pixels"],
            "historical geometry/pixels differ from evidence")
    art = ReconstructionArtifact.model_validate_json(path.read_bytes())
    require(art.source_frame_indices == evidence["source_frame_indices"]
            and art.quality.gate_thresholds == {"warp_inlier_ratio": .5,
                                              "cloud_overlap_ratio": .3},
            "historical frame/quality contract mismatch")
    return art, {"artifact_sha256": entry["sha256"],
                 "evidence_sha256": confirmation["evidence_sha256"],
                 "input_sha256": value["sha256"]}


def prepare(args, plan):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    archive = Path(args.archive_root)
    rows = plan["questions"][:1] if args.first_scene_only else plan["questions"]
    ab._write(output / "validation_plan.json", plan)
    manifest = {"panel_sha256": me.file_sha256(args.panel),
                "archive_manifests": {n: me.file_sha256(archive / n) for n in
                                     ("input_manifest.json", "artifact_manifest.json")},
                "artifacts": {}, "scenes": {}, "status": "preparing",
                "label_access": False}
    ab._write(output / "prepare_manifest.json", manifest)
    cache = read(Path(args.prepared_cache) / "prepare_manifest.json") if args.prepared_cache else {}
    if cache:
        require(cache["panel_sha256"] == manifest["panel_sha256"]
                and cache["archive_manifests"] == manifest["archive_manifests"]
                and cache["status"] == "completed", "prepared cache context changed")
    model = make_moge2_model(device="cuda:0")
    contract = me.fusion_contract(model)
    for row in rows:
        item, value = load_input(archive, row)
        old, provenance = historical_geometry(archive, item, value)
        key = value["cache_identity"]
        if key not in manifest["scenes"]:
            dest = output / "reconstructions" / key
            prior = cache.get("scenes", {}).get(key)
            if prior:
                require(me.file_sha256(prior["artifact"]) == prior["artifact_sha256"],
                        "prepared artifact changed")
                cached = ReconstructionArtifact.model_validate_json(Path(prior["artifact"]).read_bytes())
                receipt = me.validated_receipt(cached)
                require(receipt["request"]["contract"] == contract
                        and receipt["request"]["input_identity"] == key
                        and prior["provenance"] == provenance, "prepared metric identity changed")
                art = ab._copy_base(Path(prior["artifact"]), dest)
                source = "verified_prepared_cache"
            else:
                art = me.copy_geometry(old, dest).model_copy(
                    update={"reconstruction_input_identity": key})
                art = me.attach_metric_evidence(art, item.pixels, item.episode.frame_set,
                                                model, dest)
                source = "fresh_metric_inference"
            path = dest / "artifact.json"
            path.write_text(art.model_dump_json(indent=2))
            receipt = me.validated_receipt(art)
            scene = build_scene_state(art, question_type=row["task"])
            manifest["scenes"][key] = {
                "scene_name": row["scene_name"], "artifact": str(path),
                "artifact_sha256": me.file_sha256(path), "provenance": provenance,
                "source": source, "metric_status": art.scale_fusion_status,
                "metric_scale": art.metric_scale, "dispersion": art.scale_self_consistency,
                "valid_frame_ratio": receipt["valid_frame_ratio"],
                "frame_errors": [d for d in receipt["diagnostics"] if d["error"]],
                "resource": receipt["resource"], "scene": scene.model_dump(mode="json")}
            print(json.dumps(manifest["scenes"][key], default=str), flush=True)
        manifest["artifacts"][row["qa_id"]] = manifest["scenes"][key]["artifact"]
        ab.input_identity(item, manifest["artifacts"][row["qa_id"]])
        ab._write(output / "prepare_manifest.json", manifest)
    manifest["status"] = "completed"
    ab._write(output / "prepare_manifest.json", manifest)


@contextmanager
def raw_requests(output):
    """Observe the production recording client without replacing model creation."""
    original = ab._RecordingClient.chat
    directory = Path(output) / "full_requests"

    def chat(client, messages, max_tokens=8192, **kwargs):
        identity = client.store.identity
        path = directory / (ab._hash(identity)[:20] + f"-{len(client.requests):04d}.json")
        require(not path.exists(), "request audit path reused")
        row = {**identity, "messages": copy.deepcopy(messages),
               "request_sha256": ab._hash(messages), "phase": ab.current_request_phase(),
               "requested_max_tokens": max_tokens, "seed": kwargs.get("seed"),
               "actual_max_tokens": min(max_tokens, client.client.max_output_tokens),
               "temperature": client.client.temperature, "model": client.client.model,
               "status": "in_flight", "started_unix": time.time(),
               "audit_scope": "logical chat call; SDK transport retries unchanged"}
        ab._write(path, row)
        started = time.monotonic()
        try:
            answer = original(client, messages, max_tokens=max_tokens, **kwargs)
            row.update(status="completed", raw_output=answer)
            return answer
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            row.update(elapsed_s=time.monotonic() - started, usage=client.last_usage,
                       generation=client.last_generation)
            ab._write(path, row)

    ab._RecordingClient.chat = chat
    try:
        yield
    finally:
        ab._RecordingClient.chat = original


def solve(args, plan):
    from skill3d.online.config import (
        load_config, resolve_runtime, sandbox_from, active_vision_from, retrieval_policy_from)
    from skill3d.online.runner import OnlineRunConfig

    output = Path(args.output)
    prepared = read(output / "prepare_manifest.json")
    require(prepared["status"] == "completed"
            and prepared["panel_sha256"] == me.file_sha256(args.panel),
            "preparation incomplete or panel changed")
    require(set(prepared["artifacts"]) == {r["qa_id"] for r in plan["questions"]},
            "preparation does not cover exact panel")
    raw = resolve_runtime(load_config(args.config), seed=plan["seed"], formal=False)
    vision, sandbox, model = active_vision_from(raw), sandbox_from(raw), raw["vllm"]
    cfg = OnlineRunConfig(
        mode="real", seed=plan["seed"], vllm_endpoints=model["endpoints"],
        vllm_model=model["model"], max_images=model["n_frames"], max_pixels=model["max_pixels"],
        max_model_len=model["max_model_len"], max_tokens=model["max_tokens"],
        temperature=model["temperature"], cell_timeout_s=sandbox.cell_timeout_s,
        max_regen=sandbox.max_regenerate, max_solver_rounds=raw["max_solver_rounds"],
        max_retries_per_operation=raw["max_retries_per_operation"],
        finalization_rounds=raw["finalization_rounds"], eval_visual_fallback=raw["eval_visual_fallback"],
        image_layout=vision.layout, max_derived_images=vision.max_derived_images,
        retrieval_policy=retrieval_policy_from(raw),
        model_weights_sha256=os.environ.get("SKILL3D_MODEL_WEIGHTS_SHA256", ""),
        environment_sha256=os.environ.get("SKILL3D_ENVIRONMENT_SHA256", ""))
    require(all(getattr(cfg, k) == plan[k] for k in
                ("max_tokens", "max_solver_rounds", "max_retries_per_operation", "temperature")),
            "runtime budgets differ from preregistration")
    items = [load_input(Path(args.archive_root), row)[0] for row in plan["questions"]]
    for scene in prepared["scenes"].values():
        require(me.file_sha256(scene["artifact"]) == scene["artifact_sha256"],
                "prepared artifact changed before solve")
    with raw_requests(output):
        result = ab.run_skill_ablation_v11(
            items, artifact_paths=prepared["artifacts"], output_dir=output / "b01_b11",
            base_cfg=cfg, seed=plan["seed"], library_root=args.library_root,
            expected_qa_ids=[r["qa_id"] for r in plan["questions"]],
            question_types=sorted({r["task"] for r in plan["questions"]}))
    ab._write(output / "solve_receipt.json",
              {**result, "label_access": False, "scores_are_placeholders": True,
               "independent_scoring_required": True})
    require(result["status"] == "completed", "paired solver incomplete; inspect retained evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "solve"))
    parser.add_argument("--panel", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared-cache")
    parser.add_argument("--first-scene-only", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--library-root", default=str(ab.DEFAULT_LIBRARY))
    args = parser.parse_args()
    plan = panel(args.panel)
    (prepare if args.stage == "prepare" else solve)(args, plan)


if __name__ == "__main__":
    main()
