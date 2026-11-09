#!/usr/bin/env python3
"""Run the fixed choice-repair panel using sealed, already audited preparation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pilot40 as pilot


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources(root):
    manifest = read(root / "source_manifest.json")
    for row in manifest["files"]:
        path = root / "source_frozen" / row["file"]
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("frozen candidate source changed")


def verify_cache(root, *, hash_files):
    manifest = read(root / "cached_preparation_manifest.json")
    source = Path(manifest["source_root"])
    for index, row in enumerate(manifest["files"], 1):
        old, new = source / row["file"], root / "harness" / row["file"]
        for path in (old, new):
            stat = path.stat()
            if (stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]
                    or stat.st_ino != row["inode"] or stat.st_dev != row["device"]):
                raise ValueError("sealed preparation file identity changed")
        if hash_files and sha(new) != row["sha256"]:
            raise ValueError("sealed preparation bytes changed")
        if hash_files and index % 1000 == 0:
            print(json.dumps({"status": "cache_hash_verified", "files": index}), flush=True)
    return manifest


def initialize(args):
    root = args.run_dir.absolute()
    experiment = root.parent
    pilot.CURRENT["root"] = root
    sampling = read(args.sampling)
    pilot.validate_panel(sampling, args.expected_questions)
    spec = read(experiment / "experiment_spec.json")
    if (sampling["n_questions"] != 100 or sha(args.sampling) != spec["sampling_sha256"]
            or sha(args.published_inputs) != spec["published_inputs_sha256"]):
        raise ValueError("fixed panel or published source changed")
    cfg, snapshot = pilot.base_config()
    cfg.allow_final_test = True
    if (cfg.seed != 137 or cfg.temperature != 0 or cfg.max_tokens != 8192
            or cfg.online_model != "qwen3.8-flash" or not cfg.online_enable_thinking
            or cfg.online_timeout_s != 900 or cfg.max_solver_rounds != 6
            or not cfg.isolated_choice_visual_fallback or not cfg.eval_visual_fallback
            or cfg.evaluation_execution_role != "validation"
            or snapshot != spec["skill_snapshot"] or len(cfg.skills) != 24):
        raise ValueError("candidate model, budget or policy changed")
    verify_sources(experiment)
    cache = verify_cache(experiment, hash_files=False)
    identity = {
        "schema": "vsi-choice-cached-runtime/1",
        "sampling_sha256": spec["sampling_sha256"],
        "published_inputs_sha256": spec["published_inputs_sha256"],
        "source_manifest_sha256": sha(experiment / "source_manifest.json"),
        "cached_preparation_manifest_sha256": sha(
            experiment / "cached_preparation_manifest.json"),
        "solver_model": cfg.online_model, "solver_enable_thinking": True,
        "seed": cfg.seed, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
        "timeout_s": cfg.online_timeout_s, "allow_final_test": cfg.allow_final_test,
        **pilot.solver_policy_identity(cfg),
        "skill_snapshot": snapshot, "n_skills": len(cfg.skills),
        "preparation_policy": "read_only_frozen_original_preparation",
        "preparation_source_root": cache["source_root"],
        "reviewer_identity_sha256": spec["reviewer_identity_sha256"],
        "sam2_identity_sha256": spec["sam2_identity_sha256"],
        "description": spec["description"],
        "new_preparation_model_requests": 0, "new_gpu_requests": 0,
        "reference_answer_visible": False, "automatic_retry": False,
        "selected_questions": 100, "episode_model_request_cap": 12,
    }
    path = root / "runtime_identity.json"
    if path.exists() and read(path) != identity:
        raise ValueError("candidate runtime identity changed")
    if not path.exists():
        pilot.write(path, identity)
    provider = pilot.PublishedFitSetInputProvider(args.published_inputs, max_scenes=1)
    overview = pilot.SceneOverviewStore(
        args.published_inputs.parent / "scene_overviews_20261031/manifest.json")
    return root, sampling, cfg, provider, overview, SimpleNamespace(receipts=[])


def forbidden_prepare(*args, **kwargs):
    raise RuntimeError("cached-only experiment forbids preparation or cache writes")


def install_guard():
    original_record = pilot.RecordingClient.record
    original_load = pilot.load_binding

    def record(self, request, timeout):
        returned = original_record(self, request, timeout)
        if json.loads(returned.getvalue()).get("model") != "qwen3.8-flash":
            receipt = self.requests[-1]
            receipt.update(status="unavailable", error_type="ResponseModelIdentityChanged")
            pilot.write(self.root / f"{receipt['request_index']:03d}.json", receipt)
            raise RuntimeError("actual response model identity changed")
        return returned

    def load_binding(root, qa_id):
        manifest = read(root.parent / "cached_preparation_manifest.json")
        expected = next(row for row in manifest["files"]
                        if row["file"] == f"bindings/{qa_id}.json")
        if sha(root / expected["file"]) != expected["sha256"]:
            raise ValueError("cached binding bytes changed before solving")
        return original_load(root, qa_id)

    pilot.initialize = initialize
    pilot.prepare_scene = forbidden_prepare
    pilot.save_binding = forbidden_prepare
    pilot.RecordingClient.record = record
    pilot.load_binding = load_binding


def preflight(args):
    root, sampling, cfg, provider, overview, _ = initialize(args)
    verify_cache(root.parent, hash_files=True)
    source = Path(read(root.parent / "cached_preparation_manifest.json")["source_root"])
    old = {q["qa_id"]: q for q in read(source / "preflight_verified.json")["questions"]}
    verified = []
    for question in sorted(sampling["questions"], key=lambda q: (
            q["dataset"], q["scene_name"], int(q["qa_id"]))):
        qa = str(question["qa_id"])
        entry = provider.entries[qa]
        variant = pilot.variant(question, entry)
        item = provider(variant)
        episode = item.episode
        if (episode.qa_id != qa or episode.question != question["question"]
                or (episode.options or []) != (question.get("options") or [])
                or episode.dataset != question["dataset"]
                or episode.scene_name != question["scene_name"]
                or episode.split != question["source_split"] or episode.ground_truth != ""
                or pilot.canonical_task(episode.question_type) != variant["question_type"]):
            raise ValueError("published label-free public question identity changed")
        transport = provider.prepare_published(variant, item)
        descriptor = overview.descriptor(entry["scene_id"], entry=entry)
        audit = read(root / "scene_audits" / f"{entry['scene_id']}.json")
        if (qa not in audit["refreshed_qa_ids"]
                or audit["reference_answer_visible"] is not False):
            raise ValueError("original scene audit does not cover this question")
        identities = {
            "qa_id": qa, "scene_id": entry["scene_id"], "source_split": entry["split"],
            "frame_set_hash": episode.frame_set.frame_set_hash,
            "pixel_sha256": episode.frame_set.pixel_sha256,
            "source_reference_sha256": pilot.content_sha256(entry["reference"]),
            "binding_generation_sha256": transport["prepared"].stats["generation"]["sha256"],
        }
        if any(old[qa][key] != value for key, value in identities.items()):
            raise ValueError("paired frozen pixels or prepared source identity changed")
        verified.append({
            **identities, "overview_sha256": pilot.content_sha256(descriptor),
            "transport_receipt_sha256": pilot.content_sha256(transport["transport_receipt"]),
            "reference_answer_visible": False,
        })
        provider.cache.clear()
        provider._publication_files.clear()
        provider._scene_checks.clear()
        provider._scene_transport_scopes.clear()
        if len(verified) % 10 == 0:
            print(json.dumps({"status": "paired_preflight", "verified": len(verified),
                              "model_requests": 0, "gpu_requests": 0}), flush=True)
    pilot.write(root / "preflight_verified.json", {
        "schema": "vsi-choice-paired-zero-request-preflight/1",
        "questions_verified": 100, "scenes_verified": len(
            {q["scene_id"] for q in verified}),
        "sampling_sha256": sha(args.sampling),
        "source_manifest_sha256": sha(root.parent / "source_manifest.json"),
        "cached_preparation_manifest_sha256": sha(
            root.parent / "cached_preparation_manifest.json"),
        "questions": verified, "model_requests": 0, "gpu_requests": 0,
        "reference_answers_read": False, "reference_answer_visible": False,
    })


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "execute":
        sys.argv.pop(1)
        install_guard()
        import pilot40_parallel
        pilot40_parallel.main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "verify"))
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--published-inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=100)
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight(args)
    else:
        from verify_proportional_native import verify
        verify_sources(args.run_dir.parent)
        verify_cache(args.run_dir.parent, hash_files=True)
        sampling = read(args.sampling)
        verify(args.run_dir.parent, "harness",
               {str(q["qa_id"]): q for q in sampling["questions"]})


if __name__ == "__main__":
    main()
