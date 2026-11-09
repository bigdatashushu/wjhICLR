"""Five single-attempt RGB+depth questions, isolated from camera-pose trials."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time
from types import FunctionType

from xcodeapi.config import Profile

from ..traces import digest, file_digest, read, write
from .depth_comparison import runner as shared
from .depth_comparison.depth import RENDERER_VERSION
from .depth_comparison.inputs import prepare_item, request, tokenizer, TOKENIZER_SHA256
from .depth_comparison.pool import BASE_URL, MODEL, Client, inference as original_inference
from .depth_prompt import DEPTH_NOTE, COMPACT_DEPTH_NOTE, VERSION as NOTE_VERSION
from .prototype.direct import operator_lock
from .prototype.inputs import public_problem
from .prototype.protocol import PARSER_VERSION, PROMPT_VERSION

SCHEMA = "mmsi-camobj-rgb-depth-five/1"
TASK = "Positional Relationship (Cam.–Obj.)"
PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
ITEM_FIELDS = {
    "index", "source_index", "problem", "images", "api_id", "arm_order",
    "prepared_directory", "artifact_sha256", "parent_direct_payload_sha256",
}


def inference(api):
    return {**original_inference(api), "timeout_seconds": 180}


class BoundedClient(Client):
    def __init__(self, api, key):
        super().__init__(api, key)
        self.inference = inference(api)
        self.profile = Profile("responses", BASE_URL, MODEL, "XCODE_GPT_API_KEY",
                               timeout_seconds=180)


def code_identity():
    files = [PACKAGE / name for name in (
        "__init__.py", "camobj_depth_trial.py", "camobj_depth_selfcheck.py",
        "depth_prompt.py", "depth_comparison/__init__.py",
        "depth_comparison/runner.py", "depth_comparison/inputs.py",
        "depth_comparison/depth.py", "depth_comparison/pool.py",
    )]
    files += list((PACKAGE / "prototype").glob("*.py"))
    files += [ROOT / "GPTexp" / name for name in (
        "__init__.py", "traces.py", "response_policy.py", "api_pool.py",
    )]
    files += list((ROOT / "GPTexp/vendor/xcodeapi").glob("*.py"))
    return {str(path.relative_to(ROOT)): file_digest(path) for path in sorted(files)}


def check_selection(value):
    items = value["items"]
    if (value["schema"] != SCHEMA
            or digest({k: v for k, v in value.items() if k != "selection_identity"})
            != value["selection_identity"]
            or value["questions"] != 5 or value["max_requests"] != 5
            or len(items) != 5 or len(set(value["api_ids"])) != 5
            or value["reference_answer_visible"] is not False
            or value["sample_role"] != "reused_development_diagnostic"):
        raise ValueError("five_question_selection_changed")
    ids = []
    for index, item in enumerate(items):
        problem = public_problem(item["problem"])
        ids.append(problem["sample_id"])
        if (set(item) != ITEM_FIELDS or item["index"] != index
                or problem["task_type"] != TASK or len(item["images"]) != 2
                or item["arm_order"] != ["depth"]
                or item["api_id"] != value["api_ids"][index]):
            raise ValueError("question_or_single_api_binding_changed")
    if len(set(ids)) != 5:
        raise ValueError("duplicate_sample")
    return value


def prepare(directory, selected, tokenizer_path):
    check_selection(selected)
    directory = Path(directory)
    if not directory.parent.is_dir() or directory.exists():
        raise ValueError("requires_new_run_directory")
    count = tokenizer(tokenizer_path)
    directory.mkdir()
    items = [prepare_item(directory, item, count) for item in selected["items"]]
    for item in items:
        if item["payload_sha256"]["direct"] != item["parent_direct_payload_sha256"]:
            raise ValueError("official_direct_input_changed")
    write(directory / "selection.json", selected, exclusive=True)
    manifest = {
        "schema": SCHEMA, "items": items, "questions": 5, "max_requests": 5,
        "max_inflight": 5, "api_ids": selected["api_ids"],
        "inference_by_api": {api: inference(api) for api in selected["api_ids"]},
        "selection_sha256": file_digest(directory / "selection.json"),
        "code_identity": code_identity(), "environment": shared.environment_identity(),
        "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
        "renderer_version": RENDERER_VERSION, "depth_note_version": NOTE_VERSION,
        "tokenizer": {"name": "o200k_base", "sha256": TOKENIZER_SHA256},
        "parent_evidence": selected["parent_evidence"],
        "sample_role": selected["sample_role"],
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "depth_images_sent": True, "pose_context_sent": False, "task_binding_sent": False,
        "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "questions": 5, "max_requests": 5,
        "original_images_verified": True, "depth_content_hashes_verified": True,
        "parent_direct_payloads_equal": True, "gpu_forwards": 0,
        "added_text_tokens": [item["added_text_tokens"] for item in items],
        "pose_context_sent": False, "task_binding_sent": False,
    }, exclusive=True)
    validate(directory, verify_requests=True)
    return manifest


def validate(directory, *, verify_requests=False):
    directory = Path(directory)
    selected = check_selection(read(directory / "selection.json"))
    manifest, preflight = read(directory / "manifest.json"), read(directory / "preflight.json")
    if (manifest["schema"] != SCHEMA
            or digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"]
            or manifest["selection_sha256"] != file_digest(directory / "selection.json")
            or manifest["code_identity"] != code_identity()
            or manifest["environment"] != shared.environment_identity()
            or manifest["api_ids"] != selected["api_ids"]
            or manifest["parent_evidence"] != selected["parent_evidence"]
            or manifest["questions"] != 5 or manifest["max_requests"] != 5
            or manifest["max_inflight"] != 5 or len(manifest["items"]) != 5
            or manifest["inference_by_api"] != {api: inference(api) for api in selected["api_ids"]}
            or manifest["prompt_version"] != PROMPT_VERSION
            or manifest["parser_version"] != PARSER_VERSION
            or manifest["renderer_version"] != RENDERER_VERSION
            or manifest["depth_note_version"] != NOTE_VERSION
            or manifest["tokenizer"] != {"name": "o200k_base", "sha256": TOKENIZER_SHA256}
            or manifest["depth_images_sent"] is not True
            or any(manifest[k] is not False for k in (
                "reference_answer_visible", "automatic_retry", "tools_enabled",
                "pose_context_sent", "task_binding_sent"))
            or preflight["run_identity"] != manifest["run_identity"]
            or preflight["model_requests"] != 0 or preflight["reference_answers_read"] is not False):
        raise ValueError("depth_only_configuration_changed")
    for item, source in zip(manifest["items"], selected["items"]):
        if (any(item[k] != source[k] for k in ITEM_FIELDS)
                or len(item["depth_images"]) != 2 or item["added_text_tokens"] > 160
                or item["annotation"]["fixed_tokens"] > 80
                or item["annotation"]["total_tokens"] > 160
                or item["payload_sha256"]["direct"] != source["parent_direct_payload_sha256"]):
            raise ValueError("depth_only_item_changed")
        # Counts were checked with the pinned tokenizer during zero-request preparation.
        # The immutable manifest binds them; execution enforces the literal note/labels.
        expected_labels = [f"RGB{i} {r['width']}x{r['height']}"
                           for i, r in enumerate(item["images"], 1)]
        if (item["annotation"]["note"] not in (DEPTH_NOTE, COMPACT_DEPTH_NOTE)
                or expected_labels != item["annotation"]["rgb_labels"]
                or ["Depth1", "Depth2"] != item["annotation"]["depth_labels"]):
            raise ValueError("supplemental_text_changed")
        for rgb, depth in zip(item["images"], item["depth_images"]):
            if (depth["width"] != rgb["width"] or depth["height"] != rgb["height"]
                    or depth["relative_only"] is not True
                    or file_digest(directory / depth["path"]) != depth["sha256"]):
                raise ValueError("depth_image_binding_changed")
        if verify_requests:
            for arm in ("direct", "depth"):
                if digest(request(directory, item, arm)) != item["payload_sha256"][arm]:
                    raise ValueError("depth_only_payload_changed")
    return manifest


_execute_namespace = {
    **vars(shared), "validate": validate, "request": request, "inference": inference,
}
execute = FunctionType(shared.execute.__code__, _execute_namespace, "execute")
execute.__kwdefaults__ = {"client_factory": BoundedClient}
_collect = FunctionType(shared.collect.__code__, {
    **vars(shared), "inference": inference,
}, "collect")


def collect(directory, manifest):
    records, files = _collect(directory, manifest)
    files["selection.json"] = file_digest(Path(directory) / "selection.json")
    return records, files


_freeze_namespace = {**vars(shared), "validate": validate, "collect": collect}
validate_frozen = FunctionType(shared.validate_frozen.__code__, _freeze_namespace, "validate_frozen")
_freeze_namespace["validate_frozen"] = validate_frozen
freeze = FunctionType(shared.freeze.__code__, _freeze_namespace, "freeze")


def score(directory, references, expected_sha256):
    from .prototype.evaluation import reference_answers
    with operator_lock(directory):
        manifest = validate(directory)
        frozen = validate_frozen(directory, manifest)
        if read(Path(directory) / "execution.json")["status"] != "complete":
            raise ValueError("requires_complete_frozen_run_before_reference_access")
        if file_digest(references) != expected_sha256:
            raise ValueError("host_reference_hash_changed")
        answers = reference_answers(Path(references).read_bytes(), frozen["predictions"])
        rows = [{**p, "reference_answer": answers[p["sample_id"]],
                 "correct": p["status"] == "answered" and p["answer"] == answers[p["sample_id"]]}
                for p in frozen["predictions"]]
        report = {
            "schema": SCHEMA + "/score", "run_identity": manifest["run_identity"],
            "freeze_identity": frozen["freeze_identity"],
            "reference_file_sha256": expected_sha256,
            "reference_access": "host_only_after_prediction_freeze",
            "overall": {"questions": 5, "legal_answers": sum(r["status"] == "answered" for r in rows),
                        "correct": sum(r["correct"] for r in rows),
                        "accuracy": sum(r["correct"] for r in rows) / 5},
            "predictions": rows, "scored_at": time.time(),
        }
        report["score_identity"] = digest(report)
        write(Path(directory) / "score.json", report, exclusive=True)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate", "execute", "freeze", "score"))
    parser.add_argument("--directory", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--tokenizer")
    parser.add_argument("--references")
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.directory, read(args.selection), args.tokenizer)
    elif args.phase == "validate":
        result = validate(args.directory, verify_requests=True)
    elif args.phase == "execute":
        result = execute(args.directory, __import__("json").loads(sys.stdin.buffer.read()))
    elif args.phase == "score":
        result = score(args.directory, args.references, args.reference_sha256)
    else:
        result = freeze(args.directory)
    print(__import__("json").dumps({
        **{k: result[k] for k in ("run_identity", "status", "dispatched", "overall") if k in result},
        "phase": args.phase,
    }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"status":"failed","reason_code":"camobj_depth_trial_failed"}', flush=True)
        raise SystemExit(2)
