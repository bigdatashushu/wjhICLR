"""Frozen B01/B11 evaluation. No evolution, active-pointer lookup or mock fallback.

M5 is prepared once per question, before either solver sees a Skill. Each solver
gets a physical copy of that tree and fresh runtime state. JSON receipts contain
content identities, not just filenames; results are joined by qa_id.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import uuid
from dataclasses import fields, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.adapters.frame_set import frame_set_hash
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding
from skill3d.online.submission import EXECUTION_PROTOCOL_VERSION
from skill3d.routing.task_classifier import canonical_task, classify
from skill3d.schemas import (
    ObjectRecord,
    ReconstructionArtifact,
    SkillEvaluationBinding,
    SkillSpecV11,
)
from skill3d.segmentation.open_vocab_detector import capture_detector_failures
from skill3d.skills.v11_library import (
    _manifest_for,
    canonical_json_sha256,
    resolve_source_like_ref,
    validate_v11_snapshot,
)
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION
from skill3d.tools.registry import TOOL_FACE_VERSION
from skill3d.trace.store import TraceStore

SNAPSHOT_ID = "S0-v11-contract-repair"
DEFAULT_LIBRARY = Path(__file__).resolve().parents[3] / "skill_library"
ARMS = ("B01", "B11")
ARTIFACT_REFS = (
    "c2w_list", "intrinsics", "depth_maps", "point_map", "point_conf",
    "depth_conf", "track_list", "per_frame_scale_ref",
)
OBJECT_REFS = ("mask_per_frame", "pointcloud_world", "pointconf_world", "centroid_ref")


class PairingError(ValueError):
    """Missing, duplicate or inconsistent experiment identity."""


def _clean(value):
    if hasattr(value, "model_dump"):
        return _clean(value.model_dump(mode="json"))
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _encoded(value) -> bytes:
    return json.dumps(_clean(value), ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _hash(value) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(_encoded(value) + b"\n")
    temp.replace(path)


def _index(rows, *, key="qa_id") -> dict:
    out = {}
    for row in rows:
        value = str(row[key])
        if not value or value in out:
            raise PairingError(f"empty or duplicate {key}: {value!r}")
        out[value] = row
    return out


def _same_ids(expected, actual, label: str) -> None:
    if set(expected) != set(actual):
        raise PairingError(
            f"{label}: missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}")


def _ref_path(ref: str, artifact_path: Path) -> Path:
    # Existing reconstruction writers use cwd-relative paths; portable manifests
    # can use artifact-relative paths. Reject ambiguity instead of guessing.
    p = Path(ref)
    choices = {p.resolve()} if p.is_absolute() else {
        p.resolve(), (artifact_path.parent / p).resolve()}
    found = [p for p in choices if p.is_file()]
    if len(found) != 1:
        raise PairingError(f"missing or ambiguous artifact reference: {ref!r}")
    return found[0]


def _artifact_files(art: ReconstructionArtifact, path: Path) -> dict[str, Path]:
    refs = {key: getattr(art, key) for key in ARTIFACT_REFS}
    refs.update({f"confidence.{key}": value
                 for key, value in art.confidence.model_dump().items()})
    return {key: _ref_path(value, path) for key, value in refs.items() if value}


def _pixels_identity(pixels) -> list[dict]:
    return [{"shape": list(p.shape), "dtype": str(p.dtype),
             "sha256": hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()}
            for p in pixels]


def input_identity(item: EpisodeItem, artifact_path: str | Path) -> dict:
    """Validate actual frame order, source identity and every declared artifact ref."""
    ep = item.episode
    fs = ep.frame_set
    path = Path(artifact_path).resolve()
    art = ReconstructionArtifact.model_validate_json(path.read_bytes())
    if not ep.qa_id or not ep.dataset or not ep.scene_name or fs is None:
        raise PairingError(f"{ep.qa_id}: missing question/source/frame identity")
    if not art.artifact_id or not art.artifact_version:
        raise PairingError(f"{ep.qa_id}: missing artifact identity")
    if fs.frame_set_hash != frame_set_hash(fs.frame_ids):
        raise PairingError(f"{ep.qa_id}: invalid frame_set_hash")
    if art.scene_name != ep.scene_name or art.frame_set_hash != fs.frame_set_hash:
        raise PairingError(f"{ep.qa_id}: artifact scene/frame_set_hash mismatch")
    if len(set(fs.readable_frame_ids)) != len(fs.readable_frame_ids):
        raise PairingError(f"{ep.qa_id}: duplicate readable frame IDs")
    slots = [fs.frame_ids.index(fid) for fid in fs.readable_frame_ids]
    if slots != sorted(slots):
        raise PairingError(f"{ep.qa_id}: readable frame order mismatch")
    for key, expected in (
        ("frame_ids", fs.readable_frame_ids),
        ("source_frame_indices", [fs.source_frame_indices[i] for i in slots]),
        ("timestamps", [fs.timestamps[i] for i in slots]),
    ):
        if list(getattr(art, key)) != list(expected):
            raise PairingError(f"{ep.qa_id}: artifact {key} mismatch")
    if len(item.pixels) != len(slots):
        raise PairingError(f"{ep.qa_id}: actual pixel count mismatch")
    if any(p.ndim != 3 or p.shape[2] != 3 or p.size == 0 for p in item.pixels):
        raise PairingError(f"{ep.qa_id}: invalid RGB pixels")
    for key, actual, expected in (
        ("dataset", fs.dataset_id, ep.dataset),
        ("scene", fs.scene_name, ep.scene_name),
        ("episode", fs.episode_id, ep.qa_id),
    ):
        if actual and actual != expected:
            raise PairingError(f"{ep.qa_id}: FrameSet {key} mismatch")
    files = _artifact_files(art, path)
    for key in ("depth_maps", "c2w_list", "point_map", "depth_conf"):
        if key in files:
            arr = np.load(files[key], mmap_mode="r", allow_pickle=False)
            if arr.shape[0] != len(slots):
                raise PairingError(f"{ep.qa_id}: {key} frame dimension mismatch")
    value = {
        "episode": ep.model_dump(mode="json"), "source": item.source,
        "video_path": item.video_path, "pixels": _pixels_identity(item.pixels),
        "artifact_path": str(path), "artifact_json_sha256": _file_hash(path),
        "artifact_id": art.artifact_id, "artifact_version": art.artifact_version,
        "files": {key: {"path": str(p), "sha256": _file_hash(p)}
                  for key, p in sorted(files.items())},
    }
    return {**value, "sha256": _hash(value)}


def input_error_identity(item: EpisodeItem) -> dict:
    """Content identity for a preregistered question with no readable image input."""
    if item.input_error is None:
        raise PairingError(f"{item.episode.qa_id}: missing input_error record")
    value = {
        "episode": item.episode.model_dump(mode="json"),
        "source": item.source,
        "input_error": item.input_error.model_dump(mode="json"),
    }
    return {**value, "sha256": _hash(value)}


def _copy_base(source: Path, dest: Path) -> ReconstructionArtifact:
    """Copy declared arrays only: old M5 caches must not enter the experiment."""
    art = ReconstructionArtifact.model_validate_json(source.read_bytes())
    data = art.model_dump()
    dest.mkdir(parents=True)
    for key, src in _artifact_files(art, source).items():
        target = dest / (key.replace(".", "_") + src.suffix)
        shutil.copyfile(src, target)
        if key.startswith("confidence."):
            data["confidence"][key.split(".")[1]] = str(target)
        else:
            data[key] = str(target)
    return ReconstructionArtifact.model_validate(data)


def _rebase(value, source: Path, dest: Path):
    if isinstance(value, str):
        return value.replace(str(source) + "/", str(dest) + "/")
    if isinstance(value, dict):
        return {k: _rebase(v, source, dest) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase(v, source, dest) for v in value]
    return value


def _save_binding(binding: PreparedObjectBinding, root: Path) -> None:
    def pack(value, key="stats"):
        if isinstance(value, np.ndarray):
            path = root / f"binding_{key}.npy"
            np.save(path, value)
            return {"__array_ref__": str(path)}
        if isinstance(value, dict):
            return {k: pack(v, f"{key}_{k}") for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [pack(v, f"{key}_{i}") for i, v in enumerate(value)]
        return value

    for obj in binding.objects:
        for name in OBJECT_REFS:
            ref = getattr(obj, name)
            if ref:
                p = Path(ref).resolve()
                if not p.is_relative_to(root) or not p.is_file():
                    raise PairingError(f"M5 object reference escapes private tree: {ref}")
    _write(root / "binding.json", {
        "objects": binding.objects, "stats": pack(binding.stats),
        "notes": binding.notes, "materialized": binding.materialized,
    })


def _load_binding(root: Path) -> PreparedObjectBinding:
    def unpack(value):
        if isinstance(value, dict):
            if set(value) == {"__array_ref__"}:
                return np.load(value["__array_ref__"], allow_pickle=False)
            return {k: unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [unpack(v) for v in value]
        return value

    data = unpack(json.loads((root / "binding.json").read_text()))
    data["objects"] = [ObjectRecord.model_validate(o) for o in data["objects"]]
    return PreparedObjectBinding(**data)


def tree_identity(root: Path) -> dict:
    """Path-independent identity, including all M5 caches and object arrays."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PairingError(f"symlink in frozen artifact tree: {path}")
        if not path.is_file():
            continue
        digest = (_hash(_rebase(json.loads(path.read_text()), root, Path("$TREE")))
                  if path.suffix == ".json" else _file_hash(path))
        result[path.relative_to(root).as_posix()] = digest
    return {"files": result, "sha256": _hash(result)}


def _clone_tree(source: Path, dest: Path) -> None:
    tree_identity(source)  # reject links before copying
    shutil.copytree(source, dest, copy_function=shutil.copyfile)
    for path in dest.rglob("*.json"):
        # Runtime geometry schema uses NaN for unavailable diagnostic values.
        # Keep that existing serialization contract; report JSON uses null.
        path.write_text(json.dumps(_rebase(json.loads(path.read_text()), source, dest),
                                   ensure_ascii=False, sort_keys=True), encoding="utf-8")


def quality_contract() -> dict:
    """Record current code/thresholds; never claim TODO defaults are calibrated."""
    from skill3d.reconstruction_gate import m4_main_gate, scene_state

    root = Path(m4_main_gate.__file__).parent
    value = {
        "enabled": True, "metric_version": m4_main_gate.M4_GATE_VERSION,
        "default_thresholds": m4_main_gate.default_thresholds(),
        "overall_quality_threshold": scene_state.TH_OVERALL_QUALITY,
        "source_sha256": {p.name: _file_hash(p) for p in sorted(root.glob("*.py"))},
    }
    return {**value, "sha256": _hash(value)}


def _common_config(cfg: OnlineRunConfig) -> dict:
    private = {"skills", "active_snapshot_ref", "active_snapshot_manifest_sha256",
               "trace_dir", "work_dir", "recon_dir", "reuse_artifact",
               "evaluation_binding"}
    value = {f.name: getattr(cfg, f.name) for f in fields(cfg)
             if f.name not in private and f.name != "retrieval_policy"}
    value["retrieval_policy"] = cfg.retrieval_policy.to_dict()
    value["temperature"] = 0.0
    value["prompt_template_version"] = PROMPT_TEMPLATE_VERSION
    value["execution_protocol_version"] = EXECUTION_PROTOCOL_VERSION
    value["tool_face_version"] = TOOL_FACE_VERSION
    value["tool_docs_version"] = TOOL_DOCS_VERSION
    return _clean(value)


class _BoundTrace(TraceStore):
    def __init__(self, root, identity):
        super().__init__(root)
        self.identity = identity

    def append(self, topic, record):
        data = record.model_dump(mode="json") if hasattr(record, "model_dump") else dict(record)
        super().append(topic, {**data, **self.identity})


class _RecordingClient:
    def __init__(self, client, *, seed, specs, store):
        self.client, self.seed, self.specs, self.store = client, seed, specs, store
        self.requests: list[dict] = []
        self.failures: list[str] = []
        self.contract_errors: list[str] = []

    @property
    def last_usage(self):
        return getattr(self.client, "last_usage", {})

    def chat(self, messages, max_tokens=4096, **kwargs):
        plain = copy.deepcopy(messages)
        delivered = {}
        is_solver = False
        for message in plain:
            content = message.get("content")
            parts = ([{"text": content}] if isinstance(content, str) else content or [])
            for part in parts:
                text = part.get("text", "")
                is_solver = is_solver or "ReturnAnswer" in text
                # A candidate may extend the complete parent text verbatim.
                # Match the longest full body first so the parent prefix cannot
                # be counted as delivery or left behind in the common prompt.
                for spec in sorted(self.specs, key=lambda s: len(s.skill_md), reverse=True):
                    block = "\n## 参考方法\n" + spec.skill_md
                    if block in text:
                        key = f"{spec.skill_id}@{spec.version}"
                        delivered[key] = spec.content_sha256
                        text = text.replace(block, "")
                if "text" in part:
                    part["text"] = text
            if isinstance(content, str):
                message["content"] = parts[0]["text"]
        row = {"request_index": len(self.requests), "solver": is_solver,
               "seed": kwargs.get("seed"), "max_tokens": max_tokens,
               "request_sha256": _hash(messages), "common_request_sha256": _hash(plain),
               "skill_content_sha256": delivered, "success": False}
        self.requests.append(row)
        try:
            if kwargs.get("seed") != self.seed:
                raise PairingError(f"actual model seed {kwargs.get('seed')} != {self.seed}")
            if self.client is None:
                raise RuntimeError("model endpoint not configured")
            answer = self.client.chat(messages, max_tokens=max_tokens, **kwargs)
            row["success"] = True
            return answer
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.failures.append(row["error"])
            if isinstance(exc, PairingError):
                self.contract_errors.append(row["error"])
            raise
        finally:
            self.store.append("model_requests", row)


_PROGRAM_ERROR_CODES = frozenset({
    "ast_violation",
    "no_answer",
    "violation_runtime",
    "violation_syntax",
    "vllm_parse_error",
})


def _arm_result(
    out,
    client,
    identity,
    input_hash,
    config_hash,
    expected,
    *,
    options,
    reconstruction_cost_s=None,
) -> dict:
    delivered = {}
    for record in out.retrieval_records:
        data = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        delivered.update(data.get("delivered_content_sha256") or {})
    requests = [r for r in client.requests if r["solver"]]
    successful = [r for r in requests if r["success"]]
    service_errors = list(client.failures) + [
        e for r in out.rounds for e in r.get("service_errors", [])]
    incomplete = bool(service_errors) or out.final_state == "unavailable"
    errors = [r for r in out.rounds if r.get("error_code")]
    error_codes = {str(row.get("error_code") or "") for row in errors}
    delivery_ok = (all(r["skill_content_sha256"] == expected for r in requests)
                   and (not successful or delivered == expected)
                   and set(out.delivered_skill_versions) == set(delivered))
    score = None if incomplete else (
        float(bool(out.correct)) if out.is_mca else float(out.mra_value or 0.0))
    # Parsing is the same official parsing used by M12; it does not depend on GT.
    legal = out.answer is not None and out.predicted is not None
    if out.is_mca:
        legal = legal and out.predicted in list("ABCD"[:len(options or [])])
    return {
        **identity, "input_sha256": input_hash, "config_sha256": config_hash,
        "status": "incomplete" if incomplete else "completed",
        "final_state": out.final_state, "answer": out.answer, "predicted": out.predicted,
        "task": out.task or canonical_task(out.question_type),
        "metric": "Accuracy" if out.is_mca else "MRA", "score": score,
        "correct": None if incomplete else out.correct,
        "mra_value": None if incomplete else out.mra_value,
        "legal_answer": bool(legal),
        "program_error": bool(
            error_codes & _PROGRAM_ERROR_CODES
            or out.static_check_errors
            or out.synthesis_source == "vllm_parse_error"),
        "runtime_error": out.final_state == "run_error",
        "tool_contract_error": "tool_contract" in error_codes,
        "geometry_rejection": "geometry_rejected" in error_codes,
        "untrusted_geometry_used": bool(
            out.answer
            and (
                out.answer_untrusted
                or set(out.used_result_ids) & set(out.invalidated_result_ids)
            )
        ),
        "agent_rounds": int(out.agent_rounds),
        "yield_count": int(out.yield_count),
        "model_request_count": len(requests),
        "tool_call_count": sum(
            len(row.get("results") or []) for row in out.rounds
        ),
        "reconstruction_cost_s": (
            None if reconstruction_cost_s is None else float(reconstruction_cost_s)
        ),
        "errors": errors, "static_check_errors": out.static_check_errors,
        "service_errors": service_errors,
        "failure_code": out.failure_code, "notes": list(out.notes),
        "delivered_skill_versions": out.delivered_skill_versions,
        "delivered_content_sha256": delivered, "delivery_ok": delivery_ok,
        "requests": client.requests,
        "first_common_request_sha256": requests[0]["common_request_sha256"] if requests else None,
        "quality_status": out.quality_status, "main_gate_passed": out.main_gate_passed,
        "scene_route": out.scene_route, "question_tool_scope": out.question_tool_scope,
        "answer_source": out.answer_source, "image_ledger": out.image_ledger,
        "episode_status": out.episode_status,
        "input_error": out.episode_status == "input_error",
        "input_error_reason": out.input_error_reason,
    }


def _incomplete_row(identity, *, task, is_mca, input_hash, config_hash, errors):
    return {
        **identity, "input_sha256": input_hash, "config_sha256": config_hash,
        "status": "incomplete", "final_state": "unavailable", "task": task,
        "metric": "Accuracy" if is_mca else "MRA",
        "answer": None, "score": None, "legal_answer": False,
        "program_error": False, "runtime_error": False, "input_error": False,
        "tool_contract_error": False, "geometry_rejection": False,
        "untrusted_geometry_used": False, "agent_rounds": 0, "yield_count": 0,
        "model_request_count": 0, "tool_call_count": 0, "reconstruction_cost_s": None,
        "input_error_reason": "", "episode_status": "run_error", "delivery_ok": True,
        "delivered_skill_versions": [], "delivered_content_sha256": {},
        "service_errors": errors, "requests": [],
    }


def pair_results(b01: Sequence[dict], b11: Sequence[dict], expected_qa_ids) -> list[dict]:
    """Strict qa_id join, independent of result-list ordering."""
    expected = _index([{"qa_id": q} for q in expected_qa_ids])
    left, right = _index(b01), _index(b11)
    _same_ids(expected, left, "B01 results")
    _same_ids(expected, right, "B11 results")
    pairs = []
    for qa_id in expected:
        a, b = left[qa_id], right[qa_id]
        for key in ("input_sha256", "config_sha256", "task", "metric"):
            if not a.get(key) or a[key] != b.get(key):
                raise PairingError(f"{qa_id}: paired {key} missing/mismatch")
        prompts = [a.get("first_common_request_sha256"), b.get("first_common_request_sha256")]
        if all(prompts) and prompts[0] != prompts[1]:
            raise PairingError(f"{qa_id}: actual common first requests differ")
        if not a["delivery_ok"] or not b["delivery_ok"]:
            raise PairingError(f"{qa_id}: actual Skill delivery mismatch")
        completed = a["status"] == b["status"] == "completed"
        input_error_pair = bool(a.get("input_error") and b.get("input_error"))
        pairs.append({
            "qa_id": qa_id, "task": a["task"], "metric": a["metric"],
            "input_sha256": a["input_sha256"], "config_sha256": a["config_sha256"],
            "status": "completed" if completed else "incomplete",
            "comparable": completed and (all(prompts) or input_error_pair),
            "arms": {"B01": a, "B11": b},
            "delta": b["score"] - a["score"] if completed else None,
        })
    return pairs


def summarize_pairs(pairs: Sequence[dict]) -> dict:
    """All aggregate values can be recomputed from paired_results.jsonl."""
    if not pairs:
        raise PairingError("cannot summarize an empty result panel")
    _index(pairs)
    groups = {"all": list(pairs)}
    for p in pairs:
        groups.setdefault(p["task"], []).append(p)
    report = {}
    rate_fields = {
        "legal_answer_rate": "legal_answer",
        "program_error_rate": "program_error",
        "runtime_error_rate": "runtime_error",
        "input_error_rate": "input_error",
        "tool_contract_error_rate": "tool_contract_error",
        "geometry_rejection_rate": "geometry_rejection",
        "untrusted_geometry_use_rate": "untrusted_geometry_used",
    }
    mean_fields = {
        "mean_agent_rounds": "agent_rounds",
        "mean_yield_count": "yield_count",
        "mean_model_requests": "model_request_count",
        "mean_tool_calls": "tool_call_count",
        "mean_reconstruction_cost_s": "reconstruction_cost_s",
    }
    for name, rows in groups.items():
        arms = {}
        for arm in ARMS:
            records = [p["arms"][arm] for p in rows]
            scored = [r for r in records if r["score"] is not None]
            def mean(key, selected):
                return sum(float(r[key]) for r in selected) / len(selected) if selected else None
            arms[arm] = {
                "n_planned": len(records), "n_scored": len(scored),
                "n_incomplete": len(records) - len(scored),
                "Accuracy": mean("score", [r for r in scored if r["metric"] == "Accuracy"]),
                "MRA": mean("score", [r for r in scored if r["metric"] == "MRA"]),
                **{
                    output: mean(field, scored)
                    for output, field in rate_fields.items()
                },
                **{
                    output: mean(
                        field,
                        [record for record in records if record.get(field) is not None],
                    )
                    for output, field in mean_fields.items()
                },
            }
        # Never compare two averages based on different available subsets.
        complete = [p for p in rows if p["status"] == "completed"]
        delta = {}
        delta_fields = {
            "Accuracy": ("score", "Accuracy"),
            "MRA": ("score", "MRA"),
            **{output: (field, None) for output, field in rate_fields.items()},
            **{output: (field, None) for output, field in mean_fields.items()},
        }
        for key, (field, metric) in delta_fields.items():
            selected = [
                pair for pair in complete
                if (metric is None or pair["metric"] == metric)
                and pair["arms"]["B01"].get(field) is not None
                and pair["arms"]["B11"].get(field) is not None
            ]
            delta[key] = (sum(float(p["arms"]["B11"][field]) -
                              float(p["arms"]["B01"][field]) for p in selected) /
                          len(selected) if selected else None)
        report[name] = {"arms": arms, "B11_minus_B01": delta,
                        "n_complete_pairs": len(complete)}
    return {
        "status": "completed" if all(p["status"] == "completed" for p in pairs) else "incomplete",
        "n_pairs": len(pairs),
        "n_input_error_pairs": sum(
            1 for pair in pairs
            if all(pair["arms"][arm].get("input_error") for arm in ARMS)),
        "groups": report,
        "denominator_contract": (
            "Input/runtime/program/format failures score zero and remain in denominators. "
            "Service failures have null scores and make the experiment incomplete. "
            "Deltas use only the same completed pairs; partial rates are descriptive."),
    }


def run_skill_ablation_v11(
    items: Sequence[EpisodeItem], *, artifact_paths: Mapping[str, str | Path],
    output_dir: str | Path, base_cfg: OnlineRunConfig, seed: int,
    question_types: Sequence[str] = ("object_rel_distance",),
    expected_qa_ids: Sequence[str] | None = None,
    library_root: str | Path = DEFAULT_LIBRARY,
    client_factory: Callable[[str, str], object] | None = None,
    quality_confirmation: dict | None = None,
    quality_confirmations: Mapping[str, dict] | None = None,
    reconstruction_costs: Mapping[str, float] | None = None,
    _fixed_skill_arms: Mapping[str, SkillSpecV11] | None = None,
    _fixed_arm_snapshot_refs: Mapping[str, str] | None = None,
    _fixed_arm_manifest_hashes: Mapping[str, str] | None = None,
) -> dict:
    """Run frozen S0 Skill on/off with a real runner and optional test clients.

    client_factory(phase, qa_id) creates fresh clients for prepare/B01/B11.
    It is the sole fake-model seam; the program runner is never substituted.
    Output directories are create-only, so retries cannot mix experiments.
    """
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    experiment_id = f"v11-{uuid.uuid4().hex}"
    manifest = {"schema_version": "skill-ablation-v11/1.0",
                "experiment_id": experiment_id, "status": "preparing", "seed": int(seed),
                "requested_qa_ids": list(expected_qa_ids) if expected_qa_ids is not None
                else [it.episode.qa_id for it in items]}
    _write(root / "manifest.json", manifest)
    try:
        snapshot_id = SNAPSHOT_ID
        if base_cfg.mode != "real" or base_cfg.baseline != "C1_tools_program":
            raise PairingError("v11 requires real C1_tools_program (fake clients may be injected)")
        if base_cfg.evaluation_binding or base_cfg.metric_depth_model:
            raise PairingError("v11 requires normal retrieval, program solving and frozen geometry")
        wanted = {canonical_task(t) for t in question_types}
        if not wanted:
            raise PairingError("question_types must not be empty")
        fixed_pair = _fixed_skill_arms is not None
        fixed_arms = dict(_fixed_skill_arms or {})
        if fixed_pair:
            if set(fixed_arms) != set(ARMS):
                raise PairingError(
                    f"fixed Skill arms must be exactly {list(ARMS)}")
            if len(wanted) != 1:
                raise PairingError("fixed Skill evaluation requires one question type")
            for arm, spec in fixed_arms.items():
                if not isinstance(spec, SkillSpecV11):
                    raise PairingError(f"{arm}: fixed arm is not SkillSpecV11")
                if spec.question_type not in wanted:
                    raise PairingError(
                        f"{arm}: Skill question_type {spec.question_type!r} "
                        f"does not match panel {sorted(wanted)}")
            snapshot_refs = dict(_fixed_arm_snapshot_refs or {})
            manifest_hashes = dict(_fixed_arm_manifest_hashes or {})
            if set(snapshot_refs) != set(ARMS) or set(manifest_hashes) != set(ARMS):
                raise PairingError(
                    "fixed Skill evaluation requires snapshot ref/hash for both arms")
            if any(not value for value in snapshot_refs.values()):
                raise PairingError("fixed arm snapshot refs must not be empty")
            if any(
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in manifest_hashes.values()
            ):
                raise PairingError(
                    "fixed arm manifest hashes must be lowercase sha256 values")
        else:
            snapshot_refs = {}
            manifest_hashes = {}
        all_items = _index([{"qa_id": it.episode.qa_id, "item": it} for it in items])
        chosen = {q: r["item"] for q, r in all_items.items()
                  if canonical_task(r["item"].episode.question_type) in wanted}
        expected = list(expected_qa_ids) if expected_qa_ids is not None else list(chosen)
        _index([{"qa_id": q} for q in expected])
        _same_ids(expected, chosen, "question panel")
        if not chosen:
            raise PairingError("empty question panel")
        costs = {
            str(key): float(value)
            for key, value in (reconstruction_costs or {}).items()
        }
        if set(costs) - set(expected):
            raise PairingError(
                f"reconstruction costs contain unexpected qa_ids: "
                f"{sorted(set(costs) - set(expected))}")
        if any(not math.isfinite(value) or value < 0.0 for value in costs.values()):
            raise PairingError("reconstruction costs must be finite non-negative seconds")
        runnable = {q for q, item in chosen.items() if item.input_error is None}
        if runnable - set(artifact_paths):
            raise PairingError(
                f"missing artifact mappings: {sorted(runnable - set(artifact_paths))}")
        if any(it.episode.split == "final_test" for it in chosen.values()) and not base_cfg.allow_final_test:
            raise PairingError("final_test requires allow_final_test")
        cfg = replace(copy.deepcopy(base_cfg), seed=int(seed), skills=[],
                      active_snapshot_ref=snapshot_id,
                      active_snapshot_manifest_sha256="")
        common = _common_config(cfg)
        config_hash = _hash(common)
        library = Path(library_root).resolve()
        if fixed_pair:
            specs = list(fixed_arms.values())
            identities_by_arm = {
                arm: {
                    "skill_version": f"{spec.skill_id}@{spec.version}",
                    "content_sha256": spec.content_sha256,
                    "snapshot_ref": snapshot_refs[arm],
                    "manifest_sha256": manifest_hashes[arm],
                }
                for arm, spec in fixed_arms.items()
            }
            fixed_identity_sha = _hash(identities_by_arm)
            snapshot = {
                "schema_version": "fixed-skill-pair-v11/1.0",
                "snapshot_id": f"E11-{fixed_identity_sha[:20]}",
                "manifest_hash": fixed_identity_sha,
                "arms": identities_by_arm,
            }
            skill_manifest = {
                "schema_version": "fixed-skill-pair-manifest-v11/1.0",
                "manifest_sha256": fixed_identity_sha,
                "arms": identities_by_arm,
            }
            by_task = {}
        else:
            snapshot = json.loads(
                (library / f"snapshots/snapshot_{snapshot_id}.json").read_text())
            if snapshot.get("snapshot_id") != snapshot_id:
                raise PairingError("wrong frozen S0 snapshot")
            specs = validate_v11_snapshot(
                snapshot, library_root=library,
                method_context_max_chars=cfg.retrieval_policy.method_context_max_chars)
            skill_manifest = json.loads(
                resolve_source_like_ref(library, snapshot["manifest_ref"]).read_text())
            payload = {k: v for k, v in skill_manifest.items()
                       if k != "manifest_sha256"}
            if payload != _manifest_for(snapshot) or canonical_json_sha256(payload) != \
                    snapshot.get("manifest_hash") or skill_manifest.get(
                        "manifest_sha256") != snapshot.get("manifest_hash"):
                raise PairingError("frozen Skill manifest mismatch")
            by_task = {s.question_type: s for s in specs}
            if wanted - set(by_task):
                raise PairingError(
                    f"missing frozen Skills: {sorted(wanted - set(by_task))}")
        quality = quality_contract()
        if quality_confirmations is not None:
            unknown = set(quality_confirmations) - set(expected)
            if unknown:
                raise PairingError(
                    f"quality confirmations contain unexpected qa_ids: {sorted(unknown)}")
        confirmation_by_qa = {
            q: (quality_confirmations.get(q) if quality_confirmations is not None
                else quality_confirmation)
            for q in expected
        }
        for q, confirmation in confirmation_by_qa.items():
            if confirmation is None:
                continue
            if not (
                confirmation.get("confirmed") in (True, False)
                and confirmation.get("sha256", confirmation.get("quality_contract_sha256"))
                == quality["sha256"]
                and confirmation.get("evidence_ref")
            ):
                raise PairingError(
                    f"{q}: quality confirmation must match current contract and include evidence_ref")
        identities = {
            q: (input_error_identity(it) if it.input_error is not None
                else input_identity(it, artifact_paths[q]))
            for q, it in chosen.items()
        }
        # One alleged artifact identity cannot refer to different data across questions.
        seen = {}
        for q, identity in identities.items():
            if chosen[q].input_error is not None:
                continue
            key = (chosen[q].episode.dataset, identity["artifact_id"], identity["artifact_version"])
            value = {k: identity[k] for k in ("artifact_json_sha256", "pixels")}
            if key in seen and seen[key] != value:
                raise PairingError(f"{q}: reused artifact identity has different content/frames")
            seen[key] = value
        code_root = Path(runner.__file__).parents[1]
        manifest.update({
            "qa_ids": expected, "question_types": sorted(wanted),
            "evaluation_mode": (
                "fixed_parent_candidate" if fixed_pair else "skill_ablation"),
            "common_config": common, "config_sha256": config_hash,
            "inputs": identities, "snapshot": snapshot, "skill_manifest": skill_manifest,
            "quality_contract": quality,
            "quality_confirmation": quality_confirmation,
            "quality_confirmations": confirmation_by_qa if quality_confirmations is not None else None,
            "quality_confirmation_status": (
                "confirmed" if all(
                    c is not None and c.get("confirmed") is True
                    for c in confirmation_by_qa.values()
                ) else "unconfirmed"),
            "reconstruction_costs_s": dict(sorted(costs.items())),
            "client_mode": "injected" if client_factory else "vllm",
            "source_sha256": {str(p.relative_to(code_root)): _file_hash(p)
                              for p in sorted(code_root.rglob("*.py"))},
        })
        _write(root / "manifest.json", manifest)
        _write(root / "frozen_skill_snapshot.json", snapshot)
        for spec in specs:
            path = root / "skills" / spec.skill_id / spec.version / "SKILL.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(spec.skill_md, encoding="utf-8")

        def client(phase, qa_id, store):
            inner = (client_factory(phase, qa_id) if client_factory
                     else runner._make_vllm_client(cfg))
            return _RecordingClient(inner, seed=cfg.seed, specs=specs, store=store)

        prepared, prepare_errors = {}, {}
        # Freeze *all* common inputs before starting either arm.
        for q in expected:
            item = chosen[q]
            if item.input_error is not None:
                prepared[q] = {
                    "input_error": item.input_error.model_dump(mode="json"),
                    "sha256": identities[q]["sha256"],
                }
                continue
            key = hashlib.sha256(q.encode()).hexdigest()
            frozen = root / "frozen" / key
            art = _copy_base(Path(artifact_paths[q]).resolve(), frozen)
            _write(frozen / "episode.json", item.episode)
            for i, pixels in enumerate(item.pixels):
                np.save(frozen / f"frame_{i:04d}.npy", pixels)
            trace = _BoundTrace(root / "preparation" / key, {
                "experiment_id": experiment_id, "arm": "prepare",
                "run_id": f"{experiment_id}:prepare:{q}", "qa_id": q})
            llm = client("prepare", q, trace)
            local_pixels = [p.copy() for p in item.pixels]
            with capture_detector_failures() as detection_errors:
                binding = PreparedObjectBinding(*runner._bind_objects_best_effort(
                    art, local_pixels, None, item.episode.model_copy(deep=True), cfg, llm))
            if llm.contract_errors:
                raise PairingError("; ".join(llm.contract_errors))
            llm.failures.extend(detection_errors)
            if binding.stats.get("binding_error"):
                llm.failures.append(binding.stats["binding_error"])
            if llm.failures:
                prepare_errors[q] = list(llm.failures)
            _, _, art = runner._scene_from_artifact(
                art, local_pixels, binding.stats, binding.objects, item.episode,
                objects_materialized=binding.materialized, persist_quality=False)
            if _pixels_identity(local_pixels) != identities[q]["pixels"]:
                raise PairingError(f"{q}: preparation modified frozen pixels")
            _save_binding(binding, frozen)
            (frozen / "artifact.json").write_text(art.model_dump_json(), encoding="utf-8")
            prepared[q] = {"directory": str(frozen), **tree_identity(frozen),
                           "gate_thresholds": art.quality.gate_thresholds if art.quality else None}
            confirmation = confirmation_by_qa[q]
            if confirmation and confirmation.get("artifact", {}).get("sha256"):
                artifact_sha = _sha256_file(Path(artifact_paths[q]).resolve())
                if artifact_sha != confirmation["artifact"]["sha256"]:
                    raise PairingError(f"{q}: artifact hash differs from quality confirmation")
            if confirmation and confirmation.get("artifact", {}).get("frame_set_hash"):
                if confirmation["artifact"]["frame_set_hash"] != identities[q]["frame_set_hash"]:
                    raise PairingError(f"{q}: frame_set_hash differs from quality confirmation")
            if confirmation and confirmation.get("confirmed") is True and (
                prepared[q]["gate_thresholds"] != quality["default_thresholds"]
            ):
                raise PairingError(f"{q}: artifact quality thresholds differ from confirmed contract")
        manifest.update({"prepared": prepared, "preparation_errors": prepare_errors,
                         "status": "running"})
        _write(root / "manifest.json", manifest)
        results = {arm: [] for arm in ARMS}
        pairs = []
        for q in expected:
            item = chosen[q]
            task = canonical_task(item.episode.question_type)
            if item.input_error is not None:
                input_hash = identities[q]["sha256"]
                for arm in ARMS:
                    spec = fixed_arms[arm] if fixed_pair else (
                        by_task[task] if arm == "B11" else None)
                    binding = (
                        SkillEvaluationBinding(
                            mode="fixed_skill_evaluation",
                            arm="parent" if arm == "B01" else "candidate",
                            skill_id=spec.skill_id,
                            skill_version=f"{spec.skill_id}@{spec.version}",
                            content_sha256=spec.content_sha256,
                            bypassed_component="retrieval_selection",
                        )
                        if fixed_pair and spec is not None else None
                    )
                    arm_root = root / "arms" / arm / hashlib.sha256(q.encode()).hexdigest()
                    identity = {
                        "experiment_id": experiment_id,
                        "arm": arm,
                        "qa_id": q,
                        "run_id": f"{experiment_id}:{arm}:{q}",
                    }
                    trace = _BoundTrace(arm_root / "trace", identity)
                    arm_cfg = replace(
                        copy.deepcopy(cfg),
                        skills=[copy.deepcopy(spec)] if spec is not None else [],
                        reuse_artifact=None,
                        active_snapshot_ref=(
                            snapshot_refs[arm] if fixed_pair
                            else snapshot_id if arm == "B11" else "none"),
                        active_snapshot_manifest_sha256=(
                            manifest_hashes[arm] if fixed_pair
                            else snapshot["manifest_hash"] if arm == "B11" else ""),
                        evaluation_binding=binding,
                        trace_dir=str(trace.root),
                        recon_dir=str(arm_root / "artifacts"),
                        work_dir=str(arm_root / "work"),
                    )
                    llm = client(arm, q, trace)
                    out = runner.run_episode(
                        item.episode.model_copy(deep=True),
                        [],
                        arm_cfg,
                        trace_store=trace,
                        llm=llm,
                        input_error=item.input_error,
                    )
                    row = _arm_result(
                        out,
                        llm,
                        identity,
                        input_hash,
                        config_hash,
                        ({
                            f"{spec.skill_id}@{spec.version}": spec.content_sha256
                        } if spec is not None else {}),
                        options=item.episode.options,
                        reconstruction_cost_s=costs.get(q),
                    )
                    row["initial_tree_sha256"] = None
                    _write(arm_root / "result.json", row)
                    results[arm].append(row)
                pairs = pair_results(
                    results["B01"], results["B11"], expected[:len(results["B01"])])
                (root / "paired_results.jsonl").write_bytes(
                    b"".join(_encoded(pair) + b"\n" for pair in pairs))
                progress = summarize_pairs(pairs)
                progress.update({"experiment_id": experiment_id, "n_planned": len(expected)})
                if len(pairs) != len(expected):
                    progress["status"] = "running"
                _write(root / "summary.json", progress)
                continue
            frozen = Path(prepared[q]["directory"])
            input_hash = _hash({"original": identities[q]["sha256"],
                                "prepared": prepared[q]["sha256"]})
            for arm in ARMS:
                spec = fixed_arms[arm] if fixed_pair else (
                    by_task[task] if arm == "B11" else None)
                binding = (
                    SkillEvaluationBinding(
                        mode="fixed_skill_evaluation",
                        arm="parent" if arm == "B01" else "candidate",
                        skill_id=spec.skill_id,
                        skill_version=f"{spec.skill_id}@{spec.version}",
                        content_sha256=spec.content_sha256,
                        bypassed_component="retrieval_selection",
                    )
                    if fixed_pair and spec is not None else None
                )
                arm_root = root / "arms" / arm / frozen.name
                local = arm_root / "artifacts"
                _clone_tree(frozen, local)
                actual = tree_identity(local)["sha256"]
                if actual != prepared[q]["sha256"]:
                    raise PairingError(f"{q}/{arm}: copied initial inputs differ")
                identity = {"experiment_id": experiment_id, "arm": arm, "qa_id": q,
                            "run_id": f"{experiment_id}:{arm}:{q}"}
                trace = _BoundTrace(arm_root / "trace", identity)
                arm_cfg = replace(copy.deepcopy(cfg),
                                  skills=[copy.deepcopy(spec)] if spec is not None else [],
                                  reuse_artifact=str(local / "artifact.json"),
                                  active_snapshot_ref=(
                                      snapshot_refs[arm] if fixed_pair
                                      else snapshot_id if arm == "B11" else "none"),
                                  active_snapshot_manifest_sha256=(
                                      manifest_hashes[arm] if fixed_pair
                                      else snapshot["manifest_hash"]
                                      if arm == "B11" else ""),
                                  evaluation_binding=binding,
                                  trace_dir=str(trace.root), recon_dir=str(local),
                                  work_dir=str(arm_root / "work"))
                if _hash(_common_config(arm_cfg)) != config_hash:
                    raise PairingError(f"{q}/{arm}: common config changed")
                llm = client(arm, q, trace)
                if q in prepare_errors:
                    row = _incomplete_row(
                        identity, task=task, is_mca=classify(item.episode).is_mca,
                        input_hash=input_hash, config_hash=config_hash, errors=prepare_errors[q])
                    row["reconstruction_cost_s"] = costs.get(q)
                else:
                    try:
                        with capture_detector_failures() as detection_errors:
                            out = runner.run_episode(
                                item.episode.model_copy(deep=True),
                                [np.load(p, allow_pickle=False) for p in sorted(local.glob("frame_*.npy"))],
                                arm_cfg, trace_store=trace, llm=llm, prepared_binding=_load_binding(local))
                        llm.failures.extend(detection_errors)
                        row = _arm_result(
                            out, llm, identity, input_hash, config_hash,
                            ({
                                f"{spec.skill_id}@{spec.version}": spec.content_sha256
                            } if spec is not None else {}),
                            options=item.episode.options,
                            reconstruction_cost_s=costs.get(q))
                    except Exception as exc:
                        row = _incomplete_row(
                            identity, task=task, is_mca=classify(item.episode).is_mca,
                            input_hash=input_hash, config_hash=config_hash,
                            errors=[f"runner exception: {type(exc).__name__}: {exc}"])
                        row["requests"] = llm.requests
                if llm.contract_errors:
                    raise PairingError("; ".join(llm.contract_errors))
                row["initial_tree_sha256"] = actual
                _write(arm_root / "result.json", row)
                results[arm].append(row)
                if tree_identity(frozen)["sha256"] != prepared[q]["sha256"]:
                    raise PairingError(f"{q}/{arm}: frozen prepared source mutated")
                if input_identity(item, artifact_paths[q]) != identities[q]:
                    raise PairingError(f"{q}/{arm}: original input mutated")
            pairs = pair_results(results["B01"], results["B11"], expected[:len(results["B01"])])
            (root / "paired_results.jsonl").write_bytes(
                b"".join(_encoded(p) + b"\n" for p in pairs))
            progress = summarize_pairs(pairs)
            progress.update({"experiment_id": experiment_id, "n_planned": len(expected)})
            if len(pairs) != len(expected):
                progress["status"] = "running"
            _write(root / "summary.json", progress)
        for q in expected:
            if chosen[q].input_error is not None:
                continue
            if tree_identity(Path(prepared[q]["directory"]))["sha256"] != prepared[q]["sha256"]:
                raise PairingError(f"{q}: frozen source changed later in the experiment")
            if input_identity(chosen[q], artifact_paths[q]) != identities[q]:
                raise PairingError(f"{q}: original input changed later in the experiment")
        summary = summarize_pairs(pairs)
        summary.update({"experiment_id": experiment_id, "n_planned": len(expected),
                        "formal_result_eligible": (
                        all(c is not None and c.get("confirmed") is True
                            for c in confirmation_by_qa.values())
                        and client_factory is None
                        and bool(cfg.model_weights_sha256) and bool(cfg.environment_sha256))
                        and summary["status"] == "completed"
                        and summary["n_input_error_pairs"] < summary["n_pairs"]
                        and all(p["comparable"] for p in pairs)})
        manifest["status"] = summary["status"]
        _write(root / "manifest.json", manifest)
        _write(root / "summary.json", summary)
        return summary
    except Exception as exc:
        manifest.update({"status": "invalid" if isinstance(exc, PairingError) else "incomplete",
                         "error": f"{type(exc).__name__}: {exc}"})
        _write(root / "manifest.json", manifest)
        _write(root / "failure.json", {"experiment_id": experiment_id,
                                      "status": manifest["status"], "error": manifest["error"]})
        summary_path = root / "summary.json"
        failed_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        failed_summary.update({"experiment_id": experiment_id, "status": manifest["status"],
                               "error": manifest["error"], "formal_result_eligible": False})
        _write(summary_path, failed_summary)
        raise


def run_skill_pair_evaluation_v11(
    items: Sequence[EpisodeItem],
    *,
    artifact_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    base_cfg: OnlineRunConfig,
    seed: int,
    parent: SkillSpecV11,
    candidate: SkillSpecV11,
    parent_snapshot_ref: str,
    parent_manifest_sha256: str,
    candidate_snapshot_ref: str,
    candidate_manifest_sha256: str,
    expected_qa_ids: Sequence[str] | None = None,
    client_factory: Callable[[str, str], object] | None = None,
    quality_confirmation: dict | None = None,
    reconstruction_costs: Mapping[str, float] | None = None,
) -> dict:
    """Run E11 with explicit parent/candidate fixed injection on frozen inputs."""
    if parent.question_type != candidate.question_type:
        raise PairingError("parent and candidate question_type must match")
    return run_skill_ablation_v11(
        items,
        artifact_paths=artifact_paths,
        output_dir=output_dir,
        base_cfg=base_cfg,
        seed=seed,
        question_types=[parent.question_type],
        expected_qa_ids=expected_qa_ids,
        client_factory=client_factory,
        quality_confirmation=quality_confirmation,
        reconstruction_costs=reconstruction_costs,
        _fixed_skill_arms={"B01": parent, "B11": candidate},
        _fixed_arm_snapshot_refs={
            "B01": parent_snapshot_ref,
            "B11": candidate_snapshot_ref,
        },
        _fixed_arm_manifest_hashes={
            "B01": parent_manifest_sha256,
            "B11": candidate_manifest_sha256,
        },
    )
