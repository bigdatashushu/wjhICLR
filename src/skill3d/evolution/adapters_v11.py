"""Production adapters for the four-stage v11 evolution campaign."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.skill_ablation_v11 import (
    ARMS,
    _file_hash,
    _common_config,
    _hash,
    input_identity,
    input_error_identity,
    pair_results,
    quality_contract,
    run_skill_pair_evaluation_v11,
    summarize_pairs,
    tree_identity,
)
from skill3d.evolution.experience_v11 import (
    build_v11_experience_bundle_from_trace_store,
)
from skill3d.evolution.campaign_v11 import V11RevisionFormatError, _write_create_once
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig
from skill3d.online.submission import EXECUTION_PROTOCOL_VERSION
from skill3d.schemas import (
    SkillCandidateV11,
    SkillSpecV11,
    V11EvaluationArm,
    V11PairedEvaluationReceipt,
    V11PostPublishObservation,
    V11RevisionProposal,
)
from skill3d.skills.promote_atomic import read_snapshot
from skill3d.skills.registry import (
    active_snapshot_provenance,
    load_active_skills,
)
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION, _PROGRAM_SYNTH
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION
from skill3d.tools.registry import TOOL_FACE_VERSION
from skill3d.trace.store import TraceStore

REVISION_PROMPT_VERSION = "skill-revision-v11.2"
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class V11AdapterError(RuntimeError):
    """A real campaign callback cannot produce an auditable result."""


def _json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _safe_component(value: str) -> str:
    cleaned = _SAFE_COMPONENT_RE.sub("-", str(value)).strip("-")
    if not cleaned:
        raise V11AdapterError("empty filesystem identity")
    return cleaned


def model_config_identity_v11(cfg: OnlineRunConfig) -> dict:
    """Return the online model request configuration without raw endpoint URLs."""
    value = {
        "model_id": cfg.vllm_model,
        "model_weights_sha256": cfg.model_weights_sha256,
        "environment_sha256": cfg.environment_sha256,
        "endpoint_sha256": [
            hashlib.sha256(str(endpoint).encode("utf-8")).hexdigest()
            for endpoint in cfg.vllm_endpoints
        ],
        "max_tokens": int(cfg.max_tokens),
        "max_images": int(cfg.max_images),
        "max_pixels": int(cfg.max_pixels),
        "max_model_len": int(cfg.max_model_len),
        "temperature": 0.0,
    }
    return {**value, "sha256": _sha256(value)}


def solver_config_identity_v11(cfg: OnlineRunConfig, *, seed: int) -> dict:
    """Use the exact common-config projection used by the paired runner."""
    frozen = replace(
        copy.deepcopy(cfg),
        seed=int(seed),
        skills=[],
        active_snapshot_ref="identity-only",
        active_snapshot_manifest_sha256="",
        evaluation_binding=None,
    )
    value = _common_config(frozen)
    return {"config": value, "sha256": _hash(value)}


def _parse_json_response(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise V11AdapterError(
            f"offline reviser did not return one JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise V11AdapterError("offline reviser response must be a JSON object")
    return value


class V11OfflineReviser:
    """Complete-source reviser with durable response reuse and separate metadata."""

    def __init__(
        self,
        client,
        *,
        audit_root: str | Path,
        max_tokens: int = 32768,
        require_health_check: bool = True,
    ) -> None:
        self.client = client
        self.audit_root = Path(audit_root).resolve()
        self.max_tokens = int(max_tokens)
        self.require_health_check = bool(require_health_check)
        self._health_checked = False

    def _prompt(self, **kwargs) -> str:
        from skill3d.tools import REGISTRY

        parent: SkillSpecV11 = kwargs["parent"]
        experience = kwargs["experience"]
        feedback = list(kwargs.get("static_feedback") or [])
        evidence = experience.model_dump(mode="json")
        return (
            "You revise one reusable 3D reasoning method for a coding agent.\n"
            f"Protocol: {REVISION_PROMPT_VERSION}.\n"
            "Return exactly one JSON object with string fields full_skill_md and "
            "modification_reason. full_skill_md must be the complete canonical "
            "SKILL.md, including YAML frontmatter and the full Markdown body.\n"
            "Keep the method generic. Never copy case_id, qa_id, scene_id, a "
            "question, an option list, or a reference answer into the method. "
            "Do not invent tools or alter framework/model/evaluation settings. "
            "Use the cases to improve decision rules, checks, and error handling. "
            "The cases expose only question-level induction answers; no GT 3D "
            "annotation is authorized.\n\n"
            f"Parent identity: {parent.skill_id}@{parent.version} "
            f"({parent.question_type}, sha256={parent.content_sha256})\n"
            f"Parent complete SKILL.md:\n{parent.skill_md}\n"
            f"Keep the stable frontmatter name exactly {parent.name!r}.\n"
            "Current online execution contract:\n"
            f"{_PROGRAM_SYNTH.split('## 当前题目与答案要求')[0]}\n"
            f"Current Tool interfaces ({TOOL_DOCS_VERSION}):\n{REGISTRY.docs()}\n"
            "Induction experience JSON:\n"
            f"{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}\n"
            "Static-validation feedback from earlier attempts:\n"
            f"{json.dumps(feedback, ensure_ascii=False)}\n"
        )

    def _write_audit(self, path: Path, payload: dict) -> None:
        data = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        ) + "\n"
        _write_create_once(path, data.encode("utf-8"))

    def __call__(self, **kwargs) -> V11RevisionProposal:
        prompt = self._prompt(**kwargs)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the offline v11 Skill reviser. Output strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        audit_path = (
            self.audit_root
            / _safe_component(str(kwargs["campaign_id"]))
            / f"revision_{int(kwargs['attempt']):02d}.json"
        )
        exchange_path = audit_path.with_suffix(".exchange.json")
        request = {"messages": messages, "max_tokens": self.max_tokens,
                   "seed": int(kwargs.get("attempt", 0)),
                   "idempotency_key": str(kwargs["idempotency_key"])}
        if exchange_path.is_file():
            exchange = json.loads(exchange_path.read_bytes())
            if exchange["request"] != request:
                raise V11AdapterError("revision request changed during resume")
        else:
            if self.require_health_check and not self._health_checked:
                require_service = getattr(self.client, "require_service", None)
                if require_service is None:
                    raise V11AdapterError("offline client lacks require_service()")
                require_service()
                self._health_checked = True
            call = getattr(self.client, "chat_with_meta", None)
            if call is None:
                raise V11AdapterError("offline client lacks chat_with_meta()")
            meta = call(messages, max_tokens=self.max_tokens, seed=request["seed"])
            text = str(getattr(meta, "text", "") or "")
            exchange = {
                "request": request, "text": text,
                "truncated": bool(getattr(meta, "truncated", False)),
                "request_id": str(getattr(meta, "request_id", "") or "no-request-id"),
                "call": meta.manifest_fields() if callable(
                    getattr(meta, "manifest_fields", None)) else {},
            }
            self._write_audit(exchange_path, exchange)
        text, request_id = exchange["text"], exchange["request_id"]
        output_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._write_audit(audit_path, {
            "schema_version": "skill-revision-call-v11/1.0",
            "campaign_id": str(kwargs["campaign_id"]),
            "attempt": int(kwargs["attempt"]),
            "idempotency_key": str(kwargs["idempotency_key"]),
            "prompt_version": REVISION_PROMPT_VERSION,
            "prompt_sha256": prompt_sha,
            "response_sha256": output_sha,
            "exchange_sha256": hashlib.sha256(exchange_path.read_bytes()).hexdigest(),
            "call": exchange["call"],
        })
        source_ref = (
            f"{audit_path}:request:{_safe_component(request_id)}:"
            f"prompt:{prompt_sha}:response:{output_sha}"
        )
        try:
            if exchange["truncated"]:
                raise V11AdapterError("offline revision response was truncated")
            value = _parse_json_response(text)
            full_skill_md = value.get("full_skill_md")
            modification_reason = value.get("modification_reason")
            if not isinstance(full_skill_md, str) or not full_skill_md.strip():
                raise V11AdapterError("offline revision JSON lacks full_skill_md")
            if not isinstance(modification_reason, str) or not modification_reason.strip():
                raise V11AdapterError("offline revision JSON lacks modification_reason")
        except V11AdapterError as exc:
            raise V11RevisionFormatError(str(exc), source_run_ref=source_ref) from exc
        return V11RevisionProposal(
            full_skill_md=full_skill_md,
            modification_reason=modification_reason,
            source_run_ref=source_ref,
        )


@dataclass
class V11PairedEvaluator:
    """Run E11 and convert its recomputable pair rows to the strict receipt."""

    items: Sequence[EpisodeItem]
    artifact_paths: Mapping[str, str | Path]
    base_cfg: OnlineRunConfig
    output_root: str | Path
    library_root: str | Path
    expected_qa_ids: Sequence[str] | None = None
    quality_confirmation: dict | None = None
    reconstruction_costs: Mapping[str, float] | None = None
    client_factory: Callable[[str, str], object] | None = None

    def _output_dir(self, campaign_id: str) -> Path:
        return Path(self.output_root).resolve() / _safe_component(campaign_id) / "e11"

    def _parent_manifest_hash(self, snapshot_id: str) -> str:
        snapshot = read_snapshot(
            Path(self.library_root).resolve() / "snapshots",
            snapshot_id,
        )
        value = str(snapshot.get("manifest_hash") or "")
        if len(value) != 64:
            raise V11AdapterError("parent snapshot lacks a valid manifest hash")
        return value

    @staticmethod
    def _candidate_manifest_hash(candidate: SkillCandidateV11) -> str:
        return _sha256({
            "candidate_id": candidate.candidate_id,
            "parent_snapshot_id": candidate.parent_snapshot_id,
            "skill_version": candidate.candidate_skill_version,
            "content_sha256": candidate.full_skill_spec.content_sha256,
        })

    @staticmethod
    def _arm(
        rows: Sequence[dict],
        *,
        arm: str,
        spec: SkillSpecV11,
    ) -> V11EvaluationArm:
        records = [row["arms"][arm] for row in rows]
        scored = [row for row in records if row.get("score") is not None]
        runnable = [row for row in records if not row.get("input_error")]
        key = f"{spec.skill_id}@{spec.version}"
        expected = {key: spec.content_sha256}
        observed = bool(runnable) and all(
            row.get("delivery_ok")
            and set(row.get("delivered_skill_versions") or []) == {key}
            and row.get("delivered_content_sha256") == expected
            for row in runnable
        )
        return V11EvaluationArm(
            skill_version=key,
            content_sha256=spec.content_sha256,
            delivered_skill_version=key,
            delivered_content_sha256=spec.content_sha256,
            delivery_observed=observed,
            n_scored=len(scored),
            mean_score=(
                sum(float(row["score"]) for row in scored) / len(scored)
                if scored else None
            ),
            runtime_error_count=sum(
                bool(row.get("runtime_error")) for row in records),
            program_error_count=sum(
                bool(row.get("program_error")) for row in records),
            untrusted_geometry_use_count=sum(
                bool(row.get("untrusted_geometry_used")) for row in records),
            legal_answer_rate=(
                sum(bool(row.get("legal_answer")) for row in scored) / len(scored)
                if scored else None
            ),
        )

    def __call__(self, **kwargs) -> V11PairedEvaluationReceipt:
        parent: SkillSpecV11 = kwargs["parent"]
        candidate: SkillCandidateV11 = kwargs["candidate"]
        campaign_id = str(kwargs["campaign_id"])
        seed = int(kwargs["seed"])
        if candidate.parent_snapshot_id != kwargs["experience"].parent_snapshot_id:
            raise V11AdapterError("candidate and experience parent snapshots differ")
        if str(kwargs["question_type"]) != parent.question_type:
            raise V11AdapterError("evaluation question type differs from parent")
        qa_ids = [item.episode.qa_id for item in self.items]
        expected_ids = list(self.expected_qa_ids) if self.expected_qa_ids is not None else qa_ids
        if (not qa_ids or len(set(qa_ids)) != len(qa_ids)
                or len(set(expected_ids)) != len(expected_ids)
                or set(qa_ids) != set(expected_ids)):
            raise V11AdapterError("empty, duplicate, or mismatched E11 panel")
        learning_scenes = {case.scene_id for case in kwargs["experience"].cases}
        for item in self.items:
            if (item.episode.split != "inner_validation"
                    or item.episode.question_type != parent.question_type):
                raise V11AdapterError("E11 requires the target inner_validation panel")
            if not item.episode.scene_name or item.episode.scene_name in learning_scenes:
                raise V11AdapterError("learning/inner scene overlap or missing scene identity")

        model_identity = model_config_identity_v11(self.base_cfg)
        solver_identity = solver_config_identity_v11(self.base_cfg, seed=seed)
        quality_identity = quality_contract()
        expected_identities = {
            "model_id": self.base_cfg.vllm_model,
            "model_config_sha256": model_identity["sha256"],
            "quality_contract_sha256": quality_identity["sha256"],
            "solver_config_sha256": solver_identity["sha256"],
        }
        actual_identities = {
            key: kwargs[key] for key in expected_identities
        }
        if actual_identities != expected_identities:
            raise V11AdapterError(
                "campaign/evaluator frozen identities differ: "
                f"expected={expected_identities}, actual={actual_identities}")

        output = self._output_dir(campaign_id)
        if output.exists():
            required = [
                output / "manifest.json",
                output / "summary.json",
                output / "paired_results.jsonl",
            ]
            if not all(path.is_file() for path in required):
                raise V11AdapterError(
                    f"incomplete E11 directory cannot be reused: {output}")
        else:
            run_skill_pair_evaluation_v11(
                self.items,
                artifact_paths=self.artifact_paths,
                output_dir=output,
                base_cfg=self.base_cfg,
                seed=seed,
                parent=parent,
                candidate=candidate.full_skill_spec,
                parent_snapshot_ref=candidate.parent_snapshot_id,
                parent_manifest_sha256=self._parent_manifest_hash(
                    candidate.parent_snapshot_id),
                candidate_snapshot_ref=f"candidate:{candidate.candidate_id}",
                candidate_manifest_sha256=self._candidate_manifest_hash(candidate),
                expected_qa_ids=self.expected_qa_ids,
                client_factory=self.client_factory,
                quality_confirmation=self.quality_confirmation,
                reconstruction_costs=self.reconstruction_costs,
            )

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (output / "paired_results.jsonl").read_text(
                encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows or int(summary.get("n_pairs", -1)) != len(rows):
            raise V11AdapterError("E11 summary and pair rows have different denominators")
        # Re-derive both the frozen inputs and every aggregate before trusting a
        # previous directory (including one left by another candidate/config).
        expected_inputs = {
            item.episode.qa_id: input_error_identity(item) if item.input_error else
            input_identity(item, self.artifact_paths[item.episode.qa_id])
            for item in self.items
        }
        specs = {"B01": parent, "B11": candidate.full_skill_spec}
        expected_arms = {
            arm: {
                "skill_version": f"{spec.skill_id}@{spec.version}",
                "content_sha256": spec.content_sha256,
                "snapshot_ref": candidate.parent_snapshot_id if arm == "B01"
                else f"candidate:{candidate.candidate_id}",
                "manifest_sha256": self._parent_manifest_hash(candidate.parent_snapshot_id)
                if arm == "B01" else self._candidate_manifest_hash(candidate),
            } for arm, spec in specs.items()
        }
        code_root = Path(runner.__file__).parents[1]
        expected_fields = {
            "qa_ids": expected_ids,
            "question_types": [parent.question_type],
            "seed": seed,
            "evaluation_mode": "fixed_parent_candidate",
            "config_sha256": solver_identity["sha256"],
            "common_config": solver_identity["config"],
            "inputs": expected_inputs,
            "quality_contract": quality_identity,
            "quality_confirmation": self.quality_confirmation,
            "client_mode": "injected" if self.client_factory else "vllm",
            "reconstruction_costs_s": dict(self.reconstruction_costs or {}),
            "source_sha256": {str(p.relative_to(code_root)): _file_hash(p)
                              for p in sorted(code_root.rglob("*.py"))},
        }
        for key, value in expected_fields.items():
            if manifest.get(key) != value:
                raise V11AdapterError(f"E11 frozen {key} changed or missing")
        if (manifest.get("snapshot", {}).get("arms") != expected_arms
                or manifest.get("skill_manifest", {}).get("arms") != expected_arms):
            raise V11AdapterError("E11 parent/candidate identities changed")
        recomputed_pairs = pair_results(
            [pair["arms"]["B01"] for pair in rows],
            [pair["arms"]["B11"] for pair in rows], expected_ids)
        if recomputed_pairs != rows:
            raise V11AdapterError("E11 pair rows do not replay")
        recomputed = summarize_pairs(recomputed_pairs)
        if any(summary.get(key) != value for key, value in recomputed.items()):
            raise V11AdapterError("E11 summary does not match recomputed rows")
        if manifest.get("status") != recomputed["status"]:
            raise V11AdapterError("E11 manifest/row completion status differs")
        for arm, spec in specs.items():
            source = output / "skills" / spec.skill_id / spec.version / "SKILL.md"
            if not source.is_file() or source.read_text() != spec.skill_md:
                raise V11AdapterError("E11 frozen Skill source changed")
        for pair in rows:
            qa_id = pair["qa_id"]
            prepared = manifest["prepared"][qa_id]
            no_image = "input_error" in prepared
            digest = expected_inputs[qa_id]["sha256"] if no_image else prepared["sha256"]
            input_hash = digest if no_image else _hash({
                "original": expected_inputs[qa_id]["sha256"], "prepared": digest})
            key = hashlib.sha256(qa_id.encode()).hexdigest()
            if not no_image and tree_identity(output / "frozen" / key)["sha256"] != digest:
                raise V11AdapterError("E11 prepared geometry changed")
            for arm in ARMS:
                record = pair["arms"][arm]
                if (record["input_sha256"] != input_hash
                        or record["config_sha256"] != solver_identity["sha256"]
                        or record["run_id"] != f"{manifest['experiment_id']}:{arm}:{qa_id}"):
                    raise V11AdapterError("E11 arm input/run identity differs")
                if not no_image:
                    result_path = output / "arms" / arm / key / "result.json"
                    if (record.get("initial_tree_sha256") != digest
                            or json.loads(result_path.read_bytes()) != record):
                        raise V11AdapterError("E11 independent arm evidence differs")
        solver_requests = [
            request
            for pair in rows
            for arm in ARMS
            for request in pair["arms"][arm].get("requests", [])
            if request.get("solver")
        ]
        request_seed_observed = bool(solver_requests) and all(
            request.get("seed") == seed for request in solver_requests)
        panel_payload = {
            "qa_ids": manifest.get("qa_ids"),
            "question_types": manifest.get("question_types"),
            "inputs": manifest.get("inputs"),
            "config_sha256": manifest.get("config_sha256"),
        }
        panel_sha = _sha256(panel_payload)
        status = (
            "completed" if summary.get("status") == "completed" else "incomplete")
        formal = bool(
            self.quality_confirmation
            and self.quality_confirmation.get("confirmed") is True
            and self.quality_confirmation.get("sha256") == quality_identity["sha256"]
            and self.quality_confirmation.get("evidence_ref")
            and self.client_factory is None
            and bool(self.base_cfg.model_weights_sha256)
            and bool(self.base_cfg.environment_sha256)
            and recomputed["status"] == "completed"
            and recomputed["n_input_error_pairs"] < len(rows)
            and all(pair["comparable"] for pair in rows)
            and request_seed_observed
            and manifest.get("config_sha256") == solver_identity["sha256"]
            and manifest.get("quality_contract", {}).get("sha256")
            == quality_identity["sha256"]
        )
        if bool(summary.get("formal_result_eligible")) != formal:
            raise V11AdapterError("E11 formal eligibility does not replay")
        return V11PairedEvaluationReceipt(
            campaign_id=campaign_id,
            evaluation_id=str(manifest["experiment_id"]),
            panel_id=f"panel-{panel_sha[:20]}",
            panel_sha256=panel_sha,
            question_type=parent.question_type,
            seed=seed,
            request_seed_observed=request_seed_observed,
            model_id=self.base_cfg.vllm_model,
            model_config_sha256=model_identity["sha256"],
            quality_contract_sha256=quality_identity["sha256"],
            solver_config_sha256=solver_identity["sha256"],
            template_version=PROMPT_TEMPLATE_VERSION,
            execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
            tool_docs_version=TOOL_DOCS_VERSION,
            tool_face_version=TOOL_FACE_VERSION,
            status=status,
            formal_result_eligible=formal,
            frozen_panel=True,
            independent_arms=True,
            n_pairs=len(rows),
            result_refs=[
                f"{output / 'paired_results.jsonl'}#qa_id={pair['qa_id']}"
                for pair in rows
            ],
            parent=self._arm(rows, arm="B01", spec=parent),
            candidate=self._arm(
                rows, arm="B11", spec=candidate.full_skill_spec),
        )


@dataclass
class V11PostPublishVerifier:
    """Run fresh induction episodes through normal active loading and collect them."""

    items: Sequence[EpisodeItem]
    artifact_paths: Mapping[str, str | Path]
    base_cfg: OnlineRunConfig
    output_root: str | Path
    library_root: str | Path
    client_factory: Callable[[str], object] | None = None

    def _trace_dir(self, campaign_id: str) -> Path:
        return (
            Path(self.output_root).resolve()
            / _safe_component(campaign_id)
            / "post_publish_learning"
        )

    def __call__(self, **kwargs) -> V11PostPublishObservation:
        candidate: SkillCandidateV11 = kwargs["candidate"]
        campaign_id = str(kwargs["campaign_id"])
        snapshot_id = str(kwargs["snapshot_id"])
        question_type = str(kwargs["question_type"])
        if not self.items:
            raise V11AdapterError("post-publish learning panel is empty")
        if self.base_cfg.mode != "real" or self.base_cfg.evaluation_binding is not None:
            raise V11AdapterError(
                "post-publish verification requires normal real-mode retrieval")
        for item in self.items:
            if item.input_error is not None:
                raise V11AdapterError(
                    f"post-publish item {item.episode.qa_id} has input_error")
            if item.episode.split != "induction":
                raise V11AdapterError("post-publish items must use induction split")
            if item.episode.question_type != question_type:
                raise V11AdapterError(
                    "post-publish panel contains another question type")
            if item.episode.qa_id not in self.artifact_paths:
                raise V11AdapterError(
                    f"missing post-publish artifact: {item.episode.qa_id}")

        library = Path(self.library_root).resolve()
        skills, warnings, loaded_snapshot = load_active_skills(library / "snapshots")
        active_ref, manifest_hash = active_snapshot_provenance(
            library / "snapshots")
        if warnings or loaded_snapshot != snapshot_id or active_ref != snapshot_id:
            raise V11AdapterError(
                f"active loader does not expose published snapshot {snapshot_id}")
        expected_key = candidate.candidate_skill_version
        loaded = [
            spec for spec in skills
            if spec.question_type == candidate.full_skill_spec.question_type
        ]
        if len(loaded) != 1 or (
            f"{loaded[0].skill_id}@{loaded[0].version}" != expected_key
            or loaded[0].content_sha256
            != candidate.full_skill_spec.content_sha256
        ):
            raise V11AdapterError("normal active loader does not return candidate")

        trace_dir = self._trace_dir(campaign_id)
        complete = all(
            (trace_dir / name).is_file()
            for name in (
                "online_run.jsonl",
                "episode_input.jsonl",
                "episode_trace.jsonl",
                "evaluation_result.jsonl",
            )
        )
        run_id = "post-v11-" + hashlib.sha256(
            str(kwargs["idempotency_key"]).encode("utf-8")
        ).hexdigest()[:16]
        if not trace_dir.exists():
            store = TraceStore(trace_dir)
            store.claim_run(run_id)
            store = store.bind(run_id=run_id)
            outcomes = []
            for item in self.items:
                cfg = replace(
                    copy.deepcopy(self.base_cfg),
                    seed=int(self.base_cfg.seed),
                    skills=copy.deepcopy(skills),
                    active_snapshot_ref=snapshot_id,
                    active_snapshot_manifest_sha256=manifest_hash,
                    trace_dir=str(trace_dir),
                    reuse_artifact=str(
                        Path(self.artifact_paths[item.episode.qa_id]).resolve()),
                    evaluation_binding=None,
                )
                llm = (
                    self.client_factory(item.episode.qa_id)
                    if self.client_factory is not None else None
                )
                outcomes.append(runner.run_episode(
                    item.episode.model_copy(deep=True),
                    [pixels.copy() for pixels in item.pixels],
                    cfg,
                    trace_store=store.bind(qa_id=item.episode.qa_id),
                    llm=llm,
                    input_error=None,
                ))
            n_unavailable = sum(
                out.final_state == "unavailable" or any(
                    row.get("service_errors") for row in out.rounds)
                for out in outcomes)
            store.append("online_run", {
                "run_id": run_id,
                "status": "incomplete" if n_unavailable else "completed",
                "n_unavailable": n_unavailable,
                "mode": "real",
                "baseline": "C1_tools_program",
                "template_version": PROMPT_TEMPLATE_VERSION,
                "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
                "tool_docs_version": TOOL_DOCS_VERSION,
                "seed": int(self.base_cfg.seed),
                "source": "post_publish_learning",
                "n_episodes": len(self.items),
                "split": "induction",
                "label_access": False,
                "deterministic_replay": False,
                "note": "fresh post-publication induction run via normal active loading",
            })
        elif not complete:
            raise V11AdapterError(
                f"incomplete post-publish trace directory cannot be reused: {trace_dir}")

        bundle = build_v11_experience_bundle_from_trace_store(
            trace_dir,
            campaign_id=campaign_id,
            parent_snapshot_id=snapshot_id,
            parent=candidate.full_skill_spec,
            source_run_ref=f"online_run:{run_id}",
            require_real=True,
        )
        if not bundle.cases:
            raise V11AdapterError("post-publish run produced no learning cases")
        return V11PostPublishObservation(
            question_type=question_type,
            snapshot_id=snapshot_id,
            retrieved_skill_version=expected_key,
            delivered_skill_version=expected_key,
            delivered_skill_md=candidate.full_skill_spec.skill_md,
            learning_event_refs=[case.trace_ref for case in bundle.cases],
            source_run_ref=bundle.source_run_ref,
        )


__all__ = [
    "REVISION_PROMPT_VERSION",
    "V11AdapterError",
    "V11OfflineReviser",
    "V11PairedEvaluator",
    "V11PostPublishVerifier",
    "model_config_identity_v11",
    "solver_config_identity_v11",
]
