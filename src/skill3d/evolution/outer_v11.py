"""Report-only outer holdout comparison of S0 and a frozen final snapshot."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from skill3d.evaluation.skill_ablation_v11 import (
    run_skill_pair_evaluation_v11,
)
from skill3d.evolution.live_v11 import (
    REACHABLE_V11_SLOTS,
    V11ValidationPanel,
    slot_key,
)
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import SkillSpecV11
from skill3d.skills.promote_atomic import read_snapshot
from skill3d.skills.registry import active_snapshot_provenance
from skill3d.skills.v11_library import validate_v11_snapshot

OUTER_REPORT_VERSION = "v11-outer-holdout-report/1.0"


class V11OuterEvaluationError(RuntimeError):
    """Outer evidence cannot be produced without violating holdout isolation."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_create_once(path: Path, value: object) -> None:
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise V11OuterEvaluationError(
                    f"immutable outer report changed: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _by_slot(skills: list[SkillSpecV11]) -> dict[str, SkillSpecV11]:
    result = {}
    for skill in skills:
        key = slot_key(skill.question_type, skill.evidence_state)
        if key in result:
            raise V11OuterEvaluationError(
                f"snapshot contains duplicate slot: {key}"
            )
        result[key] = skill
    return result


@dataclass
class V11OuterHoldoutEvaluator:
    """Evaluate frozen snapshots on outer data without any promotion callback."""

    panels: Mapping[str, V11ValidationPanel]
    base_cfg: OnlineRunConfig
    output_root: str | Path
    library_root: str | Path
    final_snapshot_id: str
    baseline_snapshot_id: str = "S0-v11-task-tools"
    quality_confirmation: dict | None = None
    prepared_binding_store: str | Path | None = None
    prepared_binding_refs: Mapping[str, Mapping] | None = None
    client_factory: Callable[[str, str], object] | None = None
    require_final_active: bool = True

    def _snapshot(
        self,
        snapshot_id: str,
    ) -> tuple[dict, dict[str, SkillSpecV11]]:
        library = Path(self.library_root).resolve()
        snapshot = read_snapshot(
            library / "snapshots",
            snapshot_id,
        )
        skills = validate_v11_snapshot(
            snapshot,
            library_root=library,
        )
        manifest_hash = str(snapshot.get("manifest_hash") or "")
        if len(manifest_hash) != 64:
            raise V11OuterEvaluationError(
                f"snapshot lacks manifest identity: {snapshot_id}"
            )
        return snapshot, _by_slot(skills)

    def run(self) -> dict:
        expected_slots = {
            slot_key(*slot) for slot in REACHABLE_V11_SLOTS
        }
        if set(self.panels) != expected_slots:
            raise V11OuterEvaluationError(
                "outer panels must cover exactly the 19 reachable slots"
            )
        for key, panel in self.panels.items():
            if (
                panel.source_split != "outer_holdout"
                or len(panel.items) < 8
                or panel.unique_scene_count < 8
                or panel.key != key
            ):
                raise V11OuterEvaluationError(
                    f"{key}: invalid outer holdout panel contract"
                )
        library = Path(self.library_root).resolve()
        if self.require_final_active:
            active_id, _ = active_snapshot_provenance(
                library / "snapshots"
            )
            if active_id != self.final_snapshot_id:
                raise V11OuterEvaluationError(
                    "final snapshot must be frozen from the active pointer "
                    "before outer evaluation"
                )
        baseline_snapshot, baseline = self._snapshot(
            self.baseline_snapshot_id
        )
        final_snapshot, final = self._snapshot(self.final_snapshot_id)
        missing = sorted(
            (expected_slots - set(baseline))
            | (expected_slots - set(final))
        )
        if missing:
            raise V11OuterEvaluationError(
                f"snapshot slots are missing: {missing}"
            )

        root = Path(self.output_root).resolve()
        rows = {}
        for key in sorted(expected_slots):
            panel = self.panels[key]
            output = root / "slots" / key.replace("/", "__")
            if output.exists():
                required = (
                    output / "manifest.json",
                    output / "summary.json",
                    output / "paired_results.jsonl",
                )
                if not all(path.is_file() for path in required):
                    raise V11OuterEvaluationError(
                        f"{key}: incomplete outer output cannot be reused"
                    )
            else:
                run_skill_pair_evaluation_v11(
                    panel.items,
                    artifact_paths=panel.artifact_paths,
                    output_dir=output,
                    base_cfg=self.base_cfg,
                    seed=int(self.base_cfg.seed),
                    parent=baseline[key],
                    candidate=final[key],
                    parent_snapshot_ref=self.baseline_snapshot_id,
                    parent_manifest_sha256=baseline_snapshot[
                        "manifest_hash"
                    ],
                    candidate_snapshot_ref=self.final_snapshot_id,
                    candidate_manifest_sha256=final_snapshot[
                        "manifest_hash"
                    ],
                    expected_qa_ids=panel.qa_ids,
                    client_factory=self.client_factory,
                    quality_confirmation=self.quality_confirmation,
                    prepared_binding_store=self.prepared_binding_store,
                    prepared_binding_refs=(
                        {
                            qa_id: self.prepared_binding_refs[qa_id]
                            for qa_id in panel.qa_ids
                        }
                        if self.prepared_binding_refs is not None
                        else None
                    ),
                )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            rows[key] = {
                "question_type": panel.question_type,
                "evidence_state": panel.evidence_state,
                "n_pairs": summary.get("n_pairs"),
                "status": summary.get("status"),
                "formal_result_eligible": summary.get(
                    "formal_result_eligible"
                ),
                "groups": summary.get("groups"),
                "result_ref": str(output / "paired_results.jsonl"),
            }
        report = {
            "schema_version": OUTER_REPORT_VERSION,
            "purpose": "report_only",
            "source_split": "outer_holdout",
            "feeds_revision": False,
            "can_promote": False,
            "baseline_snapshot_id": self.baseline_snapshot_id,
            "baseline_manifest_sha256": baseline_snapshot[
                "manifest_hash"
            ],
            "final_snapshot_id": self.final_snapshot_id,
            "final_manifest_sha256": final_snapshot["manifest_hash"],
            "n_slots": len(rows),
            "slots": rows,
        }
        _write_create_once(root / "outer_report.json", report)
        return report


__all__ = [
    "OUTER_REPORT_VERSION",
    "V11OuterEvaluationError",
    "V11OuterHoldoutEvaluator",
]
