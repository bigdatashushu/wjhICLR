"""M15 Skill Registry：SkillSpec 语义版本 + 状态机（draft/shadow/canary/promoted/quarantined）。

语义版本规则（§7）：
- MAJOR.MINOR.PATCH；
- 前置条件变更 → 必须 MAJOR bump。v6 的前置条件是**可检索前提**
  （`applicable_question_types` / `required_evidence_signature` /
  `requires_metric_evidence` / `applicable_gate_version`）：改这些字段等于改
  "这条 Skill 在什么证据状态下会被检索到"，旧轨迹的正例不再适用，故必须 MAJOR；
- 其他模板/描述变更 → MINOR/PATCH。
Skill 状态机：draft → shadow → canary → promoted；异常 → quarantined。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from skill3d.schemas import SkillSpec, SkillState

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# 前置条件字段：变更必须 MAJOR bump（§7）。
# v6 口径：检索前提 = 题型 + 证据签名（+ 米制 gate 版本，§13.6/§17.1）。
# 这些字段一变，"这条 Skill 在哪种证据状态下会被选中"就变了 —— 旧轨迹的正例不再
# 对应同一检索条件，必须 MAJOR bump（否则新旧正例混在同一版本下，§17.2 的分桶失效）。
# `source` 也列入：real 与 mock_* 的同一 skill_id 若共用一个版本号，主表会把
# mock 轨迹算成真实贡献（§19.3 synthesis_source 口径）。
PRECONDITION_FIELDS = (
    "applicable_question_types",
    "required_evidence_signature",
    "requires_metric_evidence",
    "applicable_gate_version",
    "skill_family",
    "source",
)

# 合法状态转移
_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"shadow", "quarantined"},
    "shadow": {"canary", "quarantined"},
    "canary": {"promoted", "quarantined"},
    "promoted": {"quarantined"},
    "quarantined": {"draft"},  # 修复后重新走流程
}


class SemverError(ValueError):
    pass


class IllegalTransitionError(ValueError):
    pass


def parse_semver(semver: str) -> tuple[int, int, int]:
    m = _SEMVER_RE.match(semver)
    if not m:
        raise SemverError(f"非法语义版本: {semver}（应为 MAJOR.MINOR.PATCH）")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bump_semver(semver: str, level: str) -> str:
    major, minor, patch = parse_semver(semver)
    if level == "MAJOR":
        return f"{major + 1}.0.0"
    if level == "MINOR":
        return f"{major}.{minor + 1}.0"
    if level == "PATCH":
        return f"{major}.{minor}.{patch + 1}"
    raise SemverError(f"未知 bump 级别: {level}")


def required_bump_level(old: SkillSpec, new: SkillSpec) -> str:
    """判断 old→new 至少需要哪级 bump：前置条件变更 → MAJOR。"""
    for field in PRECONDITION_FIELDS:
        if getattr(old, field) != getattr(new, field):
            return "MAJOR"
    if old.call_graph_template != new.call_graph_template \
            or old.supported_coordinate_frames != new.supported_coordinate_frames \
            or old.validation_assertions != new.validation_assertions:
        return "MINOR"
    return "PATCH"


def validate_version_bump(old: SkillSpec, new: SkillSpec) -> None:
    """校验 new.semver 相对 old.semver 满足所需 bump 级别，否则抛 SemverError。"""
    order = {"PATCH": 0, "MINOR": 1, "MAJOR": 2}
    required = required_bump_level(old, new)
    old_v = parse_semver(old.semver)
    new_v = parse_semver(new.semver)
    if new_v <= old_v:
        raise SemverError(f"新版本 {new.semver} 必须大于旧版本 {old.semver}（候选不可变，硬约束 11）")
    actual = "PATCH"
    if new_v[0] > old_v[0]:
        actual = "MAJOR"
    elif new_v[1] > old_v[1]:
        actual = "MINOR"
    if order[actual] < order[required]:
        raise SemverError(
            f"前置条件变更（{PRECONDITION_FIELDS}）必须 MAJOR bump，实际为 {actual}"
        )


class SkillRegistry:
    """SkillSpec 注册表：语义版本 + 状态机（M15）。"""
    def __init__(self) -> None:
        # skill_id -> semver -> SkillSpec
        self._specs: dict[str, dict[str, SkillSpec]] = {}
        # skill_id -> semver -> 状态
        self._states: dict[str, dict[str, SkillState]] = {}

    def register(self, spec: SkillSpec, state: SkillState = "draft") -> None:
        parse_semver(spec.semver)  # 校验格式
        versions = self._specs.setdefault(spec.skill_id, {})
        if spec.semver in versions:
            raise SemverError(
                f"{spec.skill_id}@{spec.semver} 已存在；候选不可变，必须 bump 版本（硬约束 11）"
            )
        versions[spec.semver] = spec
        self._states.setdefault(spec.skill_id, {})[spec.semver] = state

    def register_new_version(self, old: SkillSpec, new: SkillSpec,
                             state: SkillState = "draft") -> None:
        """注册新版本：先校验 bump 级别（前置变更必须 MAJOR）。"""
        validate_version_bump(old, new)
        self.register(new, state=state)

    def get(self, skill_id: str, semver: str) -> SkillSpec | None:
        return self._specs.get(skill_id, {}).get(semver)

    def get_state(self, skill_id: str, semver: str) -> SkillState | None:
        return self._states.get(skill_id, {}).get(semver)

    def transition(self, skill_id: str, semver: str, target: SkillState) -> None:
        """状态机转移：draft→shadow→canary→promoted；任何阶段可进 quarantined。"""
        cur = self._states.get(skill_id, {}).get(semver)
        if cur is None:
            raise KeyError(f"未注册: {skill_id}@{semver}")
        if target not in _ALLOWED_TRANSITIONS[cur]:
            raise IllegalTransitionError(f"非法状态转移: {cur} → {target}")
        self._states[skill_id][semver] = target

    def list_by_state(self, state: SkillState) -> list[SkillSpec]:
        out = []
        for skill_id, versions in self._states.items():
            for semver, s in versions.items():
                if s == state:
                    out.append(self._specs[skill_id][semver])
        return out


# ------------------------------------------------------------- active 快照读取 ----

def active_snapshot_provenance(path: str | Path) -> tuple[str, str]:
    """Return ``(snapshot_id, manifest_hash)`` for the active pointer.

    This is metadata-only: the online loader still consumes only inline
    ``spec_content`` from the promoted snapshot. Missing or malformed
    provenance is represented as empty strings and never guessed.
    """
    from pathlib import Path as _Path

    from .promote_atomic import read_active_snapshot

    p = _Path(path)
    store_dir = p if p.is_dir() else p.parent
    pointer = store_dir / "active_snapshot.json"
    if not pointer.exists():
        return "genesis", ""
    try:
        snap = read_active_snapshot(store_dir)
    except Exception:  # noqa: BLE001 - provenance must not break online loading
        return "genesis", ""
    if not isinstance(snap, dict):
        return "genesis", ""
    snapshot_id = str(snap.get("snapshot_id", "genesis"))
    registered = str(snap.get("manifest_hash", "") or "")
    if not registered:
        return snapshot_id, ""
    # When the library manifest is present, independently recompute its
    # canonical payload digest. A stale registration is not provenance.
    manifest_path = store_dir.parent / "manifests" / "library_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            claimed = str(manifest.pop("manifest_sha256", "") or "")
            canonical = (json.dumps(manifest, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n").encode("utf-8")
            computed = hashlib.sha256(canonical).hexdigest()
        except Exception:  # noqa: BLE001 - malformed provenance is fail-closed
            return snapshot_id, ""
        if not claimed or claimed != computed or registered != computed:
            return snapshot_id, ""
    return snapshot_id, registered


def load_active_skills(path: str | Path) -> tuple[list[SkillSpec], list[str], str]:
    """Read only the promoted inline SkillSpecs from an active snapshot.

    ``path`` may be a snapshot store directory or its ``active_snapshot.json``
    pointer.  The loader deliberately does not scan ``sources``, generated
    files, candidates, or future candidates: only the atomically promoted
    snapshot is eligible for online retrieval.  The three-value return shape
    is kept for existing online callers.
    """
    import json
    from pathlib import Path as _Path

    from .promote_atomic import read_active_snapshot

    p = _Path(path)
    store_dir = p if p.is_dir() else p.parent
    warnings: list[str] = []
    pointer = store_dir / "active_snapshot.json"
    if not pointer.exists():
        return [], [f"无 active snapshot（{pointer} 不存在）→ 空 Skill"], "genesis"
    try:
        snap = read_active_snapshot(store_dir)
    except Exception as exc:  # noqa: BLE001 - 损坏的指针不应让在线链崩掉
        return [], [f"active snapshot 读取失败: {type(exc).__name__}: {exc}"], "genesis"
    if not isinstance(snap, dict):
        return [], ["active snapshot 不是 JSON mapping → 空 Skill"], "genesis"

    skills: list[SkillSpec] = []
    entries = snap.get("entries") or {}
    if not isinstance(entries, dict):
        return [], ["active snapshot.entries 不是 mapping → 空 Skill"], "genesis"
    for rid, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("candidate_type") != "skill":
            continue
        raw = entry.get("spec_content") or ""
        try:
            parsed = json.loads(raw)
            skills.append(SkillSpec.model_validate(parsed))
        except Exception as exc:  # noqa: BLE001 - 单条损坏不阻断其余
            warnings.append(f"条目 {rid} 无法解析为 SkillSpec: {type(exc).__name__}: {exc}")
    return skills, warnings, str(snap.get("snapshot_id", "genesis"))
