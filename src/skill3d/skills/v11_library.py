"""v11 Skill source, snapshot, and single-active-version publishing.

The v11 path is intentionally separate from the v9/v10 inline ``SkillSpec``
snapshot format. A v11 snapshot stores framework identity plus a content
addressed ``source_ref``; the referenced immutable ``SKILL.md`` remains the
only method-content source.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from skill3d.routing.task_classifier import TASK_TYPES
from skill3d.schemas import SkillCandidateV11, SkillSpecV11
from skill3d.schemas.skill import normalize_skill_markdown, parse_skill_markdown
from skill3d.skills.delivery import DEFAULT_METHOD_CONTEXT_MAX_CHARS

V11_SNAPSHOT_SCHEMA = "runtime-skill-snapshot/2.0"
V11_LIBRARY_FORMAT = "harness3d-skill-library/2.0"
V11_ENTRY_ACTIVE = "active"
V11_ENTRY_HISTORICAL = "historical"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class V11LibraryError(ValueError):
    """Raised when a v11 source, snapshot, or publication is inconsistent."""


@dataclass(frozen=True)
class LoadedSkillV11:
    """Validated source and its framework-owned runtime identity."""

    source_ref: str
    content_sha256: str
    spec: SkillSpecV11


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def validate_source_ref(source_ref: str) -> PurePosixPath:
    """Validate a repository-relative POSIX source reference."""
    value = str(source_ref or "")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise V11LibraryError(f"不安全的 source_ref: {source_ref!r}")
    if not path.parts or path.parts[0] != "versions":
        raise V11LibraryError(f"source_ref 必须位于 versions/ 下: {source_ref!r}")
    if path.name != "SKILL.md":
        raise V11LibraryError(f"source_ref 必须指向 SKILL.md: {source_ref!r}")
    return path


def resolve_source_ref(library_root: str | Path, source_ref: str) -> Path:
    """Resolve a source reference without allowing symlink/path escape."""
    root = Path(library_root).resolve()
    relative = validate_source_ref(source_ref)
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise V11LibraryError(f"source_ref 越出 Skill 库: {source_ref!r}") from exc
    return target


def load_skill_source_v11(
    library_root: str | Path,
    *,
    source_ref: str,
    skill_id: str,
    version: str,
    question_type: str,
) -> LoadedSkillV11:
    """Load one canonical complete source and bind framework identity to it."""
    path = resolve_source_ref(library_root, source_ref)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise V11LibraryError(f"{source_ref} 不是合法 UTF-8") from exc
    if normalize_skill_markdown(text) != text:
        raise V11LibraryError(f"{source_ref} 不是规范 LF/单末尾换行格式")
    front = parse_skill_markdown(text)[0]
    if path.parent.name != front["name"]:
        raise V11LibraryError(
            f"name 与目录不一致: name={front['name']!r}, directory={path.parent.name!r}")
    if question_type not in TASK_TYPES:
        raise V11LibraryError(f"未知规范题型: {question_type!r}")
    spec = SkillSpecV11(
        skill_id=skill_id,
        version=version,
        question_type=question_type,
        skill_md=text,
    )
    canonical_ref = source_ref_for(spec)
    if validate_source_ref(source_ref).as_posix() != canonical_ref:
        raise V11LibraryError(
            f"source_ref 与 Skill 身份不一致: expected={canonical_ref}, "
            f"actual={source_ref}")
    return LoadedSkillV11(
        source_ref=canonical_ref,
        content_sha256=spec.content_sha256,
        spec=spec,
    )


def source_ref_for(spec: SkillSpecV11) -> str:
    """Return the immutable v11 location for a complete source."""
    return (
        PurePosixPath("versions")
        / spec.skill_id
        / spec.version
        / spec.name
        / "SKILL.md"
    ).as_posix()


def _entry_of(
    source: LoadedSkillV11,
    *,
    parent_version: str | None,
    state: str,
    created_by: str,
    candidate_id: str = "",
    source_run_ref: str = "",
    modification_reason: str = "",
    diff_sha256: str = "",
) -> dict:
    spec = source.spec
    entry = {
        "candidate_type": "skill",
        "format": "skill-md/v11",
        "skill_id": spec.skill_id,
        "version": spec.version,
        "question_type": spec.question_type,
        "source_ref": source.source_ref,
        "content_sha256": source.content_sha256,
        "parent_version": parent_version,
        "state": state,
        "created_by": created_by,
    }
    if candidate_id:
        entry["candidate_id"] = candidate_id
    if source_run_ref:
        entry["source_run_ref"] = source_run_ref
    if modification_reason:
        entry["modification_reason"] = modification_reason
    if diff_sha256:
        entry["diff_sha256"] = diff_sha256
    return entry


def _manifest_for(snapshot: dict) -> dict:
    rows = []
    for key in sorted(snapshot["entries"]):
        entry = snapshot["entries"][key]
        rows.append({
            "skill_version": key,
            "skill_id": entry["skill_id"],
            "version": entry["version"],
            "question_type": entry["question_type"],
            "source_ref": entry["source_ref"],
            "content_sha256": entry["content_sha256"],
            "parent_version": entry.get("parent_version"),
            "state": entry["state"],
        })
    return {
        "library_format": V11_LIBRARY_FORMAT,
        "snapshot_id": snapshot["snapshot_id"],
        "parent_snapshot_id": snapshot.get("parent_snapshot_id"),
        "generation": int(snapshot.get("generation", 0)),
        "active_by_question_type": dict(snapshot["active_by_question_type"]),
        "entries": rows,
    }


def build_v11_snapshot(
    sources: Iterable[LoadedSkillV11],
    *,
    snapshot_id: str,
    parent_snapshot_id: str | None = None,
    generation: int = 0,
    created_by: str = "format_migration",
    parent_versions: Mapping[str, str] | None = None,
) -> tuple[dict, dict]:
    """Build a deterministic v11 snapshot with one active method per question type."""
    if not _SAFE_ID_RE.fullmatch(str(snapshot_id or "")):
        raise V11LibraryError(f"snapshot_id 不是安全标识符: {snapshot_id!r}")
    entries: dict[str, dict] = {}
    active_by_question_type: dict[str, str] = {}
    for source in sources:
        spec = source.spec
        key = f"{spec.skill_id}@{spec.version}"
        if key in entries:
            raise V11LibraryError(f"Skill 版本重复: {key}")
        if spec.question_type in active_by_question_type:
            raise V11LibraryError(
                f"题型 {spec.question_type} 存在多个 active Skill: "
                f"{active_by_question_type[spec.question_type]}, {key}")
        entries[key] = _entry_of(
            source, parent_version=(parent_versions or {}).get(spec.skill_id),
            state=V11_ENTRY_ACTIVE, created_by=created_by)
        active_by_question_type[spec.question_type] = key
    if not entries:
        raise V11LibraryError("v11 snapshot 至少需要一条 Skill")

    snapshot = {
        "schema_version": V11_SNAPSHOT_SCHEMA,
        "snapshot_id": str(snapshot_id),
        "parent_snapshot_id": parent_snapshot_id,
        "generation": int(generation),
        "skill_versions": sorted(entries),
        "active_by_question_type": dict(sorted(active_by_question_type.items())),
        "manifest_ref": (
            PurePosixPath("manifests") / f"manifest_{snapshot_id}.json"
        ).as_posix(),
        "manifest_hash": "",
        "entries": entries,
    }
    manifest = _manifest_for(snapshot)
    manifest_hash = canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = manifest_hash
    snapshot["manifest_hash"] = manifest_hash
    return snapshot, manifest


def validate_v11_snapshot(
    snapshot: dict,
    *,
    library_root: str | Path,
    method_context_max_chars: int = DEFAULT_METHOD_CONTEXT_MAX_CHARS,
) -> list[SkillSpecV11]:
    """Validate all references and return active v11 runtime specs."""
    if snapshot.get("schema_version") != V11_SNAPSHOT_SCHEMA:
        raise V11LibraryError(
            f"不支持的 v11 snapshot schema: {snapshot.get('schema_version')!r}")
    entries = snapshot.get("entries")
    active = snapshot.get("active_by_question_type")
    if not isinstance(entries, dict) or not isinstance(active, dict):
        raise V11LibraryError("v11 snapshot 缺少 entries/active_by_question_type mapping")
    if set(snapshot.get("skill_versions") or []) != set(entries):
        raise V11LibraryError("skill_versions 与 entries 不一致")
    if len(set(active.values())) != len(active):
        raise V11LibraryError("一个 active Skill 不能占据多个题型槽位")

    limit = int(method_context_max_chars)
    if limit <= 0:
        raise V11LibraryError("method_context_max_chars 必须为正整数")
    loaded_by_key: dict[str, SkillSpecV11] = {}
    for key, entry in sorted(entries.items()):
        if not isinstance(entry, dict):
            raise V11LibraryError(f"v11 条目不是 mapping: {key}")
        if entry.get("state") not in {V11_ENTRY_ACTIVE, V11_ENTRY_HISTORICAL}:
            raise V11LibraryError(f"v11 条目状态非法: {key} -> {entry.get('state')}")
        expected_key = f"{entry.get('skill_id')}@{entry.get('version')}"
        if key != expected_key:
            raise V11LibraryError(f"条目键与身份不一致: {key} != {expected_key}")
        source = load_skill_source_v11(
            library_root,
            source_ref=str(entry.get("source_ref") or ""),
            skill_id=str(entry.get("skill_id") or ""),
            version=str(entry.get("version") or ""),
            question_type=str(entry.get("question_type") or ""),
        )
        if source.content_sha256 != entry.get("content_sha256"):
            raise V11LibraryError(
                f"{key} 源文件 hash 不一致: snapshot={entry.get('content_sha256')}, "
                f"actual={source.content_sha256}")
        if len(source.spec.skill_md) > limit:
            raise V11LibraryError(
                f"{key} 完整 SKILL.md 为 {len(source.spec.skill_md)} 字符，"
                f"超过服务限制 {limit}")
        loaded_by_key[key] = source.spec

    active_entry_keys = {
        key for key, entry in entries.items()
        if entry.get("state") == V11_ENTRY_ACTIVE
    }
    if set(active.values()) != active_entry_keys:
        raise V11LibraryError(
            "active_by_question_type 必须完整且只引用 state=active 的条目")
    loaded: list[SkillSpecV11] = []
    for question_type, key in sorted(active.items()):
        entry = entries.get(key)
        if not isinstance(entry, dict):
            raise V11LibraryError(f"active 条目不存在: {question_type} -> {key}")
        if entry.get("question_type") != question_type:
            raise V11LibraryError(
                f"active 题型与条目不一致: {question_type} != "
                f"{entry.get('question_type')}")
        loaded.append(loaded_by_key[key])
    return loaded


def _write_create_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise V11LibraryError(f"不可变文件已存在且内容不同: {path}")
        return
    path.write_bytes(data)


def _write_pointer_atomic(store_dir: Path, snapshot_id: str) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    pointer = store_dir / "active_snapshot.json"
    temporary = store_dir / f".active_snapshot.{uuid.uuid4().hex[:8]}.tmp"
    temporary.write_bytes(_json_bytes({"snapshot_id": snapshot_id}))
    os.replace(temporary, pointer)


def write_v11_snapshot(
    library_root: str | Path,
    snapshot: dict,
    manifest: dict,
    *,
    activate: bool = False,
    method_context_max_chars: int = DEFAULT_METHOD_CONTEXT_MAX_CHARS,
) -> None:
    """Write immutable snapshot/manifest files and optionally switch the pointer."""
    root = Path(library_root)
    snapshot_id = str(snapshot.get("snapshot_id") or "")
    if not _SAFE_ID_RE.fullmatch(snapshot_id):
        raise V11LibraryError(f"snapshot_id 不是安全标识符: {snapshot_id!r}")
    validate_v11_snapshot(
        snapshot,
        library_root=root,
        method_context_max_chars=method_context_max_chars,
    )
    claimed = str(manifest.get("manifest_sha256") or "")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    if payload != _manifest_for(snapshot):
        raise V11LibraryError("manifest 内容与 snapshot 不一致")
    actual = canonical_json_sha256(payload)
    if claimed != actual or snapshot.get("manifest_hash") != actual:
        raise V11LibraryError("manifest hash 与 snapshot 登记不一致")
    manifest_path = resolve_source_like_ref(root, str(snapshot["manifest_ref"]))
    snapshot_path = root / "snapshots" / f"snapshot_{snapshot_id}.json"
    _write_create_once(manifest_path, _json_bytes(manifest))
    _write_create_once(snapshot_path, _json_bytes(snapshot))
    if activate:
        _write_pointer_atomic(root / "snapshots", str(snapshot["snapshot_id"]))


def resolve_source_like_ref(library_root: str | Path, relative_ref: str) -> Path:
    """Resolve a non-source library reference with the same path protections."""
    value = str(relative_ref or "")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise V11LibraryError(f"不安全的库内引用: {relative_ref!r}")
    root = Path(library_root).resolve()
    target = root.joinpath(*path.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise V11LibraryError(f"库内引用越界: {relative_ref!r}") from exc
    return target


def publish_v11_candidate(
    library_root: str | Path,
    candidate: SkillCandidateV11,
    *,
    expected_parent_snapshot_id: str | None = None,
    method_context_max_chars: int = DEFAULT_METHOD_CONTEXT_MAX_CHARS,
) -> tuple[dict, dict]:
    """Publish one complete candidate and replace the lineage's active pointer."""
    from skill3d.skills.promote_atomic import publish_lock, read_active_snapshot
    from skill3d.skills.registry import active_snapshot_provenance

    root = Path(library_root)
    store_dir = root / "snapshots"
    with publish_lock(store_dir):
        before = read_active_snapshot(store_dir)
        if before.get("schema_version") != V11_SNAPSHOT_SCHEMA:
            raise V11LibraryError("v11 候选只能发布到 v11 父快照")
        provenance_snapshot_id, provenance_hash = active_snapshot_provenance(store_dir)
        if provenance_snapshot_id != str(before.get("snapshot_id")) or \
                provenance_hash != str(before.get("manifest_hash") or ""):
            raise V11LibraryError("v11 父快照 manifest 校验失败，拒绝发布")
        parent_specs = {
            f"{spec.skill_id}@{spec.version}": spec
            for spec in validate_v11_snapshot(
                before,
                library_root=root,
                method_context_max_chars=method_context_max_chars,
            )
        }
        expected = expected_parent_snapshot_id or candidate.parent_snapshot_id
        if str(before.get("snapshot_id")) != str(expected):
            raise V11LibraryError(
                f"active 快照不是候选父快照: expected={expected}, "
                f"actual={before.get('snapshot_id')}")
        entries = json.loads(json.dumps(before["entries"]))
        active = dict(before["active_by_question_type"])
        parent_key = candidate.parent_skill_version
        parent = entries.get(parent_key)
        if not isinstance(parent, dict):
            raise V11LibraryError(f"父版本不在 active 快照: {parent_key}")
        spec = candidate.full_skill_spec
        if parent.get("question_type") != spec.question_type:
            raise V11LibraryError("首期修订不得改变规范题型")
        if active.get(spec.question_type) != parent_key:
            raise V11LibraryError(f"父版本不是题型 {spec.question_type} 的 active Skill")
        parent_spec = parent_specs.get(parent_key)
        if parent_spec is None:
            raise V11LibraryError(f"父版本不是当前 active Skill: {parent_key}")
        candidate_key = candidate.candidate_skill_version
        if candidate_key in entries:
            raise V11LibraryError(f"候选版本已存在: {candidate_key}")
        if spec.content_sha256 == parent_spec.content_sha256:
            raise V11LibraryError("候选与父版本内容完全相同")
        if len(spec.skill_md) > int(method_context_max_chars):
            raise V11LibraryError(
                f"候选完整 SKILL.md 为 {len(spec.skill_md)} 字符，"
                f"超过服务限制 {method_context_max_chars}")

        source_ref = source_ref_for(spec)
        source_path = resolve_source_ref(root, source_ref)
        _write_create_once(source_path, spec.skill_md.encode("utf-8"))
        loaded = load_skill_source_v11(
            root,
            source_ref=source_ref,
            skill_id=spec.skill_id,
            version=spec.version,
            question_type=spec.question_type,
        )
        source_diff = "".join(difflib.unified_diff(
            parent_spec.skill_md.splitlines(keepends=True),
            spec.skill_md.splitlines(keepends=True),
            fromfile=parent_key,
            tofile=candidate_key,
        ))
        diff_sha256 = hashlib.sha256(source_diff.encode("utf-8")).hexdigest()
        parent["state"] = V11_ENTRY_HISTORICAL
        entries[candidate_key] = _entry_of(
            loaded,
            parent_version=parent_key,
            state=V11_ENTRY_ACTIVE,
            created_by="offline_revision",
            candidate_id=candidate.candidate_id,
            source_run_ref=candidate.source_run_ref,
            modification_reason=candidate.modification_reason,
            diff_sha256=diff_sha256,
        )
        active[spec.question_type] = candidate_key
        generation = int(before.get("generation", 0)) + 1
        snapshot_id = (
            f"S{generation}-{spec.skill_id}-{spec.version}-{candidate.candidate_id}")
        snapshot = {
            "schema_version": V11_SNAPSHOT_SCHEMA,
            "snapshot_id": snapshot_id,
            "parent_snapshot_id": before["snapshot_id"],
            "generation": generation,
            "skill_versions": sorted(entries),
            "active_by_question_type": dict(sorted(active.items())),
            "manifest_ref": (
                PurePosixPath("manifests") / f"manifest_{snapshot_id}.json"
            ).as_posix(),
            "manifest_hash": "",
            "entries": entries,
        }
        manifest = _manifest_for(snapshot)
        manifest_hash = canonical_json_sha256(manifest)
        manifest["manifest_sha256"] = manifest_hash
        snapshot["manifest_hash"] = manifest_hash
        validate_v11_snapshot(
            snapshot,
            library_root=root,
            method_context_max_chars=method_context_max_chars,
        )
        write_v11_snapshot(
            root,
            snapshot,
            manifest,
            activate=False,
            method_context_max_chars=method_context_max_chars,
        )
        receipt = {
            "candidate_id": candidate.candidate_id,
            "skill_version": candidate_key,
            "parent_skill_version": parent_key,
            "snapshot_before": before["snapshot_id"],
            "snapshot_after": snapshot_id,
            "manifest_hash": manifest_hash,
            "modification_reason": candidate.modification_reason,
            "source_run_ref": candidate.source_run_ref,
            "source_diff": source_diff,
            "diff_sha256": diff_sha256,
            "rollback_ref": (
                PurePosixPath("snapshots")
                / f"snapshot_{before['snapshot_id']}.json"
            ).as_posix(),
        }
        receipt_path = root / "validation" / "promotion" / f"{candidate.candidate_id}.json"
        _write_create_once(receipt_path, _json_bytes(receipt))
        _write_pointer_atomic(store_dir, snapshot_id)
    return snapshot, receipt


__all__ = [
    "LoadedSkillV11",
    "V11LibraryError",
    "V11_ENTRY_ACTIVE",
    "V11_ENTRY_HISTORICAL",
    "V11_LIBRARY_FORMAT",
    "V11_SNAPSHOT_SCHEMA",
    "build_v11_snapshot",
    "canonical_json_sha256",
    "load_skill_source_v11",
    "publish_v11_candidate",
    "resolve_source_like_ref",
    "resolve_source_ref",
    "source_ref_for",
    "validate_source_ref",
    "validate_v11_snapshot",
    "write_v11_snapshot",
]
