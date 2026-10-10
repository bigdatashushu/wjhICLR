"""v11 active Skill loading and framework-owned semantic versions."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from skill3d.schemas import SkillSpecV11
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class SemverError(ValueError):
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


def active_snapshot_provenance(path: str | Path) -> tuple[str, str]:
    """Return ``(snapshot_id, manifest_hash)`` for the active pointer.

    Verify the current v11 manifest. Missing or malformed provenance is
    represented as an empty digest and never guessed.
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
    if not registered or snap.get("schema_version") != "runtime-skill-snapshot/2.0":
        return snapshot_id, ""
    # When the library manifest is present, independently recompute its
    # canonical payload digest. A stale registration is not provenance.
    manifest_ref = str(snap.get("manifest_ref") or "")
    if not manifest_ref:
        return snapshot_id, ""
    try:
        from .v11_library import resolve_source_like_ref

        manifest_path = resolve_source_like_ref(store_dir.parent, manifest_ref)
    except Exception:
        return snapshot_id, ""
    if not manifest_path.is_file():
        return snapshot_id, ""
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


def load_active_skills(
    path: str | Path,
) -> tuple[list[SkillSpecV11], list[str], str]:
    """Load the current v11 active snapshot; reject historical schemas."""
    from pathlib import Path as _Path

    from .promote_atomic import read_active_snapshot

    p = _Path(path)
    store_dir = p if p.is_dir() else p.parent
    pointer = store_dir / "active_snapshot.json"
    if not pointer.exists():
        return [], [f"无 active snapshot（{pointer} 不存在）→ 空 Skill"], "genesis"
    try:
        snapshot = read_active_snapshot(store_dir)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"active snapshot 读取失败: {type(exc).__name__}: {exc}") from exc
    schema = str(snapshot.get("schema_version") or "")
    if schema != "runtime-skill-snapshot/2.0":
        raise ValueError(
            "当前在线协议只接受 runtime-skill-snapshot/2.0；"
            f"收到 {schema or 'missing'}，历史快照请在对应 Git 提交运行")
    from .v11_library import validate_v11_snapshot

    snapshot_id = str(snapshot["snapshot_id"])
    skills = validate_v11_snapshot(snapshot, library_root=store_dir.parent)
    active_id, manifest_hash = active_snapshot_provenance(path)
    if active_id != snapshot_id or not manifest_hash or manifest_hash != snapshot.get("manifest_hash"):
        raise ValueError("v11 active snapshot manifest 校验失败")
    return skills, [], snapshot_id
