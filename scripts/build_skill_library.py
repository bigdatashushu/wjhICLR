#!/usr/bin/env python3
"""Build and validate the repository-local v9 Skill library.

Usage from the repository root::

    python scripts/build_skill_library.py --bootstrap /path/to/harness3d_s0_bundle.json
    python scripts/build_skill_library.py --check

``SKILL.md`` files are the only editable method sources.  Runtime JSON,
snapshots and manifests are deterministic derived files and are never read
from candidate or future-candidate directories by the online loader.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from skill3d.skills.source_compiler import (
    COMPILER_VERSION,
    SkillSourceError,
    compile_directory,
    sha256_bytes,
    sha256_file,
    verify_bundle,
)


LIBRARY_VERSION = "harness3d-skill-library/1.0"
S0_SNAPSHOT_ID = "S0-seed-20260925-v1"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_source_path(path: str) -> str:
    prefix = "skills/"
    if not path.startswith(prefix) or not path.endswith("/SKILL.md"):
        raise SkillSourceError(f"不是允许的 Skill 源路径: {path}")
    return path[len(prefix):]


def bootstrap_sources(bundle_path: Path, library_root: Path) -> dict[str, str]:
    """Verify the transport bundle and materialize only its source tree."""
    bundle = verify_bundle(bundle_path)
    if library_root.exists() and any(library_root.iterdir()):
        raise SkillSourceError(f"目标 Skill 库非空，拒绝覆盖: {library_root}")
    library_root.mkdir(parents=True, exist_ok=True)
    archive = library_root / "imports" / "harness3d_s0_bundle.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle_path, archive)
    archive_sha = sha256_file(archive)
    source_digests: dict[str, str] = {}
    for item in bundle["files"]:
        path = str(item["path"])
        if path.startswith("skills/") and path.endswith("/SKILL.md"):
            rel = _relative_source_path(path)
            dest = library_root / "skills" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = str(item["content"]).encode("utf-8")
            dest.write_bytes(data)
            source_digests[dest.relative_to(library_root).as_posix()] = sha256_bytes(data)
    if len(source_digests) != 8:
        raise SkillSourceError(f"bundle 中应有 8 个 Skill 源，实际为 {len(source_digests)}")
    # Preserve the transport bundle identity and the original document-level
    # runtime material for audit; neither directory is an online source.
    imported_meta = {
        "bundle_format": bundle["bundle_format"],
        "source_revision": bundle.get("source_revision"),
        "bundle_sha256": archive_sha,
        "file_count": bundle["file_count"],
        "source_paths": sorted(source_digests),
        "source_digests": source_digests,
    }
    _write_json(library_root / "imports" / "s0_bundle_receipt.json", imported_meta)
    return source_digests


def _compile_and_build(library_root: Path, *, check: bool) -> dict[str, Any]:
    source_root = library_root / "skills"
    generated_root = library_root / "generated"
    snapshots_root = library_root / "snapshots"
    manifests_root = library_root / "manifests"
    validation_root = library_root / "validation"
    compiled = compile_directory(source_root, require_all_tasks=True)
    if len(compiled) != 8:
        raise SkillSourceError(f"S0 必须包含 8 条 Skill，实际为 {len(compiled)}")

    index_entries: list[dict[str, Any]] = []
    spec_by_key: dict[str, dict[str, Any]] = {}
    for item in compiled:
        spec = item.spec
        key = f"{spec.skill_id}@{spec.version}"
        generated_rel = f"generated/{spec.skill_id}/{spec.version}.json"
        generated_path = library_root / generated_rel
        generated_value = spec.model_dump(mode="json")
        generated_data = _json_bytes(generated_value)
        generated_hash = sha256_bytes(generated_data)
        if check:
            if not generated_path.is_file():
                raise SkillSourceError(f"缺少派生文件: {generated_path}")
            if generated_path.read_bytes() != generated_data:
                raise SkillSourceError(f"派生 SkillSpec 已过期: {generated_path}")
        else:
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            generated_path.write_bytes(generated_data)
        source_rel = item.source_path
        # compile_directory returns a repository-relative path when called on
        # skill_library/skills; normalize it for the library manifest.
        marker = "skill_library/"
        if marker in source_rel:
            source_rel = source_rel.split(marker, 1)[1]
        index_entries.append({
            "skill_version": key,
            "skill_id": spec.skill_id,
            "version": spec.version,
            "canonical_question_type": spec.applicable_question_types[0],
            "family": spec.skill_family,
            "source_path": source_rel,
            "source_sha256": item.source_sha256,
            "generated_path": generated_rel,
            "generated_sha256": generated_hash,
            "document_schema_version": "8.0",
            "runtime_schema": "skill3d.SkillSpec/v6-compatible",
            "compiler_version": COMPILER_VERSION,
            "source_validation": "document_checked_runtime_pending",
        })
        spec_by_key[key] = generated_value

    index = {
        "library_format": LIBRARY_VERSION,
        "compiler_version": COMPILER_VERSION,
        "snapshot_id": S0_SNAPSHOT_ID,
        "source_format_version": "1.0",
        "document_export_schema_version": "8.0",
        "entries": sorted(index_entries, key=lambda x: x["skill_version"]),
    }
    index_digest = sha256_bytes(_json_bytes(index))
    index["index_sha256"] = index_digest
    index_path = generated_root / "index.json"
    if check:
        if not index_path.is_file() or _read_json(index_path) != index:
            raise SkillSourceError(f"generated/index.json 与源文件不一致: {index_path}")
    else:
        _write_json(index_path, index)

    entries: dict[str, Any] = {}
    for row in index["entries"]:
        key = row["skill_version"]
        entries[key] = {
            "root_candidate_id": row["skill_id"],
            "candidate_type": "skill",
            "spec_content": json.dumps(spec_by_key[key], ensure_ascii=False,
                                        indent=2, sort_keys=True),
            "parent_version": None,
            "created_by": "s0_import",
            "source_path": row["source_path"],
            "source_sha256": row["source_sha256"],
            "generated_path": row["generated_path"],
            "generated_sha256": row["generated_sha256"],
            "compiler_version": COMPILER_VERSION,
            "document_schema_version": "8.0",
        }
    manifest = {
        "library_format": LIBRARY_VERSION,
        "snapshot_id": S0_SNAPSHOT_ID,
        "document_schema_version": "8.0",
        "runtime_schema": "skill3d.SkillSpec/v6-compatible",
        "compiler_version": COMPILER_VERSION,
        "source_revision": "c79deab894c5274ab9bd6285b10276084216ed44",
        "entries": index["entries"],
        "runtime_verified": False,
        "production_activation": False,
        "validation_status": "document/static imported; real runtime pending",
    }
    manifest_digest = sha256_bytes(_json_bytes(manifest))
    manifest["manifest_sha256"] = manifest_digest
    manifest_path = manifests_root / "library_manifest.json"
    if check:
        if not manifest_path.is_file() or _read_json(manifest_path) != manifest:
            raise SkillSourceError(f"manifest 与源文件不一致: {manifest_path}")
    else:
        _write_json(manifest_path, manifest)

    snapshot = {
        "schema_version": "runtime-skill-snapshot/1.0",
        "document_schema_version": "8.0",
        "snapshot_id": S0_SNAPSHOT_ID,
        "parent_snapshot_id": None,
        "generation": 0,
        "skill_versions": [row["skill_version"] for row in index["entries"]],
        "manifest_hash": manifest_digest,
        "validation_receipt_refs": ["validation/s0_import_receipt.json"],
        "compatibility": {
            "skill_source_format": "harness3d-skill-source/1.0",
            "target_skill_schema": "skill3d.SkillSpec/v6-compatible",
            "document_export_schema": "8.0",
            "retriever": "canonical-task-hard-filter-before-ranking",
            "runtime_repository_revision": None,
            "runtime_verified": False,
            "production_activation": False,
        },
        "entries": entries,
    }
    snapshot_path = snapshots_root / f"snapshot_{S0_SNAPSHOT_ID}.json"
    pointer_path = snapshots_root / "active_snapshot.json"
    if check:
        if not snapshot_path.is_file() or _read_json(snapshot_path) != snapshot:
            raise SkillSourceError(f"S0 snapshot 与源文件不一致: {snapshot_path}")
        if not pointer_path.is_file() or _read_json(pointer_path) != {"snapshot_id": S0_SNAPSHOT_ID}:
            raise SkillSourceError(f"active pointer 不指向 S0: {pointer_path}")
    else:
        _write_json(snapshot_path, snapshot)
        _write_json(pointer_path, {"snapshot_id": S0_SNAPSHOT_ID})

    receipt = {
        "check_type": "s0_skill_library_static_import",
        "library_format": LIBRARY_VERSION,
        "snapshot_id": S0_SNAPSHOT_ID,
        "source_count": len(index["entries"]),
        "question_type_count": len({row["canonical_question_type"] for row in index["entries"]}),
        "bundle_verified": (library_root / "imports" / "s0_bundle_receipt.json").is_file(),
        "generated_specs_validated": True,
        "active_pointer_created": pointer_path.is_file(),
        "runtime_verified": False,
        "real_model_tool_episode": "not_run",
        "benchmark_score": "not_measured",
        "status": "passed" if not check else "passed_check",
    }
    receipt_path = validation_root / "s0_import_receipt.json"
    if check:
        if not receipt_path.is_file():
            raise SkillSourceError(f"缺少导入收据: {receipt_path}")
    else:
        _write_json(receipt_path, receipt)
    return {
        "snapshot_id": S0_SNAPSHOT_ID,
        "manifest_sha256": manifest_digest,
        "skill_count": len(index["entries"]),
        "question_type_count": len({row["canonical_question_type"] for row in index["entries"]}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", default="skill_library")
    parser.add_argument("--bootstrap", help="从 harness3d_s0_bundle.json 创建源文件")
    parser.add_argument("--check", action="store_true", help="只校验派生文件，不写派生输出")
    args = parser.parse_args()
    root = Path(args.library_root)
    try:
        if args.bootstrap:
            bootstrap_sources(Path(args.bootstrap), root)
        result = _compile_and_build(root, check=args.check)
    except (OSError, ValueError, SkillSourceError, json.JSONDecodeError) as exc:
        parser.error(f"Skill library 构建失败: {type(exc).__name__}: {exc}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
