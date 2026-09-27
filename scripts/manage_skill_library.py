#!/usr/bin/env python3
"""Incremental v9 Skill source/candidate management.

This command intentionally keeps future records outside the online loader. Only
``promote`` calls the atomic snapshot writer, and it validates inline runtime
SkillSpec content before changing the active pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from skill3d.schemas import CandidateRevision
from skill3d.skills.library import (
    candidate_revision_from_record,
    load_candidate_record,
    static_check_skill_spec,
    write_candidate_record,
)
from skill3d.skills.promote_atomic import promote
from skill3d.skills.source_compiler import (
    COMPILER_VERSION,
    SkillSourceError,
    compile_directory,
    sha256_bytes,
)


def _retrieval_limit() -> int:
    """§13.5 方法上下文上限（= 服务限制判据）。读冻结配置，读不到用默认值。"""
    from skill3d.online.config import DEFAULT_CONFIG, load_config
    from skill3d.routing.retrieval_policy import retrieval_policy_from_config

    return int(retrieval_policy_from_config(load_config(DEFAULT_CONFIG))
               .method_context_max_chars)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def compile_sources(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Compile an incremental source tree without S0's eight-task constraint."""
    compiled = compile_directory(source_root, require_all_tasks=False)
    entries: list[dict[str, Any]] = []
    output_specs: dict[str, dict[str, Any]] = {}
    for item in compiled:
        spec = item.spec
        rel = Path("generated") / spec.skill_id / f"{spec.version}.json"
        target = output_root / rel
        payload = spec.model_dump(mode="json")
        data = _json_bytes(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise SkillSourceError(f"派生 Skill 已存在且内容不同，拒绝覆盖: {target}")
        target.write_bytes(data)
        key = f"{spec.skill_id}@{spec.version}"
        output_specs[key] = payload
        entries.append({
            "skill_version": key,
            "skill_id": spec.skill_id,
            "version": spec.version,
            "canonical_question_type": spec.applicable_question_types[0],
            "family": spec.skill_family,
            "source_path": item.source_path,
            "source_sha256": item.source_sha256,
            "generated_path": rel.as_posix(),
            "generated_sha256": hashlib.sha256(data).hexdigest(),
            "compiler_version": COMPILER_VERSION,
            "source_validation": "document_checked_runtime_pending",
        })
    index = {
        "library_format": "harness3d-skill-library/1.0",
        "compiler_version": COMPILER_VERSION,
        "source_format_version": "1.0",
        "entries": sorted(entries, key=lambda row: row["skill_version"]),
    }
    index["index_sha256"] = sha256_bytes(_json_bytes(index))
    index_path = output_root / "generated" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(_json_bytes(index))
    return {"skill_count": len(entries), "index_path": str(index_path),
            "entries": index["entries"]}


def _load_revision(args: argparse.Namespace, library_root: Path) -> CandidateRevision:
    record_path = Path(args.record)
    record = load_candidate_record(record_path)
    spec_content = Path(args.spec_file).read_text(encoding="utf-8")
    return candidate_revision_from_record(
        record,
        record_path=record_path,
        spec_content=spec_content,
        revision_id=args.revision_id or record.candidate_id,
        created_by=args.created_by,
    )


def _spec_of_revision(revision: CandidateRevision) -> Any:
    """revision.spec_content → 严格校验后的 SkillSpec（静态检查的输入）。"""
    from skill3d.schemas import SkillSpec

    return SkillSpec.model_validate(json.loads(revision.spec_content))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compile_parser = sub.add_parser("compile", help="编译普通/增量 Skill 源目录")
    compile_parser.add_argument("--source-root", required=True)
    compile_parser.add_argument("--output-root", default="skill_library")

    create_parser = sub.add_parser("stage-candidate", help="原子写入 future candidate record")
    create_parser.add_argument("--library-root", default="skill_library")
    create_parser.add_argument("--record-json", required=True,
                               help="SkillLibraryCandidate JSON 文件")

    for name, help_text in (("validate-candidate", "验证 future candidate 与 runtime SkillSpec"),
                            ("promote", "显式验证并原子晋升 candidate")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--library-root", default="skill_library")
        p.add_argument("--record", required=True)
        p.add_argument("--spec-file", required=True)
        p.add_argument("--revision-id", default="")
        p.add_argument("--created-by", choices=["gpt6_induction", "gpt6_revision", "human"],
                       default="human")

    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            result = compile_sources(Path(args.source_root), Path(args.output_root))
        elif args.command == "stage-candidate":
            root = Path(args.library_root)
            record = load_candidate_record(args.record_json)
            path, digest = write_candidate_record(root, record, future=True)
            result = {"candidate_id": record.candidate_id,
                      "path": str(path), "sha256": digest, "status": "future"}
        else:
            root = Path(args.library_root)
            revision = _load_revision(args, root)
            spec = _spec_of_revision(revision)
            limit = _retrieval_limit()
            problems = static_check_skill_spec(spec, method_context_max_chars=limit)
            if args.command == "validate-candidate":
                # §14.4：静态检查不合格记 quarantine（不晋升）。
                result = {"revision_id": revision.revision_id,
                          "root_candidate_id": revision.root_candidate_id,
                          "candidate_record_ref": revision.candidate_record_ref,
                          "static_check": {"method_context_max_chars": limit,
                                           "problems": problems},
                          "status": ("quarantined" if problems else "validated")}
                if problems:
                    parser.error(
                        f"candidate {revision.revision_id} 静态检查不合格 → quarantine"
                        f"（§14.4）：{problems}")
            else:
                log: list[dict[str, Any]] = []
                snap = promote(root / "snapshots", revision.model_copy(update={"status": "promoted"}),
                               promotion_log=log, strict_skill_specs=True,
                               method_context_max_chars=limit)
                result = {"snapshot_id": snap["snapshot_id"],
                          "promotion": log[-1] if log else {},
                          "static_check": {"method_context_max_chars": limit,
                                           "problems": problems}}
    except (OSError, ValueError, json.JSONDecodeError, SkillSourceError) as exc:
        parser.error(f"Skill library 操作失败: {type(exc).__name__}: {exc}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
