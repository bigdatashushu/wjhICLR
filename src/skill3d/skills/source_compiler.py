"""Compile v9 ``SKILL.md`` sources into the runtime SkillSpec contract.

The repository-facing Skill format intentionally stays separate from the
runtime schema. The S0 bundle was authored against the document-level v8
SkillSpec, while the running code still consumes the v6-compatible
``SkillSpec`` model. This module is the explicit, deterministic migration
boundary between the two representations.

It is deliberately independent of online inference and offline governance so
that source validation can run in either environment without importing an
answer model or a provider client.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml

from skill3d.routing.task_classifier import TASK_TYPES
from skill3d.schemas import SkillSpec


COMPILER_VERSION = "harness3d-skill-compiler/1.0"
REQUIRED_SECTIONS = (
    "目标",
    "适用条件",
    "执行约束",
    "证据条件",
    "求解步骤",
    "提交前检查",
    "代码示例",
    "失败教训",
    "局限",
    "答案依据",
    "来源",
)
FAMILY_NAMES = {"counting", "metric", "relative_geometry", "route", "appearance"}
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_STEP_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")


class SkillSourceError(ValueError):
    """Raised when a source or generated artifact violates the library contract."""


@dataclass(frozen=True)
class CompiledSkill:
    """A validated runtime spec plus source and generated file identities."""

    source_path: str
    source_sha256: str
    generated_path: str
    generated_sha256: str
    spec: SkillSpec
    source_metadata: dict[str, str]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _ensure_string_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise SkillSourceError(f"{label} 必须为 mapping")
    out: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise SkillSourceError(f"{label} 的键和值必须都是字符串")
        out[key] = item
    return out


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse the YAML front matter without accepting arbitrary markdown headers."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillSourceError("SKILL.md 必须以 YAML front matter 开始")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise SkillSourceError("front matter 缺少结束标记") from exc
    raw = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(raw, dict):
        raise SkillSourceError("front matter 必须为 mapping")
    if "name" not in raw or "description" not in raw or "metadata" not in raw:
        raise SkillSourceError("front matter 必须包含 name、description、metadata")
    raw["metadata"] = _ensure_string_mapping(raw["metadata"], "metadata")
    return raw, "\n".join(lines[end + 1 :])


def parse_sections(body: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            current = match.group(1).strip()
            if current in sections:
                raise SkillSourceError(f"重复章节: {current}")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    missing = [name for name in REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise SkillSourceError(f"缺少固定章节: {missing}")
    unknown = sorted(set(sections) - set(REQUIRED_SECTIONS))
    if unknown:
        raise SkillSourceError(f"存在未登记章节: {unknown}")
    if list(sections) != list(REQUIRED_SECTIONS):
        raise SkillSourceError("固定章节顺序不符合 source format 1.0")
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _fenced_value(text: str, label: str) -> Any:
    match = re.search(r"```(?:yaml|json)\s*\n(.*?)\n```", text, re.DOTALL)
    if not match:
        raise SkillSourceError(f"{label} 缺少 yaml/json fenced block")
    try:
        value = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise SkillSourceError(f"{label} YAML/JSON 解析失败: {exc}") from exc
    return value


def _parse_steps(text: str) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for line in text.splitlines():
        match = _STEP_RE.match(line)
        if match:
            steps.append({"step_id": f"step_{int(match.group(1)):02d}",
                          "instruction": match.group(2).strip()})
    if not steps:
        raise SkillSourceError("求解步骤必须包含编号步骤")
    return steps


def _parse_bullets(text: str, label: str) -> list[str]:
    values = [line.strip()[2:].strip() for line in text.splitlines()
              if line.strip().startswith("-")]
    if not values:
        raise SkillSourceError(f"{label} 必须包含 Markdown 项目列表")
    return values


def _normalize_signature(evidence: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(evidence, dict):
        raise SkillSourceError("证据条件 fenced block 必须为 mapping")
    hard = evidence.get("hard_requirements") or {}
    prefs = evidence.get("evidence_preferences") or {}
    if not isinstance(hard, dict) or not isinstance(prefs, dict):
        raise SkillSourceError("证据条件的 hard_requirements/evidence_preferences 必须为 mapping")
    image = hard.get("image_2d")
    if image != ["available", "degraded"]:
        raise SkillSourceError("source format 的 image_2d 硬前提必须是 [available, degraded]")
    # The source format stores acceptable states. The runtime signature stores
    # the least strict accepted state, so [available, degraded] becomes
    # ``degraded``. Preserve every other declared capability for candidate
    # Skills instead of silently reducing them to the S0 visual fallback.
    signature: dict[str, str] = {}
    order = {"unavailable": 0, "degraded": 1, "available": 2}
    for capability, accepted in hard.items():
        if not isinstance(capability, str):
            raise SkillSourceError("hard_requirements 能力名必须是字符串")
        values = accepted if isinstance(accepted, list) else [accepted]
        if not values or any(str(v) not in order for v in values):
            raise SkillSourceError(
                f"hard_requirements.{capability} 必须是 available/degraded/unavailable 列表")
        signature[capability] = min((str(v) for v in values), key=lambda v: order[v])
    return signature, {"hard_requirements": hard, "evidence_preferences": prefs}


def _runtime_description(front: dict[str, Any], sections: dict[str, str]) -> str:
    return "\n\n".join([
        str(front["description"]).strip(),
        "目标:\n" + sections["目标"],
        "适用条件:\n" + sections["适用条件"],
        "执行约束:\n" + sections["执行约束"],
        "提交前检查:\n" + sections["提交前检查"],
        "局限:\n" + sections["局限"],
    ])


def _runtime_call_graph(sections: dict[str, str], evidence: dict[str, Any]) -> str:
    prefs = evidence["evidence_preferences"]
    local = prefs.get("local_requirements", {})
    visual = prefs.get("visual_route", "")
    return "\n".join([
        "Follow the numbered method steps in the delivered SKILL.md.",
        "Evidence preferences: " + json.dumps(prefs.get("preferred", []), ensure_ascii=False),
        "Local conditions: " + json.dumps(local, ensure_ascii=False),
        "Visual fallback: " + str(visual),
        "Steps:\n" + sections["求解步骤"],
    ])


def compile_skill_source(source_path: str | Path) -> tuple[SkillSpec, dict[str, str]]:
    path = Path(source_path)
    text = path.read_text(encoding="utf-8")
    front, body = parse_front_matter(text)
    sections = parse_sections(body)
    name = front["name"]
    if not isinstance(name, str) or not name:
        raise SkillSourceError("name 必须是非空字符串")
    if path.parent.name != name:
        raise SkillSourceError(
            f"name 与目录不一致: name={name!r}, directory={path.parent.name!r}"
        )
    metadata = front["metadata"]
    required_meta = {
        "harness3d-skill-id",
        "harness3d-version",
        "harness3d-question-type",
        "harness3d-family",
        "harness3d-source-format",
        "harness3d-validation",
    }
    missing = sorted(required_meta - set(metadata))
    if missing:
        raise SkillSourceError(f"metadata 缺少字段: {missing}")
    skill_id = metadata["harness3d-skill-id"]
    version = metadata["harness3d-version"]
    question_type = metadata["harness3d-question-type"]
    family = metadata["harness3d-family"]
    if question_type not in TASK_TYPES:
        raise SkillSourceError(f"未知规范题型: {question_type}")
    if family not in FAMILY_NAMES:
        raise SkillSourceError(f"未知 Skill family: {family}")
    evidence_raw = _fenced_value(sections["证据条件"], "证据条件")
    signature, evidence = _normalize_signature(evidence_raw)
    steps = _parse_steps(sections["求解步骤"])
    checks = _parse_bullets(sections["提交前检查"], "提交前检查")
    limitations = _parse_bullets(sections["局限"], "局限")
    bases = _fenced_value(sections["答案依据"], "答案依据")
    if not isinstance(bases, list) or not all(isinstance(item, str) for item in bases):
        raise SkillSourceError("答案依据必须是字符串数组")
    examples = _fenced_value(sections["代码示例"], "代码示例")
    lessons = _fenced_value(sections["失败教训"], "失败教训")
    if not isinstance(examples, list) or not all(isinstance(item, str) for item in examples):
        raise SkillSourceError("代码示例必须是字符串数组")
    if not isinstance(lessons, list) or not all(isinstance(item, str) for item in lessons):
        raise SkillSourceError("失败教训必须是字符串数组")
    requires_metric = "metric_scale" in signature
    gate_version = metadata.get("harness3d-gate-version") or None
    if requires_metric and not gate_version:
        raise SkillSourceError(
            "声明 metric_scale 硬前提的 Skill 必须提供 metadata.harness3d-gate-version")
    spec = SkillSpec(
        skill_id=skill_id,
        version=version,
        applicable_question_types=[question_type],
        required_evidence_signature=signature,
        # S0 sources omit metric_scale and therefore retain visual fallback;
        # candidate sources may declare the full metric gate explicitly.
        requires_metric_evidence=requires_metric,
        applicable_gate_version=gate_version,
        skill_family=family,
        source="real",
        description=_runtime_description(front, sections),
        call_graph_template=_runtime_call_graph(sections, evidence),
        supported_coordinate_frames=["world"],
        validation_assertions=checks,
    )
    # Pydantic validation above is the strict runtime boundary. Keep source
    # metadata outside the runtime object so extra document fields cannot leak
    # into the online prompt contract.
    return spec, metadata


def compile_directory(
    source_root: str | Path,
    *,
    require_all_tasks: bool = False,
) -> list[CompiledSkill]:
    """Compile every direct ``*/SKILL.md`` source under ``source_root``.

    A normal library can contain multiple methods for one canonical question
    type.  The S0 bootstrap is the exceptional full-panel import and enables
    ``require_all_tasks`` explicitly so that this invariant does not constrain
    later candidate or version additions.
    """
    root = Path(source_root)
    if not root.is_dir():
        raise SkillSourceError(f"Skill source 根目录不存在: {root}")
    result: list[CompiledSkill] = []
    seen_tasks: set[str] = set()
    seen_versions: set[str] = set()
    for source_path in sorted(root.glob("*/SKILL.md")):
        spec, metadata = compile_skill_source(source_path)
        task = spec.applicable_question_types[0]
        skill_version = f"{spec.skill_id}@{spec.version}"
        if skill_version in seen_versions:
            raise SkillSourceError(f"Skill id/version 重复: {skill_version}")
        seen_versions.add(skill_version)
        seen_tasks.add(task)
        result.append(CompiledSkill(
            source_path=source_path.relative_to(root.parent.parent).as_posix()
            if len(source_path.parts) >= 2 else source_path.as_posix(),
            source_sha256=sha256_file(source_path),
            generated_path="",
            generated_sha256="",
            spec=spec,
            source_metadata=metadata,
        ))
    if not result:
        raise SkillSourceError(f"Skill source 根目录为空: {root}")
    if require_all_tasks and set(seen_tasks) != set(TASK_TYPES):
        raise SkillSourceError(
            "S0 必须覆盖全部规范题型，实际缺失: "
            f"{sorted(set(TASK_TYPES) - seen_tasks)}"
        )
    return result


def validate_relative_bundle_path(path: str) -> PurePosixPath:
    p = PurePosixPath(path)
    if not p.parts or p.is_absolute() or ".." in p.parts or "\\" in path:
        raise SkillSourceError(f"bundle 路径不安全: {path!r}")
    return p


def verify_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Verify a transport bundle without writing anything."""
    path = Path(bundle_path)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("bundle_format") != "harness3d-s0-source-bundle/1.0":
        raise SkillSourceError(f"不支持的 bundle_format: {bundle.get('bundle_format')!r}")
    files = bundle.get("files")
    if not isinstance(files, list) or len(files) != bundle.get("file_count"):
        raise SkillSourceError("bundle files/file_count 不一致")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise SkillSourceError("bundle 文件条目必须为 mapping")
        rel = validate_relative_bundle_path(str(item.get("path", ""))).as_posix()
        if rel in seen:
            raise SkillSourceError(f"bundle 路径重复: {rel}")
        seen.add(rel)
        content = item.get("content")
        if not isinstance(content, str):
            raise SkillSourceError(f"bundle 内容不是字符串: {rel}")
        if sha256_bytes(content.encode("utf-8")) != item.get("sha256"):
            raise SkillSourceError(f"bundle 内容校验失败: {rel}")
    return bundle


def materialize_bundle(bundle_path: str | Path, target_root: str | Path) -> dict[str, str]:
    """Write a verified bundle into a new directory and return file digests."""
    target = Path(target_root)
    if target.exists():
        raise SkillSourceError(f"目标目录已存在，拒绝覆盖: {target}")
    bundle = verify_bundle(bundle_path)
    target.mkdir(parents=True)
    written: dict[str, str] = {}
    for item in bundle["files"]:
        rel = validate_relative_bundle_path(item["path"])
        dest = target.joinpath(*rel.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = item["content"].encode("utf-8")
        dest.write_bytes(data)
        written[rel.as_posix()] = sha256_bytes(data)
    return written


def iter_runtime_specs(generated_root: str | Path) -> Iterable[tuple[Path, SkillSpec]]:
    root = Path(generated_root)
    for path in sorted(root.glob("S*/*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        yield path, SkillSpec.model_validate(raw)


__all__ = [
    "COMPILER_VERSION",
    "CompiledSkill",
    "FAMILY_NAMES",
    "REQUIRED_SECTIONS",
    "SkillSourceError",
    "compile_directory",
    "compile_skill_source",
    "iter_runtime_specs",
    "materialize_bundle",
    "parse_front_matter",
    "parse_sections",
    "sha256_bytes",
    "sha256_file",
    "validate_relative_bundle_path",
    "verify_bundle",
]
