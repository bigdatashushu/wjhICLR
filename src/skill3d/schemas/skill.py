"""Complete SKILL.md runtime values and v11 candidate identities."""
import hashlib
import re
from typing import Optional

import yaml
from pydantic import field_validator
from . import Spec

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_V11_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_V11_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_QUESTION_TYPE_RE = re.compile(r"^[a-z0-9_]+$")


def normalize_skill_markdown(text: str) -> str:
    """Return the canonical v11 Skill source representation.

    Canonical sources use UTF-8 text, LF line endings, no NUL byte, and exactly
    one trailing newline. Whitespace inside the document is otherwise left
    untouched because the same string is delivered to the model and hashed.
    """
    if not isinstance(text, str):
        raise TypeError("skill_md 必须是字符串")
    if "\x00" in text:
        raise ValueError("skill_md 不允许包含 NUL 字节")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.rstrip("\n") + "\n"


def parse_skill_markdown(text: str) -> tuple[dict, str]:
    """Parse v11 frontmatter and return ``(metadata, body)`` without rewriting."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != "---":
        raise ValueError("SKILL.md 必须以 YAML frontmatter 开始")
    end = next(
        (index for index, line in enumerate(lines[1:], start=1)
         if line.rstrip("\n") == "---"),
        None,
    )
    if end is None:
        raise ValueError("SKILL.md frontmatter 缺少结束标记")
    try:
        front = yaml.safe_load("".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"SKILL.md frontmatter 解析失败: {exc}") from exc
    if not isinstance(front, dict):
        raise ValueError("SKILL.md frontmatter 必须是 mapping")
    name = front.get("name")
    description = front.get("description")
    if not isinstance(name, str) or not _V11_NAME_RE.fullmatch(name) or len(name) > 64:
        raise ValueError(
            "SKILL.md name 必须是不超过 64 字符的小写字母/数字/单连字符名称")
    if not isinstance(description, str) or not description.strip() \
            or len(description) > 1024:
        raise ValueError("SKILL.md description 必须是 1-1024 字符的非空字符串")
    metadata = front.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata.items()):
            raise ValueError("SKILL.md metadata 的键和值必须都是字符串")
    body = "".join(lines[end + 1:])
    if not body.strip():
        raise ValueError("SKILL.md 正文不能为空")
    return front, body


class SkillSpecV11(Spec):
    """v11 runtime value: framework identity plus the exact complete Skill source."""

    skill_id: str
    version: str
    question_type: str
    skill_md: str

    @field_validator("skill_id")
    @classmethod
    def _safe_skill_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _V11_ID_RE.fullmatch(value) or "@" in value:
            raise ValueError(f"skill_id 不是安全标识符: {value!r}")
        return value

    @field_validator("question_type")
    @classmethod
    def _valid_question_type(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _QUESTION_TYPE_RE.fullmatch(value):
            raise ValueError(f"question_type 非法: {value!r}")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _SEMVER_RE.fullmatch(value):
            raise ValueError(f"非法语义版本: {value!r}")
        return value

    @field_validator("skill_md")
    @classmethod
    def _canonical_complete_source(cls, value: str) -> str:
        normalized = normalize_skill_markdown(value)
        if value != normalized:
            raise ValueError("skill_md 必须使用 LF 且文件末尾恰好一个换行")
        parse_skill_markdown(value)
        return value

    @property
    def semver(self) -> str:
        return self.version

    @property
    def task_type(self) -> str:
        return self.question_type

    @property
    def applicable_question_types(self) -> list[str]:
        """Compatibility view for shared trace and prompt plumbing."""
        return [self.question_type]

    @property
    def name(self) -> str:
        return str(parse_skill_markdown(self.skill_md)[0]["name"])

    @property
    def description(self) -> str:
        return str(parse_skill_markdown(self.skill_md)[0]["description"])

    @property
    def body(self) -> str:
        return parse_skill_markdown(self.skill_md)[1]

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.skill_md.encode("utf-8")).hexdigest()


class SkillCandidateV11(Spec):
    """Framework-owned identity around a complete v11 candidate source."""

    candidate_id: str
    parent_snapshot_id: str
    parent_skill_version: str
    full_skill_spec: SkillSpecV11
    modification_reason: str = ""
    source_run_ref: str = ""

    @field_validator("candidate_id")
    @classmethod
    def _safe_candidate_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _V11_ID_RE.fullmatch(value):
            raise ValueError(f"candidate_id 不是安全标识符: {value!r}")
        return value

    def model_post_init(self, __context) -> None:  # noqa: D105
        parent_id, separator, parent_version = self.parent_skill_version.partition("@")
        if not separator or not _SEMVER_RE.fullmatch(parent_version):
            raise ValueError(
                f"parent_skill_version 必须是 skill_id@version: "
                f"{self.parent_skill_version!r}")
        if parent_id != self.full_skill_spec.skill_id:
            raise ValueError(
                f"候选不得跨谱系: parent={parent_id}, "
                f"candidate={self.full_skill_spec.skill_id}")
        old = tuple(int(part) for part in parent_version.split("."))
        new = tuple(int(part) for part in self.full_skill_spec.version.split("."))
        if new <= old:
            raise ValueError(
                f"候选版本 {self.full_skill_spec.version} 必须高于父版本 {parent_version}")

    @property
    def candidate_skill_version(self) -> str:
        spec = self.full_skill_spec
        return f"{spec.skill_id}@{spec.version}"


class RetrievedSkill(Spec):
    """M7 Skill 检索输出（§4 M7 字段 6 + v6 证据签名匹配）。"""

    skill_id: str = ""
    skill_version: str = ""
    score: float = 0.0
    hard_filter_passed: bool = False
    # v6：命中的证据签名（= 检索时的 EvidenceProfile 快照的投影），
    # 让"这条 Skill 是在哪种证据状态下被选中的"可审计（§17.2 分开积累）
    matched_evidence_signature: dict[str, str] = {}
    # 米制 Skill 的 gate 版本匹配结果（§13.6）
    gate_version_matched: Optional[bool] = None

    @property
    def skill_semver(self) -> str:
        """v5 字段名别名。"""
        return self.skill_version
