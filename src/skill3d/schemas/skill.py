"""Skill / 治理 Schema。

``SkillSpec`` 保留 v6-v10 的结构化运行合同，用于读取历史快照。
``SkillSpecV11`` 是 v11 的无损方法合同：完整 ``SKILL.md`` 是唯一内容源，
运行身份由框架字段承载，Markdown 正文不再拆成多份可独立修改的字段。

v6 与 v5 的三处硬性差异：

1. **检索依据从"产物 + 最低质量分"改为"题型 + 证据签名"**（D6/D8）：
   `required_evidence_signature`（能力→最低可接受状态）取代
   `requires_artifacts` + `minimum_quality` + `metric_scale_required`；
2. **允许在特定证据签名下分开积累**（§17.2）：同一题型的 Skill 可以在
   `metric_scale=unavailable` 签名与"全 available"签名下各有一套，
   **互不污染**（这是 v6 证据机制的核心收益之一）；
3. **米制 Skill 双重 fail-closed**（§13.6）：必须声明
   `requires_metric_evidence=True` + `applicable_gate_version`，
   检索（M7）与执行（M10）两次校验当前 gate 通过且版本匹配。

离线治理模型改为 DeepSeek-V4.1-Flash（§3.4）：`DeepSeekGovernanceDecision`。
"""

import hashlib
import re
from typing import Literal, Optional

import yaml
from pydantic import field_validator

from . import Spec
from .evidence import CAPABILITIES

SkillState = Literal["draft", "shadow", "canary", "promoted", "quarantined"]

# 5 个 Skill 族（§17.2 基线分组）
SKILL_FAMILIES: tuple[str, ...] = (
    "counting", "metric", "relative_geometry", "route", "appearance",
)
SkillFamily = Literal["counting", "metric", "relative_geometry", "route", "appearance"]

# 族 → 适用的规范题型（族内个体 Skill 只能声明该族的子集）
FAMILY_QUESTION_TYPES: dict[str, tuple[str, ...]] = {
    "counting": ("object_counting",),
    "metric": ("object_abs_distance", "object_size_estimation", "room_size_estimation"),
    "relative_geometry": ("object_rel_distance", "object_rel_direction"),
    "route": ("route_planning",),
    "appearance": ("obj_appearance_order",),
}

SkillSource = Literal["real", "mock_interface", "mock_replay", "mock_light"]

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


class SkillSpec(Spec):
    """题型级程序合成模板（非可执行工具/非答案，硬约束 15；v6 §5.8）。

    `required_evidence_signature` 的键必须是 `schemas.evidence.CAPABILITIES` 的成员，
    值必须是三值之一。**归纳器不得把该条件泛化/省略掉**（§17.3）—— 例如不得把
    "metric_scale>=degraded" 改写成无条件 Skill（那会让米制 Skill 在尺度不可用的
    场景里被检索出来并给出伪米制答案）。
    """

    skill_id: str
    version: str                          # MAJOR.MINOR.PATCH
    applicable_question_types: list[str]
    # 能力 → 最低可接受状态（"available" | "degraded" | "unavailable"）
    required_evidence_signature: dict[str, str] = {}
    requires_metric_evidence: bool = False
    applicable_gate_version: Optional[str] = None   # 米制 Skill 必填（§13.6）
    skill_family: SkillFamily
    source: SkillSource = "real"

    # ---- 模板内容（沿用 v4/v5 字段；进 prompt 的只有模板本身）----
    description: str = ""
    call_graph_template: str = ""
    supported_coordinate_frames: list[str] = ["world"]
    validation_assertions: list[str] = []

    # ---- v5 只读兼容（过渡期；新 Skill 一律用证据签名）----
    @property
    def semver(self) -> str:
        """v5 字段名别名（版本号语义一致）。"""
        return self.version

    @property
    def task_type(self) -> str:
        """v5 字段名别名：单一题型时返回它，否则返回族名（多题型 Skill）。"""
        qts = list(self.applicable_question_types)
        return qts[0] if len(qts) == 1 else self.skill_family

    def __init__(self, **data):
        """构造期校验（fail-closed）：签名键/值合法、米制声明完整、族与题型一致。"""
        sig = data.get("required_evidence_signature") or {}
        bad_keys = sorted(set(sig) - set(CAPABILITIES))
        if bad_keys:
            raise ValueError(
                f"required_evidence_signature 含未知能力 {bad_keys}；"
                f"词汇表见 schemas.evidence.CAPABILITIES")
        bad_vals = sorted({str(v) for v in sig.values()} - {"available", "degraded",
                                                            "unavailable"})
        if bad_vals:
            raise ValueError(
                f"required_evidence_signature 含非法状态值 {bad_vals}；"
                "只允许 available/degraded/unavailable")
        if data.get("requires_metric_evidence"):
            if "metric_scale" not in sig:
                raise ValueError(
                    "requires_metric_evidence=True 但 required_evidence_signature 未声明 "
                    "metric_scale（§13.6：归纳器不得把米制条件省略掉）")
            if not data.get("applicable_gate_version"):
                raise ValueError(
                    "米制 Skill 必须声明 applicable_gate_version（§13.6 双重 fail-closed）")
        fam = data.get("skill_family")
        if fam in FAMILY_QUESTION_TYPES:
            allowed = set(FAMILY_QUESTION_TYPES[fam])
            got = set(data.get("applicable_question_types") or [])
            extra = sorted(got - allowed)
            if extra:
                raise ValueError(
                    f"skill_family={fam} 不覆盖题型 {extra}；"
                    f"该族只允许 {sorted(allowed)}（§17.2 族基线）")
        super().__init__(**data)


class SkillCandidateV5(Spec):
    """v5 形状的候选（**历史数据只读解释用**）。

    v10 §7.2 用完整候选 `schemas.evolution.SkillCandidate` 取代了它：这个 v5 形状
    只有一串 `spec_content`、`parent_version` 可空、没有 campaign/generation/谱系，
    因此无法支撑"从具体父版本产生合法新版本"的合同。保留本类只为读旧 JSONL / 旧
    收据时**不静默升格**；新代码一律用 v10 定义。
    """

    candidate_id: str
    root_candidate_id: str
    revision_id: str
    parent_version: Optional[str]
    candidate_type: Literal["memory", "skill"]
    spec_content: str
    status: Literal["draft", "testing", "promoted", "rejected", "quarantined"]
    # 离线强模型产的 patch（v6：DeepSeek-V4.1-Flash）
    offline_patch_ref: Optional[str] = None
    induction_trace_refs: list[str] = []
    created_by: Literal["offline_induction", "offline_revision", "human"]
    created_at: str
    # v5 字段名兼容（读旧数据）
    @property
    def gpt6_patch_ref(self) -> Optional[str]:
        return self.offline_patch_ref


# v5 名称别名（只读兼容；不得用于 v10 演化链）
SkillCandidate = SkillCandidateV5


class DeepSeekGovernanceDecision(Spec):
    """离线治理裁决文本（§3.3/§3.4）。

    **纪律**：离线模型的输出**不能**直接修改 active Skill/Memory，也不能决定
    promote/reject —— promote 由确定性门 + 预注册规则决定（§3.3）。本 Schema 只承载
    "离线模型说了什么"，供审计；它不构成任何准入决定。
    """

    decision_id: str
    candidate_id: str
    review_summary: str
    semantic_risk: str
    generalization_notes: str
    offline_model: str = "DeepSeek-V4.1-Flash"
    provider: str = "deepseek"
    model_id: str = "deepseek-flash"
    prompt_version: str = ""
    # 显式标注：本裁决**不是**准入决定（确定性门才有权 promote）
    advisory_only: Literal[True] = True
    timestamp: str


# v5 名称别名（过渡期只读；新代码一律用 DeepSeekGovernanceDecision）
SkillGovernanceDecision = DeepSeekGovernanceDecision


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


__all__ = [
    "DeepSeekGovernanceDecision",
    "FAMILY_QUESTION_TYPES",
    "RetrievedSkill",
    "SKILL_FAMILIES",
    "SkillCandidate",
    "SkillCandidateV11",
    "SkillCandidateV5",
    "SkillFamily",
    "SkillGovernanceDecision",
    "SkillSource",
    "SkillSpec",
    "SkillSpecV11",
    "SkillState",
    "normalize_skill_markdown",
    "parse_skill_markdown",
]
