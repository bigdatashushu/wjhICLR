"""M15 Skill Registry：SkillSpec 语义版本 + 状态机（draft/shadow/canary/promoted/quarantined）。

语义版本规则（§7）：
- MAJOR.MINOR.PATCH；
- requires_artifacts / metric_scale_required 等前置条件变更 → 必须 MAJOR bump；
- 其他模板/描述变更 → MINOR/PATCH。
Skill 状态机：draft → shadow → canary → promoted；异常 → quarantined。
"""

from __future__ import annotations

import re

from skill3d.schemas import SkillSpec, SkillState

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# 前置条件字段：变更必须 MAJOR bump（§7）
PRECONDITION_FIELDS = ("requires_artifacts", "metric_scale_required")

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
