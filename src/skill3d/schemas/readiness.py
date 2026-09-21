"""HC34 Experiment Readiness Gate 的 Schema（§4.1 / §11.1）。

四级状态是**四个单调布尔字段**，不是单一枚举：后一层为 true 必须蕴含所有前层为
true（`real_poc_verified ⇒ connected ⇒ implemented`，`paper_eligible ⇒ 前三者`）。
违反单调性直接构造报错——把"代码存在"当成"已跑通"正是本门要堵的漏洞。
"""

from typing import Any

from pydantic import Field, model_validator

from . import Spec


class ExperimentReadiness(Spec):
    """单项能力的证据门（§11.1）。

    | 层级 | 必要证据 | 不足时禁止声称 |
    |---|---|---|
    | `implemented` | 实码存在、非 TODO 桩，单测覆盖核心合同 | 禁止声称"已实现" |
    | `connected` | 存在非自身、非测试的生产调用者 | 禁止声称"系统已具备" |
    | `real_poc_verified` | 冻结真实环境/模型/数据上通过预定义 PoC，存 receipt | 禁止声称"已跑通" |
    | `paper_eligible` | 数据隔离、统计门槛、≥3 seed、无 mock/泄漏、复现信息齐全 | 禁止进入主表 |
    """

    capability: str
    implemented: bool = False
    connected: bool = False
    real_poc_verified: bool = False
    paper_eligible: bool = False
    evidence_refs: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    # 状态推进的审计信息（§11.1：任何状态变化写 RunManifest 或独立 readiness_manifest.json）
    updated_at: str = ""
    code_commit: str = ""
    data_split_hash: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _monotonic(self) -> "ExperimentReadiness":
        """单调性：高层为真必须蕴含所有低层为真（HC34 / §11.1）。"""
        chain = [
            ("connected", self.connected, "implemented", self.implemented),
            ("real_poc_verified", self.real_poc_verified, "connected", self.connected),
            ("paper_eligible", self.paper_eligible, "real_poc_verified",
             self.real_poc_verified),
        ]
        for hi_name, hi, lo_name, lo in chain:
            if hi and not lo:
                raise ValueError(
                    f"{self.capability}: {hi_name}=True 但 {lo_name}=False —— "
                    "Readiness 四级必须单调（HC34）")
        return self

    def promote_to(self, level: str, *, evidence_refs: list[str] | None = None,
                   note: str = "") -> "ExperimentReadiness":
        """按证据推进到某一级（只升不降由调用方判断；本方法只管单调性）。"""
        order = ["implemented", "connected", "real_poc_verified", "paper_eligible"]
        if level not in order:
            raise ValueError(f"未知 Readiness 层级: {level}（取值域 {order}）")
        idx = order.index(level)
        update: dict[str, Any] = {name: True for name in order[: idx + 1]}
        if evidence_refs:
            update["evidence_refs"] = list(self.evidence_refs) + list(evidence_refs)
        if note:
            update["notes"] = (f"{self.notes} | {note}" if self.notes else note)
        return self.model_copy(update=update)

    def blocked_by(self, reason: str) -> "ExperimentReadiness":
        """记录阻断项（blockers 是"为什么还没到下一级"的显式答案）。"""
        if reason in self.blockers:
            return self
        return self.model_copy(update={"blockers": list(self.blockers) + [reason]})

    def as_row(self) -> dict:
        """表格化（§10.5 实况快照表的行格式）。"""
        return {
            "capability": self.capability,
            "implemented": self.implemented,
            "connected": self.connected,
            "real_poc_verified": self.real_poc_verified,
            "paper_eligible": self.paper_eligible,
            "blockers": list(self.blockers),
        }
