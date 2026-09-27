"""§17.1「数据访问」记录行 + §14.1 `label_access`（v9）。

规范原文（§17.1 最小真实记录表）：

    数据访问 | split、用途、组件角色、运行／候选身份、输入清单 hash、时间；生产入口实际记录

规范原文（§14.1 数据权限与经验资格）：

    "数据按 scene 分为 learning／induction、inner_validation、outer_holdout、
    final_test，同场景的题、图片、缓存和派生产物跟随划分。"
    "学习题标签允许用于离线分析并记 `label_access`；GT 3D 框、位姿和尺度默认不提供，
    不进入在线测量或标定。"
    "inner 的逐题题目、图片、标签、轨迹和错误分析只留在验证器，不交给归纳器修订。
    归纳器只获得聚合得分／失败统计及接受或拒绝决定；不能通过检索记录、调试日志或
    候选报告绕过该限制。"

三条纪律：

1. **记录必须在真实读取点产生**（"生产入口实际记录"）——不得事后补写一条看起来
   完整的访问记录；`at` 是**实际读取发生的时间**，不是报告生成时间。
2. **拒收也必须留痕**：隔离不是"悄悄过滤掉其它划分的轨迹"，被拒条目要逐条进
   `refused`（数量、原因、id 集合 hash），否则审计无法回答"到底读没读 inner"。
3. **`label_access` 只对学习划分成立**：§14.1 允许学习题标签用于离线分析（记
   `label_access`）；验证器读 inner 标签用于评分是另一回事，不套用这个字段
   （由 validator 角色与 `purpose` 表达）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import model_validator

from . import Spec

# 数据划分（§14.1 / §16.2 四层）
DATA_SPLITS = ("induction", "inner_validation", "outer_holdout", "final_test")

# 组件角色（谁在读）——只登记当前真实存在的读取方，新增读取方须先登记名字
COMPONENT_ROLES = ("inducer", "validator", "evaluator")

# 用途（读来做什么）
ACCESS_PURPOSES = ("skill_induction", "candidate_admission_panel", "learning_run")

# 拒收原因码（§14.1 硬隔离；当前只有一类：不属于本次读取允许的学习划分）
REFUSAL_REASON_CODES = frozenset({"not_in_learning_split"})


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataAccessRecord(Spec):
    """§17.1「数据访问」一行：谁（角色／运行身份）在什么时候读了哪个 split 的什么。

    `input_manifest_sha256` 由**实际读取到的条目清单**确定（不是目录哈希）：
    同一清单复算必须一致，从而让"读的是哪一批材料"可复核。
    """

    record_id: str
    at: str = ""
    split: str
    purpose: str
    component_role: str
    # 运行身份：演化 run 的 run_id（§14.2 每轮冻结一次）
    run_id: str = ""
    # 候选身份：该访问属于某个候选验证时填 revision/candidate id，否则留空
    candidate_id: str = ""
    input_manifest_sha256: str = ""
    n_items: int = 0
    # §14.1：本次读取是否用了学习题标签（成功/失败等派生标签）做离线分析
    label_access: bool = False
    # 被拒条目（隔离证据）：{"n": int, "by_reason": {code: n}, "ids_sha256": str}
    refused: dict = {}
    # 实际读取的路径/glob（生产入口传入，不猜）
    source_refs: list[str] = []
    notes: list[str] = []

    @model_validator(mode="after")
    def _check(self) -> "DataAccessRecord":
        if self.split not in DATA_SPLITS:
            raise ValueError(f"未登记的数据划分: {self.split!r}（可选 {DATA_SPLITS}）")
        if self.component_role not in COMPONENT_ROLES:
            raise ValueError(
                f"未登记的组件角色: {self.component_role!r}（可选 {COMPONENT_ROLES}）")
        if self.purpose not in ACCESS_PURPOSES:
            raise ValueError(
                f"未登记的用途: {self.purpose!r}（可选 {ACCESS_PURPOSES}）")
        # §14.1：只有学习划分的标签可以记 label_access
        if self.label_access and self.split != "induction":
            raise ValueError(
                "label_access=True 只对学习划分（induction）成立（§14.1）")
        if self.refused:
            bad = set(self.refused.get("by_reason", {})) - REFUSAL_REASON_CODES
            if bad:
                raise ValueError(f"未登记的拒收原因码: {sorted(bad)}")
            if int(self.refused.get("n", 0)) != sum(
                    int(v) for v in self.refused.get("by_reason", {}).values()):
                raise ValueError("refused.n 与 by_reason 合计不一致")
        return self


__all__ = [
    "ACCESS_PURPOSES",
    "COMPONENT_ROLES",
    "DATA_SPLITS",
    "REFUSAL_REASON_CODES",
    "DataAccessRecord",
    "utcnow_iso",
]