"""§4.7 LegacyArtifact：旧字段的**只读审计载体**（v5 HC39）。

纪律（违反即实现错误）：

- `eligible_for_runtime` / `eligible_for_statistics` 恒为 `False`（Literal[False]）；
- legacy reader **不猜单位**、不把百分数转分数、不根据互相矛盾字段推导新值，
  只保存原值 + 告警；
- 运行时、路由器、统计器与主表加载器发现 `LegacyArtifact` 必须 **hard fail**，
  并指出"需从原始帧重跑 v5 pipeline"；
- 禁止把 `LegacyArtifact` 原地 cast 成 `ReconstructionArtifact`。
"""

from typing import Literal, Optional

from . import Spec


class LegacyArtifact(Spec):
    """旧 artifact / 旧 golden 的只读解析结果（不含任何可准入字段）。"""

    source_path: str
    detected_schema_version: Optional[str] = None
    raw_fields: dict[str, object]
    deprecated_fields: set[str]
    warnings: list[str]
    eligible_for_runtime: Literal[False] = False
    eligible_for_statistics: Literal[False] = False


class LegacyArtifactError(RuntimeError):
    """当前运行时/统计遇到 legacy 数据 → hard fail（要求重跑 v5 pipeline）。"""

    def __init__(self, *, source_path: str, deprecated_fields: set[str],
                 detail: str = "") -> None:
        self.source_path = source_path
        self.deprecated_fields = set(deprecated_fields)
        self.detail = detail
        fields = sorted(self.deprecated_fields) or ["<无 schema_version>"]
        super().__init__(
            f"拒绝加载 legacy 产物 {source_path}：含 {fields}（v5 HC39：当前 Schema "
            "不兼容混写，legacy 数据不得进入运行时/准入/统计）。"
            "请用 skill3d.legacy.readers 做只读审计，并从原始帧重跑 v5 pipeline 生成新产物。"
            + (f" 细节：{detail}" if detail else ""))


class LegacyEpisodeTrace(Spec):
    """旧 Schema 的 EpisodeTrace **只读审计载体**（v9 §17.2）。

    旧 episode trace 保留自己的版本身份：6.0 的记录里没有 v9 才引入的字段，
    那不是"当时该事实不存在"，而是"当时没记这个事实"。因此**不得**把它
    `model_validate` 成当前 `EpisodeTrace` —— 缺失字段会被默认值静默填满，
    审计时无法区分两种情况。

    `eligible_for_runtime` 恒为 `False`：旧 trace 不能作为当前运行的证据。
    """

    source_path: str
    detected_schema_version: Optional[str] = None
    raw_fields: dict[str, object]
    warnings: list[str]
    eligible_for_runtime: Literal[False] = False


class LegacyEpisodeTraceError(RuntimeError):
    """当前读取入口遇到不可识别的 trace 版本 → hard fail（不猜、不静默升级）。"""

    def __init__(self, *, source_path: str, detected: Optional[str],
                 detail: str = "") -> None:
        self.source_path = source_path
        self.detected_schema_version = detected
        self.detail = detail
        super().__init__(
            f"拒绝加载 episode trace {source_path}："
            f"schema_version={detected!r} 不受当前读取入口支持。"
            "请用 skill3d.legacy.readers.read_legacy_episode_trace 做只读审计。"
            + (f" 细节：{detail}" if detail else ""))
