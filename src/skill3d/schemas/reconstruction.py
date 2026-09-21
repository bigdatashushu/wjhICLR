"""§5.2/§5.3/§5.6 重建 / 场景 / 对象 Schema（harness3D v6.0）。

v6 与 v5 的三处硬性差异（都是"规格要求"，不是实现选择）：

1. **米制路线整体替换**（D1/D2，§11）：多锚点 + log-scale 融合 + conformal 校准池
   全部废止 → artifact 不再有 `scale_anchor_fired / scale_ci_rel / scale_calibration_id /
   allowed_metric_tasks / scale_confidence` 等字段；取而代之的是
   `metric_scale / scale_self_consistency / per_frame_scale_ref / metric_model /
   metric_fusion_version / scale_fusion_status`（零样本度量深度跨帧融合）。
2. **世界系契约进 Schema**（D5，§7/§9）：`world_up` + `handedness` +
   `world_frame_status`；缺失时方向/路线类 Tool fail-closed（不猜）。
3. **scene_route × question_tool_scope 解耦**（D4，§5.3/§6.2）：`route` 一词三义废止。
   `scene_route` 只由 M4 质量决定；`question_tool_scope` 逐题派生，只收窄不新增。

`ReconstructionArtifact` 只接受 `schema_version="6.0"`；v5 及更早的尺度/G8/BA 字段
一律 hard fail（见 `LEGACY_ONLY_FIELDS`），旧产物只能经 `skill3d.legacy.readers`
只读审计，要进运行时/统计必须重跑 v6 pipeline。
"""

from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field, field_validator, model_validator

from . import Spec
from .evidence import (
    EvidenceProfile,
    MetricEvidenceGateResult,
    metric_scale_state_from_gate,
)

# 需要米制尺度的题型集合（逐题型授权的词汇表，单一事实源）
METRIC_TASK_TYPES: tuple[str, ...] = (
    "object_abs_distance", "object_size_estimation", "room_size_estimation")

# v6 §5.2：质量指标版本标识（G5 永久 not_available、G8 退役、主门=warp+重叠）
QUALITY_METRIC_VERSION: str = "v6-warp-overlap-no-g5"

# scene_route（只由 M4 质量决定）
SceneRoute = Literal["full_3d", "fallback_2d_only", "unanswerable"]
# question_tool_scope（逐题派生收窄）
QuestionToolScope = Literal["full_3d", "metric_enabled", "fallback_2d_only"]
# 答案来源（D9）
AnswerSource = Literal["tool_program", "direct_vlm_routed", "abstain", "tool_contract"]


# ---------------------------------------------------------------------------
# §5.2 质量指标（v6：M4 无真值质量门，D11）
# ---------------------------------------------------------------------------

class QualityMetrics(Spec):
    """M4 质量指标（§10；阈值全部 `[TODO_CALIBRATE]`）。

    v6 结构（D11）：

    - **主门**（§10.1）= 跨视图 warp 内点率 **且** 分组点云重叠率，两者都过才
      `main_gate_passed=True`。**多指标不得单挑**（SysCON3D 依据：前馈 backbone
      会幻觉跨视图一致性）；
    - **诊断**（§10.2）：track 重投影残差、相邻帧旋转平滑 —— 只产告警，不决定主门；
    - **conf-warp 单调自检**（§10.3）：VGGT conf 分桶 vs warp 差中位数应单调，
      不单调则降权；conf **只作软权重，不作硬门**；
    - **G5 永久 not_available**（§10.4）：本 Schema **不声明** G5 字段，
      它俩在 `ReconstructionArtifact` 上固定为 `None`；**严禁**用
      `depth_conf/point_conf`/点云密度冒充 BA 重投影残差；
    - **G8 永久退役**：本 Schema 不声明该字段，出现即 hard fail。
    """

    # ---- 主门（§10.1 交叉双指标）----
    warp_inlier_ratio: float              # 跨视图 warp 相对深度差内点率
    warp_photometric_inlier_ratio: float  # 跨视图 warp 光度内点率
    cloud_overlap_ratio: float            # 前 16 vs 后 16 帧分组点云双向重叠率
    main_gate_passed: bool
    # 主门用的阈值快照（trace 可审计："这个 pass 是按哪版阈值判的"）
    gate_thresholds: dict[str, float] = {}
    # ---- 诊断（§10.2，只产告警）----
    g1_blur_ok: float
    g2_brightness: float
    g3_motion_blur: float
    g4_frame_count: int
    g6_depth_var_coeff: float
    g7_dynamic_ratio: float
    g9_tracker_consistency: float
    g10_baseline_quality: float
    track_reproj_residual_median: Optional[float] = None
    rotation_smoothness: Optional[float] = None
    diagnostic_warnings: list[str] = []
    # ---- conf-warp 单调自检（§10.3）----
    conf_warp_monotonic: Optional[bool] = None
    conf_warp_spearman: Optional[float] = None
    # ---- 聚合（仅用于路由/报告；== NaN 时路由必须 fail-closed）----
    overall_quality: float

    @model_validator(mode="before")
    @classmethod
    def _reject_retired_metrics(cls, data: Any) -> Any:
        """G5 / G8 字段一旦出现即 fail-closed（§10.4：禁止代理值冒充）。"""
        if isinstance(data, dict):
            for banned in ("g5_reproj_err_median", "g5_reproj_err_p95",
                           "g8_bbox_coverage_min", "g11_scale_ci"):
                if banned in data:
                    raise ValueError(
                        f"QualityMetrics 不再声明 {banned}（v6 §10.4/§20："
                        "G5 永久 not_available、G8 永久退役、G11 随尺度路线废止）。"
                        "严禁用 depth_conf/point_conf 或其他代理值冒充。")
        return data


class ConfidenceMap(Spec):
    per_point_confidence: str  # 指向数组文件 ref
    coverage_count_per_frame: str = ""


# ---------------------------------------------------------------------------
# §5.2 重建产物
# ---------------------------------------------------------------------------

# v6 §5.2/§20：当前 Schema **不接受**的历史字段（出现即 hard fail）。
# 分三类：v5 多锚点+conformal 尺度路线、v5 BA 路线、v5 已退役质量指标。
LEGACY_ONLY_FIELDS: tuple[str, ...] = (
    # ---- v5 尺度路线（多锚点 + 冻结 conformal 校准池；§20 整体废止）----
    "scale_known",
    "scale_ci_rel",
    "scale_ci_abs_m",
    "scale_confidence",
    "scale_confidence_level",
    "scale_anchor_fired",
    "scale_conflict",
    "scale_calibration_id",
    "scale_calibration_dataset",
    "scale_dataset_match",
    "scale_empirical_coverage",
    "scale_nominal_coverage",
    "allowed_metric_tasks",
    "scale_method",
    "scale_source",
    # ---- v5 更早的尺度字段 ----
    "scale_ci",
    "relative_ci",
    # ---- v5 BA / sparse BA 路线（§20：BA 正式关闭）----
    "sparse_ba_receipt_ref",
    "reproj_errors",
    # ---- v5 已退役质量指标 ----
    "g8_bbox_coverage_min",
    "bbox_coverage_ratio",
    "CoverageMap",
    # ---- v5 正方形 pad 映射（BA route 专用；BA 关闭后不存在该形态）----
    "grid_transform",
)


class ReconstructionArtifact(Spec):
    """重建产物（§5.2）= 几何 ref + 帧集身份 + **世界系契约** + **度量尺度** + 质量。

    v6 不变量（违反即实现错误）：

    - `recon_method` **只允许** `vggt`（D11：BA 正式关闭；`vggt_sparse_ba` 与官方
      VGGSfRM BA 均不进入生产路线）；
    - G5 字段**固定为 `None` / `not_available`**，不进入 `overall_quality`；
    - `world_up`/`handedness` 缺失或非法 → 方向/路线类 Tool 必须 fail-closed（§9.8/§9.10）；
    - `metric_scale` 在度量融合 PoC 通过前保持 `None` + `scale_fusion_status="not_run"`；
      **不得**用未标定值冒充。
    """

    # ---- 版本字段（§5.2；与历史数据版本隔离，不兼容混写）----
    schema_version: Literal["6.0"] = "6.0"
    quality_metric_version: Literal["v6-warp-overlap-no-g5"] = "v6-warp-overlap-no-g5"
    artifact_id: str
    artifact_version: str                 # 内容寻址哈希
    scene_name: str
    recon_method: Literal["vggt"] = "vggt"

    # ---- 帧集身份（统一 FrameSet；M1 冻结后全链共用）----
    frame_ids: list[int] = []
    source_frame_indices: list[int] = []
    timestamps: list[float] = []
    frame_set_hash: str = ""

    # ---- 几何产物 ref（VGGT 前馈）----
    c2w_list: str                         # camera→world；世界系=首帧相机系，c2w[0]=I
    intrinsics: str                       # K；VGGT 输出需对齐
    depth_maps: str                       # 预处理分辨率（如 518×392）
    point_map: str
    point_conf: str
    depth_conf: str = ""                  # VGGT depth 头置信度（§10.3 软权重来源）
    track_list: Optional[str] = None

    # ---- 世界系契约（D5）----
    world_up: Optional[list[float]] = None      # 单位 3 向量；M3 从相机位姿束估计
    handedness: Optional[Literal["right", "left"]] = None
    world_frame_status: Literal["available", "degraded", "unavailable"] = "unavailable"

    # ---- 度量尺度融合（D1/D2，§11 [待实验]）----
    metric_scale: Optional[float] = None          # s_global = median_k(s_k)
    scale_self_consistency: Optional[float] = None  # 32 帧 s_k 的 std/median（离散度）
    per_frame_scale_ref: Optional[str] = None     # 32 个 s_k、离群帧列表（只读 receipt）
    metric_model: Optional[Literal["moge2", "metric3d_v2", "none"]] = None
    metric_fusion_version: Optional[str] = None
    scale_fusion_status: Literal["success", "failed", "not_run"] = "not_run"

    # ---- 质量（唯一事实源，M4）----
    quality_status: Literal["not_computed", "computed", "failed"] = "not_computed"
    quality: Optional[QualityMetrics] = None
    confidence: ConfidenceMap

    # ---- G5：永久 not_available（D11，§10.4）----
    reprojection_status: Literal["not_available"] = "not_available"
    g5_reproj_err_median: None = None
    g5_reproj_err_p95: None = None

    @field_validator("world_up")
    @classmethod
    def _world_up_unit(cls, v: Any) -> Any:
        """`world_up` 必须是**单位** 3 向量且有限；否则 hard fail（不猜、不归一化）。"""
        if v is None:
            return None
        arr = np.asarray([float(x) for x in v], dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"world_up 必须是 3 向量（收到 {v!r}）")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"world_up 含 NaN/Inf（收到 {v!r}）")
        n = float(np.linalg.norm(arr))
        if not np.isfinite(n) or abs(n - 1.0) > 1e-3:
            raise ValueError(
                f"world_up 必须是单位向量（模长={n}，收到 {v!r}）；"
                "非单位向量一律 fail-closed，不做静默归一化")
        return [float(x) for x in arr]

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_fields(cls, data: Any) -> Any:
        """当前 Schema 不接受 legacy 字段，出现即 hard fail（不静默忽略）。

        静默接受会让历史数字混进 v6 统计；旧产物必须走 `skill3d.legacy.readers`。
        """
        if not isinstance(data, dict):
            return data
        present = sorted(f for f in LEGACY_ONLY_FIELDS if f in data)
        if present:
            raise ValueError(
                f"ReconstructionArtifact 含 legacy 字段 {present}（v6 §20：当前 Schema "
                "不兼容混写）。请用 skill3d.legacy.readers 只读解析，"
                "并从原始帧重跑 v6 pipeline 生成新产物。")
        return data

    @model_validator(mode="after")
    def _v6_invariants(self) -> "ReconstructionArtifact":
        """G5 固定 None / 世界系契约自洽 / 度量融合字段自洽。"""
        # G5（§10.4）
        if self.reprojection_status != "not_available":
            raise ValueError(
                "reprojection_status 在 v6 固定为 not_available（D11：BA 正式关闭）")
        if self.g5_reproj_err_median is not None or self.g5_reproj_err_p95 is not None:
            raise ValueError("G5 标量在 v6 固定为 None（§10.4：严禁代理值冒充）")
        # 世界系契约（D5）
        if self.world_frame_status == "available" and (
                self.world_up is None or self.handedness is None):
            raise ValueError(
                "world_frame_status=available 必须同时给出 world_up 与 handedness"
                "（否则方向类 Tool 会拿到半个约定 → 必须 fail-closed）")
        # 度量融合（D1/D2）
        if self.scale_fusion_status == "success":
            if self.metric_scale is None or not np.isfinite(float(self.metric_scale)):
                raise ValueError("scale_fusion_status=success 必须有有限 metric_scale")
            if float(self.metric_scale) <= 0:
                raise ValueError(f"metric_scale 必须为正（收到 {self.metric_scale}）")
        elif self.metric_scale is not None:
            raise ValueError(
                f"scale_fusion_status={self.scale_fusion_status} 但 metric_scale="
                f"{self.metric_scale} 非空（§5.2 不变量：PoC 通过前保持 None）")
        # 质量单一事实源（硬约束 22）
        if (self.quality is None) != (self.quality_status != "computed"):
            raise ValueError(
                f"quality_status={self.quality_status} 与 quality="
                f"{'None' if self.quality is None else '有值'} 不自洽"
                "（quality=None 当且仅当状态非 computed）")
        return self


# ---------------------------------------------------------------------------
# §5.3 场景状态（scene_route × question_tool_scope 解耦）
# ---------------------------------------------------------------------------

class SceneState(Spec):
    """M4 场景状态：持久层 `artifact_ref` + 运行时活句柄 + 双路由字段 + 证据画像。

    D4 不变量（§5.3）：

    - `ROUTE_ARTIFACTS` 纯函数，**只挂 `scene_route``**；`available_artifacts` 是
      `ROUTE_ARTIFACTS[scene_route]` 再并上条件集 `{scale}`（米制题 ∧ gate_passed）；
    - prompt 头部与 scene 摘要**必须同源**（都从 `scene_route` + `question_tool_scope`
      派生），杜绝"头部 fallback、摘要 full_3d"自相矛盾；
    - `scene_route` 在同一 scene 的所有 episode 间稳定；`question_tool_scope` 可逐题不同；
    - `scene_route` **不因逐题 metric_scale 失败而改变**（metric 失败只收窄 scope）。
    """

    artifact_ref: str                     # 文件路径 / content hash
    # 运行时活句柄（只读引用，不复制、不序列化）
    artifact: Optional[Any] = Field(default=None, exclude=True)

    # D4：两个正交字段
    scene_route: SceneRoute = "fallback_2d_only"
    question_tool_scope: QuestionToolScope = "fallback_2d_only"
    # = ROUTE_ARTIFACTS[scene_route] ∪ ({scale} if 米制题 ∧ gate_passed)
    available_artifacts: set[str] = set()

    # EvidenceProfile 驱动（D6）
    evidence_profile: Optional[EvidenceProfile] = None
    metric_evidence_gate_result: Optional[MetricEvidenceGateResult] = None

    # 答案来源（D9）
    answer_source: AnswerSource = "abstain"

    # 对象与摘要
    objects: list[str] = []               # scene 级基础清单的 obj_id 列表
    summary: str = ""
    # 本题规范题型（逐题 scope 派生的第二维；空 = 未分类 → 米制 Tool 全拒）
    question_type: str = ""
    # 质量只读引用（= artifact.quality；artifact 不可用时为 None）
    quality: Optional[QualityMetrics] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _scope_narrower_than_route(self) -> "SceneState":
        """`docs(question_tool_scope) ⊆ docs(scene_route)`（§5.3 不变量，fail-closed）。

        `scene_route != full_3d` 时 scope 只能是 `fallback_2d_only` ——
        "头部 full_3d、摘要 fallback"这一类自相矛盾必须在数据层就写不进来。
        """
        if self.scene_route != "full_3d" and self.question_tool_scope == "full_3d":
            raise ValueError(
                f"question_tool_scope=full_3d 但 scene_route={self.scene_route}"
                "（§5.3：逐题只能收窄，不得新增工具）")
        if self.question_tool_scope == "metric_enabled" and self.scene_route != "full_3d":
            raise ValueError(
                "question_tool_scope=metric_enabled 要求 scene_route=full_3d"
                "（§13.2：米制能力只在 full_3d 下可能暴露）")
        return self

    # ---- 派生只读视图 ----
    @property
    def metric_gate_passed(self) -> bool:
        g = self.metric_evidence_gate_result
        return bool(g is not None and g.gate_passed)

    def metric_task_authorized(self, question_type: str) -> bool:
        """本题是否被授权使用米制 Tool：题型是米制 ∧ gate 通过 ∧ scope 允许。"""
        qt = str(question_type or "")
        return (qt in METRIC_TASK_TYPES
                and self.metric_gate_passed
                and self.question_tool_scope == "metric_enabled")

    def evidence_state(self, capability: str) -> str:
        if self.evidence_profile is None:
            return "unavailable"
        return self.evidence_profile.state(capability)


# ---------------------------------------------------------------------------
# §5.6 对象记录（scene 级基础清单，自描述）
# ---------------------------------------------------------------------------

class ObjectRecord(Spec):
    """scene 级基础清单的记录（§5.6）。

    **自描述**是硬要求（`category_name` 必须落盘）：v5 实测过"模型把 `obj_7`
    当类别名"的误用面 —— id 不含类别名，程序路径就会拿 id 去匹配类别。
    """

    obj_id: str                           # scene 级稳定 ID（跨 episode 不变）
    category_name: str                    # 自描述：消灭 "id 不含类别名" 误用面
    visible_frames: list[int] = []         # 该对象在哪些帧可见（帧槽位序号，升序）
    centroid_ref: str = ""                # 3D 质心引用（世界系，审计用）
    track_id: Optional[str] = None        # 跨帧 track
    det_conf: float = 0.0
    grounding_status: Literal["base_list", "question_targeted_fill"] = "base_list"
    duplicate_suspect: bool = False       # D6：重复嫌疑（计数降级信号）
    # ---- 运行时产物引用（M5 全量绑定产物；Tool 经 SceneHandle 只读访问）----
    mask_per_frame: str = ""              # mask 数组文件 ref
    pointcloud_world: str = ""            # 世界坐标点云 ref
    # 与点云逐点对齐的置信度 ref（§12.2 软权重来源；空 = 不可用，conf 不参与加权）
    pointconf_world: str = ""
    centroid_world: list[float] = [0.0, 0.0, 0.0]
    bbox: list[float] = []                # [min_x,min_y,min_z,max_x,max_y,max_z]

    # ---- 过渡期只读别名（v5 字段名；不参与序列化，迁移完成后删除）----
    @property
    def instance_id(self) -> str:
        return self.obj_id

    @property
    def class_hint(self) -> str:
        return self.category_name

    @property
    def confidence(self) -> float:
        return self.det_conf


# v5 名称别名（内部过渡用；新代码一律用 ObjectRecord）
ObjectInstance = ObjectRecord


__all__ = [
    "AnswerSource",
    "ConfidenceMap",
    "LEGACY_ONLY_FIELDS",
    "METRIC_TASK_TYPES",
    "ObjectInstance",
    "ObjectRecord",
    "QUALITY_METRIC_VERSION",
    "QualityMetrics",
    "QuestionToolScope",
    "ReconstructionArtifact",
    "SceneRoute",
    "SceneState",
    "metric_scale_state_from_gate",
]
