"""§5.2 重建 Schema。"""

from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from . import Spec

# v4 HC33：需要米制尺度的题型集合（逐题型授权的词汇表，单一事实源）
METRIC_TASK_TYPES: tuple[str, ...] = (
    "object_abs_distance", "object_size_estimation", "room_size_estimation")


def _ci_inconsistency_reason(data: dict) -> str:
    """HC29 不变量检查：返回非空字符串 = 不自洽（原因），空串 = 通过。

    与 `reconstruction.scale_units` 共用同一口径判定（不重复一份阈值逻辑）；
    此处只做"能不能由同一公式互推"的检查，不做业务降级。
    """
    from skill3d.reconstruction.scale_units import check_ci_rel, ci_abs_m

    s, r, a = data.get("metric_scale"), data.get("scale_ci_rel"), data.get("scale_ci_abs_m")
    if s is None and r is None and a is None:
        return ""
    if r is None and a is None:
        # v5 HC30：有点估计但**没有可用区间**（缺冻结校准器 / 融合不可用）
        # → 不可准入：无区间就不能声称经验覆盖，逐题授权必须收回。
        return ("v5 口径 CI 缺失（有 metric_scale 但无 scale_ci_rel/abs_m：未标定或"
                "无可用区间）→ 不可用于准入（硬约束 29/30）")
    if s is None or r is None or a is None:
        return f"scale/ci_rel/ci_abs 必须同时提供（收到 {s}, {r}, {a}）"
    chk = check_ci_rel(r, confidence_level=data.get("scale_confidence_level"))
    if not chk.ok:
        return chk.as_note()
    expected = ci_abs_m(float(s), float(r))
    if expected is None:
        return f"ci_abs 无法由 scale={s} × ci_rel={r} 导出"
    if abs(float(a) - expected) > 1e-6 * max(abs(expected), 1e-12):
        return f"ci_abs_m={a} ≠ scale×ci_rel={expected}（硬约束 29）"
    return ""


def _hc30_confidence_reason(data: dict) -> str:
    """HC30 数据层兜底：`medium/high` 必须由冻结校准器支撑，且授权不得超纲。

    两条规则（都 fail-closed 到 `low`）：

    1. `scale_confidence ∈ {medium, high}` 但 `scale_calibration_id` 为空 →
       未标定不得提升置信度（"不得靠常量或放宽门槛提升"）；
    2. `allowed_metric_tasks` 超出该置信档的预授权集合（例如 `medium` 却授权
       `object_abs_distance`，或 `high` 但 `ci_rel` 非有限）→ 越权授权。

    这堵住了"任何绕过 `assess_scale` 直接构造 artifact 的路径"：即使未来有新的
    生产者忘了走 v4 评估，也写不进一个未标定的 medium/high。
    """
    from skill3d.reconstruction.scale_units import pre_authorized_metric_tasks

    conf = str(data.get("scale_confidence", "low") or "low")
    if conf == "low":
        return ""
    if not data.get("scale_calibration_id"):
        return (f"scale_confidence={conf} 但无 scale_calibration_id"
                "（HC30：未标定一律 low）")
    allowed = {str(t) for t in (data.get("allowed_metric_tasks") or set())}
    if not allowed:
        return ""
    pre = pre_authorized_metric_tasks(conf, data.get("scale_ci_rel"),
                                      tuple(sorted(allowed)))
    over = allowed - pre
    if over:
        return (f"allowed_metric_tasks 越权授权 {sorted(over)}"
                f"（confidence={conf}, ci_rel={data.get('scale_ci_rel')}；HC33）")
    return ""


class ConfidenceMap(Spec):
    per_point_confidence: str  # 指向数组文件 ref
    coverage_count_per_frame: str


# v5 HC38：G8 已**永久退役**——当前 Schema 不声明 `g8_bbox_coverage_min`、
# 不声明 `CoverageMap`，也不存在任何替代的几何覆盖门。旧字段只能由
# `skill3d.legacy.readers` 读取为 `LegacyArtifact`（只读审计，不入准入/路由/统计）。
LEGACY_ONLY_FIELDS: tuple[str, ...] = (
    "scale_ci",              # 旧绝对 CI 半宽（旧公式）
    "relative_ci",           # 旧未注明置信水平的相对 CI
    "g8_bbox_coverage_min",  # 已退役的 G8
    "bbox_coverage_ratio",   # 旧 CoverageMap 内部字段
    "CoverageMap",
)


class QualityMetrics(Spec):
    """G1–G11 重建/输入质量指标（附录 A；阈值全部 TODO_CALIBRATE）。

    v5 活动指标集合 = G1–G4、G6、G7、G9–G11，**外加条件项 G5**：
    G5 只在 `ReconstructionArtifact.reprojection_status == "computed"` 时为有限值，
    否则必须为 `None`（v5 HC37：正式 `vggt` 主线无真 BA → `not_available`）。
    G8 已永久退役，本 Schema 不含该字段（HC38）。
    """

    g1_blur_ok: float
    g2_brightness: float
    g3_motion_blur: float
    g4_frame_count: int
    # v5 HC37：G5 是**可选**指标；`None` = 未计算（从 overall_quality 分母排除，
    # 不得补 1.0/历史值/代理值）。有限值仅允许伴随 reprojection_status="computed"。
    g5_reproj_err_median: Optional[float] = None
    g5_reproj_err_p95: Optional[float] = None
    g6_depth_var_coeff: float
    g7_dynamic_ratio: float
    g9_tracker_consistency: float
    g10_baseline_quality: float
    g11_scale_ci: float
    overall_quality: float

    @model_validator(mode="before")
    @classmethod
    def _reject_retired_g8(cls, data: Any) -> Any:
        """G8 字段一旦出现即 fail-closed（HC38：不计算、不序列化、不门控）。

        不做"读进来忽略掉"的静默兼容——静默接受会让旧 golden/旧 artifact 被当成
        当前口径的产物混入统计。旧数据请走 `skill3d.legacy.readers`。
        """
        if isinstance(data, dict) and "g8_bbox_coverage_min" in data:
            raise ValueError(
                "QualityMetrics 不再声明 g8_bbox_coverage_min（v5 HC38：G8 永久退役）。"
                "旧产物请用 skill3d.legacy.readers 只读解析，并从原始帧重跑 v5 pipeline。")
        return data


class ImageGridTransform(Spec):
    """§9 坐标层：`original 像素 → VGGT-depth-grid` 的仿射映射（v4/A-8 新增）。

    **为什么必须显式记录**：BA route 的硬前提是正方形输入（官方 `track_predict`
    断言 `height == width`），官方预处理是**中心 pad 到正方形再缩放**
    （`load_and_preprocess_images_square`）。pad 让映射不再是纯缩放，而是
    `dst = (src + pad_offset) × scale`；若下游仍按"纯缩放"处理 mask，
    mask 与深度网格会错位最多约 `pad_offset` 个像素（C-7 类缺陷）。
    故把映射当**数据**落进 artifact，M5 按它做最近邻重采样。

    feed-forward 主线（非正方形、无 pad）写 `padded_to_square=False` 的纯缩放映射，
    行为与 v3 完全一致（老 artifact 缺该字段时按纯缩放回退）。
    """

    source_hw: list[int]          # 原始帧 (H, W)，如 [480, 640]
    grid_hw: list[int]            # 深度/点图网格 (H, W)，如 [518, 518]
    pad_side: int = 0             # 中心 pad 后的正方形边长（原图像素）
    pad_offset_xy: list[int] = [0, 0]
    scale_x: float = 1.0
    scale_y: float = 1.0
    padded_to_square: bool = False
    method: str = "resize"        # resize | center_pad_then_resize


class ScaleAnchorEvidence(Spec):
    """单个尺度锚点的证据（§4.1 v4；硬约束 31）。

    每个锚点必须保存：来源、尺度估计、不确定性、残差、是否被接受、拒收原因码、
    参与帧。物体尺寸先验一律 `[TODO_CALIBRATE]`，**不得**当无误差真值。
    """

    anchor_type: Literal["camera_height_floor", "door", "table", "chair", "other"]
    scale_estimate: float          # 该锚点单独给出的尺度（m / rel-unit）
    ci_rel: float                  # 该锚点的相对不确定度（半宽分数，HC29 口径）
    residual: float                # log 尺度空间中到融合解的距离
    accepted: bool                 # 是否进入最终融合
    reason_code: str               # ok / outlier / conflict / geometry_gate / low_weight …
    source_frame_ids: list[int] = []
    # 原始可读证据（审计/论文用；不参与任何判定）
    anchor_name: str = ""
    measured: Optional[float] = None
    prior_m: Optional[float] = None
    weight: float = 0.0
    note: str = ""


class ReconstructionArtifact(Spec):
    """重建产物 = 几何 ref + 帧集身份 + 尺度 + **质量单一事实源**（§4 M4 / 硬约束 22）。

    `quality_status` 三态：`not_computed`（默认，保证旧数据反序列化兼容）/
    `computed` / `failed`。`quality=None` 当且仅当状态非 `computed`。
    route 判定必须 fail-closed：状态非 computed 或 quality 为 None 或
    `overall_quality` 为 NaN/非有限值 → **不得**停在 `full_3d`。

    v5 版本纪律（HC39）：本 Schema 只接受 `schema_version="5.0"`；
    `quality_metric_version` 固定活动指标集合；G5 由 `reprojection_status` 门控
    （HC37：正式 `vggt` 主线无真 BA → `not_available` + G5=None，禁止代理值）；
    旧尺度/G8 字段一律不接受（HC38/HC39）。
    """

    # ---- v5 版本字段（HC39：与历史数据版本隔离，不兼容混写）----
    schema_version: Literal["5.0"] = "5.0"
    quality_metric_version: Literal["v5-no-g8-g5-optional"] = "v5-no-g8-g5-optional"
    artifact_id: str
    artifact_version: str  # 内容寻址哈希
    scene_name: str
    recon_method: Literal["vggt", "vggt_sparse_ba", "dust3r_mast3r", "colmap"]
    # ---- 帧集身份（统一 FrameSet，硬约束 21；M1 冻结后全链共用）----
    frame_ids: list[int] = []
    source_frame_indices: list[int] = []
    timestamps: list[float] = []
    frame_set_hash: str = ""
    # §9：原图→深度网格 的仿射映射（BA route 的正方形 pad 会改变它；缺省 = 纯缩放）
    grid_transform: Optional[ImageGridTransform] = None
    # ---- 几何产物 ref ----
    c2w_list: str  # camera→world SE(3) 序列 ref（VGGT extrinsic 取逆，c2w[0]=I）
    intrinsics: str
    depth_maps: str
    point_map: str
    point_conf: str
    track_list: Optional[str]
    # ---- 尺度（v4：统一口径 + 多锚点 + 经验校准；硬约束 29–33）----
    metric_scale: Optional[float]
    scale_known: bool
    # v4 HC29：相对 CI **半宽分数**（[0,+∞)），指定 `scale_confidence_level` 下有效。
    # 旧 `scale_ci`（绝对 m，旧公式）保留只读，**不得用于准入**（§10.2/§10.5）。
    scale_ci_rel: Optional[float] = None
    # v4 HC29：`= metric_scale * scale_ci_rel`（m）；与上式不自洽即 fail-closed 为 low
    scale_ci_abs_m: Optional[float] = None
    # 该 CI 对应的置信水平（如 0.90）；必须与冻结校准器一致
    scale_confidence_level: Optional[float] = None
    # v4 HC31：多锚点证据（来源/估计/不确定性/残差/接受状态）
    scale_anchor_fired: list[ScaleAnchorEvidence] = []
    # v4 HC31：锚点间尺度比超过冲突阈值 → 显式记录，且不得返回虚假的 medium
    scale_conflict: bool = False
    # v4 HC32：在线只加载冻结校准器；未标定时为 None → 置信度恒 low
    scale_calibration_id: Optional[str] = None
    # v5.1：校准器来源数据集 + 是否与被评测数据集同源（论文口径必须可审计）
    scale_calibration_dataset: str = ""
    scale_dataset_match: Optional[bool] = None
    # v4 HC32/§7：名义 coverage 与经验 coverage 都要报
    scale_empirical_coverage: Optional[float] = None
    scale_nominal_coverage: Optional[float] = None
    # v4 HC33：当前尺度评估**实际授权**的米制题型（逐题型授权，不是全局开关）
    allowed_metric_tasks: set[str] = set()
    # ---- 质量（唯一事实源，D-5 / 硬约束 22）----
    quality_status: Literal["not_computed", "computed", "failed"] = "not_computed"
    quality: Optional[QualityMetrics] = None
    confidence: ConfidenceMap
    # ---- 尺度锚定证据（论文消融用；旧绝对 CI 字段 scale_ci 已按 HC39 移除）----
    # D-2：默认 **low**（未标定一律 low，不得进主表）。显式 null 在校验期归一为 low，
    # 堵住"None = 不做置信度否决"这种 fail-open 读法（§3 M7 硬过滤谓词）。
    scale_confidence: Literal["high", "medium", "low"] = "low"
    scale_method: str = ""
    # 尺度来源（§10 D-2 / RunManifest：把"哪个先验锚出来的"记进复现清单）
    scale_source: str = ""
    # ---- 重投影证据（G5 显式可用性，v5 HC35/37）----
    # 正式 `vggt` 主线（无真 BA）**固定** `not_available` + G5=None；
    # 只有 `vggt_sparse_ba` 通过 §10.1 PoC 后才允许 `computed`。
    reprojection_status: Literal["not_available", "computed", "failed"] = "not_available"
    # G5 残差数组 ref（BA 产物才有）
    reproj_errors: Optional[str] = None
    # G5 残差标量（仅 reprojection_status="computed" 时允许有限值，否则必须 None）
    g5_reproj_err_median: Optional[float] = None
    g5_reproj_err_p95: Optional[float] = None
    # `vggt_sparse_ba` 的 PoC 回执 ref（§4.5 SparseBAReceipt）；正式主线为 None
    sparse_ba_receipt_ref: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_fields(cls, data: Any) -> Any:
        """HC38/HC39：当前 Schema 不接受旧尺度/G8 字段，出现即 hard fail。

        不做静默忽略：`scale_ci`、`relative_ci`、`g8_bbox_coverage_min` 等旧字段
        与 v5 口径不可互推，静默接受会把历史数字混进 v5 统计。旧产物必须走
        `skill3d.legacy.readers`（只读审计），要进运行时/统计必须重跑 v5 pipeline。
        """
        if not isinstance(data, dict):
            return data
        present = sorted(f for f in LEGACY_ONLY_FIELDS if f in data)
        if present:
            raise ValueError(
                f"ReconstructionArtifact 含 legacy 字段 {present}（v5 HC38/39："
                "当前 Schema 不兼容混写）。请用 skill3d.legacy.readers 只读解析，"
                "并从原始帧重跑 v5 pipeline 生成新产物。")
        return data

    @model_validator(mode="before")
    @classmethod
    def _g5_availability_consistency(cls, data: Any) -> Any:
        """v5 HC37：G5 有限值 **当且仅当** `reprojection_status="computed"`。

        - 状态非 `computed` 而 G5 有有限值 → 视为"用代理值冒充重投影残差"，hard fail；
        - 状态 `computed` 但 G5 缺失 → hard fail（既然声明算过就必须有值）。
        NaN 与 `None` 等价（`ser_json_inf_nan="constants"` 下 NaN 可落盘，读回要一致）。
        """
        if not isinstance(data, dict):
            return data
        status = str(data.get("reprojection_status", "not_available") or "not_available")
        finite = []
        for k in ("g5_reproj_err_median", "g5_reproj_err_p95"):
            v = data.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv == fv and abs(fv) != float("inf"):
                finite.append(k)
        if status != "computed" and finite:
            raise ValueError(
                f"reprojection_status={status} 但 {finite} 为有限值："
                "G5 只能用真 BA 的重投影残差（HC37 禁止代理值/缺省补分）。")
        if status == "computed" and len(finite) != 2:
            raise ValueError(
                "reprojection_status=computed 必须同时给出有限的 "
                "g5_reproj_err_median / g5_reproj_err_p95（HC37）。")
        return data

    @field_validator("scale_confidence", mode="before")
    @classmethod
    def _null_scale_confidence_is_low(cls, v: Any) -> Any:
        """旧 artifact 可能显式写 `scale_confidence: null`。

        `None` **不等于**"不做置信度否决"，它是"没锚定过"——一律归一为 `low`
        （fail-closed，§3 M7：metric_scale_required 只接受 medium/high）。
        """
        return "low" if v is None else v

    @field_validator("allowed_metric_tasks", mode="before")
    @classmethod
    def _unknown_metric_tasks_rejected(cls, v: Any) -> Any:
        """未在 HC33 词汇表内的题型名不得进入授权集合（否则门控静默失效）。"""
        if v is None:
            return set()
        names = {str(x) for x in v}
        unknown = names - set(METRIC_TASK_TYPES)
        if unknown:
            raise ValueError(
                f"allowed_metric_tasks 含未知题型 {sorted(unknown)}；"
                f"词汇表见 schemas.reconstruction.METRIC_TASK_TYPES")
        return names

    @model_validator(mode="before")
    @classmethod
    def _scale_ci_v4_consistency(cls, data: Any) -> Any:
        """HC29 + HC30 数据层兜底：口径不自洽 / 未标定提升置信度 → fail-closed。

        - HC29：`scale_ci_abs_m ≈ metric_scale * scale_ci_rel`，否则降 low + 清空授权；
        - HC30：`medium/high` 必须有 `scale_calibration_id` 且授权不越纲，否则降 low。

        一律**降级**而非抛异常：artifact 是既成事实，反序列化旧数据时抛异常会让整条
        链不可用；"降为 low"才是硬约束 29/30 要求的更安全失败方式。新写回的 artifact
        由 M3 的 `assess_scale`/`apply_scale_assessment` 保证合规（生产端保证），
        本校验器是消费端兜底（旧数据 / 手工构造 / 篡改）。
        """
        if not isinstance(data, dict):
            return data
        bad = _ci_inconsistency_reason(data) or _hc30_confidence_reason(data)
        if not bad:
            return data
        out = dict(data)
        tasks = out.get("allowed_metric_tasks")
        already_downgraded = (out.get("scale_confidence") == "low"
                              and not (tasks or set()))
        if already_downgraded:
            return out
        out["scale_confidence"] = "low"
        out["allowed_metric_tasks"] = set()
        # HC31：`scale_conflict` **只**表示"锚点间尺度比超冲突阈值"，不得被用作
        # "口径降级"的标记（那会把两类事实混在一起，让报告误报锚点冲突）。
        # 降级事实由 `scale_method` 的原因串 + 清空的授权集合承载。
        method = str(out.get("scale_method") or "")
        out["scale_method"] = f"{method} | [HC29/30 fail-closed] {bad}".strip(" |")
        return out


class SceneState(Spec):
    """M4 场景状态：持久层 `artifact_ref` + 运行时活句柄 + route/产物集合。

    不持有独立 quality 副本（一切质量信息从 artifact 派生，硬约束 22）。
    `available_artifacts` 与 `tools.contract.ROUTE_ARTIFACTS` 同源（硬约束 23）。
    """

    artifact_ref: str
    route: Literal["full_3d", "fallback_2d_only", "unanswerable"]
    frame: Literal["world", "camera", "image"]
    scale_known: bool
    objects: list[str]  # ObjectInstance id ref
    summary: str  # 给 Synthesizer 的场景摘要
    # G-11：尺度置信档位（high/medium/low）；low 时 measurement 题降级 2D-only。
    # 与 ReconstructionArtifact 同口径：默认 low，显式 null 也归一为 low。
    scale_confidence: Literal["high", "medium", "low"] = "low"
    # v5 HC31：锚点冲突必须显式可见（不得用 getattr 兜底成 False 而看不到真冲突）
    scale_conflict: bool = False
    # v4 HC29/HC33：v4 口径的相对 CI 半宽（分数）与逐题型授权
    scale_ci_rel: Optional[float] = None
    # v5 HC38：本版**不设置** geometric_coverage 门；trace/报告只能写 not_defined，
    # 不得写 "coverage passed"（旧字段 scale_ci 已按 HC39 移除）。
    coverage_gate_status: Literal["not_defined"] = "not_defined"
    # v4 HC33：当前尺度评估授权的米制题型（从 artifact 派生，逐题门控的唯一依据）
    allowed_metric_tasks: set[str] = set()
    # v4 HC33：本 episode 的规范题型（逐题授权的第二维；空 = 未分类 → 米制 Tool 全拒）
    question_type: str = ""
    # M4：route → 可用重建产物集合（D-3 静态裁剪的执行期依据）
    available_artifacts: set[str] = set()
    # 运行时活句柄（只读引用，不参与序列化/哈希；持久层用 artifact_ref）
    artifact: Optional[Any] = Field(default=None, exclude=True)
    # 质量只读引用（= artifact.quality；artifact 不可用时为 None）
    quality: Optional[QualityMetrics] = Field(default=None, exclude=True)

    @field_validator("scale_confidence", mode="before")
    @classmethod
    def _null_scale_confidence_is_low(cls, v: Any) -> Any:
        """同 ReconstructionArtifact：显式 null 一律归一为 low（fail-closed）。"""
        return "low" if v is None else v

    @field_validator("allowed_metric_tasks", mode="before")
    @classmethod
    def _unknown_metric_tasks_rejected(cls, v: Any) -> Any:
        """同 ReconstructionArtifact：未知题型名不得进入授权集合。"""
        if v is None:
            return set()
        names = {str(x) for x in v}
        unknown = names - set(METRIC_TASK_TYPES)
        if unknown:
            raise ValueError(
                f"SceneState.allowed_metric_tasks 含未知题型 {sorted(unknown)}；"
                f"词汇表见 schemas.reconstruction.METRIC_TASK_TYPES")
        return names

    def metric_task_authorized(self, question_type: str) -> bool:
        """逐题型授权判定（HC33）：`question_type ∈ allowed_metric_tasks`。

        `question_type` 为空 = 尚未分类 → **不授权**（fail-closed：米制 Tool 必须先
        知道题型才能证明自己被授权）。
        """
        return bool(question_type) and str(question_type) in self.allowed_metric_tasks


class ObjectInstance(Spec):
    """M5 对象绑定/分割输出（§4 M5 字段 6）。"""

    instance_id: str
    class_hint: str
    mask_per_frame: str  # mask 数组文件 ref
    pointcloud_world: str  # 世界坐标点云 ref
    centroid_world: list[float]
    bbox: list[float]
    confidence: float
    # M5 掩码支撑帧（帧集槽位序号，升序）。用于"物体首次出现顺序"这类**逐帧可见性**
    # 题型的程序化作答（mask 已在 M5 内存中，这里只落 32 个整数，不额外占显存）。
    # 空列表 = 该对象在 32 帧内没有有效掩码（unverified/分割失败）。
    visible_frames: list[int] = []
