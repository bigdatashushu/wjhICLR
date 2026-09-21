"""M6 几何 Tool 单测（v6 §8 矩阵 / §9 逐 Tool 定义 / §12 距离原语）。

v6 工具面（14 个）与 v5 的差异是本文件的重心：

- 删除：`relative_direction`（方向向量版）/ `object_size_longest_dim` /
  `object_distance_meters` / `room_size_m2`；
- 新增/改名：`relative_direction_of`（三个参数都是对象 id）/ `object_3d_extent` /
  `camera_object_distance` / `plane_fit_room_size` / `count_objects` /
  `surface_distance_between_objects`；
- **每个 Tool 必须显式声明 `requires_evidence`**（§17.4 硬约束 7）；暴露与否由
  `EvidenceProfile` × `question_tool_scope` 决定，而不是逐题硬编码（§7.2）；
- 米制 Tool（`requires_evidence` 含 `metric_scale`）只在 `metric_enabled` 下可见，
  且**执行期二次校验**（缺系数 → `domain_value`，scope 不符 → `tool_contract`）。
"""

import json

import numpy as np
import pytest

import skill3d.tools  # noqa: F401  - 触发 Tool 注册
from skill3d.schemas import (
    ConfidenceMap,
    EvidenceProfile,
    ObjectRecord,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
)
from skill3d.schemas.evidence import (
    GATE_SUBCONDITIONS,
    GATE_VERSION,
    MetricEvidenceGateResult,
)
from skill3d.tools import REGISTRY, SceneHandle, call_tool
from skill3d.tools.contract import (
    EVIDENCE_METRIC_SCALE,
    SCOPE_FALLBACK_2D_ONLY,
    SCOPE_FULL_3D,
    SCOPE_METRIC_ENABLED,
    ArtifactUnavailableError,
    ToolContractError,
    available_artifacts_for,
    question_tool_scope_of,
)
from skill3d.tools.registry import ToolArgValidationError, ToolNotFoundError

# v6 工具面（§8/§9 的 14 个 Tool）；出现别的名字说明工具面被改动了
V6_TOOLS = {
    "list_objects", "count_objects", "object_centroid", "object_3d_extent",
    "plane_fit_room_size", "robust_distance", "camera_object_distance",
    "surface_distance_between_objects", "relative_direction_of",
    "object_visible_frames", "connectivity_graph", "exists_in_scene",
    "reproject", "euclidean_distance",
}

UP = [0.0, 1.0, 0.0]
_K = np.array([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]])


# ------------------------------------------------------------------ 夹具 ----

def _quality(**over) -> QualityMetrics:
    base = dict(warp_inlier_ratio=0.9, warp_photometric_inlier_ratio=0.9,
                cloud_overlap_ratio=0.9, main_gate_passed=True,
                g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.0,
                g4_frame_count=32, g6_depth_var_coeff=0.1, g7_dynamic_ratio=0.0,
                g9_tracker_consistency=0.9, g10_baseline_quality=0.5,
                overall_quality=0.9)
    base.update(over)
    return QualityMetrics(**base)


def _artifact(*, world_up=UP, handedness="right", metric_scale=None, **over):
    """v6 artifact：世界系契约齐备；米制尺度默认 None（融合未跑 = 未授权）。"""
    present = world_up is not None and handedness is not None
    base = dict(
        artifact_id="a", artifact_version="v", scene_name="s", c2w_list="c",
        intrinsics="k", depth_maps="d", point_map="p", point_conf="pc",
        quality_status="computed", quality=_quality(),
        world_up=world_up, handedness=handedness,
        world_frame_status="available" if present else "unavailable",
        confidence=ConfidenceMap(per_point_confidence=""))
    if metric_scale is not None:
        base.update(metric_scale=metric_scale, scale_fusion_status="success",
                    scale_self_consistency=0.02)
    base.update(over)
    return ReconstructionArtifact(**base)


def _scene(*, metric: bool = False, question_type: str = "",
           world_up=UP, handedness="right", track_consensus: str = "available",
           objs=(), artifact=None) -> SceneState:
    """v6 SceneState：`question_tool_scope` 逐题派生；米制由 gate ∧ 题型共同决定。"""
    art = artifact or _artifact(world_up=world_up, handedness=handedness,
                                metric_scale=(2.0 if metric else None))
    metric_q = question_type in ("object_abs_distance", "object_size_estimation",
                                 "room_size_estimation")
    gate = MetricEvidenceGateResult(gate_passed=metric, gate_version=GATE_VERSION,
                                    sub_results={n: metric for n in GATE_SUBCONDITIONS})
    profile = EvidenceProfile(
        geometry_3d="available",
        world_frame="available" if world_up is not None else "unavailable",
        metric_scale="available" if metric else "unavailable",
        object_detection="available", track_consensus=track_consensus,
        object_grounding="available")
    return SceneState(
        artifact_ref="artifact://demo", artifact=art, scene_route="full_3d",
        question_tool_scope=question_tool_scope_of(
            "full_3d", metric_question=metric_q, gate_passed=metric),
        available_artifacts=set(available_artifacts_for(
            "full_3d", question_type=question_type, gate_passed=metric,
            metric_question=metric_q)),
        evidence_profile=profile, metric_evidence_gate_result=gate,
        objects=[o.obj_id for o in objs], summary="demo scene",
        question_type=question_type)


def _obj(oid: str, category: str, centroid, *, pts=None, track=None,
         dup: bool = False, vis=None, bbox=None) -> ObjectRecord:
    return ObjectRecord(
        obj_id=oid, category_name=category, centroid_world=list(centroid),
        pointcloud_world=(_points_ref(oid, pts) if pts is not None else ""),
        track_id=track, duplicate_suspect=dup, visible_frames=vis or [0, 1],
        bbox=list(bbox or [0.0] * 6), det_conf=0.9)


_POINT_DIR = None


def _points_ref(oid: str, pts) -> str:
    """把点集落到临时 .npy（`SceneHandle` 只读 `ObjectRecord` 自带的 ref）。"""
    import tempfile
    from pathlib import Path

    global _POINT_DIR
    if _POINT_DIR is None:
        _POINT_DIR = Path(tempfile.mkdtemp(prefix="toolpts-"))
    p = _POINT_DIR / f"{oid}.npy"
    np.save(p, np.asarray(pts, dtype=np.float64))
    return str(p)


def _handle(scene: SceneState, *, objs=(), c2w=np.eye(4)[None, ...],
            intrinsics=_K, objects_materialized=True,
            point_map=None, metric_scale=None) -> SceneHandle:
    h = SceneHandle(scene, objects=list(objs), c2w_list=c2w, intrinsics=intrinsics,
                    objects_materialized=objects_materialized, metric_scale=metric_scale)
    if point_map is not None:
        h.set_point_map(point_map)
    return h


def _cluster(center, n: int = 400, spread: float = 0.02, seed: int = 0) -> np.ndarray:
    """物体点云：≥ N_min(100) 个点，距离可判（低分位 ≈ 最近表面）。"""
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=np.float64) + rng.normal(0, spread, size=(n, 3))


def _room_point_map(lo=(-2.0, -1.0, -3.0), hi=(2.0, 1.0, 3.0), n: int = 40) -> np.ndarray:
    """矩形房间的 6 面采样点（地面对角线 = √(4²+6²)，面积 = 4×6）。"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    zs = np.linspace(lo[2], hi[2], n)
    faces = []
    g = np.stack(np.meshgrid(xs, zs, indexing="ij"), -1).reshape(-1, 2)
    faces.append(np.column_stack([g[:, 0], np.full(len(g), lo[1]), g[:, 1]]))   # 地面
    g = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    for z in (lo[2], hi[2]):
        faces.append(np.column_stack([g[:, 0], g[:, 1], np.full(len(g), z)]))
    g = np.stack(np.meshgrid(ys, zs, indexing="ij"), -1).reshape(-1, 2)
    for x in (lo[0], hi[0]):
        faces.append(np.column_stack([np.full(len(g), x), g[:, 0], g[:, 1]]))
    return np.vstack(faces)


# ------------------------------------------------------------- 工具面（§8）----

def test_tool_face_is_exactly_the_v6_set():
    """工具面就是 §8/§9 的 14 个：v5 的 4 个已删除 Tool 不得回归。"""
    assert set(REGISTRY.names()) == V6_TOOLS
    for gone in ("relative_direction", "object_size_longest_dim",
                 "object_distance_meters", "room_size_m2"):
        assert gone not in REGISTRY


def test_every_tool_declares_evidence_and_metric_flag_is_derived_from_it():
    """§17.4：每个 Tool 显式声明 `requires_evidence`；米制 Tool 由该声明判出。"""
    for name in REGISTRY.names():
        declared = REGISTRY.requires_evidence(name)
        assert isinstance(declared, list)
        spec = REGISTRY.spec(name)
        assert set(spec.tolerates_degraded) <= set(declared)
        assert REGISTRY.is_metric_tool(name) == (EVIDENCE_METRIC_SCALE in declared)
    metric_tools = {n for n in REGISTRY.names() if REGISTRY.is_metric_tool(n)}
    assert metric_tools == {"object_3d_extent", "plane_fit_room_size",
                            "camera_object_distance"}


def test_metric_tools_only_visible_under_metric_scope():
    """米制 Tool 只在 `metric_enabled` 出现（§5.3/D4：逐题只收窄，唯一例外是米制追加）。"""
    metric_tools = {n for n in REGISTRY.names() if REGISTRY.is_metric_tool(n)}
    for scope in (SCOPE_FALLBACK_2D_ONLY, SCOPE_FULL_3D):
        assert metric_tools.isdisjoint(REGISTRY.names_for_scope(scope)), scope
    assert metric_tools <= set(REGISTRY.names_for_scope(SCOPE_METRIC_ENABLED))
    docs = REGISTRY.docs(SCOPE_METRIC_ENABLED)
    for name in metric_tools:
        assert f"- {name}(" in docs
    assert "- camera_object_distance(" not in REGISTRY.docs(SCOPE_FULL_3D)


# --------------------------------------------------------- 纯算术 / 相机 ----

def test_euclidean_distance():
    r = call_tool("euclidean_distance", {"point_a": [0, 0, 0], "point_b": [3, 4, 0]},
                  _handle(_scene()))
    assert r.error is None
    assert json.loads(r.value) == pytest.approx(5.0)
    assert r.source == "real"
    assert len(r.request_digest) == 64


def test_unknown_tool_raises():
    with pytest.raises(ToolNotFoundError):
        call_tool("not_a_tool", {}, _handle(_scene()))


def test_arg_validation():
    with pytest.raises(ToolArgValidationError):
        call_tool("euclidean_distance", {"point_a": [0, 0, 0]}, _handle(_scene()))


def test_reproject():
    h = _handle(_scene())
    # 世界点 (0,0,2)，c2w=I，f=100 → uv = (0,0)
    r = call_tool("reproject", {"p3d": [0.0, 0.0, 2.0], "frame_idx": 0}, h)
    assert json.loads(r.value) == [pytest.approx(0.0), pytest.approx(0.0)]
    # 世界点 (1,0,2) → u = 100*1/2 = 50
    r2 = call_tool("reproject", {"p3d": [1.0, 0.0, 2.0], "frame_idx": 0}, h)
    assert json.loads(r2.value)[0] == pytest.approx(50.0)
    # 相机后方的点 → Tool 确定性异常进 error
    r3 = call_tool("reproject", {"p3d": [0.0, 0.0, -1.0], "frame_idx": 0}, h)
    assert r3.error is not None and r3.error_code == "domain_value"


# ------------------------------------------------------------ 对象清单（§9.1/§9.2）----

def test_list_objects_returns_self_describing_records():
    """`list_objects` 返回**自描述**记录（含 `category_name`），不是裸 id 列表。

    §5.6/§8：v5 实测过"模型把 `obj_7` 当类别名"的误用面 —— v6 把类别名直接放进
    记录里，并要求计数走 `count_objects`（而不是数 `list_objects` 的长度）。
    """
    o = _obj("obj_7", "door", (0.0, 0.0, 1.0), track="t7", vis=[3, 4, 9])
    h = _handle(_scene(objs=[o]), objs=[o])
    recs = json.loads(call_tool("list_objects", {}, h).value)
    assert isinstance(recs, list) and len(recs) == 1
    rec = recs[0]
    assert rec["obj_id"] == "obj_7"
    assert rec["category_name"] == "door"          # 自描述（v6 新增）
    assert rec["track_id"] == "t7"
    assert rec["visible_frames"] == [3, 4, 9]
    assert "duplicate_suspect" in rec and "grounding_status" in rec
    # 类别筛选走 category_filter（子串、大小写不敏感）
    assert len(json.loads(call_tool("list_objects", {"category_filter": "DOOR"}, h).value)) == 1
    assert json.loads(call_tool("list_objects", {"category_filter": "chair"}, h).value) == []
    # id 是符号，不含类别名（不能拿 id 当类别判断）
    assert "door" not in rec["obj_id"]


def test_count_objects_counts_distinct_tracks():
    """§9.2：按 **track 共识**计数，不数清单长度；重复嫌疑显式降级。"""
    o_a = _obj("obj_0", "chair", (0.0, 0.0, 1.0), track="t0")
    o_dup = _obj("obj_1", "chair", (0.0, 0.0, 1.1), track="t0", dup=True)   # 同 track
    o_b = _obj("obj_2", "chair", (0.5, 0.0, 1.0), track="t1")
    o_c = _obj("obj_3", "chair", (1.0, 0.0, 1.0))                          # 无 track
    objs = [o_a, o_dup, o_b, o_c]
    h = _handle(_scene(objs=objs), objs=objs)
    out = json.loads(call_tool("count_objects", {"category_name": "chair"}, h).value)
    assert set(out) >= {"count", "n_distinct_tracks", "duplicate_suspect",
                        "evidence_degraded"}
    assert out["count"] == 3                 # t0 去重后 1 + t1 + 无 track 的 1
    assert out["n_distinct_tracks"] == 3
    assert out["n_records"] == 4             # 清单长度是 4（直接数就会系统性错）
    assert out["duplicate_suspect"] is True
    assert out["evidence_degraded"] is False
    # 类别过滤同样生效（"chai" 子串命中 4 条；"table" 不命中）
    assert json.loads(call_tool("count_objects", {"category_name": "table"}, h).value)["count"] == 0


def test_count_objects_marks_degraded_evidence():
    """track_consensus=degraded → 暴露但答案带 `evidence_degraded` 标记（§7.2/§9.2）。"""
    o = _obj("obj_0", "chair", (0.0, 0.0, 1.0), track="t0")
    h = _handle(_scene(objs=[o], track_consensus="degraded"), objs=[o])
    r = call_tool("count_objects", {"category_name": "chair"}, h)
    assert json.loads(r.value)["evidence_degraded"] is True
    assert r.degraded_evidence == ["evidence_degraded:track_consensus"]


def test_exists_in_scene():
    o = _obj("obj_0", "chair", (0.0, 0.0, 1.0), track="t0")
    h = _handle(_scene(objs=[o]), objs=[o])
    assert json.loads(call_tool("exists_in_scene", {"name": "chair"}, h).value) is True
    assert json.loads(call_tool("exists_in_scene", {"name": "sofa"}, h).value) is False


# --------------------------------------------------- 单对象量（§9.3–§9.5）----

def test_object_centroid_returns_normalized_and_metric():
    o = _obj("obj_0", "chair", (1.0, 0.0, 2.0), track="t0")
    h = _handle(_scene(objs=[o]), objs=[o])
    out = json.loads(call_tool("object_centroid", {"obj_id": "obj_0"}, h).value)
    assert out["centroid_normalized"] == pytest.approx([1.0, 0.0, 2.0])
    assert out["category_name"] == "chair" and out["track_id"] == "t0"
    # 米制坐标只在尺度可用时附上
    assert out["centroid_metric"] is None
    h_metric = _handle(_scene(metric=True, question_type="object_abs_distance", objs=[o]),
                       objs=[o], metric_scale=2.0)
    out_m = json.loads(call_tool("object_centroid", {"obj_id": "obj_0"}, h_metric).value)
    assert out_m["centroid_metric"] == pytest.approx([2.0, 0.0, 4.0])


def test_object_3d_extent_applies_metric_scale():
    """§9.4：米制 extent = 归一化 extent × 尺度（面积按平方）；缺系数即 fail-closed。"""
    pts = _cluster((0.0, 0.0, 2.0))
    o = _obj("obj_0", "chair", (0.0, 0.0, 2.0), pts=pts, track="t0")
    scale = 2.0
    h = _handle(_scene(metric=True, question_type="object_size_estimation", objs=[o]),
                objs=[o], metric_scale=scale)
    out = json.loads(call_tool("object_3d_extent", {"obj_id": "obj_0"}, h).value)
    assert out["n_valid_points"] > 0
    for norm, metric in zip(out["extent_normalized"], out["extent_metric"]):
        assert metric == pytest.approx(norm * scale)

    # 尺度不可用（未授权）→ 米制 Tool 不在 scope 内 → 执行期 fail-closed
    h_no = _handle(_scene(objs=[o]), objs=[o])
    with pytest.raises(ToolContractError) as ei:
        call_tool("object_3d_extent", {"obj_id": "obj_0"}, h_no)
    assert ei.value.error_code == "tool_contract"

    # scope 允许（metric_enabled）但产物里没有换算系数 → domain_value（第三道保险）
    no_coeff = _scene(metric=True, question_type="object_size_estimation", objs=[o],
                      artifact=_artifact(metric_scale=None))
    h_scope = _handle(no_coeff, objs=[o])
    out_no = call_tool("object_3d_extent", {"obj_id": "obj_0"}, h_scope)
    assert out_no.error_code == "domain_value" and out_no.value == "null"


def test_plane_fit_room_size_area_scales_with_square():
    """§9.5：地面矩形对角线/面积；`room_area_m2` 按尺度**平方**换算。"""
    scale = 3.0
    h = _handle(_scene(metric=True, question_type="room_size_estimation"),
                metric_scale=scale, point_map=_room_point_map())
    out = json.loads(call_tool("plane_fit_room_size", {}, h).value)
    # 房间地面 4×6 → 对角线 √(4²+6²)、面积 24（归一化单位）
    assert out["room_diagonal_normalized"] == pytest.approx(np.hypot(4.0, 6.0), rel=1e-3)
    expected_area = 24.0 * scale * scale
    assert out["room_area_m2"] == pytest.approx(expected_area, rel=1e-3)

    # 未注入点图 → 受控 domain_value（不得崩成服务故障）
    h_no_pm = _handle(_scene(metric=True, question_type="room_size_estimation"),
                      metric_scale=scale)
    r = call_tool("plane_fit_room_size", {}, h_no_pm)
    assert r.error_code == "domain_value"


def test_object_visible_frames_from_binding():
    """逐帧可见性原语：供"物体首次出现顺序"题型按 min(可见帧) 排序。"""
    o = _obj("obj_0", "basket", (0.0, 0.0, 1.0), vis=[4, 5, 9, 12], track="t0")
    h = _handle(_scene(objs=[o]), objs=[o])
    r = call_tool("object_visible_frames", {"obj_id": "obj_0"}, h)
    assert r.error is None and json.loads(r.value) == [4, 5, 9, 12]


def test_connectivity_graph_returns_graph_or_fails_closed():
    """§9.10：连通性图需要世界系约定 + 点图；缺任一即受控错误。"""
    o = _obj("obj_0", "chair", (0.0, 0.0, 1.0), track="t0")
    pm = _room_point_map()
    h = _handle(_scene(objs=[o]), objs=[o], point_map=pm)
    out = json.loads(call_tool("connectivity_graph", {}, h).value)
    assert set(out) >= {"nodes", "edges", "traversable_matrix"}

    r = call_tool("connectivity_graph", {}, _handle(_scene(objs=[o]), objs=[o]))
    assert r.error_code == "domain_value" and r.value == "null"


# ------------------------------------------------------------ 距离（§9.6/§9.7）----

def test_robust_distance_works_without_metric_scale():
    """§9.6/§12.3：rel_distance 官方口径（观察点→对象）**不需要**米制尺度。

    尺度不可用时：`distance_normalized` 照给、`distance_metric=None` 并显式标
    `metric_scale_unavailable` —— 绝不拿世界单位冒充米制（硬约束 23）。
    """
    pts = _cluster((0.0, 0.0, 2.0))
    o = _obj("obj_0", "door", (0.0, 0.0, 2.0), pts=pts, track="t0")
    h = _handle(_scene(question_type="object_rel_distance", objs=[o]), objs=[o])
    assert "scale" not in h.available_artifacts
    out = json.loads(call_tool("robust_distance",
                               {"reference": "camera", "target": "obj_0"}, h).value)
    assert out["distance_normalized"] == pytest.approx(1.95, abs=0.1)   # ≈2.0 的低分位
    assert out["distance_metric"] is None
    assert "metric_scale_unavailable" in out["degradation_flags"]
    assert out["quantile_q"] > 0 and out["n_valid_points"] > 0

    # 尺度可用时额外给米制值（同一低分位 × 尺度）
    h_metric = _handle(_scene(metric=True, question_type="object_abs_distance", objs=[o]),
                       objs=[o], metric_scale=2.0)
    out_m = json.loads(call_tool("robust_distance",
                                 {"reference": "camera", "target": "obj_0"},
                                 h_metric).value)
    assert out_m["distance_metric"] == pytest.approx(out_m["distance_normalized"] * 2.0)


def test_camera_object_distance_fails_without_metric_scale():
    """§9.6/§12.3 abs_distance：米制题专用；尺度不可用时双层 fail-closed。"""
    pts = _cluster((0.0, 0.0, 2.0))
    o = _obj("obj_0", "door", (0.0, 0.0, 2.0), pts=pts, track="t0")

    # ① scope 层：非米制 scope 下 Tool 不在允许集合 → tool_contract
    h_plain = _handle(_scene(objs=[o]), objs=[o])
    with pytest.raises(ToolContractError) as ei:
        call_tool("camera_object_distance", {"obj_id": "obj_0"}, h_plain)
    assert ei.value.error_code == "tool_contract"

    # ② 执行层：scope=metric_enabled 但产物里没有换算系数 → domain_value（不返回伪米制值）
    no_coeff = _scene(metric=True, question_type="object_abs_distance", objs=[o],
                      artifact=_artifact(metric_scale=None))
    r = call_tool("camera_object_distance", {"obj_id": "obj_0"}, _handle(no_coeff, objs=[o]))
    assert r.error_code == "domain_value" and r.value == "null"

    # ③ 系数齐备 → 米制值 = 归一化 × 尺度
    h_ok = _handle(_scene(metric=True, question_type="object_abs_distance", objs=[o]),
                   objs=[o], metric_scale=2.0)
    out = json.loads(call_tool("camera_object_distance", {"obj_id": "obj_0"}, h_ok).value)
    assert out["distance_metric"] == pytest.approx(out["distance_normalized"] * 2.0)


def test_surface_distance_between_objects_uses_both_pointsets():
    """§9.7：独立 Tool（双向 NN 低分位），**不参与** rel_distance 官方口径作答。"""
    a = _obj("obj_0", "table", (0.0, 0.0, 1.0), pts=_cluster((0.0, 0.0, 1.0)), track="t0")
    b = _obj("obj_1", "chair", (0.0, 0.0, 3.0), pts=_cluster((0.0, 0.0, 3.0)), track="t1")
    h = _handle(_scene(objs=[a, b]), objs=[a, b])
    out = json.loads(call_tool("surface_distance_between_objects",
                               {"obj_a": "obj_0", "obj_b": "obj_1"}, h).value)
    assert out["surface_distance_normalized"] == pytest.approx(1.9, abs=0.2)
    assert out["n_nn_samples"] > 0
    assert "point_contamination_suspect" in out
    # 同一对象 → 域值错误（不返回 0 假装"表面距离为零"）
    r = call_tool("surface_distance_between_objects",
                  {"obj_a": "obj_0", "obj_b": "obj_0"}, h)
    assert r.error_code == "domain_value"

    # 非米制工具不受尺度失败连累（§7.2：单项失败只收回依赖它的工具）
    assert {"list_objects", "robust_distance", "surface_distance_between_objects"} <= set(
        REGISTRY.names_for_scope(_scene(objs=[a, b]).question_tool_scope))


# ------------------------------------------------- 方向：世界系契约（§9.8/D5）----

def _direction_objects():
    """观察者在原点、面向 +z 的 door；target 由调用方指定。"""
    return (_obj("obj_0", "whiteboard", (0.0, 0.0, 0.0), track="t0"),
            _obj("obj_1", "door", (0.0, 0.0, 2.0), track="t1"))


def _direction_handle(target_pos, *, world_up=UP, handedness="right"):
    o0, o1 = _direction_objects()
    o2 = _obj("obj_2", "laptop", target_pos, track="t2")
    objs = [o0, o1, o2]
    scene = _scene(world_up=world_up, handedness=handedness, objs=objs)
    return _handle(scene, objs=objs)


def _ask_direction(handle) -> dict:
    return json.loads(call_tool(
        "relative_direction_of",
        {"observer_id": "obj_0", "facing_at_id": "obj_1", "target_id": "obj_2"},
        handle).value)


def test_relative_direction_of_uses_object_reference_frame():
    """§9.8：三个参数都是对象 id（facing_at 是**对象**，不是方向向量）。

    回归背景：v5 的 `relative_direction` 的 `facing` 是方向向量，模型实测传了对象
    质心（一个点）→ 静默算出与题目无关的方位。v6 用 id 表达"面向某物体"。
    """
    # 面向 +z：target 在 +x → left；在 -x → right；正前 → front；正后 → behind
    assert _ask_direction(_direction_handle((5.0, 0.0, 0.1)))["direction"] == "left"
    assert _ask_direction(_direction_handle((-5.0, 0.0, 0.1)))["direction"] == "right"
    assert _ask_direction(_direction_handle((0.0, 0.0, 5.0)))["direction"] == "front"
    assert _ask_direction(_direction_handle((0.0, 0.0, -5.0)))["direction"] == "behind"

    # 类别名同样可解析（大小写不敏感）
    o0, o1 = _direction_objects()
    o2 = _obj("obj_2", "laptop", (5.0, 0.0, 0.1), track="t2")
    h = _handle(_scene(objs=[o0, o1, o2]), objs=[o0, o1, o2])
    r = call_tool("relative_direction_of",
                  {"observer_id": "whiteboard", "facing_at_id": "DOOR", "target_id": "laptop"}, h)
    assert json.loads(r.value)["direction"] == "left"


def test_relative_direction_of_reports_world_contract_used():
    """输出必须自报用了哪套世界系约定（`world_up_used` / `handedness_used`，§9.8）。"""
    out = _ask_direction(_direction_handle((5.0, 0.0, 0.1)))
    assert set(out) == {"direction", "world_up_used", "handedness_used"}
    assert out["world_up_used"] == pytest.approx(UP)
    assert out["handedness_used"] == "right"


def test_relative_direction_handedness_and_up_flip_the_answer():
    """世界系约定变了，同一个几何的左右必须跟着翻（§9.8 判据 `(f×d)·u < 0`）。"""
    target = (5.0, 0.0, 0.1)
    right_handed = _ask_direction(_direction_handle(target))                     # up=+y
    left_handed = _ask_direction(_direction_handle(target, handedness="left"))
    assert right_handed["direction"] == "left"
    assert left_handed["direction"] == "right"        # 镜像世界 → 左右互换
    # 竖直轴取反 → 同样互换（up 是唯一的"上下"事实源）
    flipped_up = _ask_direction(_direction_handle(target, world_up=[0.0, -1.0, 0.0]))
    assert flipped_up["direction"] == "right"


def test_relative_direction_projects_onto_world_up_plane():
    """判定必须在**垂直于 world_up 的平面**内做：沿 up 平移 target 不得改变左右。"""
    # up = +y：把 target 抬高 1000 单位，水平分量不变 → 结论必须不变
    base = (2.0, 0.0, 0.0)
    moved = (2.0, 1000.0, 0.0)
    assert _ask_direction(_direction_handle(base))["direction"] == \
        _ask_direction(_direction_handle(moved))["direction"]
    # 倾斜的 world_up（俯仰 26.6°）同样成立：沿 up 平移不改变投影
    tilt = np.array([0.0, 1.0, 0.5])
    tilt = (tilt / np.linalg.norm(tilt)).tolist()
    p = np.array([2.0, 0.0, 0.0])
    up_v = np.asarray(tilt)
    far = (p + 1000.0 * up_v).tolist()
    assert _ask_direction(_direction_handle(tuple(p), world_up=tilt))["direction"] == \
        _ask_direction(_direction_handle(far, world_up=tilt))["direction"]


def test_world_up_comes_from_artifact_contract_not_geometry():
    """世界系约定来自 artifact（M3 落盘、M4 校验），**不再**从场景几何猜。

    历史缺陷（保留作记录）：v5 用"包围盒最小 extent 轴"当竖直轴，实测选到了水平轴
    → 四个 rel_direction 题全错。v6 的 `_up_axis` 启发式已删除（§9.8 fail-closed）。
    """
    o0, o1 = _direction_objects()
    o2 = _obj("obj_2", "laptop", (5.0, 0.0, 0.1), track="t2")
    objs = [o0, o1, o2]
    # 对象 bbox 全是 0（无几何信息）+ 倾斜的 world_up：只要 artifact 有约定，答案就确定
    tilt = [0.0, 0.8944271909999159, 0.4472135954999579]
    h = _handle(_scene(world_up=tilt, handedness="right", objs=objs), objs=objs)
    out = _ask_direction(h)
    assert out["world_up_used"] == pytest.approx(tilt)
    assert out["direction"] in ("front", "behind", "left", "right")
    assert not hasattr(h, "_up_axis")               # v5 启发式已彻底移除


def test_relative_direction_of_fails_closed_without_world_up():
    """§9.8：`world_up`/`handedness` 缺失 → fail-closed，**绝不**退回无符号启发式。"""
    o0, o1 = _direction_objects()
    o2 = _obj("obj_2", "laptop", (5.0, 0.0, 0.1), track="t2")
    objs = [o0, o1, o2]

    # ① 证据层：world_frame=unavailable → Tool 被收回（执行期二次校验也拦）
    scene = _scene(world_up=None, handedness=None, objs=objs)
    assert scene.evidence_state("world_frame") == "unavailable"
    assert "relative_direction_of" not in REGISTRY.names_for_scope(
        scene.question_tool_scope, evidence_profile=scene.evidence_profile)
    handle = _handle(scene, objs=objs)
    with pytest.raises(ArtifactUnavailableError) as ei:
        call_tool("relative_direction_of",
                  {"observer_id": "obj_0", "facing_at_id": "obj_1", "target_id": "obj_2"},
                  handle)
    assert ei.value.missing == ["world_frame"]

    # ② 第三道保险：证据说可用、但产物里真的没有约定 → domain_value（不猜方向）
    scene2 = scene.model_copy(update={
        "evidence_profile": scene.evidence_profile.model_copy(
            update={"world_frame": "available"})})
    out = call_tool("relative_direction_of",
                    {"observer_id": "obj_0", "facing_at_id": "obj_1", "target_id": "obj_2"},
                    _handle(scene2, objs=objs))
    assert out.error_code == "domain_value" and out.value == "null"
    assert "world_up" in (out.error or "")


def test_horizontal_degenerate_direction_is_domain_error():
    """target 与 observer 只差竖直方向 → 水平方位未定义 → domain_value（不猜）。"""
    h = _direction_handle((0.0, -3.0, 0.0))       # 正下方
    r = call_tool("relative_direction_of",
                  {"observer_id": "obj_0", "facing_at_id": "obj_1", "target_id": "obj_2"}, h)
    assert r.error_code == "domain_value" and r.value == "null"
