"""SceneHandle：Tool 访问重建产物的唯一只读句柄（硬约束 17）。

Tool 不得自行读原始文件、不得绕过置信度图、不得访问未挂载路径。
SceneHandle 只持有内存中的数组/对象结构，不暴露任何文件系统路径。

v6 变化（D4/D5/D6）：
- 可用产物由 `scene_route` × `question_tool_scope` 共同决定（`scale` 只在
  `metric_enabled` 时可用），不再有 `scale_known` 中间量；
- `world_up` / `handedness` 从 artifact 的**世界系契约**读入（M3 落盘、M4 校验），
  缺失时 `up_direction()` 返回 None → 方向类 Tool 必须 fail-closed（§9.8）；
- `object_points()` 让距离原语拿到对象 3D 点集（句柄负责加载与缓存，
  Tool 仍然不接触路径）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from skill3d.schemas import ObjectRecord, SceneState

from .contract import (
    ARTIFACT_INTRINSICS,
    ARTIFACT_OBJECTS,
    ARTIFACT_POSES,
    ARTIFACT_SCALE,
    route_aware_available,
)


class SceneHandle:
    """SceneState 的只读运行期句柄。

    - objects: ObjectRecord 列表（内存结构，非文件路径）
    - c2w_list / intrinsics: numpy 数组序列（由重建管线加载后注入）
    - 不提供任何 open()/路径访问接口（`object_points` 是唯一的数组读取入口，
      且只读 `ObjectRecord` 自带的 ref）
    """

    def __init__(
        self,
        scene_state: SceneState,
        objects: Optional[list[ObjectRecord]] = None,
        c2w_list: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        quality_overall: Optional[float] = None,
        objects_materialized: Optional[bool] = None,
        metric_scale: Optional[float] = None,
    ) -> None:
        self._state = scene_state
        # 世界系 → 米 的换算系数。优先用显式传入值（mock_light 的合成几何），
        # 否则读运行时 artifact。
        self._metric_scale = metric_scale
        self._objects = {o.obj_id: o for o in (objects or [])}
        # 硬约束 23：objects **产物是否已产出**与"产物里有没有对象"是两件事。
        # - M5 未跑/失败 → 产物不存在 → `objects` 不可用 → Tool 抛 ArtifactUnavailableError；
        # - M5 跑过但一个对象都没绑到 → 产物存在且为空 → `exists_in_scene` 返回 False。
        # 未显式声明时保守按"空列表视为未产出"处理（老调用点语义不变）。
        self._objects_materialized = (
            bool(objects) if objects_materialized is None else bool(objects_materialized)
        )
        # category_name 小写索引，供 exists_in_scene / 名称解析
        self._by_name: dict[str, str] = {}
        for o in objects or []:
            self._by_name.setdefault(o.category_name.lower(), o.obj_id)
        self._c2w = None if c2w_list is None else np.asarray(c2w_list, dtype=np.float64)
        self._k = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        self._quality_overall = quality_overall
        # 对象 3D 点集缓存（距离原语用；只读 ObjectRecord 的 ref）
        self._points_cache: dict[str, np.ndarray] = {}
        # 对象逐点置信度缓存
        self._conf_cache: dict[str, Optional[np.ndarray]] = {}
        # 世界系点图 (N,H,W,3)（平面拟合/连通性图用；由重建管线注入）
        self._point_map: Optional[np.ndarray] = None

    # ---- 只读元信息 ----
    @property
    def frame(self) -> str:
        return "world"

    @property
    def scene_route(self) -> str:
        """scene 级路由（只由 M4 质量决定；v6 D4）。"""
        return self._state.scene_route

    @property
    def question_tool_scope(self) -> str:
        """题级工具集范围（逐题派生收窄；v6 D4）。"""
        return self._state.question_tool_scope

    @property
    def question_type(self) -> str:
        """本 episode 的规范题型（逐题授权的第二维）。"""
        return str(getattr(self._state, "question_type", "") or "")

    @property
    def allowed_metric_tasks(self) -> set[str]:
        """本题被授权使用米制 Tool 的题型集合（空 = 未授权）。

        v6：授权由 `MetricEvidenceGate` 主导（gate 通过 ∧ 题型是米制），
        返回单元素集合或空集，供 `check_metric_task_contract` 的一致性校验使用。
        """
        qt = self.question_type
        if qt and self._state.metric_task_authorized(qt):
            return {qt}
        return set()

    @property
    def summary(self) -> str:
        return self._state.summary

    @property
    def metric_scale(self) -> Optional[float]:
        """世界系 → 米 的换算系数（米/世界单位）。

        米制 Tool 必须用它把世界单位换算成米，**不得**直接把世界单位当米返回。
        未融合成功（None 或非有限）→ 米制 Tool fail-closed（硬约束 23）。
        """
        if self._metric_scale is not None:
            return float(self._metric_scale)
        art = getattr(self._state, "artifact", None)
        v = getattr(art, "metric_scale", None)
        return None if v is None else float(v)

    @property
    def metric_gate_passed(self) -> bool:
        return self._state.metric_gate_passed

    @property
    def evidence_profile(self):
        """当前 EvidenceProfile（Tool 可见性与降级标记的判据；无则 None）。"""
        return getattr(self._state, "evidence_profile", None)

    @property
    def conf_warp_monotonic(self) -> Optional[bool]:
        """VGGT conf 的 conf-warp 单调性自检结果（§10.3）。

        `True` = conf 可作软权重；`False` = 不单调 → 降权；`None` = 未自检
        → **完全不使用 conf 过滤**（保守；绝不把未验证的 conf 当硬门）。
        """
        art = getattr(self._state, "artifact", None)
        q = getattr(art, "quality", None)
        return getattr(q, "conf_warp_monotonic", None)

    @property
    def world_up(self) -> Optional[np.ndarray]:
        """世界系"上"方向（M3 落盘的世界系契约；缺失 → None → 方向 Tool fail-closed）。"""
        art = getattr(self._state, "artifact", None)
        v = getattr(art, "world_up", None)
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            return None
        n = float(np.linalg.norm(arr))
        if not np.isfinite(n) or n < 1e-9:
            return None
        return arr / n

    @property
    def handedness(self) -> Optional[str]:
        art = getattr(self._state, "artifact", None)
        v = getattr(art, "handedness", None)
        return None if v is None else str(v)

    @property
    def quality_overall(self) -> Optional[float]:
        return self._quality_overall

    @property
    def quality_status(self) -> str:
        """质量状态（来自 artifact 的单一事实源；无 artifact 时 not_computed）。"""
        art = getattr(self._state, "artifact", None)
        return str(getattr(art, "quality_status", "not_computed"))

    @property
    def available_artifacts(self) -> set[str]:
        """可用产物集合（v6：`scene_route` × `question_tool_scope` 共同决定）。

        句柄上真正装载/可用的东西会进一步收窄集合（例如未注入 c2w → 不可称 poses
        可用，避免"声明可用但实际读不到"的静默错答）。

        `scale` 只在**本题被授权使用米制 Tool**（题型是米制 ∧ gate 通过 ∧
        scope=metric_enabled）时可用；这样"米制失败"只收回米制工具，
        `depth/poses/point_cloud/objects` 与相对几何 Tool 不受影响（§7.2/§7.3）。
        """
        declared = route_aware_available(
            getattr(self._state, "available_artifacts", None), self.scene_route)
        if self._c2w is None:
            declared -= {ARTIFACT_POSES}
        if self._k is None:
            declared -= {ARTIFACT_INTRINSICS, ARTIFACT_POSES}
        if not self._objects_materialized:
            declared -= {ARTIFACT_OBJECTS}
        if not self._state.metric_task_authorized(self.question_type):
            declared -= {ARTIFACT_SCALE}
        return declared

    @property
    def objects_materialized(self) -> bool:
        """objects 产物是否已产出（决定 `exists_in_scene` 返回 False 还是抛错）。"""
        return self._objects_materialized

    @property
    def frame_set_hash(self) -> str:
        return str(getattr(self._state, "frame_set_hash", "") or "")

    # ---- 对象访问 ----
    def list_objects(self) -> list[str]:
        return sorted(self._objects.keys())

    def object_records(self) -> list[ObjectRecord]:
        """自描述对象清单（§8：`list_objects()` 返回含 category_name 的记录）。"""
        return [self._objects[oid] for oid in self.list_objects()]

    def list_objects_by_name(self, name: str = "") -> list[str]:
        """按 category_name 子串过滤对象 id（空 = 全部）。大小写不敏感。

        计数题必须靠这个（`exists_in_scene` 是布尔，拿它当计数器只能得到 0/1）。
        """
        h = str(name or "").strip().lower()
        if not h:
            return self.list_objects()
        out = []
        for oid in self.list_objects():
            obj = self._objects.get(oid)
            val = str(getattr(obj, "category_name", "") or "").lower()
            if h in val or (val and val in h):
                out.append(oid)
        return out

    def object_category_name(self, object_id: str) -> str:
        obj = self._objects.get(str(object_id))
        return str(getattr(obj, "category_name", "") or "")

    def resolve_object_id(self, name_or_id: str) -> str:
        """按 obj_id 或 category_name（大小写不敏感）解析对象。"""
        if name_or_id in self._objects:
            return name_or_id
        hit = self._by_name.get(name_or_id.lower())
        if hit is None:
            raise KeyError(f"对象不存在: {name_or_id}")
        return hit

    def get_object(self, name_or_id: str) -> ObjectRecord:
        return self._objects[self.resolve_object_id(name_or_id)]

    def object_points(self, name_or_id: str) -> np.ndarray:
        """对象世界系 3D 点集 (N,3)（距离原语用）。

        只读 `ObjectRecord.pointcloud_world` 这个 ref —— Tool 仍然不接触文件路径。
        ref 为空/文件缺失 → 返回空 (0,3) 数组（调用方按 N_min 判 degraded，
        **不**在这里伪造点）。
        """
        oid = self.resolve_object_id(name_or_id)
        if oid in self._points_cache:
            return self._points_cache[oid]
        ref = str(getattr(self._objects[oid], "pointcloud_world", "") or "")
        pts = np.zeros((0, 3), dtype=np.float64)
        if ref:
            p = Path(ref)
            if p.is_file():
                try:
                    arr = np.load(p)
                except Exception:  # noqa: BLE001 - 坏文件不阻断（降级为无点）
                    arr = None
                if arr is not None:
                    arr = np.asarray(arr, dtype=np.float64)
                    pts = arr if arr.ndim == 2 else np.zeros((0, 3), dtype=np.float64)
        self._points_cache[oid] = pts
        return pts

    def object_centroid_world(self, name_or_id: str) -> np.ndarray:
        obj = self.get_object(name_or_id)
        return np.asarray(obj.centroid_world, dtype=np.float64)

    def set_object_points(self, name_or_id: str, points) -> None:
        """注入对象世界系点集（内存路径：重建管线/合成场景已有数组时用）。

        仍然**不暴露路径**：只是把调用方已有的数组放进只读缓存，Tool 侧行为与
        `object_points()` 完全一致（后续调用直接命中缓存）。真实路径用
        `ObjectRecord.pointcloud_world` 文件 ref 时不需要调用本方法。
        """
        oid = self.resolve_object_id(name_or_id)
        arr = np.asarray(points, dtype=np.float64)
        self._points_cache[oid] = (arr if arr.ndim == 2 and arr.shape[1] == 3
                                   else arr.reshape(-1, 3))

    def object_point_conf(self, name_or_id: str) -> Optional[np.ndarray]:
        """对象点云的逐点置信度 (N,)（VGGT conf，经 conf-warp 自检后才被使用）。

        ref 缺失/文件损坏 → None（调用方把 conf 当作"不可用"，**不**伪造 1.0）。
        """
        oid = self.resolve_object_id(name_or_id)
        if oid in self._conf_cache:
            return self._conf_cache[oid]
        ref = str(getattr(self._objects[oid], "pointconf_world", "") or "")
        conf: Optional[np.ndarray] = None
        p = Path(ref) if ref else None
        if p is not None and p.is_file():
            try:
                arr = np.asarray(np.load(p), dtype=np.float64).reshape(-1)
            except Exception:  # noqa: BLE001 - 坏文件不阻断（降级为 conf 不可用）
                arr = None
            if arr is not None and arr.size:
                pts = self.object_points(oid)
                if pts.shape[0] == arr.size:
                    conf = arr
        self._conf_cache[oid] = conf
        return conf

    def connectivity_graph(self) -> dict:
        """场景连通性/可达图（§9.10；route_planning 用）。

        做法（阈值全 `[TODO_CALIBRATE]`）：世界系点云投影到地面平面 → 高度带
        占据栅格 → 自由空间连通域 → 对象/起点按所在自由域判定可达。
        `world_up` 缺失 → 抛错（fail-closed，绝不猜竖直方向）。
        """
        up = self.world_up
        if up is None:
            raise KeyError("世界系约定缺失（world_up=None）→ 无法在正确平面上建连通图")
        pm = self.get_point_map()
        if pm is None or pm.size == 0:
            raise KeyError("未注入世界系点图，无法构建连通性图")
        from skill3d.tools.connectivity import build_connectivity_graph

        return build_connectivity_graph(
            point_map=np.asarray(pm, dtype=np.float64).reshape(-1, 3),
            up=up,
            objects=[(o.obj_id, np.asarray(o.centroid_world, dtype=np.float64))
                     for o in self._objects.values()],
            start=self.camera_center(0),
        )

    def up_direction(self) -> Optional[np.ndarray]:
        """兼容入口：等价于 `world_up`（v6 起来自 artifact 的世界系契约）。

        历史缺陷（保留作记录）：v5 曾在这里用"包围盒最小 extent 轴"猜竖直轴，
        实测 `acd95847c5` 的最小 extent 是 z，而相机上方向主分量是 **y** →
        `relative_direction` 在错误的平面里算 left/right，四个 rel_direction 题全错。
        v6 的世界系契约由 M3 的位姿束估计给出，**不再**从场景几何猜。
        """
        return self.world_up

    def camera_center(self, frame_idx: int = 0) -> np.ndarray:
        """相机中心（世界系）。主相机 = 首帧（世界系原点，c2w[0]=I）。"""
        if self._c2w is None:
            raise KeyError("SceneHandle 未注入 c2w_list（重建产物缺失）")
        return np.asarray(self._c2w[frame_idx][:3, 3], dtype=np.float64)

    def exists(self, name_or_id: str) -> bool:
        try:
            self.resolve_object_id(name_or_id)
            return True
        except KeyError:
            return False

    # ---- 相机参数 ----
    def get_c2w(self, frame_idx: int) -> np.ndarray:
        if self._c2w is None:
            raise KeyError("SceneHandle 未注入 c2w_list（重建产物缺失）")
        return self._c2w[frame_idx]

    def get_intrinsics(self, frame_idx: int) -> np.ndarray:
        if self._k is None:
            raise KeyError("SceneHandle 未注入 intrinsics（重建产物缺失）")
        return self._k[frame_idx] if self._k.ndim == 3 else self._k

    def get_point_map(self) -> Optional[np.ndarray]:
        """世界系点图 (N,H,W,3)（平面拟合/连通性图用；未注入则 None）。"""
        return self._point_map

    def set_point_map(self, point_map: Optional[np.ndarray]) -> None:
        self._point_map = (None if point_map is None
                           else np.asarray(point_map, dtype=np.float64))

    # ---- 场景包围盒（世界系）----
    def scene_bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """由所有对象 bbox（[min_x,min_y,min_z,max_x,max_y,max_z]）求并集。"""
        if not self._objects:
            raise ValueError("场景中无对象，无法求包围盒")
        mins, maxs = [], []
        for o in self._objects.values():
            if len(o.bbox) != 6:
                raise ValueError(f"对象 {o.obj_id} bbox 维度异常: {o.bbox}")
            mins.append(np.asarray(o.bbox[:3], dtype=np.float64))
            maxs.append(np.asarray(o.bbox[3:], dtype=np.float64))
        return np.minimum.reduce(mins), np.maximum.reduce(maxs)

    # ---- 状态指纹（mock_replay 校验用）----
    def state_digest(self) -> str:
        payload = {
            "artifact_ref": self._state.artifact_ref,
            "scene_route": self._state.scene_route,
            "question_tool_scope": self._state.question_tool_scope,
            "question_type": self.question_type,
            "objects": sorted(self._objects.keys()),
            "available_artifacts": sorted(self.available_artifacts),
            "frame_set_hash": self.frame_set_hash,
            "c2w": None if self._c2w is None else hashlib.sha256(self._c2w.tobytes()).hexdigest(),
            "k": None if self._k is None else hashlib.sha256(self._k.tobytes()).hexdigest(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
