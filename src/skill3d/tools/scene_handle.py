"""SceneHandle：Tool 访问重建产物的唯一只读句柄（硬约束 17）。

Tool 不得自行读原始文件、不得绕过置信度图、不得访问未挂载路径。
SceneHandle 只持有内存中的数组/对象结构，不暴露任何文件系统路径。
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

from skill3d.schemas import ObjectInstance, SceneState

from .contract import (
    ARTIFACT_INTRINSICS,
    ARTIFACT_OBJECTS,
    ARTIFACT_POSES,
    ARTIFACT_SCALE,
    route_aware_available,
)


class SceneHandle:
    """SceneState 的只读运行期句柄。

    - objects: ObjectInstance 列表（内存结构，非文件路径）
    - c2w_list / intrinsics: numpy 数组序列（由重建管线加载后注入）
    - 不提供任何 open()/路径访问接口
    """

    def __init__(
        self,
        scene_state: SceneState,
        objects: Optional[list[ObjectInstance]] = None,
        c2w_list: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        quality_overall: Optional[float] = None,
        objects_materialized: Optional[bool] = None,
        metric_scale: Optional[float] = None,
    ) -> None:
        self._state = scene_state
        # v5 HC29：世界系 → 米 的换算系数（`scale_ci_abs_m = metric_scale × scale_ci_rel`）。
        # 优先用显式传入值（mock_light 的合成几何），否则读运行时 artifact。
        self._metric_scale = metric_scale
        self._objects = {o.instance_id: o for o in (objects or [])}
        # 硬约束 23：objects **产物是否已产出**与"产物里有没有对象"是两件事。
        # - M5 未跑/失败 → 产物不存在 → `objects` 不可用 → Tool 抛 ArtifactUnavailableError；
        # - M5 跑过但一个对象都没绑到 → 产物存在且为空 → `exists_in_scene` 返回 False。
        # 未显式声明时保守按"空列表视为未产出"处理（老调用点语义不变）。
        self._objects_materialized = (
            bool(objects) if objects_materialized is None else bool(objects_materialized)
        )
        # class_hint 小写索引，供 exists_in_scene / 名称解析
        self._by_hint: dict[str, str] = {}
        for o in objects or []:
            self._by_hint.setdefault(o.class_hint.lower(), o.instance_id)
        self._c2w = None if c2w_list is None else np.asarray(c2w_list, dtype=np.float64)
        self._k = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        self._quality_overall = quality_overall

    # ---- 只读元信息 ----
    @property
    def frame(self) -> str:
        return self._state.frame

    @property
    def scale_known(self) -> bool:
        return self._state.scale_known

    @property
    def allowed_metric_tasks(self) -> set[str]:
        """v4 HC33：本次尺度评估授权的米制题型（逐题门控的唯一依据）。"""
        return {str(t) for t in (getattr(self._state, "allowed_metric_tasks", None) or set())}

    @property
    def question_type(self) -> str:
        """v4 HC33：本 episode 的规范题型（逐题授权的第二维）。"""
        return str(getattr(self._state, "question_type", "") or "")

    @property
    def route(self) -> str:
        return self._state.route

    @property
    def summary(self) -> str:
        return self._state.summary

    @property
    def metric_scale(self) -> Optional[float]:
        """世界系 → 米 的换算系数（米/世界单位）。

        HC29 定义：`scale_ci_abs_m = metric_scale × scale_ci_rel`（m）。米制 Tool
        必须用它把世界单位换算成米，**不得**直接把世界单位当米返回。
        未标定/未锚定（None 或非有限）→ 米制 Tool fail-closed（硬约束 23）。
        """
        if self._metric_scale is not None:
            return float(self._metric_scale)
        art = getattr(self._state, "artifact", None)
        v = getattr(art, "metric_scale", None)
        return None if v is None else float(v)

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
        """route → 可用产物集合（与 `tools.contract.ROUTE_ARTIFACTS` 同源）。

        句柄上真正装载/可用的东西会进一步收窄集合（例如未注入 c2w → 不可称 poses 可用，
        避免"声明可用但实际读不到"的静默错答）。

        v4（HC33）：`scale` 只在本场景**至少有一个米制题型被授权**时才可用；逐题的
        收窄（`question_type ∈ allowed_metric_tasks`）由
        `contract.route_artifacts_for_question` 与执行期 `check_metric_task_contract`
        完成。这样 `low` 档只收回米制工具，**不**影响 `depth/poses/point_cloud/objects`。
        """
        declared = route_aware_available(
            getattr(self._state, "available_artifacts", None), self.route)
        if self._c2w is None:
            declared -= {ARTIFACT_POSES}
        if self._k is None:
            declared -= {ARTIFACT_INTRINSICS, ARTIFACT_POSES}
        if not self._objects_materialized:
            declared -= {ARTIFACT_OBJECTS}
        if not (self._state.scale_known and self.allowed_metric_tasks):
            # 尺度未锚定（D-2）或没有任何米制题型被授权（HC33）→ `scale` 产物不可用：
            # metric Tool 必须 fail-closed（硬约束 23），而不是拿相对单位当真值作答
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

    def list_objects_by_hint(self, hint: str = "") -> list[str]:
        """按 class_hint 子串过滤对象 id（空 = 全部）。大小写不敏感。

        计数题必须靠这个（`exists_in_scene` 是布尔，拿它当计数器只能得到 0/1）。
        """
        h = str(hint or "").strip().lower()
        if not h:
            return self.list_objects()
        out = []
        for oid in self.list_objects():
            obj = self._objects.get(oid)
            hint_val = str(getattr(obj, "class_hint", "") or "").lower()
            if h in hint_val or (hint_val and hint_val in h):
                out.append(oid)
        return out

    def object_class_hint(self, object_id: str) -> str:
        obj = self._objects.get(str(object_id))
        return str(getattr(obj, "class_hint", "") or "")

    def resolve_object_id(self, name_or_id: str) -> str:
        """按 instance_id 或 class_hint（大小写不敏感）解析对象。"""
        if name_or_id in self._objects:
            return name_or_id
        hit = self._by_hint.get(name_or_id.lower())
        if hit is None:
            raise KeyError(f"对象不存在: {name_or_id}")
        return hit

    def get_object(self, name_or_id: str) -> ObjectInstance:
        return self._objects[self.resolve_object_id(name_or_id)]

    def up_direction(self) -> Optional[np.ndarray]:
        """世界系**竖直向上**单位向量（由相机位姿导出）；无法确定时返回 None。

        相机约定为 OpenCV（+y_cam 指向图像下方），故世界系里"图像上方" =
        `-c2w[:3, 1]`。手持扫描中相机基本竖直，多帧平均即重力反方向。

        为什么不能靠"包围盒最小 extent 轴"猜（2026-09-21 真实缺陷）：那个启发式
        ① 无符号（分不清上下）；② 在长条形房间里会把水平轴当成竖轴。实测
        `acd95847c5` 的最小 extent 是 z（1.818 < 1.865 < 2.282），而相机上方向
        的主分量是 **y**（|−0.726| 最大）→ `relative_direction` 在错误的平面里算
        left/right，四个 rel_direction 题全错。
        """
        if self._c2w is None:
            return None
        R = np.asarray(self._c2w, dtype=np.float64)[:, :3, :3]
        if R.size == 0:
            return None
        up = -R[:, :, 1].mean(axis=0)
        n = float(np.linalg.norm(up))
        if not np.isfinite(n) or n < 1e-9:
            return None
        return up / n

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

    # ---- 场景包围盒（世界系）----
    def scene_bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """由所有对象 bbox（[min_x,min_y,min_z,max_x,max_y,max_z]）求并集。"""
        if not self._objects:
            raise ValueError("场景中无对象，无法求包围盒")
        mins, maxs = [], []
        for o in self._objects.values():
            if len(o.bbox) != 6:
                raise ValueError(f"对象 {o.instance_id} bbox 维度异常: {o.bbox}")  # TODO bbox 格式约定
            mins.append(np.asarray(o.bbox[:3], dtype=np.float64))
            maxs.append(np.asarray(o.bbox[3:], dtype=np.float64))
        return np.minimum.reduce(mins), np.maximum.reduce(maxs)

    # ---- 状态指纹（mock_replay 校验用）----
    def state_digest(self) -> str:
        payload = {
            "artifact_ref": self._state.artifact_ref,
            "route": self._state.route,
            "frame": self._state.frame,
            "scale_known": self._state.scale_known,
            "objects": sorted(self._objects.keys()),
            "available_artifacts": sorted(self.available_artifacts),
            "frame_set_hash": self.frame_set_hash,
            "c2w": None if self._c2w is None else hashlib.sha256(self._c2w.tobytes()).hexdigest(),
            "k": None if self._k is None else hashlib.sha256(self._k.tobytes()).hexdigest(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
