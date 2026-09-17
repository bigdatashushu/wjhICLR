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
    ) -> None:
        self._state = scene_state
        self._objects = {o.instance_id: o for o in (objects or [])}
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
    def route(self) -> str:
        return self._state.route

    @property
    def summary(self) -> str:
        return self._state.summary

    @property
    def quality_overall(self) -> Optional[float]:
        return self._quality_overall

    # ---- 对象访问 ----
    def list_objects(self) -> list[str]:
        return sorted(self._objects.keys())

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
            "c2w": None if self._c2w is None else hashlib.sha256(self._c2w.tobytes()).hexdigest(),
            "k": None if self._k is None else hashlib.sha256(self._k.tobytes()).hexdigest(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
