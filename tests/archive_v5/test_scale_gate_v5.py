"""v5 尺度可用性（`scale_known` + `allowed_metric_tasks`）单测 —— 只读归档。

原位置：`tests/unit/test_v3_gap_fixes.py::test_scale_artifact_hidden_when_scale_unusable`
（v5 期间正文，逐字保留，**不做适配、不做修补**；辅助 `_scene` / `_obj` 一并保留）。

**v6 废止原因**（《系统架构 v6》§5.3/§13/§20）：

- v5 用 `scale_known`（整场景的布尔中间量）+ `allowed_metric_tasks`（逐题型授权集合）
  两个字段表达"米制能不能用"；v6 把这两个字段整体删除（`route` 一词三义也被
  `scene_route` × `question_tool_scope` 取代），米制可用性改为
  `EvidenceProfile.metric_scale`（三值）由 **MetricEvidenceGate** 的 6 项子条件决定；
- `available_artifacts` 不再由 `allowed_metric_tasks` 参与计算，而是
  `ROUTE_ARTIFACTS[scene_route] ∪ ({scale} if 米制题 ∧ gate_passed)`；
- `SceneHandle` 上不再有 `scale_known`，`available_artifacts` 每次按
  `metric_task_authorized()`（题型 ∈ 米制 ∧ gate 通过 ∧ scope=metric_enabled）现算。

**v6 替代物（回归入口）**：
- `tests/unit/test_v3_gap_fixes.py::test_scale_artifact_requires_metric_gate_and_metric_question`
  （融合未跑 → 任何题型都没有 `scale`；融合成功 + receipt + 米制题 → 有；非米制题 → 无）；
- `tests/unit/test_v6_schema.py::test_scope_allows_narrowing_only` /
  `::test_registry_docs_tracks_evidence_and_scope`（scope 收窄与证据驱动的 Tool 暴露）；
- `tests/unit/test_metric_fusion.py`、`tests/unit/test_v6_schema.py`（gate 6 项子条件）。

详见 `tests/archive_v5/README.md`。
"""

from __future__ import annotations

import numpy as np

# 以下 import 是 v5 正文里的（可能已随 v6 删除/改名）——归档件保持原样，不做适配
from skill3d.reconstruction_gate.scene_state import quality_gate  # noqa: F401
from skill3d.sandbox.kernel import RestrictedNamespaceKernel  # noqa: F401
from skill3d.schemas import ConfidenceMap, ReconstructionArtifact, SceneState  # noqa: F401
from skill3d.tools import REGISTRY  # noqa: F401
from skill3d.tools.scene_handle import SceneHandle


def _scene(route="full_3d", scale_known=False, metric_tasks=None) -> SceneState:
    return SceneState(artifact_ref="a", route=route, frame="world",
                      scale_known=scale_known, objects=[], summary="s",
                      allowed_metric_tasks=set(metric_tasks or set()))


def _obj():
    from skill3d.schemas import ObjectInstance

    return ObjectInstance(instance_id="obj_0", class_hint="table", mask_per_frame="",
                          pointcloud_world="", centroid_world=[0.0, 0.0, 0.0],
                          bbox=[0.0] * 6, confidence=0.9)


def test_scale_artifact_hidden_when_scale_unusable():
    """尺度不可用时 `scale` 不得声明为可用（硬约束 23：metric Tool 必须 fail-closed）。

    v4 HC33：`scale` 可用性还要求**至少一个米制题型被授权**（逐题型授权），
    所以"scale_known=True 但 allowed_metric_tasks=∅"同样必须收回 `scale`。
    """
    unusable = SceneHandle(_scene("full_3d"), objects=[_obj()],
                           c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                           intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                           objects_materialized=True)
    assert "scale" not in unusable.available_artifacts     # scale_known=False
    # scale_known=True 但无任何米制授权 → 仍不可用（v4 HC33）
    no_auth = SceneHandle(_scene("full_3d", scale_known=True), objects=[_obj()],
                          c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                          intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                          objects_materialized=True)
    assert "scale" not in no_auth.available_artifacts
    # HC33：收回 scale 不影响非米制 3D 产物
    assert {"depth", "poses", "point_cloud", "objects"} <= no_auth.available_artifacts
    # 有授权 → 可用
    usable = SceneHandle(_scene("full_3d", scale_known=True,
                                metric_tasks={"object_abs_distance"}), objects=[_obj()],
                         c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                         intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                         objects_materialized=True)
    assert "scale" in usable.available_artifacts
