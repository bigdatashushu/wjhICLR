"""场景连通性/可达图（§9.10 `connectivity_graph`，route_planning 题型的确定性原语）。

做法（全部阈值 `[TODO_CALIBRATE]`）：

1. 用 `world_up` 建地面平面正交基，把世界系点云投影成 (x, y, h)；
2. 以点云的**高度低分位**为地面高度 h_floor，在 `[h_floor + band_lo, h_floor + band_hi]`
   高度带内的点视为**障碍**（墙/家具的主体都落在这一带；纯地面点落在地面附近）；
3. 障碍按占据栅格聚合，并做半径膨胀（把 agent 当作有体积的点，避免"贴着墙走"）；
4. 自由空间做连通域标记（`scipy.ndimage.label`，8 邻域）；
5. 对象与起点按所在自由域判定可达：同域 → 有边，边权 = 地面平面欧氏距离。

诚实边界：这是**几何可达性**（自由空间是否连通），不是真实导航规划；它不判
"门是否关着"、不建模可移动障碍。`traversable_matrix` 只表示"两点在同一连通自由域"。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

# ---- 阈值（全部 TODO_CALIBRATE）----
GRID_CELL_REL: float = 0.02        # TODO_CALIBRATE: 栅格边长 / 场景尺度
OBSTACLE_BAND_LO_REL: float = 0.01  # TODO_CALIBRATE: 障碍高度带下界 / 场景尺度
OBSTACLE_BAND_HI_REL: float = 1.00  # TODO_CALIBRATE: 障碍高度带上界 / 场景尺度
AGENT_RADIUS_REL: float = 0.03     # TODO_CALIBRATE: agent 半径 / 场景尺度（膨胀量）
FLOOR_PERCENTILE: float = 10.0     # TODO_CALIBRATE: 地面高度取点云高度第 10 百分位
MAX_GRID_CELLS: int = 4_000_000    # 内存护栏：栅格单元上限（超限自动放大格子）
CONNECTIVITY_VERSION: str = "connectivity-v6"


def _ground_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由 up 建地面平面正交基 (e1, e2)，确定性（不依赖随机/特征分解顺序）。"""
    u = np.asarray(up, dtype=np.float64)
    u = u / float(np.linalg.norm(u))
    # 取与 u 最不平行的坐标轴做种子，保证数值稳定且确定
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(u)))] = 1.0
    e1 = np.cross(u, seed)
    n1 = float(np.linalg.norm(e1))
    if n1 < 1e-12:  # pragma: no cover - u 与所有坐标轴都平行时不可能
        raise ValueError("无法由 world_up 构造地面基（退化向量）")
    e1 = e1 / n1
    e2 = np.cross(u, e1)
    return e1, e2 / float(np.linalg.norm(e2))


def build_connectivity_graph(
    *,
    point_map: np.ndarray,
    up: np.ndarray,
    objects: Sequence[tuple[str, np.ndarray]],
    start: Optional[np.ndarray] = None,
) -> dict:
    """构建连通性/可达图（§9.10）。

    - `point_map`：(M,3) 世界系点云（已展平）；
    - `up`：世界系单位上方向（缺失时调用方必须 fail-closed，不得猜）；
    - `objects`：[(obj_id, centroid_world)]；
    - `start`：起点（通常相机中心），可选。

    返回 `{nodes, edges, traversable_matrix, start_node, grid, notes}`：
    `nodes` 为 `[start?] + obj_ids`；`traversable_matrix[i][j]` 表示 i、j 在同一
    自由连通域内（几何可达）。
    """
    pts = np.asarray(point_map, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] < 3:
        raise ValueError("点云有效点不足，无法构建连通性图")

    e1, e2 = _ground_basis(up)
    # 场景尺度：用点云在三个主轴上的跨度中位数（对离群点稳健）
    span = float(np.median(np.ptp(pts, axis=0)))
    if not np.isfinite(span) or span <= 0:
        span = float(np.max(np.ptp(pts, axis=0)))
    if not np.isfinite(span) or span <= 0:
        raise ValueError("点云跨度为 0，无法估计场景尺度")

    xy = np.stack([pts @ e1, pts @ e2], axis=1)
    h = pts @ np.asarray(up, dtype=np.float64) / float(
        np.linalg.norm(np.asarray(up, dtype=np.float64)))
    h_floor = float(np.percentile(h, FLOOR_PERCENTILE))

    origin = xy.min(axis=0)
    extent = xy.max(axis=0) - origin
    cell = max(GRID_CELL_REL * span, 1e-9)
    while True:
        nx = int(np.ceil(extent[0] / cell)) + 1
        ny = int(np.ceil(extent[1] / cell)) + 1
        if nx * ny <= MAX_GRID_CELLS or cell > span:
            break
        cell *= 2.0

    ij = np.floor((xy - origin) / cell).astype(np.int64)
    ij = np.clip(ij, 0, [nx - 1, ny - 1])

    band_lo = h_floor + OBSTACLE_BAND_LO_REL * span
    band_hi = h_floor + OBSTACLE_BAND_HI_REL * span
    in_band = (h >= band_lo) & (h <= band_hi)

    occupied = np.zeros((ny, nx), dtype=bool)
    if np.any(in_band):
        occupied[ij[in_band, 1], ij[in_band, 0]] = True

    free = ~occupied
    # agent 半径膨胀：把障碍向外扩，等价于把自由空间向内缩
    r_cells = int(np.ceil(AGENT_RADIUS_REL * span / cell))
    if r_cells > 0 and occupied.any():
        from scipy.ndimage import binary_dilation

        struct = np.ones((2 * r_cells + 1, 2 * r_cells + 1), dtype=bool)
        free = ~binary_dilation(occupied, structure=struct)

    from scipy.ndimage import label

    labels, n_comp = label(free, structure=np.ones((3, 3), dtype=np.int32))

    def _cell_of(p: np.ndarray) -> Optional[tuple[int, int]]:
        v = np.asarray(p, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(v)):
            return None
        q = np.array([float(v @ e1), float(v @ e2)])
        if np.any(q < origin) or np.any(q > origin + extent):
            # 落在点云范围外：夹到边界（房间外沿的物体常略超点云范围）
            q = np.clip(q, origin, origin + extent)
        c = np.floor((q - origin) / cell).astype(np.int64)
        i, j = int(np.clip(c[1], 0, ny - 1)), int(np.clip(c[0], 0, nx - 1))
        return (i, j)

    def _label_of(p: Optional[np.ndarray]) -> int:
        if p is None:
            return 0
        c = _cell_of(p)
        if c is None:
            return 0
        return int(labels[c[0], c[1]])

    node_ids: list[str] = []
    node_labels: list[int] = []
    node_xy: list[list[float]] = []
    start_label = 0
    if start is not None:
        start_label = _label_of(np.asarray(start, dtype=np.float64))
        node_ids.append("start")
        node_labels.append(start_label)
        node_xy.append([float(np.asarray(start, dtype=np.float64) @ e1),
                        float(np.asarray(start, dtype=np.float64) @ e2)])
    for oid, cen in objects:
        v = np.asarray(cen, dtype=np.float64)
        node_ids.append(str(oid))
        node_labels.append(_label_of(v))
        node_xy.append([float(v @ e1), float(v @ e2)])

    n = len(node_ids)
    traversable = [[False] * n for _ in range(n)]
    edges: list[dict] = []
    for i in range(n):
        traversable[i][i] = True
        for j in range(i + 1, n):
            # 可达 ⟺ 同属一个非零自由连通域
            reach = (node_labels[i] != 0 and node_labels[i] == node_labels[j])
            traversable[i][j] = traversable[j][i] = bool(reach)
            if reach:
                d = float(np.linalg.norm(np.asarray(node_xy[i]) - np.asarray(node_xy[j])))
                edges.append({"a": node_ids[i], "b": node_ids[j], "cost": d})

    notes = [
        f"连通性图 v{CONNECTIVITY_VERSION}：{n} 节点 / {len(edges)} 可达边；"
        f"自由连通域 {int(n_comp)} 个；栅格 {nx}×{ny}，cell={cell:.4g}（场景尺度 {span:.4g}）；"
        f"障碍膨胀 r={r_cells} 格；阈值全 TODO_CALIBRATE",
    ]
    if n_comp > 1:
        notes.append(f"[warn] 自由空间被切成 {int(n_comp)} 个连通域 —— "
                     "可能真有隔断，也可能是阈值过严（TODO_CALIBRATE）")

    return {
        "nodes": node_ids,
        "edges": edges,
        "traversable_matrix": traversable,
        "node_components": node_labels,
        "start_node": "start" if start is not None else None,
        "grid": {"nx": nx, "ny": ny, "cell": cell, "n_components": int(n_comp)},
        "version": CONNECTIVITY_VERSION,
        "notes": notes,
    }


__all__ = ["build_connectivity_graph", "CONNECTIVITY_VERSION"]
