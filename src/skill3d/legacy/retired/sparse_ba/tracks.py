"""多帧 track 合并与不变量检查（§10.1 L0）。

track = 同一物理 3D 点跨帧的观测链。本模块只做**纯逻辑**合并与校验：
GPU 侧（SuperPoint/ALIKED + LightGlue + RANSAC）产出的是逐 pair 匹配，
这里把它们并成 track 并拒绝非法结构。

**必须满足的不变量**（L0 硬门）：

1. 同帧单观测：一条 track 在同一帧最多出现一次（同帧多观测 = 匹配歧义）；
2. 无循环冲突：合并时若两观测已在同一 track，不得再次合并（union 语义）；
3. 有限坐标：非有限坐标的观测直接丢弃（不得进入三角化）；
4. 最小 track 长度：默认 `MIN_TRACK_LENGTH`，短 track 不参与 BA。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from . import MIN_TRACK_LENGTH


@dataclass
class Observation:
    """一次观测：`(frame_id, keypoint_id, xy)`。"""

    frame_id: int
    keypoint_id: int
    xy: tuple[float, float]


@dataclass
class Track:
    track_id: int
    observations: list[Observation] = field(default_factory=list)

    def frames(self) -> list[int]:
        return [o.frame_id for o in self.observations]

    def has_duplicate_frame(self) -> bool:
        frames = self.frames()
        return len(frames) != len(set(frames))

    def all_finite(self) -> bool:
        return all(np.isfinite(o.xy[0]) and np.isfinite(o.xy[1])
                   for o in self.observations)


@dataclass
class TrackMergeReport:
    tracks: list[Track] = field(default_factory=list)
    n_input_matches: int = 0
    n_dropped_nonfinite: int = 0
    n_dropped_duplicate_frame: int = 0
    n_dropped_short: int = 0
    n_merge_cycles_rejected: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def summary(self) -> str:
        return (f"tracks={self.n_tracks} matches={self.n_input_matches} "
                f"drop_nonfinite={self.n_dropped_nonfinite} "
                f"drop_dup_frame={self.n_dropped_duplicate_frame} "
                f"drop_short={self.n_dropped_short} "
                f"cycles_rejected={self.n_merge_cycles_rejected}")


class _UnionFind:
    """观测级并查集（`(frame, keypoint)` 为节点）。"""

    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}

    def add(self, node: tuple[int, int]) -> None:
        self._parent.setdefault(node, node)

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        self.add(node)
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:   # 路径压缩
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """合并两节点；返回是否发生了新合并（False = 已在同一集合）。"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[rb] = ra
        return True


def merge_tracks(
    matches: Iterable[Sequence[tuple[tuple[int, int], tuple[float, float]]]],
    *,
    min_length: int = MIN_TRACK_LENGTH,
) -> TrackMergeReport:
    """把逐 pair 匹配并成 track（HC36 前端输出的唯一消费入口）。

    参数形态：每个 pair 的匹配列表 =
    `[((frame_a, kp_a), (xa, ya)), ((frame_b, kp_b), (xb, yb)), ...]` 成对出现
    ——即每次匹配贡献两个 (frame, keypoint) 节点。为便于调用，接受**扁平的
    (node, xy) 序列**：相邻两项构成一次匹配。

    **不做**任何按内容的启发式（不读题目/答案，也不按置信度删点）：
    只做结构性拒绝（非有限坐标、同帧重复、过短）。
    """
    uf = _UnionFind()
    coords: dict[tuple[int, int], tuple[float, float]] = {}
    rep = TrackMergeReport()

    flat: list[tuple[tuple[int, int], tuple[float, float]]] = []
    for group in matches:
        flat.extend(list(group))
    if len(flat) % 2 != 0:
        rep.problems.append(f"匹配序列长度为奇数（{len(flat)}）：无法成对解释")
    n_match_events = len(flat) // 2
    rep.n_input_matches = n_match_events

    for i in range(n_match_events):
        (na, xya) = flat[2 * i]
        (nb, xyb) = flat[2 * i + 1]
        if not (np.isfinite(xya[0]) and np.isfinite(xya[1])
                and np.isfinite(xyb[0]) and np.isfinite(xyb[1])):
            rep.n_dropped_nonfinite += 1
            continue
        if na == nb:                       # 同帧匹配 → 无几何意义
            rep.n_dropped_duplicate_frame += 1
            continue
        uf.add(na)
        uf.add(nb)
        coords[na] = xya
        coords[nb] = xyb
        if not uf.union(na, nb):
            # 已在同一集合：合并成环（同一 track 内的重复观测）
            rep.n_merge_cycles_rejected += 1

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in coords:
        groups.setdefault(uf.find(node), []).append(node)

    tracks: list[Track] = []
    for idx, (_, nodes) in enumerate(sorted(groups.items(),
                                            key=lambda kv: sorted(kv[1]))):
        obs = [Observation(frame_id=int(n[0]), keypoint_id=int(n[1]),
                           xy=(float(coords[n][0]), float(coords[n][1])))
               for n in sorted(nodes)]
        t = Track(track_id=idx, observations=obs)
        if t.has_duplicate_frame():
            rep.n_dropped_duplicate_frame += 1
            continue
        if len(obs) < int(min_length):
            rep.n_dropped_short += 1
            continue
        tracks.append(t)

    rep.tracks = tracks
    return rep


def validate_tracks(tracks: Sequence[Track], *,
                    min_length: int = MIN_TRACK_LENGTH) -> tuple[bool, list[str]]:
    """L0 不变量检查（§10.1）：返回 (是否通过, 问题清单)。"""
    problems: list[str] = []
    for t in tracks:
        if t.has_duplicate_frame():
            problems.append(f"track {t.track_id} 同帧多观测: {sorted(t.frames())}")
        if not t.all_finite():
            problems.append(f"track {t.track_id} 含非有限坐标")
        if len(t.observations) < int(min_length):
            problems.append(f"track {t.track_id} 长度 {len(t.observations)} "
                            f"< 最小长度 {min_length}")
    ids = [t.track_id for t in tracks]
    if len(ids) != len(set(ids)):
        problems.append("track_id 重复")
    return (not problems), problems


def tracks_to_arrays(tracks: Sequence[Track]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """展开为 `(frame_ids, keypoint_ids, xy)` 三个数组（喂 PyCOLMAP 三角化）。"""
    f: list[int] = []
    k: list[int] = []
    xy: list[tuple[float, float]] = []
    for t in tracks:
        for o in t.observations:
            f.append(o.frame_id)
            k.append(o.keypoint_id)
            xy.append(o.xy)
    return (np.asarray(f, dtype=np.int64), np.asarray(k, dtype=np.int64),
            np.asarray(xy, dtype=np.float64).reshape(-1, 2))


def track_length_histogram(tracks: Sequence[Track],
                           max_len: Optional[int] = None) -> dict[int, int]:
    """track 长度直方图（L1/L2 报告用）。"""
    hist: dict[int, int] = {}
    for t in tracks:
        n = len(t.observations)
        if max_len is not None and n > max_len:
            n = max_len
        hist[n] = hist.get(n, 0) + 1
    return dict(sorted(hist.items()))
