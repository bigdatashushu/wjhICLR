"""预注册 pair graph（§10.1 L0：确定、与题目无关、覆盖全部 32 帧）。

**纪律**：

- pair graph 只由帧数决定：不读问题、不读答案、不读图像内容；
- 同一配置的 `pair_graph_hash` 必须稳定（内容寻址：`sha256(json)`）；
- 必须覆盖全部帧（每个 frame 至少出现在一个 pair 里），否则某些帧的位姿
  在 BA 中不受约束；
- 相邻 pair（短基线，用于三角化）与有限跨帧 pair（长基线，用于约束漂移）
  分开预注册，数量都固定，不得按结果调参。

本模块是**纯 P1 逻辑**（`[已实现]`，CPU 可测），GPU 侧只消费它的输出。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PairGraphConfig:
    """pair graph 预注册配置（`TODO_CALIBRATE`：全部为起始参考值）。"""

    n_frames: int = 32
    adjacent_stride: int = 1     # 相邻帧步长（1 = 逐帧相邻）
    adjacent_span: int = 2       # 每个节点向前连的相邻帧数
    skip_steps: tuple[int, ...] = (4, 8)   # 跨帧 pair 的跳距（长基线）
    skip_stride: int = 4         # 跨帧 pair 的采样步长

    def as_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "adjacent_stride": self.adjacent_stride,
            "adjacent_span": self.adjacent_span,
            "skip_steps": list(self.skip_steps),
            "skip_stride": self.skip_stride,
        }


@dataclass
class PairGraph:
    config: PairGraphConfig
    pairs: list[tuple[int, int]] = field(default_factory=list)
    pair_graph_hash: str = ""

    def as_dict(self) -> dict:
        return {
            "config": self.config.as_dict(),
            "n_pairs": len(self.pairs),
            "pairs": [[a, b] for a, b in self.pairs],
            "pair_graph_hash": self.pair_graph_hash,
        }


def build_pair_graph(config: PairGraphConfig | None = None) -> PairGraph:
    """构造确定性的 pair graph（与题目/答案无关）。"""
    cfg = config or PairGraphConfig()
    n = int(cfg.n_frames)
    if n < 2:
        raise ValueError(f"pair graph 需要 >=2 帧，收到 {n}")
    pairs: set[tuple[int, int]] = set()

    step = max(1, int(cfg.adjacent_stride))
    span = max(1, int(cfg.adjacent_span))
    for i in range(n):
        for k in range(1, span + 1):
            j = i + k * step
            if j < n:
                pairs.add((i, j))

    for d in cfg.skip_steps:
        d = int(d)
        if d <= 0 or d >= n:
            continue
        stride = max(1, int(cfg.skip_stride))
        for i in range(0, n - d, stride):
            pairs.add((i, i + d))

    ordered = sorted(pairs)
    h = hashlib.sha256(json.dumps(
        {"config": cfg.as_dict(), "pairs": [list(p) for p in ordered]},
        sort_keys=True).encode()).hexdigest()
    return PairGraph(config=cfg, pairs=ordered, pair_graph_hash=h)


def uncovered_frames(graph: PairGraph) -> list[int]:
    """未被任何 pair 覆盖的帧（L0 硬门：必须为空）。"""
    seen: set[int] = set()
    for a, b in graph.pairs:
        seen.add(int(a))
        seen.add(int(b))
    return [i for i in range(graph.config.n_frames) if i not in seen]


def duplicate_pairs(graph: PairGraph) -> list[tuple[int, int]]:
    """重复 pair（L0 硬门：必须为空；重复会浪费算力且破坏确定性计数）。"""
    counts: dict[tuple[int, int], int] = {}
    for p in graph.pairs:
        counts[p] = counts.get(p, 0) + 1
    return sorted(p for p, c in counts.items() if c > 1)


def self_pairs(graph: PairGraph) -> list[tuple[int, int]]:
    """自环 pair（必须为空）。"""
    return [(a, b) for a, b in graph.pairs if a == b]


def l0_pair_graph_ok(graph: PairGraph) -> tuple[bool, list[str]]:
    """L0 合同检查（§10.1）：返回 (是否通过, 问题清单)。"""
    problems: list[str] = []
    if not graph.pairs:
        problems.append("pair graph 为空")
    unc = uncovered_frames(graph)
    if unc:
        problems.append(f"未覆盖帧: {unc}")
    dup = duplicate_pairs(graph)
    if dup:
        problems.append(f"重复 pair: {dup[:5]}")
    sp = self_pairs(graph)
    if sp:
        problems.append(f"自环 pair: {sp[:5]}")
    if not graph.pair_graph_hash:
        problems.append("pair_graph_hash 为空")
    return (not problems), problems
