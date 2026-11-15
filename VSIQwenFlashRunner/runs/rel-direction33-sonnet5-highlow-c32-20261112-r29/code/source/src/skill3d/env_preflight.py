"""运行前环境预检（v9 §17.1「环境依赖」+ §18.1 有区分力的验收）。

存在的理由是一次真实的静默降级：base conda env 是 numpy 1.26.4 + scipy 1.18.0
（scipy 要求 numpy≥2.0），`import scipy.spatial` 直接抛 `np.long`；加了临时 shim
之后 `cKDTree` 构造仍然抛 `ValueError: NoneType copy mode not allowed`。此时

- M4 主门的子项被 `except Exception` 吞成 warning（`m4_main_gate` 的 fail-closed
  设计本身是对的），于是 `main_gate_passed=False` → `scene_route=fallback_2d_only`；
- 整条在线链照常跑完，只是**所有几何/米制能力静默失效**，看起来像"场景质量不够"
  而不是"环境坏了"。

环境层面的故障必须在启动时吵闹（§12：不能把工程故障伪装成方法效果）。判据是
**行为**而不是版本号：直接调用运行时真正依赖的 scipy 能力，因为版本号组合的
兼容矩阵会漂移，而"这个函数现在能不能用"不会。
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Callable, Optional

# 冻结实验环境（README §环境 / `本机环境实测.md` §8）。预检只用于**报错提示**，
# 不作为判据 —— 判据始终是下面的行为检查。
FROZEN_ENV_PYTHON = "/home/cvailab/anaconda3/envs/skill3d-exp/bin/python"


class RuntimeEnvironmentError(RuntimeError):
    """环境依赖不可用 → 拒绝启动，而不是降级后照跑。"""


def _probe_cKDTree() -> None:
    """m4_main_gate（点云重叠）与 distance_primitives（最近邻距离）依赖。"""
    import numpy as np
    from scipy.spatial import cKDTree

    pts = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tree = cKDTree(pts)                 # 旧 numpy 下这里抛 copy=None
    dist, _ = tree.query(pts, k=1)
    if not np.all(np.isfinite(dist)):
        raise ValueError("cKDTree.query 返回非有限距离")


def _probe_rankdata() -> None:
    """m4_main_gate 的一致性/秩相关子项依赖。"""
    from scipy.stats import rankdata

    ranks = rankdata([1.0, 2.0, 2.0, 3.0])
    if list(ranks) != [1.0, 2.5, 2.5, 4.0]:
        raise ValueError(f"rankdata 结果异常: {list(ranks)}")


def _probe_ndimage() -> None:
    """connectivity（route_planning 的可通行图）依赖。"""
    import numpy as np
    from scipy.ndimage import binary_dilation, label

    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    if int(label(mask)[1]) != 1:
        raise ValueError("ndimage.label 连通分量数异常")
    if not binary_dilation(mask).any():
        raise ValueError("ndimage.binary_dilation 无输出")


# (名称, 探针, 影响的运行时能力)
_PROBES: tuple[tuple[str, Callable[[], None], str], ...] = (
    ("scipy.spatial.cKDTree", _probe_cKDTree,
     "M4 点云重叠子项 / 最近邻距离原语（object_abs_distance、object_rel_distance）"),
    ("scipy.stats.rankdata", _probe_rankdata, "M4 一致性子项"),
    ("scipy.ndimage.label", _probe_ndimage, "connectivity_graph（route_planning）"),
)


def _version_of(module: str) -> str:
    try:
        mod = __import__(module)
        return str(getattr(mod, "__version__", "") or "")
    except Exception as exc:  # noqa: BLE001 - 版本读不到本身就是一条环境事实
        return f"<unavailable: {type(exc).__name__}>"


def describe_environment() -> dict[str, Any]:
    """环境依赖快照，供 RunManifest「环境依赖」留档（§17.1）。"""
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": _version_of("numpy"),
        "scipy": _version_of("scipy"),
        "opencv": _version_of("cv2"),
    }


def runtime_dependency_problems() -> list[str]:
    """逐项真实调用运行时依赖；返回人类可读的问题清单（空 = 全部可用）。"""
    problems: list[str] = []
    for name, probe, impact in _PROBES:
        try:
            probe()
        except Exception as exc:  # noqa: BLE001 - 探测失败即环境问题，不冒泡
            problems.append(f"{name} 不可用（{type(exc).__name__}: {exc}）；影响：{impact}")
    return problems


def assert_runtime_dependencies(*, context: str = "online") -> dict[str, Any]:
    """环境不可用则 refusal（fail-closed）；可用则返回环境快照。

    绝不在依赖缺失时"照常跑完"：M4 子项的 fail-closed 会把环境故障记成
    `fallback_2d_only`，那是把工程故障伪装成场景质量。
    """
    env = describe_environment()
    problems = runtime_dependency_problems()
    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        raise RuntimeEnvironmentError(
            f"[{context}] 运行环境依赖检查未通过，已拒绝启动（不在降级状态下出结果）：\n"
            f"{detail}\n"
            f"  当前解释器：{env['python_executable']}"
            f"（python {env['python']} / numpy {env['numpy']} / scipy {env['scipy']}）\n"
            f"  本项目冻结实验环境：{FROZEN_ENV_PYTHON}\n"
            "  注意：numpy 与 scipy 的版本必须自洽（scipy 需要匹配的 numpy 主版本）；"
            "不要用测试进程内 shim 掩盖该问题 —— shim 只能补一个符号，补不了 ABI。")
    return env
