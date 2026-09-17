"""M3 可选稠密化：gsplat 3DGS 训练（§4 M3）。

gsplat Apache-2.0，比原版 3DGS 少 4x 显存；训练 6-10GB、~20-30min/场景。
lazy import；可选模块，失败不阻断主线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

GSPLAT_CONFIG: Optional[str] = None  # TODO_USER_INPUT: gsplat simple_trainer 配置


def train_gsplat(colmap_dir: str | Path, output_dir: str | Path) -> Optional[str]:
    """基于 VGGT/COLMAP 导出的 sparse 模型训练 3DGS 稠密化。

    返回 PLY 路径；gsplat 不可用或训练失败返回 None（可选模块降级）。
    TODO: gsplat simple_trainer 具体 CLI/API 以官方 repo 为准（§4 M3 字段 5）。
    """
    try:
        import torch  # lazy import
        import gsplat  # noqa: F401  # lazy import
    except Exception:
        return None

    # TODO: 调用 gsplat simple_trainer（或 python -m examples.simple_trainer）
    # 输入 colmap_dir（demo_colmap.py --use_ba 输出），输出 PLY
    return None
