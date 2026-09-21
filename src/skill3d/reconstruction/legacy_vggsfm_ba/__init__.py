"""`reconstruction/legacy_vggsfm_ba`：官方 VGGSfM tracker + PyCOLMAP BA 的**只读历史码**。

**v5 HC35（已确认设计）：该路线已被真实 24 GiB OOM 证据否决，不是生产 route。**

- 不再作为生产 route、系统完成条件或主结果前置条件；
- 默认配置 `enable_official_vggsfm_ba=False`；生产启用请求直接抛
  `UnsupportedConfigurationError`（不静默回退、不"尽力而为"）；
- 只保留失败 receipt、历史代码与可复现实验说明，供论文失败分析引用；
- 生产 BA 候选只有 `vggt_sparse_ba`（见 `reconstruction/sparse_ba/`），
  且必须先通过 §10.1 的 L0→L1→L2 门槛。

历史实现（`route.py`）保留原样，仅在显式复现模式（`allow_repro=True`）下可调用，
用于重放 OOM 失败证据；`recon_method` 名称 `vggt_ba` 也一并废弃（v5 只认
`vggt` 与 `vggt_sparse_ba`）。
"""

from __future__ import annotations

from typing import Any

# ---- 生产开关（默认关闭；HC35 要求默认配置不得触发官方 BA）----
OFFICIAL_VGGSFM_BA_ENABLED: bool = False

# 失败证据引用（论文失败分析/审计用；真实数值见 data/receipts 与 Readiness manifest）
OOM_EVIDENCE_REFS: tuple[str, ...] = (
    "data/readiness_manifest.json#ba_route",
    "问题报告v4.md",
)
REJECTION_REASON: str = "rejected_on_24g_oom"


class UnsupportedConfigurationError(RuntimeError):
    """生产配置请求了一个 v5 明确否决的能力（HC35：官方 VGGSfM BA）。"""


def assert_official_ba_disabled(*, enable_official_vggsfm_ba: bool = False,
                                context: str = "") -> None:
    """生产路径的统一护栏：任何启用官方 BA 的请求都直接报错（fail-closed）。"""
    if enable_official_vggsfm_ba:
        raise UnsupportedConfigurationError(
            "官方 VGGSfM tracker + PyCOLMAP BA 已在 24 GiB RTX 4090 环境被真实 OOM 证据否决"
            "（v5 HC35），不得作为生产 route 启用。默认配置必须为 "
            "enable_official_vggsfm_ba=false；生产 BA 候选只有 `vggt_sparse_ba`"
            "（尚未通过 §10.1 PoC，默认关闭）。"
            + (f" 触发位置：{context}" if context else ""))


def run_official_ba_repro(*args: Any, allow_repro: bool = False, **kwargs: Any) -> Any:
    """**仅复现实验**入口：显式 `allow_repro=True` 才转发到历史实现。

    生产代码不得调用本函数；`allow_repro=False`（默认）即 hard fail，
    这样"偷偷把官方 BA 接回主线"会被立刻发现。
    """
    if not allow_repro:
        raise UnsupportedConfigurationError(
            "官方 VGGSfM BA 只能在显式复现模式下运行（allow_repro=True），"
            "用于重放 OOM 失败证据；生产主线固定 `vggt` feed-forward（HC35）。")
    from .route import run_ba_route  # 延迟导入：历史码只在复现时加载

    return run_ba_route(*args, **kwargs)


# 历史符号的只读别名（审计/复现脚本用；生产代码请勿 import 本模块）
def __getattr__(name: str) -> Any:  # pragma: no cover - 仅复现工具使用
    from . import route as _route

    if hasattr(_route, name):
        return getattr(_route, name)
    raise AttributeError(f"legacy_vggsfm_ba 无属性 {name!r}")


__all__ = [
    "OFFICIAL_VGGSFM_BA_ENABLED",
    "OOM_EVIDENCE_REFS",
    "REJECTION_REASON",
    "UnsupportedConfigurationError",
    "assert_official_ba_disabled",
    "run_official_ba_repro",
]


def probe_pycolmap() -> Any:  # pragma: no cover - 复现工具
    """历史探测函数（pycolmap 版本/签名）。"""
    from .route import probe_pycolmap as _p

    return _p()


def historical_route_module() -> Any:  # pragma: no cover - 复现工具
    """返回历史实现模块（只读；不得用于生产 route）。"""
    from . import route

    return route
