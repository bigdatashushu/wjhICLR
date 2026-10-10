"""Read the frozen v11 YAML runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG = "configs/config.yaml"

# 与 configs/config.yaml 的默认值保持一致（yaml 缺失键时的兜底）
_FALLBACK: dict[str, Any] = {
    "seed": 0,
    "max_solver_rounds": 6,
    "max_retries_per_operation": 3,
    "finalization_rounds": 1,
    "paths": {
        "data_root": "data",
        "vsi_bench_meta": "data/vsi_bench_meta",
        "raw_videos": "data/raw_videos",
        "raw_video_fallbacks": [],
        "reconstructions": "data/reconstructions",
        "trace_store": "data/traces",
        "skill_library": "skill_library",
        "active_snapshot": "skill_library/snapshots/active_snapshot.json",
    },
    "frame_sampling": {"n_frames": 32, "strategy": "uniform"},
    # v11 检索策略缺省（与 configs/config.yaml 的 `retrieval:` 段一致）。
    # 缺键时用默认值跑，但检索记录里的 `config_source` 会记 "default" ——
    # 缺省值不会被冒充成"已冻结配置"。
    "retrieval": {
        "config_version": "ret-v11-deterministic-1",
        "method_context_max_chars": 8000,
    },
    "vllm": {
        "model": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "quantization": "none",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.90,
        "n_frames": 32,
        "max_pixels": 131072,
        "tensor_parallel_size": 1,
        "dp_world_size": 8,
        "base_port": 8100,
        "temperature": 0.0,
    },
    "sandbox": {
        "cell_timeout_s": 120,
        "max_regenerate": 3,
    },
    # v9 §9.4 主动图像布局（缺键时的默认，与 configs/config.yaml 一致）
    "active_vision": {"layout": "derived_plus_originals_v1", "max_derived_images": 8},
    "split_config": "configs/vsi_bench_split.yaml",
    # v6 §20：v4/v5 的 `scale` 段（冻结 conformal 校准器 / 标定池 / 逐题型授权校准）
    # 随"放弃一切需要校准的路线"整体废止，键已删除；米制尺度改由
    # `reconstruction/metric_fusion.py`（零样本度量深度跨帧融合，§11）在推理期产出，
    # 其模型卡/阈值属 [待实验]/[TODO_CALIBRATE]，PoC 通过后再登记新键。
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str | Path] = None) -> dict:
    """读主配置并与默认值合并；文件不存在时回退默认值（不报错，便于 CLI 先跑）。"""
    cfg_path = Path(path) if path is not None else Path(DEFAULT_CONFIG)
    if not cfg_path.is_file():
        return dict(_FALLBACK)
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"配置格式错误（应为 mapping）: {cfg_path}")
    return _deep_merge(_FALLBACK, loaded)


def load_yaml(path: str | Path) -> dict:
    """读任意配置 yaml（split）。"""
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置格式错误（应为 mapping）: {p}")
    return data


@dataclass
class VLLMSettings:
    """§4 M8：本地 vLLM（DP×8，不 TP）。endpoints 为空 = 未配置。"""

    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    quantization: str = "awq"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.92
    tensor_parallel_size: int = 1
    dp_world_size: int = 8
    base_port: int = 8100
    temperature: float = 0.0
    # 32 帧输入与单帧像素上限（§1.2 / §4 M1；实测 32 帧 @131072 → ~9703 prompt tokens）
    n_frames: int = 32
    max_pixels: int = 131072
    endpoints: list[str] = field(default_factory=list)

    @property
    def configured(self) -> bool:
        return bool(self.endpoints)


@dataclass
class PathSettings:
    """§2 数据/产物路径。"""

    data_root: str = "data"
    vsi_bench_meta: str = "data/vsi_bench_meta"
    raw_videos: str = "data/raw_videos"
    # §5.2"可重试加载"的来源副本根（用户 2026-09-27 裁定：重试 = 换来源/换副本）。
    # 缺省空 = 只尝试主目录里的同名副本，不隐式启用任何镜像。
    raw_video_fallbacks: list[str] = field(default_factory=list)
    reconstructions: str = "data/reconstructions"
    trace_store: str = "data/traces"
    skill_library: str = "skill_library"
    active_snapshot: str = "skill_library/snapshots/active_snapshot.json"


@dataclass
class SandboxSettings:
    """§13.2 沙箱参数。"""

    cell_timeout_s: int = 120
    max_regenerate: int = 3


def paths_from(cfg: dict) -> PathSettings:
    return PathSettings(**(cfg.get("paths") or {}))


def vllm_from(cfg: dict) -> VLLMSettings:
    data = dict(cfg.get("vllm") or {})
    data.pop("endpoints", None)
    return VLLMSettings(**data)


def sandbox_from(cfg: dict) -> SandboxSettings:
    return SandboxSettings(**(cfg.get("sandbox") or {}))


@dataclass
class ActiveVisionSettings:
    """§9.4 主动图像的声明布局（图像上限内怎么装图）。"""

    layout: str = "derived_plus_originals_v1"
    max_derived_images: int = 8


def active_vision_from(cfg: dict) -> ActiveVisionSettings:
    data = dict((cfg or {}).get("active_vision") or {})
    return ActiveVisionSettings(
        layout=str(data.get("layout", "derived_plus_originals_v1")),
        max_derived_images=int(data.get("max_derived_images", 8)))


def retrieval_policy_from(cfg: dict):
    """读取 v11 固定查找合同与完整正文长度上限。

    唯一读取点是这里 —— 检索策略不得在运行期按题/按场景临时构造。
    """
    from skill3d.routing.retrieval_policy import retrieval_policy_from_config

    return retrieval_policy_from_config(cfg)
