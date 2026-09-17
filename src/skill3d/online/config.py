"""在线链配置读取（§13 配置样例）。

§4 M21 首选 Hydra；本机环境未安装 hydra-core，故用 PyYAML 直读 `configs/*.yaml`
（键名与 §13 样例一致，接口不变，后续接入 Hydra 时替换本模块实现即可）。

仅读取，不写配置；阈值一律来自 yaml，代码内不硬编码实验阈值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG = "configs/config.yaml"

# 与 configs/config.yaml 的默认值保持一致（yaml 缺失键时的兜底）
_FALLBACK: dict[str, Any] = {
    "mode": "online",
    "seed": 0,
    "paths": {
        "data_root": "data",
        "vsi_bench_meta": "data/vsi_bench_meta",
        "raw_videos": "data/raw_videos",
        "reconstructions": "data/reconstructions",
        "trace_store": "data/traces",
        "memory_db": "data/memory_lancedb",
        "skill_registry": "data/skill_registry",
        "active_snapshot": "data/active_snapshot.json",
    },
    "frame_sampling": {"n_frames": 32, "strategy": "uniform"},
    "vllm": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "quantization": "awq",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.92,
        "tensor_parallel_size": 1,
        "dp_world_size": 8,
        "base_port": 8100,
        "temperature": 0.0,
    },
    "sandbox": {
        "image": "skill3d-sandbox:latest",
        "network": "none",
        "read_only": True,
        "cpus": 4,
        "memory": "8g",
        "pids_limit": 256,
        "cell_timeout_s": 120,
        "max_regenerate": 3,
    },
    "split_config": "configs/vsi_bench_split.yaml",
    "admission_thresholds": "configs/admission_thresholds.yaml",
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
    """读任意配置 yaml（split / admission_thresholds）。"""
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
    reconstructions: str = "data/reconstructions"
    trace_store: str = "data/traces"
    memory_db: str = "data/memory_lancedb"
    skill_registry: str = "data/skill_registry"
    active_snapshot: str = "data/active_snapshot.json"


@dataclass
class SandboxSettings:
    """§13.2 沙箱参数。"""

    image: str = "skill3d-sandbox:latest"
    network: str = "none"
    read_only: bool = True
    cpus: int = 4
    memory: str = "8g"
    pids_limit: int = 256
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
