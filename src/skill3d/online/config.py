"""Read the frozen v11 YAML runtime configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import math
import os
from pathlib import Path
import re
from typing import Any, Optional
from urllib.parse import urlsplit

import yaml

DEFAULT_CONFIG = "configs/config.yaml"
SOURCE_ROOT = Path(__file__).resolve().parents[3]
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class RuntimeConfig(dict):
    """Effective values with audit metadata kept outside the YAML fields."""

    def __init__(self, values: dict, *, path: Path, sources: dict):
        super().__init__(values)
        self.path = str(path)
        self.sources = sources


def resolve_path(path: str | Path) -> Path:
    """M1: repository resources have one root, independent of the caller's cwd."""
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (SOURCE_ROOT / value).resolve()


def expand_values(value: Any, _chain: tuple[str, ...] = ()) -> Any:
    """Expand parsed string values; environment text can never become YAML syntax."""
    if isinstance(value, dict):
        return {key: expand_values(item, _chain) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_values(item, _chain) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.groups()
        if name in _chain:
            raise ValueError(f"Cyclic configuration variable: {' -> '.join((*_chain, name))}")
        if name in os.environ:
            replacement = os.environ[name]
            # Match the existing shell ${VAR:-default} convention for empty values.
            if replacement == "" and default is not None:
                replacement = default
        elif default is not None:
            replacement = default
        else:
            raise ValueError(f"Missing configuration variable: {name}")
        return expand_values(replacement, (*_chain, name))

    result = _PLACEHOLDER.sub(replace, value)
    if "${" in result:
        raise ValueError("Unresolved or malformed configuration placeholder")
    return result

# 与 configs/config.yaml 的默认值保持一致（yaml 缺失键时的兜底）
_FALLBACK: dict[str, Any] = {
    "seed": 0,
    "max_solver_rounds": 6,
    "max_retries_per_operation": 3,
    "finalization_rounds": 1,
    "eval_visual_fallback": True,
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
        "max_tokens": 8192,
        "endpoints": [],
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
    """Read and validate before sampling; missing explicit inputs never fall back."""
    cfg_path = resolve_path(path if path is not None else DEFAULT_CONFIG)
    loaded = load_yaml(cfg_path)
    allowed = set(_FALLBACK) | {"input_diagnostics", "detection"}
    unknown = set(loaded) - allowed
    if unknown:
        raise ValueError(f"Unknown runtime configuration fields: {sorted(unknown)}")
    cfg = RuntimeConfig(_deep_merge(deepcopy(_FALLBACK), loaded),
                        path=cfg_path, sources={})
    for section in ("paths", "frame_sampling", "vllm", "sandbox", "active_vision", "retrieval"):
        if not isinstance(cfg[section], dict):
            raise ValueError(f"{section} must be a mapping")
    for key, value in cfg.items():
        if isinstance(value, dict):
            for name in value:
                cfg.sources[f"{key}.{name}"] = (
                    "yaml" if name in loaded.get(key, {}) else "default")
        else:
            cfg.sources[key] = "yaml" if key in loaded else "default"
    _integer(cfg["seed"], "seed", minimum=0)
    for key in ("max_solver_rounds", "max_retries_per_operation", "finalization_rounds"):
        _integer(cfg[key], key, minimum=0 if key == "finalization_rounds" else 1)
    if not isinstance(cfg["eval_visual_fallback"], bool):
        raise ValueError("eval_visual_fallback must be boolean")
    cfg["paths"] = asdict(paths_from(cfg))
    if not isinstance(cfg["split_config"], str) or not cfg["split_config"]:
        raise ValueError("split_config must be a non-empty path")
    cfg["split_config"] = str(resolve_path(cfg["split_config"]))
    cfg["vllm"] = asdict(vllm_from(cfg))
    sandbox_from(cfg)
    active_vision_from(cfg)
    resolve_runtime(cfg)
    return cfg


def load_yaml(path: str | Path) -> dict:
    """Shared strict parsed-value reader for runtime, split and GPU configurations."""
    p = resolve_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置格式错误（应为 mapping）: {p}")
    return expand_values(data)


def resolve_detector_endpoint(experiment: dict | None = None, *, environ=None,
                              require_detector: bool = False) -> dict:
    """Resolve once: explicit launcher environment > original YAML > default."""
    env = os.environ if environ is None else environ
    experiment = experiment or {}
    if "HARNESS3D_DETECTOR_ENDPOINT" in env:
        value, source = env["HARNESS3D_DETECTOR_ENDPOINT"], "env"
    elif "detector_endpoint" in experiment:
        value = experiment["detector_endpoint"]
        source = experiment.get("detector_endpoint_source", "yaml")
    else:
        value, source = "http://127.0.0.1:20022", "default"
    if not isinstance(value, str):
        raise ValueError("experiment.detector_endpoint must be a string; empty disables it")
    endpoint = value.strip()
    if endpoint:
        try:
            parsed = urlsplit(endpoint)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path not in ("", "/") or parsed.port is None
                    or not 1 <= parsed.port <= 65535):
                raise ValueError("expected http(s)://host:port")
        except ValueError as exc:
            raise ValueError(f"Invalid detector endpoint {endpoint!r}: {exc}") from exc
    elif require_detector:
        raise ValueError("Detector is explicitly disabled but REQUIRE_DETECTOR=true")
    if source not in ("env", "yaml", "default"):
        raise ValueError("Invalid detector endpoint source")
    return {"endpoint": endpoint, "source": source, "enabled": bool(endpoint)}


def _integer(value: Any, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _settings(kind, data: Any, section: str):
    if not isinstance(data, dict):
        raise ValueError(f"{section} must be a mapping")
    unknown = set(data) - {item.name for item in fields(kind)}
    if unknown:
        raise ValueError(f"Unknown {section} fields: {sorted(unknown)}")
    return kind(**data)


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
    max_tokens: int = 8192
    # 32 帧输入与单帧像素上限（§1.2 / §4 M1；实测 32 帧 @131072 → ~9703 prompt tokens）
    n_frames: int = 32
    max_pixels: int = 131072
    endpoints: list[str] = field(default_factory=list)

    def __post_init__(self):
        for name in ("max_model_len", "max_tokens", "n_frames", "max_pixels",
                     "tensor_parallel_size", "dp_world_size", "base_port"):
            _integer(getattr(self, name), f"vllm.{name}")
        if self.base_port > 65535:
            raise ValueError("vllm.base_port must be <= 65535")
        if self.max_tokens >= self.max_model_len:
            raise ValueError("vllm.max_tokens must leave a positive input context budget")
        for name, low, high in (("temperature", 0, 2), ("gpu_memory_utilization", 0, 1)):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not low <= value <= high
                    or (name == "gpu_memory_utilization" and value == 0)):
                raise ValueError(f"vllm.{name} is outside its valid range")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("vllm.model must be non-empty")
        if not isinstance(self.quantization, str) or not self.quantization:
            raise ValueError("vllm.quantization must be non-empty")
        if not isinstance(self.endpoints, list):
            raise ValueError("vllm.endpoints must be a list of URLs")
        for endpoint in self.endpoints:
            if not isinstance(endpoint, str):
                raise ValueError("vllm.endpoints must contain URLs")
            parsed = urlsplit(endpoint)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(f"Invalid vLLM endpoint: {endpoint}")
            if parsed.username or parsed.password:
                raise ValueError("vLLM endpoint credentials must not be embedded in URLs")
            try:
                parsed.port
            except ValueError as exc:
                raise ValueError("Invalid vLLM endpoint port") from exc

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
    settings = _settings(PathSettings, cfg.get("paths", {}), "paths")
    for item in fields(settings):
        value = getattr(settings, item.name)
        if item.name == "raw_video_fallbacks":
            if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
                raise ValueError("paths.raw_video_fallbacks must be a list of paths")
            setattr(settings, item.name, [str(resolve_path(v)) for v in value])
        else:
            if not isinstance(value, str) or not value:
                raise ValueError(f"paths.{item.name} must be a non-empty path")
            setattr(settings, item.name, str(resolve_path(value)))
    return settings


def vllm_from(cfg: dict) -> VLLMSettings:
    return _settings(VLLMSettings, cfg.get("vllm", {}), "vllm")


def resolve_runtime(cfg: dict, *, seed=None, n_frames=None, endpoints=None,
                    model=None, formal: bool = False) -> dict:
    """M1 precedence: explicit CLI overrides YAML/default; preserve 0 and []."""
    sources = getattr(cfg, "sources", {})
    if seed is not None:
        cfg["seed"] = seed
        sources["seed"] = "cli"
    _integer(cfg["seed"], "seed", minimum=0)
    sampling = cfg["frame_sampling"]
    if set(sampling) - {"n_frames", "strategy"}:
        raise ValueError("Unknown frame_sampling fields")
    if n_frames is not None:
        sampling["n_frames"] = n_frames
        sources["frame_sampling.n_frames"] = "cli"
    _integer(sampling["n_frames"], "frame_sampling.n_frames")
    if sampling.get("strategy", "uniform") != "uniform":
        raise ValueError("frame_sampling.strategy must be uniform")
    if endpoints is not None:
        cfg["vllm"]["endpoints"] = endpoints
        sources["vllm.endpoints"] = "cli"
    if model is not None:
        cfg["vllm"]["model"] = model
        sources["vllm.model"] = "cli"
    settings = vllm_from(cfg)
    if sampling["n_frames"] > settings.n_frames:
        raise ValueError("Input frame count exceeds the configured model image capacity")
    if formal and (sampling["n_frames"] != 32 or settings.n_frames < 32):
        raise ValueError("Formal experiments require the frozen 32-frame input contract")
    return cfg


def config_receipt(cfg: dict) -> dict:
    """Retain provenance without serializing secret-bearing extension fields."""
    def public(value):
        if isinstance(value, dict):
            return {key: ("<redacted>" if any(word in str(key).lower()
                        for word in ("api_key", "password", "secret", "authorization"))
                          else public(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [public(item) for item in value]
        return value

    return {"path": getattr(cfg, "path", ""), "effective": public(dict(cfg)),
            "sources": dict(getattr(cfg, "sources", {}))}


def sandbox_from(cfg: dict) -> SandboxSettings:
    settings = _settings(SandboxSettings, cfg.get("sandbox", {}), "sandbox")
    _integer(settings.cell_timeout_s, "sandbox.cell_timeout_s")
    _integer(settings.max_regenerate, "sandbox.max_regenerate", minimum=0)
    return settings


@dataclass
class ActiveVisionSettings:
    """§9.4 主动图像的声明布局（图像上限内怎么装图）。"""

    layout: str = "derived_plus_originals_v1"
    max_derived_images: int = 8


def active_vision_from(cfg: dict) -> ActiveVisionSettings:
    settings = _settings(ActiveVisionSettings, cfg.get("active_vision", {}), "active_vision")
    _integer(settings.max_derived_images, "active_vision.max_derived_images", minimum=0)
    if settings.layout != "derived_plus_originals_v1":
        raise ValueError("Unsupported active_vision.layout")
    return settings


def retrieval_policy_from(cfg: dict):
    """读取 v11 固定查找合同与完整正文长度上限。

    唯一读取点是这里 —— 检索策略不得在运行期按题/按场景临时构造。
    """
    from skill3d.routing.retrieval_policy import retrieval_policy_from_config

    return retrieval_policy_from_config(cfg)
