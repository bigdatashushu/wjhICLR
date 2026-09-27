"""M21 版本锁定：pip freeze hash + docker digest + checkpoint sha256 + git HEAD → RunManifest。

subprocess 真实实现；单项失败记 "unknown"（不阻断）。
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from skill3d.schemas import RunManifest


def _run_cmd(cmd: list[str], cwd: str | None = None, timeout: int = 60) -> str:
    """执行命令取 stdout；失败返回 "unknown"。"""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             timeout=timeout, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def pip_freeze_hash() -> str:
    """pip freeze 内容 sha256。"""
    out = _run_cmd(["pip", "freeze"])
    if out == "unknown":
        return "unknown"
    return hashlib.sha256(out.encode("utf-8")).hexdigest()


def docker_digest(image: str) -> str:
    """docker image inspect 取 digest；docker 不可用 → "unknown"。"""
    return _run_cmd(["docker", "image", "inspect", "--format", "{{.Id}}", image])


def checkpoint_sha256(path: str) -> str:
    """checkpoint 文件 sha256；文件不存在 → "unknown"。"""
    p = Path(path)
    if not p.is_file():
        return "unknown"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo_dir: str) -> str:
    return _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir)


def config_hash(config_path: str) -> str:
    p = Path(config_path)
    if not p.is_file():
        return "unknown"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inference_env_versions() -> dict:
    """推理环境版本记录（§16.4：所有主结果必须记录推理环境）。

    取 vLLM / torch / transformers 版本；未安装记 "absent"（不虚构版本号）。
    v9 §17.1：numpy/scipy 同属"环境依赖" —— 它们的版本组合决定 M4 主门与距离
    原语是否真的可用，只记模型框架会漏掉这一层。
    """
    import importlib

    out: dict = {}
    for mod in ("vllm", "torch", "transformers", "sam2", "pycolmap"):
        try:
            m = importlib.import_module(mod)
            out[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - 未安装 → 显式 absent
            out[mod] = "absent"
    try:
        from skill3d.env_preflight import describe_environment
    except Exception:  # noqa: BLE001 - 环境模块不可用不阻断 manifest 落盘
        return out
    for k, v in describe_environment().items():
        out.setdefault(k, v)
    return out


def build_run_manifest(docker_image: str, checkpoint_path: str,
                       config_path: str, repo_dir: str,
                       mlflow_run_id: str = "unknown",
                       *, split_version: str = "", seed: int | None = None,
                       split_config_path: str = "") -> RunManifest:
    manifest = RunManifest(
        code_commit=git_head(repo_dir),
        docker_digest=docker_digest(docker_image),
        checkpoint_sha256=checkpoint_sha256(checkpoint_path),
        pip_freeze_hash=pip_freeze_hash(),
        config_hash=config_hash(config_path),
        mlflow_run_id=mlflow_run_id,
    )
    # §16.4：split version / split 配置哈希 / seed / 推理环境版本一并落盘
    return manifest.model_copy(update={
        "split_version": split_version,
        "split_config_hash": (config_hash(split_config_path)
                              if split_config_path else ""),
        "seed": seed,
        "inference_env": inference_env_versions(),
    })


def write_run_manifest(manifest: RunManifest, out_path: str | Path,
                       extra: dict | None = None) -> Path:
    """把 RunManifest 落盘 JSON（§16.4：论文 artifact 审计用）。"""
    import json

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.model_dump()
    if extra:
        data.update(extra)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
