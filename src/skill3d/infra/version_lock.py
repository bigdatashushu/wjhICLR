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


def build_run_manifest(docker_image: str, checkpoint_path: str,
                       config_path: str, repo_dir: str,
                       mlflow_run_id: str = "unknown") -> RunManifest:
    return RunManifest(
        code_commit=git_head(repo_dir),
        docker_digest=docker_digest(docker_image),
        checkpoint_sha256=checkpoint_sha256(checkpoint_path),
        pip_freeze_hash=pip_freeze_hash(),
        config_hash=config_hash(config_path),
        mlflow_run_id=mlflow_run_id,
    )
