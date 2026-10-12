"""Shared launch modes, frozen working-tree sources, and environment identities."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RECEIPT_ENV = "SKILL3D_CODE_RECEIPT"
MODE_ENV = "SKILL3D_RUN_MODE"
_EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".cache",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist",
    "checkpoints", "weights", "logs", "wandb",
}
_EXCLUDED_ROOTS = {
    "data", "dataset", "datasets", "results", "output", "outputs", "runs",
    "source_snapshots", "launcher_configs", "launcher_inputs", "launcher_logs", "environment",
}
_EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".log", ".npy", ".npz", ".pt", ".pth", ".bin",
    ".safetensors", ".ckpt", ".mp4", ".mov", ".avi", ".mkv", ".env",
}


class StartupError(RuntimeError):
    """Launch inputs cannot be frozen or verified."""


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise StartupError(result.stderr.decode(errors="replace").strip())
    return result.stdout.decode("utf-8", errors="surrogateescape").strip()


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_mode(cfg: dict, *, dev=False, smoke=False, dry_run=False) -> str:
    return ("development" if dev or smoke or dry_run
            or not cfg["experiment"].get("formal", False) else "formal")


def validate_existing_environment(root: Path, cfg: dict) -> dict:
    """Validate a user-selected interpreter without installing into it."""
    from importlib.metadata import distributions, version
    from packaging.requirements import Requirement

    expected_python = str(cfg["environment"]["python"])
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if expected_python != actual_python:
        raise StartupError(f"Existing Python {actual_python} != required {expected_python}")
    declarations = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    pinned_exceptions = cfg["environment"].get("existing_dependency_versions", {})
    exceptions = {}
    for text in declarations:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        installed = version(requirement.name)
        if requirement.name in pinned_exceptions and pinned_exceptions[requirement.name] != installed:
            raise StartupError(f"Existing {requirement.name} does not match its explicit version pin")
        if requirement.specifier and installed not in requirement.specifier:
            if pinned_exceptions.get(requirement.name) != installed:
                raise StartupError(f"Existing {requirement.name}=={installed} does not satisfy {text}")
            exceptions[requirement.name] = {"installed": installed, "declared": text}
    if version("vllm") != str(cfg["environment"]["vllm_version"]):
        raise StartupError("Existing vLLM version does not match configured version")
    import importlib
    import inspect

    modules = ["sam2"]
    if cfg["weights"]["moge2"].get("enabled"):
        modules.append("moge.model.v2")
    sources = {}
    for name in modules:
        module = importlib.import_module(name)
        source = Path(inspect.getfile(module)).resolve()
        sources[name] = {"path": str(source), "sha256": sha256(source)}
    return {
        "mode": "validated_existing", "python": sys.executable,
        "python_version": sys.version, "contract": environment_contract(root, cfg),
        "declared_dependency_exceptions": exceptions,
        "packages": sorted((dist.metadata["Name"], dist.version) for dist in distributions()
                           if dist.metadata["Name"]),
        "import_sources": sources,
    }


def configure_run(cfg: dict, *, dev=False, smoke=False, dry_run=False) -> dict:
    value = copy.deepcopy(cfg)
    experiment = value["experiment"]
    mode = run_mode(value, dev=dev, smoke=smoke, dry_run=dry_run)
    experiment.update(run_mode=mode, formal=mode == "formal")
    if mode == "development":
        # A previous commit's confirmation must not be presented as development
        # source approval. Runtime geometry gates still compute normally.
        experiment["quality_confirmation"] = ""
    if smoke:
        experiment.update(induction_limit=2, inner_limit=2, post_publish_limit=1,
                          reconstruction_sampling_per_task=4, run_evolution=False)
        if experiment.get("metric_validation"):
            # This route validates one frozen scene before loading the solver.
            experiment["metric_validation"]["prepare_only"] = True
            experiment["metric_validation"]["first_scene_only"] = True
    return value


def _included(relative: str) -> bool:
    path = Path(relative)
    return (not any(part in _EXCLUDED_DIRS for part in path.parts)
            and path.parts[0] not in _EXCLUDED_ROOTS
            and path.suffix not in _EXCLUDED_SUFFIXES
            and not path.name.endswith(".local.env"))


def _source_paths(root: Path) -> tuple[list[str], list[str]]:
    def listed(*args):
        output = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", *args])
        return [os.fsdecode(name) for name in output.split(b"\0") if name]
    original = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "-z", "--name-only", "HEAD"])
    original_names = [os.fsdecode(name) for name in original.split(b"\0") if name]
    tracked = [name for name in original_names + listed("--cached") if _included(name)]
    untracked = [name for name in listed("--others", "--exclude-standard") if _included(name)]
    return sorted(set(tracked)), sorted(set(untracked))


def freeze_working_tree(root: Path, destination: Path) -> dict:
    """Copy current tracked/untracked code; never copy Git, environments or weights."""
    root, destination = root.resolve(), destination.resolve()
    if destination == root or root.is_relative_to(destination):
        raise StartupError("Source snapshot must not contain the working repository")
    if destination.exists():
        raise StartupError(f"Source snapshot already exists: {destination}")
    commit = git(root, "rev-parse", "HEAD")
    if not COMMIT_RE.fullmatch(commit):
        raise StartupError("Working source must have a Git commit")
    status = git(root, "status", "--porcelain=v1")
    tracked, untracked = _source_paths(root)
    names = sorted(set(tracked + untracked))
    destination.mkdir(parents=True)
    snapshot = destination / "source"
    snapshot.mkdir()
    files = []
    try:
        for name in names:
            source = root / name
            if not source.exists():
                continue  # Preserve a working-tree deletion by leaving it absent.
            if not source.is_file():
                raise StartupError(f"Unsupported source entry: {source}")
            # Materialize internal links; never preserve a link to mutable source.
            if not source.resolve().is_relative_to(root):
                raise StartupError(f"Source link points outside the repository: {source}")
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            executable = bool(source.stat().st_mode & 0o111)
            target.chmod(0o555 if executable else 0o444)
            files.append({"path": name, "size": target.stat().st_size,
                          "sha256": sha256(target), "executable": executable})
        if not (snapshot / "scripts/run_gpu_experiment_v11.py").is_file():
            raise StartupError("Working source is missing the GPU experiment runner")
        # Preserve the ignored offline metadata as an input, separately from code.
        input_files = []
        metadata = root / "data/vsi_bench_meta/test.jsonl"
        if metadata.is_file():
            target = snapshot / "data/vsi_bench_meta/test.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(metadata, target)
            input_files.append({"path": target.relative_to(snapshot).as_posix(),
                                "sha256": sha256(target), "size": target.stat().st_size})
        diff = subprocess.check_output(
            ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", *tracked])
        (destination / "working_tree.diff").write_bytes(diff)
        (destination / "git_status.txt").write_text(status + "\n")
        if (git(root, "rev-parse", "HEAD") != commit
                or git(root, "status", "--porcelain=v1") != status
                or _source_paths(root) != (tracked, untracked)
                or any(sha256(root / row["path"]) != row["sha256"] for row in files)):
            raise StartupError("Working source changed while creating its snapshot; retry launch")
        receipt = {
            "schema_version": "experiment-source/1.0", "run_mode": "development",
            "base_commit": commit, "git_dirty": bool(status),
            "original_source": str(root), "source_root": str(snapshot),
            "files": files, "source_sha256": hashlib.sha256(encoded(files)).hexdigest(),
            "untracked_source_files": untracked,
            "deleted_source_files": [name for name in tracked if not (root / name).exists()],
            "diff_sha256": hashlib.sha256(diff).hexdigest(), "input_files": input_files,
        }
        write_json(destination / "code_identity.json", receipt)
        return receipt
    except Exception:
        shutil.rmtree(destination)
        raise


def verify_source_receipt(path: Path, root: Path) -> dict:
    receipt = json.loads(path.read_text())
    if (receipt.get("schema_version") != "experiment-source/1.0"
            or Path(receipt["source_root"]).resolve() != root.resolve()
            or not COMMIT_RE.fullmatch(receipt.get("base_commit", ""))):
        raise StartupError("Code identity does not describe the executing source")
    if hashlib.sha256(encoded(receipt["files"])).hexdigest() != receipt["source_sha256"]:
        raise StartupError("Source file manifest identity changed")
    for row in receipt["files"] + receipt.get("input_files", []):
        relative = Path(row["path"])
        target = root / relative
        if relative.is_absolute() or ".." in relative.parts or target.is_symlink():
            raise StartupError(f"Invalid snapshot reference: {relative}")
        if (not target.is_file() or target.stat().st_size != row["size"]
                or sha256(target) != row["sha256"]):
            raise StartupError(f"Frozen source changed: {relative}")
    expected = {row["path"] for row in receipt["files"]}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and _included(p.relative_to(root).as_posix())}
    if actual != expected:
        raise StartupError("Frozen source contains missing or additional code files")
    return receipt


def current_code_identity(root: Path | None = None) -> dict:
    value = os.environ.get(RECEIPT_ENV)
    if not value:
        return {}
    path = Path(value)
    receipt = json.loads(path.read_text())
    if root is not None and Path(receipt["source_root"]).resolve() != root.resolve():
        return {}
    return {key: receipt[key] for key in (
        "run_mode", "base_commit", "git_dirty", "source_sha256", "source_root")}


def environment_contract(root: Path, cfg: dict, python_version: str | None = None) -> dict:
    """Exclude commits, prompt edits, budgets and resource thresholds from installs."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    environment = cfg["environment"]
    value = {
        "policy_version": "gpu-environment/2",
        "python_version": python_version or ".".join(map(str, sys.version_info[:2])),
        "configured_python": str(environment.get("python", "")),
        "existing_dependency_versions": environment.get("existing_dependency_versions", {}),
        "project_dependencies": pyproject["project"]["dependencies"],
        "requires_python": pyproject["project"].get("requires-python"),
        "build_system": pyproject["build-system"],
        "vllm_version": str(environment["vllm_version"]),
        "sam2_code_revision": str(environment["sam2_code_revision"]),
        "moge_enabled": bool(cfg["weights"]["moge2"].get("enabled")),
        "moge_code_revision": (str(environment["moge_code_revision"])
                               if cfg["weights"]["moge2"].get("enabled") else None),
        "additional_runtime_dependencies": ["Pillow", "einops", "safetensors", "huggingface_hub"],
        "dependency_files": {
            p.relative_to(root).as_posix(): sha256(p)
            for name in ("requirements*.txt", "constraints*.txt", "uv.lock", "poetry.lock")
            for p in root.glob(name) if p.is_file()},
    }
    return {**value, "sha256": hashlib.sha256(encoded(value)).hexdigest()}
