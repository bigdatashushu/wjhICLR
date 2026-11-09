"""Freeze local overrides and shared dependencies without copying other runners."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LAYOUT = {"schema": "vsi-qwen-flash-source-layout/1", "mode": "frozen-merged"}
SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".sh", ".j2", ".jinja"}
EXCLUDED_DIRS = {
    "__pycache__", ".git", ".venv", "venv", ".cache", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "runs", "prepared_cache", "logs", "weights", "checkpoints",
}
TOP_LEVEL = {"run.py", "experiment.sh", "README.md", "AGENTS.md", ".gitignore", "migration.json"}


class SourceError(RuntimeError):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resource_root(root=ROOT):
    """Seed/config resources live in the shared repo until a run freezes them."""
    root = Path(root).resolve()
    marker = root / "source_layout.json"
    if marker.is_file():
        if json.loads(marker.read_text()) != LAYOUT:
            raise SourceError("Invalid frozen source layout")
        return root
    return root.parent


def _allowed(path, base):
    relative = path.relative_to(base)
    return (not any(part in EXCLUDED_DIRS for part in relative.parts)
            and not path.name.startswith(".env")
            and ".local." not in path.name
            and (path.suffix in SUFFIXES or path.name == ".gitkeep"
                 or relative.as_posix() in TOP_LEVEL))


def source_inventory(root=ROOT):
    """Explicit source trees only; no Git-wide scan or credential-file reads."""
    root = Path(root).resolve()
    shared = resource_root(root)
    sources = {}

    def add(path, relative, owner):
        if path.is_symlink() or not path.is_file():
            raise SourceError("Source entries must be regular files")
        sources[relative] = {
            "source": path, "owner": owner, "size": path.stat().st_size,
            "sha256": sha256(path), "executable": bool(path.stat().st_mode & 0o111),
        }

    def tree(base, prefix, owner):
        if not base.is_dir():
            raise SourceError(f"Required source tree is missing: {prefix}")
        for path in sorted(base.rglob("*")):
            if path.is_file() and _allowed(path, base):
                add(path, (Path(prefix) / path.relative_to(base)).as_posix(), owner)

    tree(shared / "src/skill3d", "src/skill3d", "shared")
    tree(shared / "skill_library", "skill_library", "shared")
    for name in ("configs/config.yaml", "configs/vsi_bench_split.yaml", "pyproject.toml"):
        add(shared / name, name, "shared")
    # Retain the small vendored implementation, without its environments/weights.
    if (shared / "third_party/vggt").is_dir():
        tree(shared / "third_party/vggt", "third_party/vggt", "shared")
    for name in ("src", "scripts", "configs"):
        tree(root / name, name, "local")
    for name in sorted(TOP_LEVEL):
        if (root / name).is_file():
            add(root / name, name, "local")
    if (root / "source_layout.json").is_file():
        add(root / "source_layout.json", "source_layout.json", "local")
    return sources


def inventory_rows(inventory):
    return [{"path": name, "size": row["size"], "sha256": row["sha256"],
             "executable": row["executable"]} for name, row in sorted(inventory.items())]


def source_report(root=ROOT):
    root = Path(root).resolve()
    inventory = source_inventory(root)
    rows = inventory_rows(inventory)
    modules = {}
    for name, row in sorted(inventory.items()):
        if row["owner"] != "local" or not name.startswith("src/skill3d/") or not name.endswith(".py"):
            continue
        if name.endswith("/__init__.py"):
            continue
        module_name = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
        spec = importlib.util.find_spec(module_name)
        if spec is None or Path(spec.origin).resolve() != row["source"].resolve():
            raise SourceError("A local VSI module resolved to another implementation")
        modules[module_name] = str(row["source"])
    return {
        "runner_root": str(root), "shared_resource_root": str(resource_root(root)),
        "source_sha256": hashlib.sha256(encoded(rows)).hexdigest(),
        "local_files": sum(row["owner"] == "local" for row in inventory.values()),
        "shared_files": sum(row["owner"] == "shared" for row in inventory.values()),
        "local_module_paths": modules, "model_requests": 0,
    }


def _git_identity(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
        if result.returncode:
            raise SourceError("Source Git provenance is unavailable")
        return result.stdout.decode(errors="surrogateescape").strip()
    commit = git("rev-parse", "HEAD")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise SourceError("Invalid source base commit")
    return commit, bool(git("status", "--porcelain=v1"))


def development_identity(root=ROOT):
    root = Path(root).resolve()
    commit, dirty = _git_identity(resource_root(root))
    rows = inventory_rows(source_inventory(root))
    return {
        "run_mode": "development", "base_commit": commit, "git_dirty": dirty,
        "source_root": str(root), "source_sha256": hashlib.sha256(encoded(rows)).hexdigest(),
    }


def freeze_source(root, destination):
    """Materialize one merged, hash-validated source tree; preserve truthful Git provenance."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    shared = resource_root(root)
    if destination.exists():
        raise SourceError("Source snapshot already exists")
    if destination == root or root.is_relative_to(destination):
        raise SourceError("Source snapshot must not contain the runner")
    inventory = source_inventory(root)
    commit, dirty = _git_identity(shared)
    destination.mkdir(parents=True)
    snapshot = destination / "source"
    snapshot.mkdir()
    rows = []
    try:
        for row in inventory_rows(inventory):
            original = inventory[row["path"]]["source"]
            target = snapshot / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            if target.stat().st_size != row["size"] or sha256(target) != row["sha256"]:
                raise SourceError("Source changed while copying")
            target.chmod(0o555 if row["executable"] else 0o444)
            rows.append(row)
        marker = snapshot / "source_layout.json"
        marker.write_bytes(encoded(LAYOUT))
        marker.chmod(0o444)
        rows = [row for row in rows if row["path"] != "source_layout.json"]
        rows.append({"path": "source_layout.json", "size": marker.stat().st_size,
                     "sha256": sha256(marker), "executable": False})
        rows.sort(key=lambda row: row["path"])
        if (inventory_rows(source_inventory(root)) != inventory_rows(inventory)
                or _git_identity(shared)[0] != commit):
            raise SourceError("Source changed while freezing")
        receipt = {
            "schema_version": "experiment-source/1.0", "run_mode": "formal",
            "base_commit": commit, "git_dirty": dirty,
            "original_source": str(root), "source_root": str(snapshot),
            "snapshot_policy": "vsi-local-overlay/1",
            "files": rows, "source_sha256": hashlib.sha256(encoded(rows)).hexdigest(),
            "input_files": [],
            "file_origins": {name: row["owner"] for name, row in sorted(inventory.items())
                             if name != "source_layout.json"},
        }
        (destination / "code_identity.json").write_bytes(encoded(receipt))
        return receipt
    except BaseException:
        shutil.rmtree(destination)
        raise
