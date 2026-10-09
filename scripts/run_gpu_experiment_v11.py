#!/usr/bin/env python3
"""Download pinned weights and run the real v11 GPU experiment pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import yaml

_COMMIT_RE = __import__("re").compile(r"^[0-9a-f]{40}$")


class GPUExperimentError(RuntimeError):
    """A prerequisite or subprocess failed; no result may be claimed."""


def _read_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise GPUExperimentError("GPU experiment config must be a YAML mapping")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise GPUExperimentError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _gpu_inventory() -> list[dict]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise GPUExperimentError("nvidia-smi not found")
    result = subprocess.run(
        [
            binary,
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise GPUExperimentError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        index, name, memory, driver = [part.strip() for part in line.split(",", 3)]
        rows.append({
            "index": int(index),
            "name": name,
            "memory_total_mib": int(memory),
            "driver_version": driver,
        })
    return rows


def _preflight(
    root: Path,
    work_dir: Path,
    cfg: dict,
    *,
    dry_run: bool,
) -> dict:
    environment = cfg["environment"]
    experiment = cfg["experiment"]
    if platform.system() != "Linux" and not dry_run:
        raise GPUExperimentError("GPU experiment launcher requires Linux")
    if any("final_test" in str(value) for value in experiment.values()):
        raise GPUExperimentError("this launcher must not contain final_test inputs")
    if experiment.get("formal"):
        if not experiment.get("accept_model_licenses") and not dry_run:
            raise GPUExperimentError(
                "formal run requires experiment.accept_model_licenses=true")
        if not experiment.get("accept_vsi_bench_license") and not dry_run:
            raise GPUExperimentError(
                "formal run requires experiment.accept_vsi_bench_license=true")
        confirmation = Path(
            str(experiment.get("quality_confirmation") or "")).expanduser()
        if not confirmation.is_file() and not dry_run:
            raise GPUExperimentError(
                "formal run requires an existing quality_confirmation JSON")
    video_value = str(experiment.get("video_root") or "")
    video_root = Path(video_value).expanduser() if video_value else None
    if (
        (
            video_root is None
            or not video_root.is_dir()
            or not any(video_root.rglob("*.mp4"))
        )
        and not dry_run
    ):
        raise GPUExperimentError(
            "real GPU pipeline requires a non-empty licensed VSI-Bench video_root")
    free_gib = shutil.disk_usage(work_dir.parent).free / (1024 ** 3)
    required_disk = float(environment.get("min_free_disk_gib", 80))
    if free_gib < required_disk and not dry_run:
        raise GPUExperimentError(
            f"free disk {free_gib:.1f} GiB < required {required_disk:.1f} GiB")
    gpus = [] if dry_run else _gpu_inventory()
    required_count = int(environment.get("min_gpu_count", 2))
    required_memory = float(environment.get("min_gpu_memory_gib", 20)) * 1024
    if not dry_run and len(gpus) < required_count:
        raise GPUExperimentError(
            f"found {len(gpus)} GPUs, need at least {required_count}")
    qwen_gpu = int(cfg["runtime"]["qwen_gpu"])
    geometry_gpu = int(cfg["runtime"]["geometry_gpu"])
    if qwen_gpu == geometry_gpu:
        raise GPUExperimentError(
            "qwen_gpu and geometry_gpu must differ; Qwen/VGGT/SAM2 sharing is forbidden")
    if not dry_run and (
        qwen_gpu not in {gpu["index"] for gpu in gpus}
        or geometry_gpu not in {gpu["index"] for gpu in gpus}
    ):
        raise GPUExperimentError("configured GPU index is absent")
    selected = [
        gpu for gpu in gpus
        if gpu["index"] in {qwen_gpu, geometry_gpu}
    ]
    if not dry_run and any(
        gpu["memory_total_mib"] < required_memory for gpu in selected
    ):
        raise GPUExperimentError(
            f"selected GPUs need at least {required_memory / 1024:.1f} GiB")
    commit = _git(root, "rev-parse", "HEAD")
    if not _COMMIT_RE.fullmatch(commit):
        raise GPUExperimentError("current code is not at a full Git commit")
    dirty = _git(root, "status", "--porcelain")
    if experiment.get("formal") and dirty and not dry_run:
        raise GPUExperimentError(
            "formal run requires a clean detached/pinned code checkout")
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "code_commit": commit,
        "git_dirty": bool(dirty),
        "free_disk_gib": round(free_gib, 2),
        "gpus": gpus,
    }


def _validate_weight_config(cfg: dict) -> None:
    for name, entry in cfg["weights"].items():
        if name == "moge2" and not entry.get("enabled"):
            continue
        revision = str(entry.get("revision") or "")
        if not _COMMIT_RE.fullmatch(revision):
            raise GPUExperimentError(
                f"weights.{name}.revision must be a 40-char commit")
        if not str(entry.get("repo_id") or ""):
            raise GPUExperimentError(f"weights.{name}.repo_id is empty")


def _snapshot_files(root: Path) -> list[dict]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "/.cache/" in path.as_posix():
            continue
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    if not files:
        raise GPUExperimentError(f"downloaded snapshot is empty: {root}")
    return files


def _existing_weight_path(entry: dict) -> tuple[Path | None, str]:
    value = str(entry.get("existing_path") or "")
    if not value:
        return None, ""
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        return None, ""
    main_ref = path / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = path / "snapshots" / revision
        if _COMMIT_RE.fullmatch(revision) and snapshot.is_dir():
            return snapshot, revision
    revision = path.name if _COMMIT_RE.fullmatch(path.name) else ""
    return path, revision


def _download_weights(
    cfg: dict,
    weight_root: Path,
    *,
    dry_run: bool,
) -> tuple[dict, dict[str, Path]]:
    _validate_weight_config(cfg)
    if dry_run:
        paths = {}
        for name, entry in cfg["weights"].items():
            if name == "moge2" and not entry.get("enabled"):
                continue
            existing, _revision = _existing_weight_path(entry)
            paths[name] = existing or weight_root / name
        return {"status": "dry_run", "models": {}}, paths
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise GPUExperimentError(
            "huggingface_hub is required to download weights") from exc
    models: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    for name, entry in cfg["weights"].items():
        if name == "moge2" and not entry.get("enabled"):
            continue
        revision = str(entry["revision"])
        existing, existing_revision = _existing_weight_path(entry)
        if existing is not None:
            destination = existing
            source = "existing_verified_cache"
        else:
            destination = weight_root / name / revision
            destination.mkdir(parents=True, exist_ok=True)
            try:
                snapshot_download(
                    repo_id=str(entry["repo_id"]),
                    revision=revision,
                    local_dir=str(destination),
                    token=os.environ.get("HF_TOKEN") or None,
                )
            except Exception as exc:
                raise GPUExperimentError(
                    f"failed to download {name} at pinned revision {revision}: {exc}"
                ) from exc
            source = "downloaded_pinned_revision"
        files = _snapshot_files(destination)
        models[name] = {
            "repo_id": str(entry["repo_id"]),
            "revision": revision,
            "existing_revision": existing_revision,
            "source": source,
            "local_path": str(destination),
            "n_files": len(files),
            "total_bytes": sum(row["size"] for row in files),
            "files": files,
        }
        paths[name] = destination
    manifest = {
        "schema_version": "skill3d-weight-manifest-v11/1.0",
        "status": "complete",
        "models": models,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _json_bytes(manifest)).hexdigest()
    _write_json(weight_root / "weight_manifest.json", manifest)
    return manifest, paths


def _command_env(
    root: Path,
    cfg: dict,
    weights: dict[str, Path],
) -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = [str(root / "src"), str(root / "third_party" / "vggt")]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SKILL3D_VGGT_CHECKPOINT": str(weights["vggt"]),
        "SKILL3D_SAM2_CHECKPOINT": str(
            weights["sam2"] / cfg["weights"]["sam2"]["checkpoint_file"]),
        "SKILL3D_SAM2_CONFIG": str(cfg["weights"]["sam2"]["config"]),
        "SKILL3D_ALLOW_HF_DOWNLOAD": "",
    })
    detector = str(cfg["experiment"].get("detector_endpoint") or "")
    if detector:
        environment["SKILL3D_DETECTOR_ENDPOINT"] = detector
    return environment


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    commands: list[dict],
    stage: str,
) -> None:
    commands.append({"stage": stage, "command": command})
    print("+", " ".join(command), flush=True)
    if dry_run:
        return
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    if result.returncode:
        raise GPUExperimentError(
            f"stage {stage} failed with exit code {result.returncode}")


def _wait_vllm(base_url: str, model_name: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status != 200:
                    raise GPUExperimentError(
                        f"vLLM health returned HTTP {response.status}")
            with urlopen(f"{base_url}/v1/models", timeout=10) as response:
                payload = json.load(response)
            names = {
                str(row.get("id") or "") for row in payload.get("data", [])
            }
            if model_name not in names:
                raise GPUExperimentError(
                    f"served model {model_name!r} not in {sorted(names)}")
            return
        except (OSError, URLError, ValueError, GPUExperimentError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5)
    raise GPUExperimentError(
        f"vLLM did not become ready within {timeout_s}s: {last_error}")


def _start_vllm(
    root: Path,
    run_root: Path,
    cfg: dict,
    weights: dict[str, Path],
    env: dict[str, str],
    *,
    dry_run: bool,
    commands: list[dict],
) -> subprocess.Popen | None:
    runtime = cfg["runtime"]
    command = [
        shutil.which("vllm") or "vllm",
        "serve",
        str(weights["qwen"]),
        "--served-model-name",
        str(runtime["served_model_name"]),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(runtime["max_model_len"]),
        "--gpu-memory-utilization",
        str(runtime["gpu_memory_utilization"]),
        "--limit-mm-per-prompt",
        json.dumps({
            "image": int(runtime["n_images"]),
            "video": 1,
        }),
        "--mm-processor-kwargs",
        json.dumps({"max_pixels": int(runtime["max_pixels"])}),
        "--host",
        str(runtime["host"]),
        "--port",
        str(runtime["port"]),
    ]
    commands.append({"stage": "serve_qwen", "command": command})
    print("+", " ".join(command), flush=True)
    if dry_run:
        return None
    log_path = run_root / "vllm.log"
    log = log_path.open("ab")
    process_env = dict(env)
    process_env["CUDA_VISIBLE_DEVICES"] = str(runtime["qwen_gpu"])
    process = subprocess.Popen(
        command,
        cwd=root,
        env=process_env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base_url = f"http://{runtime['host']}:{runtime['port']}"
    try:
        _wait_vllm(
            base_url,
            str(runtime["served_model_name"]),
            float(runtime["startup_timeout_s"]),
        )
    except Exception:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        raise
    return process


def _quality_confirmation_ok(path: Path) -> bool:
    from skill3d.evaluation.skill_ablation_v11 import quality_contract

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    contract = quality_contract()
    return bool(
        value.get("confirmed") is True
        and value.get("sha256") == contract["sha256"]
        and value.get("evidence_ref")
    )


def _pipeline(
    root: Path,
    run_root: Path,
    cfg: dict,
    weights: dict[str, Path],
    *,
    dry_run: bool,
) -> list[dict]:
    experiment = cfg["experiment"]
    runtime = cfg["runtime"]
    commands: list[dict] = []
    env = _command_env(root, cfg, weights)
    geometry_env = dict(env)
    geometry_env["CUDA_VISIBLE_DEVICES"] = str(runtime["geometry_gpu"])
    endpoint = f"http://{runtime['host']}:{runtime['port']}"
    python = sys.executable
    video_root = str(Path(str(experiment["video_root"])).expanduser())
    config_path = str(root / "configs" / "config.yaml")
    recon_dir = run_root / "reconstructions"
    question_type = str(experiment["question_type"])
    datasets = str(experiment.get("datasets") or "")
    seed = int(experiment["seed"])
    sampling = int(experiment["reconstruction_sampling_per_task"])

    reconstruction = [
        python,
        "-m",
        "skill3d.reconstruction.run",
        "--config",
        config_path,
        "--source",
        "vsi_bench",
        "--split",
        "induction,inner_validation",
        "--video-root",
        video_root,
        "--recon-dir",
        str(recon_dir),
        "--question-types",
        question_type,
        "--limit",
        str(sampling),
        "--episodes-per-scene",
        "1",
        "--n-frames",
        str(runtime["n_images"]),
        "--gpus",
        "0",
    ]
    if datasets:
        reconstruction.extend(["--datasets", datasets])
    if cfg["weights"]["moge2"].get("enabled"):
        reconstruction.append("--moge2")
    _run(
        reconstruction,
        cwd=root,
        env=geometry_env,
        dry_run=dry_run,
        commands=commands,
        stage="reconstruction",
    )

    process = _start_vllm(
        root,
        run_root,
        cfg,
        weights,
        env,
        dry_run=dry_run,
        commands=commands,
    )
    try:
        parent_trace = run_root / "parent_learning_trace"
        parent_manifest = run_root / "parent_learning_manifest.json"
        parent = [
            python,
            "-m",
            "skill3d.online.eval",
            "--config",
            config_path,
            "--mode",
            "real",
            "--source",
            "vsi_bench",
            "--split",
            "induction",
            "--video-root",
            video_root,
            "--recon-dir",
            str(recon_dir),
            "--question-types",
            question_type,
            "--limit",
            str(experiment["induction_limit"]),
            "--seed",
            str(seed),
            "--vllm-endpoint",
            endpoint,
            "--vllm-model",
            str(runtime["served_model_name"]),
            "--trace-dir",
            str(parent_trace),
            "--no-memory",
            "--run-manifest",
            str(parent_manifest),
        ]
        if datasets:
            parent.extend(["--datasets", datasets])
        _run(
            parent,
            cwd=root,
            env=geometry_env,
            dry_run=dry_run,
            commands=commands,
            stage="parent_learning",
        )

        ablation_dir = run_root / "b01_b11"
        ablation = [
            python,
            str(root / "scripts" / "run_skill_ablation_v11.py"),
            "--config",
            config_path,
            "--source",
            "vsi_bench",
            "--split",
            "inner_validation",
            "--video-root",
            video_root,
            "--recon-dir",
            str(recon_dir),
            "--question-types",
            question_type,
            "--limit",
            str(experiment["inner_limit"]),
            "--seed",
            str(seed),
            "--vllm-endpoint",
            endpoint,
            "--vllm-model",
            str(runtime["served_model_name"]),
            "--output-dir",
            str(ablation_dir),
        ]
        if datasets:
            ablation.extend(["--datasets", datasets])
        confirmation = str(experiment.get("quality_confirmation") or "")
        if confirmation:
            ablation.extend(["--quality-confirmation", confirmation])
        _run(
            ablation,
            cwd=root,
            env=geometry_env,
            dry_run=dry_run,
            commands=commands,
            stage="b01_b11",
        )

        if experiment.get("run_evolution"):
            if not os.environ.get("DEEPSEEK_API_KEY") and not dry_run:
                raise GPUExperimentError(
                    "run_evolution=true requires DEEPSEEK_API_KEY")
            campaign = [
                python,
                str(root / "scripts" / "run_evolution_campaign_v11.py"),
                "--config",
                config_path,
                "--campaign-id",
                str(experiment["campaign_id"]),
                "--question-type",
                question_type,
                "--parent-trace-dir",
                str(parent_trace),
                "--source",
                "vsi_bench",
                "--video-root",
                video_root,
                "--recon-dir",
                str(recon_dir),
                "--inner-limit",
                str(experiment["inner_limit"]),
                "--post-limit",
                str(experiment["post_publish_limit"]),
                "--seed",
                str(seed),
                "--quality-confirmation",
                confirmation,
                "--library-root",
                str(root / "skill_library"),
                "--run-root",
                str(run_root / "evolution"),
                "--vllm-endpoint",
                endpoint,
                "--vllm-model",
                str(runtime["served_model_name"]),
            ]
            if datasets:
                campaign.extend(["--datasets", datasets])
            _run(
                campaign,
                cwd=root,
                env=geometry_env,
                dry_run=dry_run,
                commands=commands,
                stage="evolution",
            )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pinned-weight real GPU bootstrap for v11 experiments")
    parser.add_argument(
        "--config",
        default="configs/gpu_experiment_v11.yaml",
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("v11-%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = work_dir / "runs" / run_id
    if run_root.exists() and not args.dry_run:
        print(f"GPUExperimentError: run directory already exists: {run_root}",
              file=sys.stderr)
        return 2
    run_root.mkdir(parents=True, exist_ok=args.dry_run)
    try:
        cfg = _read_config(config_path)
        preflight = _preflight(
            root, work_dir, cfg, dry_run=args.dry_run)
        if cfg["experiment"].get("formal") and not args.dry_run:
            confirmation = Path(
                str(cfg["experiment"]["quality_confirmation"])).expanduser()
            if not _quality_confirmation_ok(confirmation):
                raise GPUExperimentError(
                    "quality_confirmation does not match current quality contract")
        weight_manifest, weights = _download_weights(
            cfg,
            work_dir / "weights",
            dry_run=args.dry_run,
        )
        pipeline = {
            "schema_version": "skill3d-gpu-pipeline-v11/1.0",
            "run_id": run_id,
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "preflight": preflight,
            "weight_manifest_sha256": weight_manifest.get("manifest_sha256", ""),
            "formal_requested": bool(cfg["experiment"].get("formal")),
            "final_test_touched": False,
            "status": "downloaded",
            "commands": [],
        }
        _write_json(run_root / "pipeline_manifest.json", pipeline)
        if args.download_only:
            print(json.dumps(pipeline, ensure_ascii=False, indent=2))
            return 0
        commands = _pipeline(
            root,
            run_root,
            cfg,
            weights,
            dry_run=args.dry_run,
        )
        pipeline["commands"] = commands
        pipeline["status"] = "dry_run" if args.dry_run else "completed"
        _write_json(run_root / "pipeline_manifest.json", pipeline)
        print(json.dumps(pipeline, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "skill3d-gpu-pipeline-v11/1.0",
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "final_test_touched": False,
        }
        _write_json(run_root / "failure.json", failure)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
