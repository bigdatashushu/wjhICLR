"""Dataset registry, frozen source, stage receipts, and conservative continuation.

Only the Python standard library is imported here. A dry run never imports the
scientific runtime, reads model credentials, creates directories, or sends I/O
to services. Commands are argv arrays, never evaluated by a shell.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import string
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "harness3d-experiment-campaign/1"
PROFILE_SCHEMA = "harness3d-experiment-profile/1"
STAGES = (
    "seed", "initial_validation", "learning", "freeze", "final_validation",
    "paired_score", "baseline32", "baseline_score", "seal",
)
TOKENS = {"data_root", "run_root", "source_root", "python", "dataset"}
NATIVE_INPUTS = {
    "panel", "metadata", "published_inputs", "qa_split", "online_config", "seed_library",
}
SECRET_KEYS = {"api_key", "token", "password", "secret", "authorization", "credential"}


class CampaignError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def git(root, *args):
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    require(process.returncode == 0, "Source Git identity is unavailable")
    return process.stdout.decode("utf-8", errors="surrogateescape").strip()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, indent=2) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def unique(pairs):
        value = {}
        for key, child in pairs:
            require(key not in value, "Duplicate JSON field")
            value[key] = child
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def expand(value, context):
    if isinstance(value, str):
        fields = [field for _, field, _, _ in string.Formatter().parse(value)
                  if field is not None]
        require(set(fields) <= TOKENS, "Unknown template placeholder")
        return value.format(**context)
    if isinstance(value, list):
        return [expand(child, context) for child in value]
    if isinstance(value, dict):
        return {key: expand(child, context) for key, child in value.items()}
    return value


def public_profile(value):
    """Inline secrets are forbidden; credential values belong to process env."""
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.lower() not in SECRET_KEYS, "Inline credentials are forbidden")
            public_profile(child)
    elif isinstance(value, list):
        for child in value:
            public_profile(child)


def validate_argv(argv):
    require(isinstance(argv, list) and bool(argv)
            and all(isinstance(word, str) and word and "\0" not in word for word in argv),
            "Stage argv must be a nonempty array of strings")
    for word in argv:
        require(not any(word.lower().startswith(prefix) for prefix in (
            "--api-key", "--password", "--token=", "--secret", "authorization:")),
            "Pass credentials through the process environment, not command arguments")


def validate_profile(profile):
    require(profile.get("schema") == PROFILE_SCHEMA, "Unsupported experiment profile")
    public_profile(profile)
    backend = profile.get("backend")
    require(backend in {"prepared-v11", "commands"}, "Unknown experiment backend")
    require(isinstance(profile.get("inputs"), dict), "Profile requires input paths")
    require(all(isinstance(name, str) and isinstance(path, str) and path
                for name, path in profile["inputs"].items()), "Invalid input paths")
    if backend == "commands":
        stages = profile.get("stages")
        require(isinstance(stages, list) and bool(stages), "Command profile requires stages")
        names = set()
        for stage in stages:
            name = stage.get("name", "")
            require(name and all(c.isalnum() or c in "_-" for c in name)
                    and name not in names, "Invalid or duplicate stage name")
            names.add(name)
            validate_argv(stage.get("argv"))
            require(stage.get("resume") in {"never", "receipt", "native-v11"},
                    "Each command requires an explicit resume policy")
            receipt = stage.get("receipt") or {}
            require(isinstance(receipt.get("path"), str)
                    and isinstance(receipt.get("equals"), dict) and receipt["equals"],
                    "Every stage requires a completion receipt contract")
    else:
        require(set(profile["inputs"]) == NATIVE_INPUTS, "Incomplete prepared input paths")
        runtime = profile.get("runtime") or {}
        for name in ("workers", "skill_owners", "reviser_concurrency",
                     "episode_concurrency", "request_concurrency", "scene_cache_size"):
            require(type(runtime.get(name)) is int and runtime[name] > 0,
                    f"{name} must be a positive integer")
        require(1 <= runtime["reviser_concurrency"] <= runtime["skill_owners"] <= 4,
                "Reviser concurrency must fit the 1..4 library owners")
        require(type(runtime.get("max_requests")) is int and runtime["max_requests"] >= 0,
                "max_requests must be a nonnegative integer")
        require(type(runtime.get("seed")) is int, "Seed must be an integer")
        rate = runtime.get("requests_per_second")
        require(isinstance(rate, (int, float)) and math.isfinite(rate) and rate >= 0,
                "Request rate must be finite and nonnegative")
        baseline = profile.get("baseline") or {}
        require(baseline.get("protocol") == "prepared-32"
                and baseline.get("scope") in {"panel", "validation"}
                and type(baseline.get("workers")) is int
                and 0 < baseline["workers"] <= runtime["request_concurrency"],
                "Native baseline requires prepared-32 and bounded workers")
        require(runtime.get("learning_input_mode") in {
            "aggregate_v1", "full_cases_v1", "analysis_synthesis_v1",
            "diagnosis_v1", "diagnosis_multimodal_v1"},
            "Unknown learning input mode")
        require(type(runtime.get("analysis_concurrency", 1)) is int
                and 1 <= runtime.get("analysis_concurrency", 1) <= 64,
                "Analysis concurrency must be within 1..64")
        require(type(runtime.get("overlap_owner_slots", False)) is bool,
                "Owner-slot overlap must be boolean")
        require(type(runtime.get("prefetch_scenes", False)) is bool,
                "Scene prefetch must be boolean")
        require(runtime.get("offline_frame_storage", "per_variant_v1")
                in {"per_variant_v1", "shared_scene_v2"}, "Unknown offline frame storage")
        caps = runtime.get("revision_request_caps", {})
        require(isinstance(caps, dict), "Revision request caps must be a slot map")
        from skill3d.evolution.fitset_stream_v11 import FitSetEvolutionConfig
        try:
            FitSetEvolutionConfig(revision_request_caps=caps)
        except (TypeError, ValueError):
            raise CampaignError("Revision request caps are invalid") from None
    return profile


def plan_stages(profile, context):
    if profile["backend"] == "commands":
        stages = expand(profile["stages"], context)
    else:
        stages = [{
            "name": name,
            "argv": [
                context["python"], "-B", f"{context['source_root']}/scripts/experiment_stage_v11.py",
                "--run-root", context["run_root"], "--stage", name,
            ],
            "resume": "native-v11" if name in {
                "initial_validation", "learning", "final_validation", "baseline32"} else "receipt",
            "receipt": {
                "path": f"{context['run_root']}/stages/{name}/receipt.json",
                "equals": {"status": "complete", "stage": name},
            },
            "artifacts": [f"{context['run_root']}/stages/{name}"],
        } for name in STAGES]
    for stage in stages:
        validate_argv(stage["argv"])
        for path in [stage["receipt"]["path"], *stage.get("artifacts", [])]:
            target = Path(path).resolve()
            require(target.is_relative_to(Path(context["run_root"]).resolve()),
                    "Stage receipts/artifacts must be inside the new run directory")
    return stages


def path_identity(path):
    path = Path(path).resolve()
    require(path.exists(), f"Required input does not exist: {path}")
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    require(bool(paths), f"Required input directory is empty: {path}")
    rows = []
    for child in paths:
        require(not child.is_symlink(), "Input trees cannot contain symbolic links")
        rows.append({
            "path": str(child), "size": child.stat().st_size, "sha256": file_hash(child),
        })
    return rows


def verify_files(rows):
    for row in rows:
        path = Path(row["path"])
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == row["size"] and file_hash(path) == row["sha256"],
                f"Frozen input or artifact changed: {path}")


def pins(profile):
    return {name: path_identity(path) for name, path in profile["inputs"].items()}


def get_field(value, key):
    for part in key.split("."):
        require(isinstance(value, dict) and part in value, "Completion receipt lacks required field")
        value = value[part]
    return value


def check_receipt(stage):
    path = Path(stage["receipt"]["path"])
    require(path.is_file() and not path.is_symlink(),
            f"Stage {stage['name']} has no completion receipt")
    value = read_json(path)
    for key, expected in stage["receipt"]["equals"].items():
        actual = get_field(value, key)
        require(type(actual) is type(expected) and actual == expected,
                f"Stage {stage['name']} is incomplete")
    return value


def collect_artifacts(stage):
    paths = {Path(stage["receipt"]["path"]).resolve()}
    for value in stage.get("artifacts", []):
        root = Path(value)
        require(root.exists(), f"Stage artifact is missing: {root}")
        paths.update([root] if root.is_file() else (p for p in root.rglob("*") if p.is_file()))
    rows = []
    for path in sorted(paths):
        require(not path.is_symlink(), "Stage artifact contains symbolic link")
        rows.append({"path": str(path), "size": path.stat().st_size, "sha256": file_hash(path)})
    return rows


@contextmanager
def run_lock(root):
    with (root / "campaign.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CampaignError("Another launcher owns this run") from None
        yield


def unknown_requests(root):
    path = root / "requests" / "state.json"
    if path.is_file():
        state = read_json(path)
        require(not any(row.get("status") == "started" for row in state.get("requests", [])),
                "Unresolved model request: continuation requires an audited response receipt")


def child_environment(root):
    env = os.environ.copy()
    # Never write environment values to the run. Keys stay with this process.
    env.update(
        PYTHONPATH=str(root / "code/source/src"),
        PYTHONDONTWRITEBYTECODE="1",
        SKILL3D_CODE_RECEIPT=str(root / "code/code_identity.json"),
        SKILL3D_RUN_MODE="formal",
    )
    return env


def execute_stage(root, stage):
    argv = list(stage["argv"])
    if stage["resume"] == "native-v11":
        argv.append("--resume")
    child = subprocess.Popen(argv, cwd=root / "code/source", env=child_environment(root),
                             start_new_session=True)
    old_handlers = {}

    def stop(signum, _frame):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop)
        returncode = child.wait()
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
    require(returncode == 0, f"Stage {stage['name']} stopped (exit {returncode}); no automatic retry")


def validate_run(manifest):
    from skill3d.infra.experiment_startup import verify_source_receipt
    body = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    require(manifest.get("schema") == SCHEMA and digest(body) == manifest.get("identity_sha256"),
            "Campaign manifest identity changed")
    root = Path(manifest["run_root"])
    source = verify_source_receipt(root / "code/code_identity.json", root / "code/source")
    require(source["source_sha256"] == manifest["source_sha256"], "Campaign source identity changed")
    verify_files(manifest["interpreter_pins"])
    for rows in manifest["input_pins"].values():
        verify_files(rows)
    require(pins(manifest["profile"]) == manifest["input_pins"], "Input directory contents changed")


def run_pipeline(manifest, *, resume):
    root = Path(manifest["run_root"])
    with run_lock(root):
        validate_run(manifest)
        state_path = root / "campaign_state.json"
        state = read_json(state_path) if state_path.exists() else {
            "identity_sha256": manifest["identity_sha256"], "stages": {}, "status": "running",
        }
        require(state.get("identity_sha256") == manifest["identity_sha256"],
                "Campaign state belongs to another run")
        stages = manifest["stages"]
        require(set(state["stages"]) <= {row["name"] for row in stages}, "Unknown stage in state")
        # Verify every completed stage before dispatching any child.
        for stage in stages:
            saved = state["stages"].get(stage["name"])
            if saved and saved["status"] == "complete":
                check_receipt(stage)
                verify_files(saved["artifacts"])
                require(collect_artifacts(stage) == saved["artifacts"],
                        "Completed stage directory changed")
        for stage in stages:
            name = stage["name"]
            saved = state["stages"].get(name)
            if saved and saved["status"] == "complete":
                print(f"[{name}] already complete", flush=True)
                continue
            unknown_requests(root)
            if saved:
                require(resume, "Use --resume for an interrupted run")
                if Path(stage["receipt"]["path"]).is_file():
                    check_receipt(stage)
                    state["stages"][name] = {
                        "status": "complete", "artifacts": collect_artifacts(stage),
                    }
                    atomic(state_path, state)
                    continue
                require(stage["resume"] == "native-v11",
                        f"Stage {name} has uncertain completion; receipt-only continuation refused")
            # Re-check identities between phases, not only on entry.
            validate_run(manifest)
            state.update(status="running", current_stage=name)
            state["stages"][name] = {"status": "started", "started_at": time.time()}
            atomic(state_path, state)
            print(f"[{name}] starting", flush=True)
            try:
                execute_stage(root, stage)
                check_receipt(stage)
                unknown_requests(root)
                state["stages"][name] = {
                    "status": "complete", "artifacts": collect_artifacts(stage),
                    "completed_at": time.time(),
                }
                atomic(state_path, state)
            except BaseException as exc:
                state.update(status="stopped", exception_type=type(exc).__name__)
                atomic(state_path, state)
                raise
        state.update(status="complete", current_stage=None)
        atomic(state_path, state)
        print(f"Experiment complete: {root}", flush=True)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", help="registry name: vsi-bench, revsi, or custom with --config")
    result.add_argument("--data-root")
    result.add_argument("--config", help="JSON experiment profile")
    result.add_argument("--registry", default=str(ROOT / "configs/experiments/datasets.json"))
    result.add_argument("--output-dir")
    result.add_argument("--python", help="existing experiment interpreter; never installs packages")
    result.add_argument("--workers", type=int)
    result.add_argument("--revisers", type=int)
    result.add_argument("--max-requests", type=int)
    result.add_argument("--dry-run", action="store_true", help="print plan; no writes, network, or model calls")
    result.add_argument("--check", action="store_true", help="check local inputs only; no writes or requests")
    result.add_argument("--resume", action="store_true", help="continue a new-launcher run at --output-dir")
    result.add_argument("--list-datasets", action="store_true")
    return result


def resolve_new(args):
    registry_path = Path(args.registry).resolve()
    registry = read_json(registry_path)
    require(registry.get("schema") == "harness3d-dataset-registry/1", "Unknown dataset registry")
    dataset = args.dataset or "vsi-bench"
    entry = None
    for name, candidate in registry["datasets"].items():
        if dataset == name or dataset in candidate.get("aliases", []):
            dataset, entry = name, candidate
            break
    require(dataset and all(c.isalnum() or c in "_.-" for c in dataset)
            and dataset not in {".", ".."}, "Invalid dataset name")
    require(entry is not None or args.config, "Unknown dataset; supply --config for custom data")
    profile_path = Path(args.config).resolve() if args.config else (
        registry_path.parent / entry["profile"]).resolve()
    template = read_json(profile_path)
    overrides = {"workers": args.workers, "reviser_concurrency": args.revisers,
                 "max_requests": args.max_requests}
    for key, value in overrides.items():
        if value is not None:
            template.setdefault("runtime", {})[key] = value
    validate_profile(template)
    data = Path(args.data_root or (ROOT / entry["data_root"] if entry else profile_path.parent)).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output_dir or ROOT / "runs" / dataset / stamp).resolve()
    require(output != ROOT and not ROOT.is_relative_to(output), "Output would contain source repository")
    require(output != data and not data.is_relative_to(output)
            and not output.is_relative_to(data), "Output must be separate from dataset inputs")
    # Preserve a venv's symlink spelling: resolving it launches base Python and
    # silently drops that environment's installed scientific dependencies.
    interpreter = str(Path(args.python or sys.executable).absolute())
    context = {
        "data_root": str(data), "run_root": str(output), "source_root": str(ROOT),
        "python": interpreter, "dataset": dataset,
    }
    profile = expand(template, context)
    for path in profile["inputs"].values():
        target = Path(path).resolve()
        require(not target.is_relative_to(output) and not output.is_relative_to(target),
                "Output must not overlap any input path")
    return dataset, output, interpreter, profile, template, profile_path, context


def launch(args):
    if args.list_datasets:
        registry = read_json(args.registry)
        for name, row in registry["datasets"].items():
            print(f"{name}: {row['description']}")
        return 0
    if args.resume:
        require(args.output_dir, "--resume requires --output-dir")
        root = Path(args.output_dir).resolve()
        manifest = read_json(root / "campaign_manifest.json")
        require(manifest["run_root"] == str(root), "Run directory has moved")
        for key, value in {"dataset": args.dataset, "data_root": args.data_root,
                           "python": args.python}.items():
            if value is not None:
                expected = manifest[key]
                actual = value if key == "dataset" else str(
                    Path(value).absolute() if key == "python" else Path(value).resolve())
                require(actual == expected, f"{key} changed during resume")
        for key, value in {"workers": args.workers, "reviser_concurrency": args.revisers,
                           "max_requests": args.max_requests}.items():
            if value is not None:
                require(value == manifest["profile"]["runtime"][key],
                        f"{key} changed during resume; create a new run")
        if args.config:
            require(file_hash(args.config) == manifest["profile_file_sha256"],
                    "Experiment profile changed during resume")
        if args.dry_run:
            print(encoded({"dataset": manifest["dataset"], "run_root": str(root),
                           "stages": manifest["stages"], "mode": "resume-preview"}).decode(), end="")
            return 0
        validate_run(manifest)
        if args.check:
            print("Frozen run inputs and source verified; model_requests=0")
            return 0
        return run_pipeline(manifest, resume=True)
    dataset, root, interpreter, profile, template, profile_path, context = resolve_new(args)
    preview_context = {**context, "source_root": str(root / "code/source")}
    stages = plan_stages(profile, preview_context)
    if args.dry_run:
        print(encoded({
            "dataset": dataset, "run_root": str(root), "backend": profile["backend"],
            "inputs": profile["inputs"], "runtime": profile.get("runtime"),
            "baseline": profile.get("baseline"), "stages": stages, "model_requests": 0,
        }).decode(), end="")
        return 0
    require(Path(interpreter).is_file(), "Experiment interpreter does not exist")
    require(sys.version_info >= (3, 11), "Execution requires Python 3.11+; select it with --python")
    input_pins = pins(profile)
    if profile["backend"] == "prepared-v11":
        require(profile["runtime"]["max_requests"] > 0,
                "Supply --max-requests or freeze a positive budget in the profile")
    if args.check:
        print(f"Local inputs verified ({len(input_pins)} anchors); model_requests=0")
        return 0
    require(not root.exists(), "Output already exists; use --resume or choose a new directory")
    require(not git(ROOT, "status", "--porcelain=v1"),
            "Commit launcher source before a formal run")
    from skill3d.infra.experiment_startup import freeze_working_tree
    root.mkdir(parents=True)
    try:
        receipt = freeze_working_tree(ROOT, root / "code")
        require(not receipt["git_dirty"], "Source changed during launch")
        receipt["run_mode"] = "formal"
        atomic(root / "code/code_identity.json", receipt)
        # Source-owned seed/config inputs follow the immutable snapshot.
        frozen_context = {**context, "source_root": str(root / "code/source")}
        frozen_profile = expand(template, frozen_context)
        require(pins(profile) == input_pins, "Inputs changed while freezing source")
        body = {
            "schema": SCHEMA, "dataset": dataset, "run_root": str(root),
            "data_root": context["data_root"], "python": interpreter,
            "profile": frozen_profile, "profile_file_sha256": file_hash(profile_path),
            "source_sha256": receipt["source_sha256"], "input_pins": pins(frozen_profile),
            "interpreter_pins": path_identity(Path(interpreter).resolve()) + (
                path_identity(Path(interpreter).parent.parent / "pyvenv.cfg")
                if (Path(interpreter).parent.parent / "pyvenv.cfg").is_file() else []),
            "stages": plan_stages(frozen_profile, frozen_context),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest = {**body, "identity_sha256": digest(body)}
        atomic(root / "campaign_manifest.json", manifest)
    except BaseException:
        # Retain an explicit marker; never adopt a half-created run.
        atomic(root / "initialization_failure.json", {"status": "incomplete"})
        raise
    return run_pipeline(manifest, resume=False)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        require(not (args.dry_run and args.check), "--dry-run and --check are mutually exclusive")
        if (args.python and not args.dry_run and not args.list_datasets
                and Path(args.python).absolute() != Path(sys.executable).absolute()):
            # The coordinator and scientific children share the selected
            # existing interpreter; no install or environment mutation.
            return subprocess.call([args.python, "-B", str(ROOT / "scripts/run_experiment.py"),
                                    *(sys.argv[1:] if argv is None else argv)])
        return launch(args)
    except CampaignError as exc:
        print(f"Launcher stopped: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Launcher interrupted; child processes drained", file=sys.stderr)
        return 130
    except Exception as exc:
        # Third-party/JSON exceptions can contain private data. Persist type only.
        print(f"Launcher stopped: {type(exc).__name__}", file=sys.stderr)
        return 2
