"""Explicit real-service wiring, zero-inference preflight and durable live runs."""
from __future__ import annotations

import base64
import contextlib
import io
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import uuid
from urllib.parse import urlsplit

import numpy as np

from action_audit import audit_run, seal_record
from action_bridge import ActionBridge, NDJSONTransport, v2_environment
from action_contract import public_hash
from action_host import ActionHost
from action_identity import (ARMS, COMPARE_ARMS, RERUN_ARMS, ENVIRONMENTS,
                             configuration_identity, protocol_identity, source_identity)
from action_solver import ActionSolver, MODEL_CONFIG, make_qwen_client
from budget import RunBudget, policy_for
from evidence import CombinedDINO, CurrentEvidencePreparer, PerceptionBackends
from methods import load_methods
from protocol import ContractError, OperationalError, ROOT, canonical, json_object, select_tasks, sha256

SCHEMA = "embodied-live-config/1"


def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def private_output(path):
    path = Path(path).absolute()
    if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve())
            or not path.parent.is_dir() or path.exists()):
        raise ContractError("live_output_must_be_new_in_workspace")
    return path


def write_new(path, value):
    with open(path, "xb") as stream:
        os.chmod(path, 0o600)
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_config(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size > 65536:
        raise ContractError("live_config_path")
    config = json_object(path.read_bytes())
    expected = {"schema", "selection", "arms", "integration_step_limit",
                "require_identical_initial_observation", "navigation_python", "household_python",
                "render_gpu", "perception_gpu", "detector", "reviewer", "sam2", "moge2"}
    if (not expected <= set(config) or set(config) - expected - {"manipulation_python", "task_cards"}
            or config["schema"] != SCHEMA
            or config["arms"] not in (list(ARMS), list(COMPARE_ARMS), list(RERUN_ARMS))
            or type(config["selection"]) is not list or not 1 <= len(config["selection"]) <= 180):
        raise ContractError("live_config_shape")
    if "task_cards" in config and type(config["task_cards"]) is not bool:
        raise ContractError("live_task_card_mode")
    for key in ("render_gpu", "perception_gpu"):
        if type(config[key]) is not int or config[key] not in {0, 1, 3, 7}:
            raise ContractError("live_gpu_not_authorized")
    limit = config["integration_step_limit"]
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 20):
        raise ContractError("live_integration_step_limit")
    if type(config["require_identical_initial_observation"]) is not bool:
        raise ContractError("live_pairing_policy")
    # Reject secrets rather than copying arbitrary configuration into receipts.
    shapes = {
        "detector": {"endpoint", "token_env", "model_sha256", "backend_config_sha256",
                     "deployment_receipt"},
        "reviewer": {"endpoint", "expected_identity_sha256"},
        "sam2": {"checkpoint", "checkpoint_sha256", "config"},
        "moge2": {"checkpoint", "checkpoint_sha256"},
    }
    for key, fields in shapes.items():
        if type(config[key]) is not dict or set(config[key]) != fields:
            raise ContractError("live_backend_config_shape")
        for name, value in config[key].items():
            if name.endswith("sha256") and (type(value) is not str or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ContractError("live_backend_hash")
    for key in ("detector", "reviewer"):
        endpoint = urlsplit(config[key]["endpoint"])
        if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1"
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                or endpoint.path not in {"", "/"}):
            raise ContractError("live_endpoint_must_be_loopback")
    if config["detector"]["token_env"] != "HARNESS3D_PERCEPTION_TOKEN":
        raise ContractError("live_token_environment")
    for key in ("navigation_python", "household_python"):
        if not Path(config[key]).is_absolute() or not Path(config[key]).is_file():
            raise ContractError("official_python_missing")
    if any(i.get("environment") == "eb_manipulation" for i in config["selection"]):
        if ("manipulation_python" not in config or not Path(config["manipulation_python"]).is_absolute()
                or not Path(config["manipulation_python"]).is_file()):
            raise ContractError("manipulation_python_missing")
    return config


def select_public(config):
    from household_protocol import SUBSETS, source, verify_environment
    selection = []
    for item in config["selection"]:
        if (type(item) is not dict or set(item) != {"environment", "subset", "index"}
                or item["environment"] not in ENVIRONMENTS or type(item["index"]) is not int
                or item["index"] < 0):
            raise ContractError("live_selection_shape")
        environment, subset, index = item["environment"], item["subset"], item["index"]
        if environment == "eb_navigation":
            digest = select_tasks(ROOT / "upstream", [subset], index + 1)[index].input_sha256
        elif environment == "eb_manipulation":
            from manipulation_worker import asset_catalog
            catalog = asset_catalog()
            if not any(r["subset"] == subset and r["index"] == index for r in catalog["selection"]):
                raise ContractError("manipulation_task_not_installed")
            digest = sha256(canonical(catalog))
        else:
            kind = environment[3:]
            if subset not in SUBSETS[kind]:
                raise ContractError("live_selection_subset")
            if index >= 5:
                raise ContractError("household_expansion_selection")
            if kind == "alfred":
                assets = json_object((ROOT / "action_alfred_data.json").read_bytes())
                if subset != assets["subset"] or index not in assets["indexes"]:
                    raise ContractError("alfred_expansion_assets_selection")
            verify_environment(kind)
            name = ("embodiedbench/envs/eb_alfred/data/splits/splits.json" if kind == "alfred"
                    else f"embodiedbench/envs/eb_habitat/datasets/{subset}.pickle")
            digest = sha256(source(name))  # no pickle or GT deserialization in host
        selection.append({**item, "episode_id": f"{environment}:{subset}:{index:03d}",
                          "source_sha256": digest})
    if len({i["episode_id"] for i in selection}) != len(selection):
        raise ContractError("live_selection_duplicate")
    return selection


class PinnedDetector:
    """One authenticated request. Public deployment receipt pins its process."""
    def __init__(self, config):
        self.config = config
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.identity = {k: config[k] for k in ("model_sha256", "backend_config_sha256")}

    def call(self, route, body=None):
        token = os.environ.get(self.config["token_env"], "")
        if not token:
            raise OperationalError("detector_auth_missing")
        request = urllib.request.Request(
            self.config["endpoint"].rstrip("/") + route,
            data=None if body is None else canonical(body),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
        with self.opener.open(request, timeout=10 if body is None else 60) as response:
            data = response.read(4 * 1024**2 + 1)
        if len(data) > 4 * 1024**2 or token.encode() in data:
            raise OperationalError("detector_response_rejected")
        return json_object(data)

    def health(self):
        receipt = json_object(Path(self.config["deployment_receipt"]).read_bytes())
        proc = Path("/proc") / str(receipt["pid"])
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if (int(fields[19]) != int(receipt["start_ticks"])
                or sha256((proc / "cmdline").read_bytes()) != receipt["cmdline_sha256"]):
            raise OperationalError("detector_deployment_changed")
        result = self.call("/health")
        if result.get("model_name") != "grounding_dino" or result.get("status") not in {"healthy", "健康"}:
            raise OperationalError("detector_health_invalid")
        return {"healthy": True, "deployment_receipt_sha256":
                file_hash(self.config["deployment_receipt"]), "identity_sha256": public_hash(self.identity),
                "identity_basis": "frozen deployment process and configured model pins"}

    def detect(self, rgb, prompt, **kwargs):
        from skill3d.segmentation.open_vocab_detector import Detection
        image = io.BytesIO()
        from PIL import Image
        Image.fromarray(rgb).save(image, format="JPEG", quality=90)
        body = {"image": base64.b64encode(image.getvalue()).decode(),
                "text_prompt": prompt, "box_threshold": .25}
        result = self.call("/infer", body)
        if result.get("success") is False or type(result.get("detections")) is not list:
            raise OperationalError("detector_result_invalid")
        rows, h, w = [], *rgb.shape[:2]
        if len(result["detections"]) > 256:
            raise OperationalError("detector_response_cap")
        for row in result["detections"]:
            box = [float(x) for x in row["bbox"]]
            confidence = float(row["confidence"])
            if len(box) != 4 or not all(math.isfinite(x) for x in (*box, confidence)) or not 0 <= confidence <= 1:
                raise OperationalError("detector_row_invalid")
            x0, y0, x1, y1 = box
            x0, x1 = sorted((max(0., min(x0, w - 1)), max(0., min(x1, w - 1))))
            y0, y1 = sorted((max(0., min(y0, h - 1)), max(0., min(y1, h - 1))))
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                rows.append(Detection(str(row["label"]).strip(), (x0, y0, x1, y1), confidence))
        return rows


def worker_command(config, item, work, *, probe=False):
    environment = item["environment"]
    key = ("navigation_python" if environment == "eb_navigation" else
           "manipulation_python" if environment == "eb_manipulation" else "household_python")
    python = config[key]
    command = [python, "-B", str(ROOT / "action_worker_v2.py"), "--environment", environment,
               "--subset", item["subset"], "--work-dir", str(work), "--official-runtime-verified"]
    if environment == "eb_navigation":
        command += ["--gpu-device", str(config["render_gpu"])]
    elif environment == "eb_manipulation":
        # glibc reads this search path at process startup, before runtime().
        command = ["/usr/bin/env", "LD_LIBRARY_PATH=" + str(ROOT / "manipulation-runtime/coppeliasim"),
                   *command]
    return command + (["--probe"] if probe else [])


def preflight(config, work):
    """No Qwen calls, no simulator launch, no perception inference."""
    if platform.system() != "Linux":
        raise ContractError("live_requires_verified_linux_runtime")
    selection = select_public(config)
    runtimes = {}
    for item in selection:
        environment = item["environment"]
        if environment in runtimes:
            continue
        reply = subprocess.run(worker_command(config, item, work, probe=True),
                               env=v2_environment(work), capture_output=True, timeout=30)
        if reply.returncode != 0 or len(reply.stdout) > 65536:
            raise OperationalError("official_runtime_probe_failed")
        row = json_object(reply.stdout)
        if (row["schema"] != "embodied-v2-runtime-check/1" or row["runtime_ready"] is not True
                or row["environment"] != environment or row["model_requests"] != 0
                or row["simulator_launches"] != 0 or any(
                    sha256((ROOT / name).read_bytes()) != digest
                    for name, digest in row["source_sha256"].items())):
            raise OperationalError("official_runtime_probe_identity")
        runtimes[environment] = row
    detector = PinnedDetector(config["detector"])
    detector_health = detector.health()
    from skill3d.small_image.backends import InternCategoryAdapter, SAM2ImageAdapter, MoGe2Adapter
    reviewer = InternCategoryAdapter({**config["reviewer"], "timeout_s": 60})
    reviewer.client.identity()
    for key in ("sam2", "moge2"):
        if file_hash(config[key]["checkpoint"]) != config[key]["checkpoint_sha256"]:
            raise ContractError("perception_checkpoint_changed")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(config["perception_gpu"]),
                      CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    segmenter = SAM2ImageAdapter(identity={"checkpoint_sha256": config["sam2"]["checkpoint_sha256"]},
                                checkpoint=config["sam2"]["checkpoint"], config=config["sam2"]["config"])
    depth = MoGe2Adapter(identity={"checkpoint_sha256": config["moge2"]["checkpoint_sha256"]},
                        checkpoint=config["moge2"]["checkpoint"], device="cuda:0",
                        infer_kwargs={"resolution_level": 6})
    preparer = CurrentEvidencePreparer(PerceptionBackends(
        detector=CombinedDINO(identity=detector.identity, endpoint=config["detector"]["endpoint"],
                              detect_fn=detector.detect),
        segmenter=segmenter, reviewer=reviewer, depth_estimator=depth))
    # Exercise the native boundary without any model/agent execution.
    from sandbox_linux import linux_command
    from sandbox_process import minimal_environment
    bootstrap = ("import sys; sys.path[:0]=" + repr([str(ROOT), str(ROOT.parent / "src")])
                 + "; from sandbox_linux import seal_linux; import json; print(json.dumps(seal_linux()))")
    sealed = subprocess.run(linux_command(sys.executable, bootstrap),
                            env=minimal_environment(work), capture_output=True, timeout=10)
    if sealed.returncode != 0:
        raise OperationalError("linux_isolation_probe_failed")
    receipt = seal_record({
        "schema": "embodied-live-preflight/1", "status": "ready", "runtime": runtimes,
        "selection": selection, "live_config_sha256": public_hash(config),
        "evidence_config_sha256": preparer.config_sha256,
        "detector": detector_health, "review_identity_sha256": reviewer.expected_identity_sha256,
        "checkpoint_sha256": {k: config[k]["checkpoint_sha256"] for k in ("sam2", "moge2")},
        "isolation": json_object(sealed.stdout), "live_requests": 0,
        "official_simulator_launches": 0, "automatic_retry": False, "reference_answer_visible": False,
    })
    return receipt, preparer


def live_check(config_path, work):
    config = load_config(config_path)
    return preflight(config, work)[0]


def live_run(config_path, output, *, compare=False):
    from action_run import startup_failure_report
    output = private_output(output)
    config = load_config(config_path)
    arms = config.get("arms", list(ARMS))
    if compare and (arms != list(COMPARE_ARMS) or config["integration_step_limit"] is not None):
        raise ContractError("compare_requires_two_arms_and_full_episodes")
    run_dir = output.parent / (output.stem + "-episodes")
    if run_dir.exists():
        raise ContractError("live_run_directory_exists")
    run_dir.mkdir(mode=0o700)
    receipt, preparer = preflight(config, run_dir)
    if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        raise ContractError("live_model_auth_missing")
    methods, selection = load_methods(), receipt["selection"]
    environments = [i["environment"] for i in selection]
    task_cards = config.get("task_cards", True)
    capture = compare or "eb_manipulation" in environments or task_cards
    run_id = "live-" + uuid.uuid4().hex
    manifest = {
        "schema": "embodied-paired-run/1", "run_id": run_id, "arms": arms,
        "selection": selection, "denominator_per_arm": len(selection),
        "model_config_sha256": public_hash(MODEL_CONFIG),
        "budget_sha256_by_environment": {e: public_hash(policy_for(e)) for e in environments},
        "method_sha256_by_environment": {e: methods[e].content_sha256 for e in environments},
        "method_library_sha256": next(iter(methods.values())).manifest_sha256,
        "evidence_config_sha256": preparer.config_sha256,
        "input_policy": "current_only", "execution_policy": "single_action", "offline_fake": False,
        "automatic_retry": False, "reference_answer_visible": False,
        "pairing": "same public task/source and policy; action trajectories may diverge",
        "implementation_source": source_identity(), "protocol": protocol_identity(),
        "configuration": configuration_identity(methods, preparer.config_sha256),
        "live_preflight": receipt, "live_config_sha256": public_hash(config),
        "integration_step_limit": config["integration_step_limit"],
        "require_identical_initial_observation": config["require_identical_initial_observation"],
        "capture_complete_trajectory": capture,
        "task_cards": task_cards,
    }
    seal_record(manifest)
    write_new(run_dir / "manifest.json", manifest)
    reports, budgets, stopped, trajectories = [], {}, False, []
    # Task-major ordering provides integration coverage before longer selections.
    roots = {arm: RunBudget(run_id + ":" + arm, environments) for arm in arms}
    for ordinal, item in enumerate(selection):
        for arm in arms:
            budget = roots[arm].episode(item["episode_id"], item["environment"])
            method = methods[item["environment"]] if arm in {"B11", "Full"} else None
            transport = solver = host = None
            trajectory = None
            if capture:
                from action_trajectory import Trajectory
                trajectory = Trajectory(run_dir / f"trajectory-{ordinal:03d}-{arm}",
                                        episode_id=item["episode_id"], arm=arm)
            launched = False
            if stopped:
                report = startup_failure_report(roots[arm].run_id, item, arm, budget,
                                                method.content_sha256 if method else "")
                report["terminal_reason"] = "not_started_after_operational_stop"
            else:
                work = run_dir / f"{ordinal:03d}-{arm}"
                work.mkdir(mode=0o700)
                try:
                    transport = NDJSONTransport(worker_command(config, item, work), work_dir=work,
                                                environment=item["environment"])
                    bridge = ActionBridge(transport, environment=item["environment"],
                                          episode_id=item["episode_id"], budget=budget)
                    solver = ActionSolver(make_qwen_client(), arm=arm, environment=item["environment"],
                                          method=method, trajectory=trajectory, task_cards=task_cards)
                    preparer.trajectory = trajectory if arm != "Direct" else None
                    host = ActionHost(bridge, solver, run_id=roots[arm].run_id, arm=arm,
                                      evidence_preparer=None if arm == "Direct" else preparer,
                                      method_sha256=method.content_sha256 if method else "",
                                      trajectory=trajectory)
                    host.integration_step_limit = config["integration_step_limit"]
                    launched = True  # official open may fail after launching
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        report = host.run(subset=item["subset"], index=item["index"],
                                          source_sha256=item["source_sha256"])
                except Exception:
                    report = (host.report(reason="host_operational_failure", aborted=True) if host else
                              startup_failure_report(roots[arm].run_id, item, arm, budget,
                                                     method.content_sha256 if method else ""))
                finally:
                    if transport:
                        transport.close()
                stopped = report["status"] == "aborted"
            report.update(model_requests=solver.receipts if solver else [],
                          paired_manifest_sha256=manifest["sha256"], source_sha256=item["source_sha256"],
                          offline_fake=False, official_open_attempted=launched,
                          implementation_source_sha256=manifest["implementation_source"]["sha256"],
                          protocol_sha256=manifest["protocol"]["sha256"],
                          prepared_config_sha256=manifest["configuration"]["sha256"])
            reports.append(seal_record(report))
            write_new(run_dir / f"attempt-{ordinal:03d}-{arm}.json", report)
            if trajectory:
                complete = trajectory.finish(report)
                trajectories.append({
                    "episode_id": item["episode_id"], "arm": arm,
                    "directory": trajectory.directory.name, "trajectory": complete})
            print(canonical({"event": "episode_complete", "episode_id": item["episode_id"], "arm": arm,
                             "status": report["status"], "reason": report["terminal_reason"],
                             "steps": report["environment_steps"],
                             "qwen": report["budget"]["reserved"].get("qwen", 0)}).decode(), flush=True)
    budgets = {arm: roots[arm].summary() for arm in arms}
    closure = all(r["status"] != "aborted" and (config["integration_step_limit"] is None
                  or r["environment_steps"] >= config["integration_step_limit"]
                  or r["terminal_reason"] == "official_done") for r in reports)
    result = seal_record({
        "schema": "embodied-live-run/1", "manifest": manifest, "attempts": reports, "run_budgets": budgets,
        "live_requests": sum(sum(r["budget"]["reserved"].get(k, 0) for k in
                                 ("qwen", "reviews", "dino", "sam", "moge")) for r in reports),
        "official_simulator_launches": sum(bool(r["observations"]) for r in reports),
        "official_open_attempts": sum(r["official_open_attempted"] for r in reports),
        "deployments": 0, "status": "passed" if closure else "failed",
    })
    result["audit"] = audit_run(result)
    write_new(output, result)
    if capture:
        write_new(run_dir / "trajectories.json", {
            "schema": "embodied-comparison-trajectories/1", "run_sha256": result["sha256"],
            "manifest": manifest, "audit": result["audit"], "episodes": trajectories})
    return result
