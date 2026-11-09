"""Validate a separate run configuration while reusing the frozen simulator worker."""
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from target_detection_r05.protocol import Task
from .sources import ContractError, LIMITS


def validate_detection(detection):
    if (not isinstance(detection, dict) or set(detection) != {
            "backend", "service", "temperature", "max_tokens", "group_iou", "page_size", "max_crop_groups"}
            or detection["backend"] != "qwen3vl8b"
            or detection["group_iou"] != 0.8 or detection["page_size"] != 12
            or detection["max_crop_groups"] != 2):
        raise ContractError("unsupported_qwen_detection_configuration")
    if (type(detection["temperature"]) not in (int, float)
            or not 0 <= detection["temperature"] <= 2
            or type(detection["max_tokens"]) is not int or not 1 <= detection["max_tokens"] <= 8192):
        raise ContractError("invalid_detection_sampling")
    service = detection["service"]
    fields = {"transport", "ssh_target", "base_url", "model", "timeout_s", "auth_process_pid",
              "auth_process_start_ticks", "auth_process_boot_id", "model_snapshot", "model_files_sha256"}
    if not isinstance(service, dict) or set(service) != fields:
        raise ContractError("unexpected_detector_service_fields")
    url = urlsplit(service["base_url"])
    if (service["transport"] not in ("local", "ssh")
            or not re.fullmatch(r"[A-Za-z0-9_.@-]+", service["ssh_target"])
            or url.scheme not in ("http", "https") or not url.hostname
            or url.username or url.password or url.query or url.fragment or url.path != "/v1"
            or service["model"] != "Qwen3-VL-8B-Instruct-FP8"
            or type(service["timeout_s"]) not in (int, float) or not 1 <= service["timeout_s"] <= 600):
        raise ContractError("invalid_detector_service")
    if (any(type(service[k]) is not int or service[k] <= 0
            for k in ("auth_process_pid", "auth_process_start_ticks"))
            or not re.fullmatch(r"[0-9a-f-]{36}", service["auth_process_boot_id"])
            or not Path(service["model_snapshot"]).is_absolute()):
        raise ContractError("invalid_detector_process_identity")
    hashes = service["model_files_sha256"]
    if (not isinstance(hashes, dict) or set(hashes) != {
            "config.json", "generation_config.json", "model.safetensors.index.json"}
            or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values())):
        raise ContractError("invalid_detector_model_metadata_identity")


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"schema", "model", "runtime", "episodes", "n_shots", "max_requests", "detection", "execution"}
    if set(config) != required or config["schema"] != "qwen3vl-target-detection-config/1":
        raise ContractError("invalid_qwen_configuration")
    model = config["model"]
    if set(model) != {"name", "api_key_env", "base_url_env", "response_format",
                      "max_tokens", "temperature", "timeout_s", "extra_body"}:
        raise ContractError("unexpected_model_config")
    for key in ("api_key_env", "base_url_env"):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+", model[key]):
            raise ContractError("invalid_model_environment_name")
    if not isinstance(model["name"], str) or not model["name"]:
        raise ContractError("invalid_model_name")
    if model["response_format"] != "json_object":
        raise ContractError("planner_requires_json_object_mode")
    if type(model["max_tokens"]) is not int or not 1 <= model["max_tokens"] <= 32768:
        raise ContractError("invalid_token_budget")
    if (type(model["temperature"]) not in (int, float) or not 0 <= model["temperature"] <= 2
            or type(model["timeout_s"]) not in (int, float) or not 1 <= model["timeout_s"] <= 600):
        raise ContractError("invalid_sampling_or_timeout")
    extra = model["extra_body"]
    if not isinstance(extra, dict) or not set(extra) <= {"seed", "enable_thinking", "reasoning_effort"}:
        raise ContractError("unsupported_model_extra")
    if "seed" in extra and type(extra["seed"]) is not int:
        raise ContractError("invalid_seed")
    if "enable_thinking" in extra and type(extra["enable_thinking"]) is not bool:
        raise ContractError("invalid_thinking_flag")
    if "reasoning_effort" in extra and extra["reasoning_effort"] not in (
            "none", "minimal", "low", "medium", "high", "xhigh"):
        raise ContractError("invalid_reasoning_effort")
    runtime = config["runtime"]
    required_runtime = {"transport", "ssh_target", "code_root", "source_root", "work_root",
                        "python", "gpu", "timeout_s", "environment_seed"}
    if not required_runtime <= set(runtime) or not set(runtime) <= required_runtime | {
            "manipulation_catalog", "manipulation_dataset", "gpu_min_free_mb"}:
        raise ContractError("unexpected_runtime_config")
    if runtime["transport"] not in ("local", "ssh") or not re.fullmatch(
            r"[A-Za-z0-9_.@-]+", runtime["ssh_target"]):
        raise ContractError("invalid_transport")
    for key in ("code_root", "source_root", "work_root", "manipulation_catalog", "manipulation_dataset"):
        if key in runtime and not Path(runtime[key]).is_absolute():
            raise ContractError("runtime_paths_must_be_absolute")
    # The reused worker requires its work directory under its own package.
    # A new detector-specific namespace avoids touching any historical run.
    namespace = (Path(runtime["code_root"]) / "target_detection_r05/runs/qwen3vl8b").resolve()
    work = Path(runtime["work_root"]).resolve()
    if work != namespace and namespace not in work.parents:
        raise ContractError("work_root_must_use_qwen3vl8b_namespace")
    minimum = runtime.get("gpu_min_free_mb", 0)
    if type(minimum) is not int or minimum < 0:
        raise ContractError("invalid_gpu_headroom")
    if set(runtime["python"]) != set(LIMITS) or set(runtime["gpu"]) != set(LIMITS):
        raise ContractError("four_runtime_entries_required")
    if any(not Path(p).is_absolute() for p in runtime["python"].values()):
        raise ContractError("python_paths_must_be_absolute")
    if runtime["gpu"]["alfred"] is not None or runtime["gpu"]["manipulation"] is not None:
        raise ContractError("software_rendering_gpu_must_be_null")
    if any(type(runtime["gpu"][k]) is not int or runtime["gpu"][k] not in (0, 1, 3, 7)
           for k in ("navigation", "habitat")):
        raise ContractError("gpu_outside_authorized_set")
    if (type(runtime["environment_seed"]) is not int
            or type(runtime["timeout_s"]) not in (int, float) or not 1 <= runtime["timeout_s"] <= 600):
        raise ContractError("invalid_runtime_budget")
    for field in ("n_shots", "max_requests"):
        if set(config[field]) != set(LIMITS) or any(type(v) is not int for v in config[field].values()):
            raise ContractError("four_integer_environment_budgets_required")
    if any(not 0 <= v <= {"navigation": 3, "alfred": 10, "habitat": 10, "manipulation": 4}[k]
           for k, v in config["n_shots"].items()):
        raise ContractError("unsupported_example_count")
    if any(not 1 <= v <= LIMITS[k] + (10 if k in ("alfred", "habitat") else 0)
           for k, v in config["max_requests"].items()):
        raise ContractError("invalid_request_budget")
    execution = config["execution"]
    if (execution != {"max_episodes": 1, "max_api_requests": 1, "max_gpu_episodes": 1,
                      "per_environment": {kind: 1 for kind in LIMITS}}):
        raise ContractError("qwen_trial_requires_serial_execution")
    tasks = []
    for row in config["episodes"]:
        if set(row) != {"environment", "subset", "index"}:
            raise ContractError("unexpected_episode_fields")
        tasks.append(Task(**row))
    if not tasks or len({t.identity for t in tasks}) != len(tasks):
        raise ContractError("empty_or_duplicate_selection")
    validate_detection(config["detection"])
    return config, tasks
