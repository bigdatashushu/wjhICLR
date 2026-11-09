"""Small explicit configuration; endpoint and credentials are environment-only."""
import json
import re
from pathlib import Path

from .protocol import Task
from .sources import ContractError, LIMITS


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"schema", "model", "runtime", "episodes", "n_shots", "max_requests", "detection"}
    if not required <= set(config) or not set(config) <= required | {"execution"}:
        raise ContractError("unexpected_config_fields")
    if config["schema"] != "target-detection-config/1":
        raise ContractError("invalid_config_version")
    model = config["model"]
    if set(model) != {"name", "api_key_env", "base_url_env", "response_format",
                      "max_tokens", "temperature", "timeout_s", "extra_body"}:
        raise ContractError("unexpected_model_config")
    for key in ("api_key_env", "base_url_env"):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+", model[key]):
            raise ContractError("invalid_model_environment_name")
    if not isinstance(model["name"], str) or not model["name"]:
        raise ContractError("invalid_model_name")
    if model["response_format"] not in ("official_schema", "json_object"):
        raise ContractError("invalid_response_format")
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
    runtime_required = {"transport", "ssh_target", "code_root", "source_root", "work_root",
                        "python", "gpu", "timeout_s", "environment_seed"}
    if not runtime_required <= set(runtime) or not set(runtime) <= runtime_required | {
            "manipulation_catalog", "manipulation_dataset", "gpu_min_free_mb"}:
        raise ContractError("unexpected_runtime_config")
    for key in ("manipulation_catalog", "manipulation_dataset"):
        if key in runtime and not Path(runtime[key]).is_absolute():
            raise ContractError("asset_paths_must_be_absolute")
    minimum = runtime.get("gpu_min_free_mb", 0)
    if type(minimum) is not int or minimum < 0:
        raise ContractError("invalid_gpu_headroom")
    execution = config.get("execution", {
        "max_episodes": 1, "max_api_requests": 1, "max_gpu_episodes": 1,
        "per_environment": {kind: 1 for kind in LIMITS}})
    if set(execution) != {"max_episodes", "max_api_requests", "max_gpu_episodes", "per_environment"}:
        raise ContractError("unexpected_execution_config")
    if any(type(execution[key]) is not int or not 1 <= execution[key] <= 32
           for key in ("max_episodes", "max_api_requests", "max_gpu_episodes")):
        raise ContractError("invalid_concurrency")
    if (set(execution["per_environment"]) != set(LIMITS)
            or any(type(v) is not int or not 1 <= v <= execution["max_episodes"]
                   for v in execution["per_environment"].values())):
        raise ContractError("invalid_environment_concurrency")
    if runtime["transport"] not in ("local", "ssh") or not re.fullmatch(
            r"[A-Za-z0-9_.@-]+", runtime["ssh_target"]):
        raise ContractError("invalid_transport")
    for key in ("code_root", "source_root", "work_root"):
        if not Path(runtime[key]).is_absolute():
            raise ContractError("runtime_paths_must_be_absolute")
    # New writable state must stay under the independent package.
    if Path(runtime["code_root"]) / "target_detection_v2" not in Path(runtime["work_root"]).parents:
        raise ContractError("work_root_must_be_under_target_detection_v2")
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
        if set(config[field]) != set(LIMITS):
            raise ContractError("four_environment_budgets_required")
        if any(type(v) is not int for v in config[field].values()):
            raise ContractError("integer_budget_required")
    if any(not 0 <= v <= {"navigation": 3, "alfred": 10, "habitat": 10, "manipulation": 4}[k]
           for k, v in config["n_shots"].items()):
        raise ContractError("unsupported_example_count")
    if any(not 1 <= v <= LIMITS[k] + (10 if k in ("alfred", "habitat") else 0)
           for k, v in config["max_requests"].items()):
        raise ContractError("invalid_request_budget")
    tasks = []
    for row in config["episodes"]:
        if set(row) != {"environment", "subset", "index"}:
            raise ContractError("unexpected_episode_fields")
        tasks.append(Task(**row))
    if not tasks or len({t.identity for t in tasks}) != len(tasks):
        raise ContractError("empty_or_duplicate_selection")
    if model["response_format"] != "json_object":
        raise ContractError("target_tools_require_json_object_mode")
    detection = config["detection"]
    if detection != {"box_threshold": 0.15, "group_iou": 0.8, "page_size": 12,
                     "max_crop_groups": 2,
                     "model_sha256": "7788da2cb34aac2b766c112ea0ec4b3ea3c8010b48cc55224b39e2444ac2264e"}:
        raise ContractError("unsupported_detection_configuration")
    if execution["max_episodes"] != 1:
        raise ContractError("trial_requires_serial_environments")
    return config, tasks
