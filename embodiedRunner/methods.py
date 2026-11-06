"""Load one complete environment method from the task-local frozen library."""
from pathlib import Path

from action_contract import public_hash
from action_runtime.delivery import plan_delivery
from action_runtime.execution_contract import ActionExecutionContract
from action_runtime.frozen_method import load_frozen_method
from action_tools import TOOL_NAMES
from protocol import ContractError, json_object

ROOT = Path(__file__).resolve().parent / "methods"


def load_methods(root=ROOT):
    root = Path(root)
    manifest = json_object((root / "manifest.json").read_text(encoding="utf-8"))
    if (set(manifest) != {"schema", "methods"}
            or manifest["schema"] != "embodied-method-library/1"
            or not isinstance(manifest["methods"], list) or len(manifest["methods"]) != 4):
        raise ContractError("method_manifest_shape")
    digest = public_hash(manifest)
    methods = {}
    for row in manifest["methods"]:
        metadata = {key: value for key, value in row.items() if key != "source_ref"}
        method = load_frozen_method(
            root, source_ref=row["source_ref"], metadata=metadata, manifest_sha256=digest)
        if (method.environment in methods or set(method.tools) != TOOL_NAMES
                or method.required_capabilities != ("current_rgb",)
                or set(method.optional_capabilities) != {
                    "candidate_category", "relative_depth", "relative_points"}):
            raise ContractError("method_environment_or_capabilities")
        plan_delivery([method], max_chars=ActionExecutionContract().method_context_max_chars)
        methods[method.environment] = method
    return methods
