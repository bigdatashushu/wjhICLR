"""Bind only solver model identity while preserving the original adapter bytes."""
from __future__ import annotations

from copy import deepcopy

from . import adapters
from .traces import digest, write


def bound_spec(spec, model):
    if model != "gpt-5.6-terra":
        raise ValueError("unexpected_terra_binding_model")
    bound = deepcopy(spec)
    changed = []
    for field, key in (("solver", "expected_model"), ("config", "model")):
        if field in bound:
            original = bound[field].get(key)
            if original not in {"gpt-6.1-sol", model}:
                raise ValueError("unexpected_original_model_binding")
            bound[field][key] = model
            if original != model:
                changed.append(field + "." + key)
    restored = deepcopy(bound)
    for field, key in (("solver", "expected_model"), ("config", "model")):
        if field in spec:
            if key in spec[field]:
                restored[field][key] = spec[field][key]
            else:
                restored[field].pop(key, None)
    if restored != spec:
        raise ValueError("non_model_solver_field_changed")
    return bound, changed


def mind_config(value, model):
    from MindcubeRunner.common import Config

    class TerraConfig(Config):
        def __post_init__(self):
            actual = self.model
            if actual != model:
                raise ValueError("unexpected_terra_model")
            object.__setattr__(self, "model", "qwen3.8-flash")
            try:
                super().__post_init__()
            finally:
                object.__setattr__(self, "model", actual)

    # The original adapter uses the same base-config compatibility step and timeout.
    return TerraConfig(**{**value, "model": model, "timeout_s": 36000})


def solve(spec, dataset, client, output, run_identity):
    bound, changed = bound_spec(spec, client.model)
    write(output / "model_binding.json", {
        "schema": "gptexp-runtime-model-binding/1", "model": client.model,
        "original_spec_sha256": digest(spec), "bound_spec_sha256": digest(bound),
        "changed_fields": changed, "non_model_fields_identical": True,
        "run_identity": run_identity, "reference_answers_read": False,
        "adapter_bytes_unchanged": True, "worker_bytes_unchanged": True}, exclusive=True)
    original = adapters.mind_config
    try:
        if dataset == "mindcube-tiny":
            adapters.mind_config = lambda value: mind_config(value, client.model)
        return adapters.solve(bound, dataset, client, output, run_identity)
    finally:
        adapters.mind_config = original
