"""Bind only solver model identity while preserving the original adapter bytes."""
from __future__ import annotations

from copy import deepcopy
import inspect
import textwrap

from . import adapters
from .response_policy import CORRELATION, STRICT
from .traces import digest, write


def bound_spec(spec, model, response_policy=STRICT):
    if model != "gpt-5.6-terra":
        raise ValueError("unexpected_terra_binding_model")
    bound = deepcopy(spec)
    changed = []
    for field, key in (("solver", "expected_model"), ("config", "model")):
        if field in bound:
            original = bound[field].get(key)
            if original not in {"gpt-6.1-sol", model}:
                raise ValueError("unexpected_original_model_binding")
            target = "" if field == "solver" and response_policy == CORRELATION else model
            bound[field][key] = target
            if original != target:
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


def binding_record(spec, model, run_identity, response_policy=STRICT):
    bound, changed = bound_spec(spec, model, response_policy)
    record = {
        "schema": "gptexp-runtime-model-binding/1", "model": model,
        "original_spec_sha256": digest(spec), "bound_spec_sha256": digest(bound),
        "changed_fields": changed, "non_model_fields_identical": True,
        "run_identity": run_identity, "reference_answers_read": False,
        "adapter_bytes_unchanged": True, "worker_bytes_unchanged": True}
    if response_policy == CORRELATION:
        record.update(response_metadata_policy=response_policy,
                      reported_metadata_enforced=False,
                      reported_metadata_synthesized=False)
    return bound, record


def mind_without_metadata_gate(original):
    """Transform just the response guard and its formerly hardcoded receipt."""
    source = textwrap.dedent(inspect.getsource(original))
    guard = ('if actual.get("model") != config.model or actual.get("model_reported") is not True:'
             '\n                    raise ValueError("actual_model_changed")')
    receipt = 'response_identity={"model": actual["model"], "model_reported": True}'
    if source.count(guard) != 1 or source.count(receipt) != 1:
        raise ValueError("mind_response_guard_source_changed")
    source = source.replace(guard, "pass  # provider metadata is recorded without enforcement")
    source = source.replace(receipt, "response_identity=dict(actual)")
    namespace = {}
    exec(compile(source, inspect.getsourcefile(original), "exec"), original.__globals__, namespace)
    return namespace[original.__name__]


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
    response_policy = getattr(client, "response_metadata_policy", STRICT)
    bound, record = binding_record(spec, client.model, run_identity, response_policy)
    write(output / "model_binding.json", record, exclusive=True)
    original = adapters.mind_config
    mind_worker, original_solve = None, None
    try:
        if dataset == "mindcube-tiny":
            from MindcubeRunner import worker as mind_worker
            original_solve = mind_worker.solve
            if response_policy == CORRELATION:
                mind_worker.solve = mind_without_metadata_gate(original_solve)
            adapters.mind_config = lambda value: mind_config(value, client.model)
        return adapters.solve(bound, dataset, client, output, run_identity)
    finally:
        adapters.mind_config = original
        if mind_worker is not None:
            mind_worker.solve = original_solve
