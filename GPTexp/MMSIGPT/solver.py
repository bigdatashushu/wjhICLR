"""MMSI dependency overrides without mutating a shared module or frozen code."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import hashlib
import json
from types import FunctionType

from skill3d.small_image.runner import SmallImageSolver

from .program import assemble_program
from .prompts import build_solver_prompt
from .public_input import PROMPT_VERSION
from .runtime import MMSIKernel
from .tools import TOOL_VERSION


class MMSISolver(SmallImageSolver):
    """Reuse the frozen budget and solve loop with per-call dependency bindings."""
    def __init__(self, *args, execution_trace=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_trace = execution_trace if execution_trace is not None else []
        self.interface_adaptations = []

    def solve(self, problem, **kwargs):
        self.execution_trace.clear()
        self.interface_adaptations.clear()
        original = SmallImageSolver.solve
        bindings = dict(original.__globals__)
        bindings.update(
            RestrictedNamespaceKernel=partial(
                MMSIKernel, problem=problem, execution_trace=self.execution_trace,
                adaptations=self.interface_adaptations),
            build_solver_prompt=build_solver_prompt,
            assemble_program=partial(assemble_program, adaptations=self.interface_adaptations),
        )
        solve = FunctionType(
            original.__code__, bindings, original.__name__,
            original.__defaults__, original.__closure__,
        )
        solve.__kwdefaults__ = original.__kwdefaults__
        value = replace(solve(self, problem, **kwargs), prompt_version=PROMPT_VERSION,
                        tool_version=TOOL_VERSION, receipt_sha256="")
        digest = hashlib.sha256(json.dumps(
            value.to_dict(), sort_keys=True, ensure_ascii=True, allow_nan=False,
            separators=(",", ":")).encode("ascii")).hexdigest()
        return replace(value, receipt_sha256=digest)
