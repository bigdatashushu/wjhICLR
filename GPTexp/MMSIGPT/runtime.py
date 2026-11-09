"""MMSI context and tool exposure over the read-only shared kernel."""
from __future__ import annotations

import inspect

from skill3d.sandbox.kernel import (
    AnswerTerminate, ControlTerminate, RestrictedNamespaceKernel, ToolNamespace,
    YieldTerminate,
)
from skill3d.schemas import AnswerPayload

from .public_input import INTERFACE_VERSION, MMSIContext, freeze, public_input


class MMSIKernel(RestrictedNamespaceKernel):
    def __init__(self, *args, problem, execution_trace=None, adaptations=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._public_input = freeze(public_input(problem, self._ns["frames"]))
        self.execution_trace = execution_trace if execution_trace is not None else []
        self.interface_adaptations = adaptations if adaptations is not None else []
        self._ns["AnswerPayload"] = self._answer_payload
        self._refresh_tools()

    def _refresh_tools(self):
        scene = self._scene
        names = set(self._registry.names_for_scope(
            scene.question_tool_scope, scene.available_artifacts,
            evidence_profile=scene.evidence_profile,
            question_type=scene.question_type, local_facts=scene.local_facts,
        )) if self._tools_enabled else set()
        for name in self._registry.names():
            self._ns.pop(name, None)
        functions = {name: self._tool_functions[name] for name in names}
        self._ns["tools"] = ToolNamespace(functions)
        self._ns.update(functions)
        # This pipeline has no observation-reader protocol.
        self._ns.pop("ReadObservation", None)

    def _answer_payload(self, *args, **kwargs):
        basis = kwargs.get("basis")
        if isinstance(basis, str) and basis.startswith("visual_estimate:"):
            kwargs["basis"] = "visual_estimate"
            self.interface_adaptations.append({
                "interface_version": INTERFACE_VERSION,
                "stage": "answer_payload",
                "adaptations": ["visual_estimate_annotation"],
            })
        return AnswerPayload(*args, **kwargs)

    def _invoke_entry(self):
        if self.answer_slot.given or self.yield_slot.given:
            return None
        entry = self._ns.get("solve")
        if not callable(entry):
            return None
        ctx = MMSIContext(
            scene=self._ns["scene"], tools=self._ns["tools"], frames=self._ns["frames"],
            input=self._public_input, options=self._public_input["options"],
            question=self._public_input["question"],
        )
        try:
            params = [p for p in inspect.signature(entry).parameters.values()
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            entry(ctx) if params else entry()
        except AnswerTerminate:
            return "answer"
        except (YieldTerminate, ControlTerminate):
            return "yield"
        return None

    def reset_user_namespace(self):
        super().reset_user_namespace()
        self._refresh_tools()

    def set_tools_enabled(self, enabled):
        super().set_tools_enabled(enabled)
        self._refresh_tools()

    def retarget_scene(self, scene):
        super().retarget_scene(scene)
        self._refresh_tools()

    def run_cell(self, source, *args, **kwargs):
        from ..adapters import jsonable
        from .evidence import capture

        start = len(self.interface_adaptations)
        self._ns.pop("evidence_audit", None)
        cell = super().run_cell(source, *args, **kwargs)
        evidence = capture(
            self._ns.get("evidence_audit"), n_views=len(self._ns["frames"]),
            labels=tuple(row["label"] for row in self._public_input["options"]),
            answer=cell.answer,
        )
        self.execution_trace.append(jsonable({
            "program_source": source, "execution": cell,
            "tool_calls": self.cell_calls, "tool_results": self.cell_results,
            "authorization_receipts": self.authorization_receipts[
                self._cell_authorization_mark:],
            "observation_reads": self.observation_reads[self._cell_read_mark:],
            "contract_violations": self.contract_violations[self._cell_violation_mark:],
            "answer_payload": self.answer_slot.payload,
            "interface_adaptations": self.interface_adaptations[start:],
            "evidence_audit": evidence,
        }))
        return cell
