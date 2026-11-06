"""Task-local execution controls and equivalence with unchanged QA primitives."""
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import io

import numpy as np
from PIL import Image
import pytest

from action_contract import (ActionPayload, EpisodeBinding, DecisionSnapshot, ExecutionEnvelope,
                             freeze, public_observation, validate_action_payload)
from action_runtime.ast_guard import ast_guard
from action_runtime.action_check import check_action
from action_runtime.control_profile import ActionContextView, EMBODIED_ACTIONS_PROFILE
from action_runtime.kernel import RestrictedNamespaceKernel
from action_runtime.segment import execute_segment
from action_runtime.delivery import plan_delivery, render_skill_entry
from action_runtime.frozen_method import FrozenMethod, load_frozen_method
from skill3d.small_image.tools import build_registry


def digest(data):
    return hashlib.sha256(data).hexdigest()


def geometry():
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    prepared = SimpleNamespace(
        frames=[frame], candidates=[], provenance={},
        geometries={0: {"depth": np.ones((500, 500), dtype=np.float32),
                        "valid_mask": np.ones((500, 500), dtype=bool), "points": None}})
    return prepared, SimpleNamespace(input_sha256=digest(b"input"), benchmark="fake")


def action_kernel():
    prepared, problem = geometry()
    registry, scene = build_registry(prepared, problem)
    stream = io.BytesIO()
    Image.fromarray(prepared.frames[0]).save(stream, format="PNG")
    binding = EpisodeBinding("run", "episode", "task", "eb_navigation", "base", problem.input_sha256)
    obs = public_observation(
        binding, png=stream.getvalue(), instruction="Find chair", environment_step=0,
        mutation_revision=1, observation_revision=0, action_table=["forward", "right"],
        public_feedback="Ready")
    snapshot = DecisionSnapshot(obs, "decision", "", digest(b"budget"))
    envelope = ExecutionEnvelope("episode", "session", 1, 0, "decision", snapshot.snapshot_sha256)
    ctx = ActionContextView(
        obs.episode_id, snapshot.decision_id, 0, snapshot.snapshot_sha256,
        obs.action_table_sha256, obs.action_table, obs.instruction, obs.public_feedback,
        freeze(obs.public_memory), snapshot.capabilities, 0, obs.view_id)
    kernel = RestrictedNamespaceKernel(
        registry, scene, frames=list(prepared.frames), episode_id="episode",
        profile=EMBODIED_ACTIONS_PROFILE, public_context=ctx, action_constructor=ActionPayload,
        action_validator=lambda payload: validate_action_payload(
            payload, snapshot, envelope, session_id="session", max_actions=20,
            actual_result_ids=[r.result_id for r in kernel.tool_results],
            authorize_result=lambda *_: True))
    return kernel


SUBMIT = """def solve(ctx):
    SubmitActions(ActionPayload(
        decision_id=ctx.decision_id, observation_revision=ctx.observation_revision,
        snapshot_sha256=ctx.snapshot_sha256, action_table_sha256=ctx.action_table_sha256,
        actions=[0], evidence_refs=[]))
    point_depth(0, 5, 5)
    SubmitActions(None)
"""


def test_submission_terminates_before_tools_or_second_submission():
    kernel = action_kernel()
    assert ast_guard(SUBMIT, set(kernel._registry.names()), profile=EMBODIED_ACTIONS_PROFILE).ok
    result = execute_segment(kernel, SUBMIT, program_id="program")
    assert result.cell.terminated == "actions" and result.cell.submission.actions == (0,)
    assert result.cell.answer is None and result.trace.calls == []


@pytest.mark.parametrize("source", [
    "ReturnAnswer('A')", "answer = AnswerPayload('A')", "ctx.reward",
    "ctx.scene", "SubmitActions = print", "ctx.snapshot_sha256 = 'other'",
    "import os", "import numpy as np\nnp.load('/private/tmp/sentinel')",
    "x = ctx.tools.step(0)", "x = (1).__class__",
    "import numpy as np\nnp.ctypeslib",
])
def test_answer_and_host_access_denied(source):
    assert not ast_guard(source, set(), profile=EMBODIED_ACTIONS_PROFILE).ok


def test_alias_cannot_rewrite_host_slot_and_frames_are_readonly():
    kernel = action_kernel()
    result = execute_segment(kernel, "submit = SubmitActions\nsubmit.given = True\n" + SUBMIT, program_id="p")
    assert result.cell.terminated == "actions" and result.cell.submission.actions == (0,)
    kernel.reset_user_namespace()
    broken = kernel.run_cell("frames[0][0, 0, 0] = 1\n" + SUBMIT)
    assert broken.error_code and broken.submission is None


@pytest.mark.parametrize("source", [
    "import numpy as np\nnp.lib.npyio",
    "import numpy as np\nalias = np\nalias.load('sentinel.npy')",
])
def test_numerical_facade_denies_hidden_apis_at_runtime(source):
    # Attribute aliases need runtime enforcement in addition to the AST guard.
    kernel = action_kernel()
    result = kernel.run_cell(source + "\n" + SUBMIT)
    assert result.error_type == "AttributeError" and result.submission is None


def test_reset_clears_user_variables_preserves_tool_ledger():
    kernel = action_kernel()
    first = execute_segment(kernel, "result = point_depth(0, 5, 5)\nYieldObservations([result['result_id']], 'inspect')",
                            program_id="a")
    assert first.cell.terminated == "yield" and len(first.trace.results) == 1
    second = execute_segment(kernel, SUBMIT, program_id="b", reset_namespace=True)
    assert "result" not in kernel._ns
    assert len(kernel.tool_results) == 1 and second.trace.results == []
    assert second.cell.terminated == "actions"


def test_task_local_qa_default_matches_original_execution_and_guard():
    from skill3d.sandbox.kernel import RestrictedNamespaceKernel as Original
    from skill3d.sandbox.ast_guard import ast_guard as original_guard
    prepared, problem = geometry()
    kernels = []
    source = "r = point_depth(0, 5, 5)\nReturnAnswer(r['data']['depth'])"
    for cls in (Original, RestrictedNamespaceKernel):
        registry, scene = build_registry(prepared, problem)
        kernels.append(cls(registry, scene, frames=list(prepared.frames), call_id_factory=lambda: "call"))
    cells = [k.run_cell(source) for k in kernels]
    assert [(c.answer, c.terminated, c.error_code, c.stdout) for c in cells] == [
        ("1", "answer", None, ""), ("1", "answer", None, "")]
    results = [k.cell_results[0].model_copy(update={"latency_ms": 0.0}).model_dump() for k in kernels]
    assert results[0] == results[1]
    assert kernels[0].authorization_receipts == kernels[1].authorization_receipts
    assert ast_guard(source).model_dump() == original_guard(source).model_dump()
    segment = execute_segment(kernels[1], "ReturnAnswer('A')", program_id="p", reset_namespace=True)
    assert segment.cell.answer == "A" and segment.trace.results == []
    assert not check_action("ReturnAnswer(1)", profile=EMBODIED_ACTIONS_PROFILE).ok
    with pytest.raises(ValueError):
        ast_guard(SUBMIT, set(), profile=replace(EMBODIED_ACTIONS_PROFILE))


def test_frozen_method_full_delivery_cap_failure_and_path(tmp_path):
    text = "---\nmethod_id: navigation\n---\nUse current RGB and official feedback.\n"
    method = FrozenMethod(
        "navigation", "1.0.0", "eb_navigation", "embodied-actions/1", ("current_rgb",), (),
        ("point_depth",), text, digest(text.encode()), digest(b"manifest"), "navigation/SKILL.md")
    delivery = plan_delivery([method], max_chars=1000)
    assert delivery.entries[0].text == text == render_skill_entry(method)
    delivery.mark_request_failed()
    assert delivery.delivered_content_sha256 == {}
    delivery.mark_delivered()
    assert delivery.delivered_content_sha256 == {"navigation@1.0.0": digest(text.encode())}
    with pytest.raises(ValueError):
        plan_delivery([method], max_chars=10)
    with pytest.raises(ValueError):
        replace(method, skill_md=text + "different\n")
    metadata = {"method_id": method.skill_id, "version": method.version,
                "environment": method.environment, "control_profile": method.control_profile,
                "required_capabilities": list(method.required_capabilities),
                "optional_capabilities": [], "tools": list(method.tools),
                "content_sha256": method.content_sha256}
    with pytest.raises(ValueError):
        load_frozen_method(tmp_path, source_ref="../outside.md", metadata=metadata,
                           manifest_sha256=method.manifest_sha256)
