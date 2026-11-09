"""Run real child processes; no models, perception services or simulators."""
from dataclasses import replace
import io
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from action_contract import (DecisionSnapshot, EpisodeBinding, ExecutionEnvelope,
                             public_observation)
from action_tools import ResultStore
from protocol import ContractError, sha256
from sandbox_process import DEFAULT_LIMITS, execute_isolated, minimal_environment, resident_memory
from sandbox_wire import pack_prepared, pack_snapshot, unpack_snapshot

pytestmark = pytest.mark.skipif(platform.system() not in {"Darwin", "Linux"}, reason="OS policy evidence")


def context():
    rgb = np.zeros((500, 500, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    binding = EpisodeBinding("fake", "episode", "task", "eb_navigation", "base", sha256(b"input"))
    obs = public_observation(
        binding, png=stream.getvalue(), instruction="Go to chair", environment_step=0,
        mutation_revision=1, observation_revision=0, action_table=["forward", "right"],
        public_feedback="Ready")
    snapshot = DecisionSnapshot(obs, "decision", "", sha256(b"budget"),
                                capabilities=("current_rgb", "relative_depth"))
    envelope = ExecutionEnvelope("episode", "session", 1, 0, "decision", snapshot.snapshot_sha256)
    prepared = SimpleNamespace(
        frames=[rgb], candidates=[], provenance={},
        geometries={0: {"depth": np.ones((500, 500), dtype=np.float32),
                        "valid_mask": np.ones((500, 500), dtype=bool), "points": None}})
    return binding, snapshot, envelope, prepared


SUBMIT = """def solve(ctx):
    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,
        ctx.snapshot_sha256, ctx.action_table_sha256, [0]))
"""


def run(source=SUBMIT, **kwargs):
    binding, snapshot, envelope, prepared = context()
    return execute_isolated(source, snapshot, binding, envelope, prepared=prepared, **kwargs)


def test_real_segment_submit_numeric_tools_and_sanitized_output():
    result = run("import numpy as np\nprint('synthetic-private-text')\n" + SUBMIT.replace(
        "    SubmitActions", "    r = point_depth(0, 5, 5)\n    assert r['data']['depth'] == 1\n    SubmitActions"))
    assert result.error_code is None, result
    assert result.submission.actions == (0,) and result.tool_calls == 1
    assert result.receipt["isolation_sealed"] and result.receipt["resources"]["memory_monitor_observed"]
    assert "synthetic-private-text" not in repr(result)


def test_public_context_factory_rechecks_hashes_and_ignores_no_extra_fields():
    binding, snapshot, _envelope, prepared = context()
    assert unpack_snapshot(pack_snapshot(snapshot, binding))[1] == snapshot
    value = pack_snapshot(snapshot, binding)
    value["snapshot"]["observation"]["reward"] = 1
    with pytest.raises(ContractError):
        unpack_snapshot(value)
    prepared.frames[0][1, 1] = 1
    with pytest.raises(ContractError):
        pack_prepared(snapshot, prepared)


@pytest.mark.parametrize("source", [
    "open('/private/tmp/sentinel')",
    "import socket\nsocket.socket()",
    "import subprocess\nsubprocess.run(['true'])",
    "import numpy as np\nnp.fromfile('/private/tmp/sentinel')",
    "import numpy as np\nnp.ctypeslib",
    "x = frames[0]\nx.setflags(write=True)",
])
def test_generated_file_network_native_and_frame_access_rejected(source):
    result = run(source + "\n" + SUBMIT)
    assert result.error_code and result.submission is None


def test_infinite_loop_and_output_limit_are_unrecoverable():
    result = run("while True:\n    pass", limits={**DEFAULT_LIMITS, "wall_s": 2, "cpu_s": 1})
    assert result.error_code == "timeout" and result.submission is None
    result = run("try:\n    print('x' * 100000)\nexcept Exception:\n    pass\n" + SUBMIT)
    assert result.error_code == "output_limit" and result.submission is None


def test_parent_memory_limit_terminates_numerical_allocation():
    result = run("import numpy as np\nx = np.ones((160000000,), dtype=np.uint8)\n" + SUBMIT,
                 limits={**DEFAULT_LIMITS, "memory_bytes": 150 * 1024**2})
    assert result.error_code == "oom" and result.submission is None
    assert result.receipt["resources"]["memory_monitor_observed"]


def test_tool_budget_reader_cap_and_candidate_surface_denial():
    result = run("for i in range(13):\n    point_depth(0,5,5)\n" + SUBMIT)
    assert result.error_code == "tool_budget" and result.submission is None
    assert result.tool_calls == 12 and len(result.receipt["tool_receipts"]) == 12
    result = run("candidate_depth_summary('candidate')\n" + SUBMIT)
    assert result.error_code == "tool_contract" and result.submission is None
    assert result.receipt["tool_receipts"][0]["allowed"] is False
    assert "args" not in result.receipt["tool_receipts"][0]
    result = run("r = point_depth(0,5,5)\nfor i in range(4):\n"
                 "    ReadObservation(r['result_id'], ['depth'])\n" + SUBMIT)
    assert result.error_code and result.submission is None and result.reader_calls == 3


def test_same_decision_yield_reader_and_previous_dependency_then_revoke():
    binding, snapshot, envelope, prepared = context()
    first = execute_isolated(
        "r = point_depth(0, 5, 5)\nYieldObservations([r['result_id']], 'inspect')",
        snapshot, binding, envelope, prepared=prepared)
    assert first.error_code is None, first
    result_id = first.yielded_result_ids[0]
    second = execute_isolated(
        f"page = ReadObservation('{result_id}', ['depth'])\n" + SUBMIT,
        snapshot, binding, envelope, prepared=prepared, previous_results=first.results)
    assert second.error_code is None and second.reader_calls == 1
    store = ResultStore(snapshot, second.results)
    assert store.authorize(result_id, snapshot)
    store.revoke()
    assert not store.authorize(result_id, snapshot)
    stale = replace(snapshot, decision_id="next")
    with pytest.raises(ContractError):
        ResultStore(stale, second.results)


def test_task_state_only_rebind_preserves_result_authorization_and_reader_cap():
    from copy import deepcopy
    from audit_action_trajectory import audit_result_contexts
    from action_contract import public_hash
    binding, snapshot, envelope, prepared = context()
    first = execute_isolated(
        "r = point_depth(0,5,5)\nYieldObservations([r['result_id']], 'inspect')",
        snapshot, binding, envelope, prepared=prepared)
    result_id = first.yielded_result_ids[0]
    store = ResultStore(snapshot, first.results, max_reads=1)
    page = store.read(result_id, ["depth"])
    updated = replace(snapshot, task_state={"goal": {"id": "find_chair"}})
    rebound = store.rebind_task_state(updated)
    assert store.revoked and not store.authorize(result_id, snapshot)
    assert rebound.authorize(result_id, updated) and not rebound.authorize(result_id, snapshot)
    assert rebound.records[result_id]["response"] == store.records[result_id]["response"]
    assert rebound.records[result_id]["authorization"] == store.records[result_id]["authorization"]
    assert rebound.records[result_id]["sha256"] != store.records[result_id]["sha256"]
    assert rebound.reads == [page]
    events = [
        {"kind": "decision", "snapshot": snapshot.public_dict()},
        {"kind": "execution", "results": list(store.records.values())},
        {"kind": "decision_context", "snapshot": updated.public_dict()},
        {"kind": "result_context", "previous_snapshot_sha256": snapshot.snapshot_sha256,
         "snapshot_sha256": updated.snapshot_sha256, "reason": "task_state_only",
         "results": list(rebound.records.values())},
    ]
    audit_result_contexts(events)
    tampered = deepcopy(events)
    row = tampered[-1]["results"][0]
    row["response"]["data"]["depth"] = 999
    row["sha256"] = public_hash({k: v for k, v in row.items() if k != "sha256"})
    with pytest.raises(ContractError, match="trajectory_result_rebind_receipt"):
        audit_result_contexts(tampered)
    with pytest.raises(ContractError, match="reader_cap"):
        rebound.read(result_id)
    with pytest.raises(ContractError, match="result_rebind_not_task_state_only"):
        store.rebind_task_state(updated)


@pytest.mark.parametrize("field,value", [
    ("decision_id", "next"), ("evidence_revision", 1),
    ("evidence_manifest_sha256", sha256(b"new evidence")),
    ("capabilities", ("current_rgb",)), ("budget_sha256", sha256(b"new budget")),
    ("observation", None),
])
def test_result_rebind_rejects_non_task_state_changes(field, value):
    _, snapshot, _, _ = context()
    store = ResultStore(snapshot)
    if field == "observation":
        value = replace(snapshot.observation, observation_revision=1)
    updated = replace(snapshot, task_state={"goal": {"id": "chair"}}, **{field: value})
    with pytest.raises(ContractError, match="result_rebind_not_task_state_only"):
        store.rebind_task_state(updated)
    assert not store.revoked


def test_task_state_rebind_revalidates_original_tool_rows():
    binding, snapshot, envelope, prepared = context()
    first = execute_isolated(
        "r = point_depth(0,5,5)\nYieldObservations([r['result_id']], 'inspect')",
        snapshot, binding, envelope, prepared=prepared)
    store = ResultStore(snapshot, first.results)
    store.records[first.yielded_result_ids[0]]["response"]["data"]["depth"] = 999
    with pytest.raises(ContractError, match="result_scope_mismatch"):
        store.rebind_task_state(replace(snapshot, task_state={"goal": {"id": "chair"}}))
    assert not store.revoked


def test_ast_rejection_preserves_prior_result_ledger_and_zero_caps_deny_new_access():
    binding, snapshot, envelope, prepared = context()
    first = execute_isolated(
        "r = point_depth(0,5,5)\nYieldObservations([r['result_id']], 'inspect')",
        snapshot, binding, envelope, prepared=prepared)
    broken = execute_isolated(
        "import os", snapshot, binding, envelope, prepared=prepared, previous_results=first.results)
    assert broken.error_code == "violation_syntax" and broken.results == first.results
    no_tools = execute_isolated(
        "point_depth(0,5,5)\n" + SUBMIT, snapshot, binding, envelope, prepared=prepared,
        previous_results=broken.results, limits={**DEFAULT_LIMITS, "tool_calls": 0, "reads": 0})
    assert no_tools.error_code == "tool_budget" and no_tools.tool_calls == 0
    assert no_tools.results == first.results and no_tools.submission is None
    submitted = execute_isolated(
        SUBMIT, snapshot, binding, envelope, prepared=prepared, previous_results=broken.results,
        limits={**DEFAULT_LIMITS, "tool_calls": 0, "reads": 0})
    assert submitted.error_code is None and submitted.results == first.results


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS seatbelt probe")
def test_minimal_environment_and_no_extra_fd_inheritance(tmp_path):
    assert not any("SECRET" in k or "TOKEN" in k or "PROXY" in k
                   for k in minimal_environment(tmp_path))
    # Independently exercise the actual second-stage OS seal through trusted
    # Python, bypassing the AST. Read an already-open sentinel FD as well.
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("synthetic")
    script = """import os, socket, subprocess, sys
from sandbox_worker import seal_macos
fd = os.open(sys.argv[1], os.O_RDONLY)
seal_macos()
checks = {}
for name, fn in [('file', lambda: open(sys.argv[1]).read()),
                 ('fd', lambda: os.read(fd, 20)),
                 ('write', lambda: open(sys.argv[1], 'w')),
                 ('network', lambda: socket.socket().connect(('127.0.0.1', 9))),
                 ('process', lambda: subprocess.run(['/usr/bin/true']))]:
    try:
        fn()
        checks[name] = False
    except OSError:
        checks[name] = True
print(checks)
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", script, str(sentinel)],
        env={**minimal_environment(tmp_path), "PYTHONPATH":
             str(Path(__file__).parents[1]) + ":" + str(Path(__file__).parents[2] / "src")},
        capture_output=True, text=True, timeout=5)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "{'file': True, 'fd': True, 'write': True, 'network': True, 'process': True}"
