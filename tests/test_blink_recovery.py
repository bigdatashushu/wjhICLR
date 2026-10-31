"""Offline recovery checks use real request recording, accounting and identities."""
from copy import deepcopy
import json
import shutil

import pytest

from skill3d.evaluation import blink_experiment as experiment
from skill3d.evaluation import blink_recovery as recovery
from skill3d.synthesis.client_errors import ServiceUnavailable
from test_blink_experiment import fake_client, fixture_campaign


def stopped_parent(tmp_path, monkeypatch):
    # Generate an actual prefix through the campaign owner and worker. Only
    # process dispatch and the provider are mocked; no GT is opened.
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-placeholder")
    campaign, _ = fixture_campaign(tmp_path, monkeypatch)
    body = deepcopy(campaign.manifest)
    body.pop("content_sha256")
    body["profile"]["model"]["timeout_s"] = 180
    body["profile"]["runtime"]["max_requests"] = 1341
    body["git_commit"] = "a"*40
    (campaign.root/"manifest.json").write_text(json.dumps(experiment.seal(body)))
    state_path = campaign.root/"budget/state.json"
    state = json.loads(state_path.read_text())
    assert state["requests"] == []
    state["policy"]["max_requests"] = 1341
    state_path.write_text(json.dumps(state))
    campaign = experiment.Campaign(campaign.root, experiment.read(campaign.root/"manifest.json"))
    (campaign.root/"operator.lock").touch()
    base = fake_client(['{"answer":"B"}']*24+[TimeoutError("test-only-private-detail")])
    base.timeout_s = 180
    monkeypatch.setattr(experiment, "BailianClient", lambda **kwargs: base)

    class LocalPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, worker, spec):
            class Future:
                def result(self):
                    return worker(spec)
            return Future()

    monkeypatch.setattr(experiment, "ProcessPoolExecutor", LocalPool)
    for p in campaign.tests[:24]:
        campaign.operation(p, "", "direct", mode="direct")
    with pytest.raises(ServiceUnavailable):
        campaign.operation(campaign.tests[24], "", "direct", mode="direct")
    repo = tmp_path/"repair-source"
    shutil.copytree(campaign.root/"code/source", repo)
    (repo/"src/skill3d/evaluation").mkdir()
    (repo/"src/skill3d/evaluation/blink_recovery.py").write_text("# repair fixture\n")
    original_command = recovery.subprocess.check_output

    def git_identity(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)]:
            return "a"*40+"\n" if "rev-parse" in command else "?? src/skill3d/evaluation/blink_recovery.py\n"
        return original_command(command, **kwargs)

    monkeypatch.setattr(recovery.subprocess, "check_output", git_identity)
    return campaign, base, repo


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_recovery_preserves_parent_prefix_budget_and_24_outputs_and_retries_only_failed(
        tmp_path, monkeypatch):
    parent, base, repo = stopped_parent(tmp_path, monkeypatch)
    before = snapshot(parent.root)
    plan = recovery.inspect(parent.root)
    assert plan["model_requests"] == 0 and not plan["reference_answers_read"]
    assert len(plan["reused_direct_operations"]) == 24
    assert plan["remaining_requests"] == 1316
    child = tmp_path/"recovery-run"
    manifest = recovery.prepare(repo, parent.root, child, authorized_failure_sha256=
                                plan["authorized_failure_result_sha256"])
    assert snapshot(parent.root) == before
    assert (child/"budget/state.json").read_bytes() == (parent.root/"budget/state.json").read_bytes()
    assert manifest["content_sha256"] != parent.manifest["content_sha256"]
    assert manifest["profile"]["model"]["timeout_s"] == 600
    assert manifest["recovery"]["repair_source_uncommitted"] is True
    assert len(base.captured) == 25
    campaign = experiment.Campaign(child, manifest)
    campaign.verify()
    for p in campaign.tests[:24]:
        row = campaign.operation(p, "", "direct", mode="direct")
        assert row["identity"]["campaign"] == parent.manifest["content_sha256"]
    assert len(base.captured) == 25
    replacement = fake_client(['{"answer":"B"}'])
    replacement.timeout_s = 600
    monkeypatch.setattr(experiment, "BailianClient", lambda **kwargs: replacement)
    retried = campaign.operation(campaign.tests[24], "", "direct", mode="direct")
    assert retried["status"] == "complete" and len(replacement.captured) == 1
    assert retried["campaign"] == manifest["content_sha256"]
    assert retried["model"]["timeout_s"] == 600
    assert campaign.operation(campaign.tests[24], "", "direct", mode="direct") == retried
    assert len(replacement.captured) == 1
    usage = campaign.governor.summary()
    assert usage["reserved_requests"] == 26 and usage["error_requests"] == 1
    assert usage["policy"]["max_requests"] == 1341
    state = json.loads((child/"budget/state.json").read_text())
    assert state["requests"][:25] == json.loads(before["budget/state.json"])["requests"]
    assert snapshot(parent.root) == before


@pytest.mark.parametrize("violation", ["authorization", "source_scope", "output_in_parent"])
def test_preparation_rejects_unapproved_changes_before_writing_child(
        tmp_path, monkeypatch, violation):
    parent, _, repo = stopped_parent(tmp_path, monkeypatch)
    plan = recovery.inspect(parent.root)
    before = snapshot(parent.root)
    child = tmp_path/"recovery-run"
    authorization = plan["authorized_failure_result_sha256"]
    if violation == "authorization":
        authorization = "0"*64
    elif violation == "source_scope":
        (repo/"src/skill3d/minimal.py").write_text("# unapproved solver change\n")
    else:
        child = parent.root/"child"
    with pytest.raises(ValueError):
        recovery.prepare(repo, parent.root, child, authorized_failure_sha256=authorization)
    assert not child.exists() and snapshot(parent.root) == before


@pytest.mark.parametrize("violation", ["budget_reset", "erased_error", "new_error", "changed_origin"])
def test_recovery_refuses_budget_or_lineage_changes(tmp_path, monkeypatch, violation):
    parent, _, repo = stopped_parent(tmp_path, monkeypatch)
    plan = recovery.inspect(parent.root)
    child = tmp_path/"recovery-run"
    manifest = recovery.prepare(repo, parent.root, child, authorized_failure_sha256=
                                plan["authorized_failure_result_sha256"])
    campaign = experiment.Campaign(child, manifest)
    if violation == "changed_origin":
        (child/"origin/budget/state.json").write_text("{}")
    else:
        state_path = child/"budget/state.json"
        state = json.loads(state_path.read_text())
        if violation == "budget_reset":
            state["requests"] = []
        elif violation == "erased_error":
            state["requests"][24]["status"] = "completed"
        else:
            row = {**state["requests"][24], "request_id": 25}
            state["requests"].append(row)
        state_path.write_text(json.dumps(state))
    with pytest.raises((ValueError, RuntimeError)):
        campaign.verify()


def test_failed_recovery_request_stops_without_automatic_retry(tmp_path, monkeypatch):
    parent, _, repo = stopped_parent(tmp_path, monkeypatch)
    plan = recovery.inspect(parent.root)
    child = tmp_path/"recovery-run"
    manifest = recovery.prepare(repo, parent.root, child, authorized_failure_sha256=
                                plan["authorized_failure_result_sha256"])
    campaign = experiment.Campaign(child, manifest)
    replacement = fake_client([TimeoutError("test-only-private-detail")])
    replacement.timeout_s = 600
    monkeypatch.setattr(experiment, "BailianClient", lambda **kwargs: replacement)
    with pytest.raises(ServiceUnavailable):
        campaign.operation(campaign.tests[24], "", "direct", mode="direct")
    with pytest.raises(RuntimeError, match="unresolved_or_failed_prior_requests"):
        campaign.operation(campaign.tests[24], "", "direct", mode="direct")
    assert len(replacement.captured) == 1
    assert campaign.governor.summary()["reserved_requests"] == 26
    assert campaign.governor.summary()["error_requests"] == 2
