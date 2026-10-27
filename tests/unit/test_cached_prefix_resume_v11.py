"""Synthetic-only interrupted solver continuation, with all real transport trapped."""

from copy import copy, deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_cached_episode_recovery_v11 as fixtures
from skill3d.evaluation import fitset_v11 as fitset
from skill3d.evaluation.cached_episode_recovery_v11 import (
    PREFIX_RESUME_SCHEMA, CachedPrefixRecoverySpec, CachedPrefixResumeClient,
    RecoveryStop, evidence_sha256, open_cached_episode_recovery,
    recovery_accounting, resumed_seeded_gate_client,
)
from skill3d.schemas import SkillSpecV11
from skill3d.online.runner import OnlineRunConfig
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import describe_request, payload_bytes
from skill3d.synthesis.request_context import request_phase
from skill3d.synthesis.request_governor import GovernedClient, RequestGovernor, RequestGovernorConfig

_real_governor_request = RequestGovernor.request


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload_bytes(value))


class SyntheticDeployment:
    def __init__(self, root, run_identity):
        self.physical_root = root
        self.logical_root = root.parent / "original-logical-system"
        self.deployment_id = "synthetic-deployment"
        self.path = root / "deployments" / self.deployment_id / "continuation.json"
        self.raw = b"synthetic immutable deployment"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.raw)
        self.identity_sha256 = fixtures.sha(self.raw)
        self.value = {"run_identity": run_identity, "preserved_files": {}}

    def physical_path(self, path):
        return self.physical_root / Path(path).relative_to(self.logical_root)

    def logical_path(self, path):
        return str(self.logical_root / Path(path).relative_to(self.physical_root))

    def seal(self, path):
        path.chmod(0o600)
        raw = path.read_bytes()
        self.value["preserved_files"][path.relative_to(self.physical_root).as_posix()] = {
            "identity": {"sha256": fixtures.sha(raw), "size": len(raw)}, "archive_ref": None,
        }


@pytest.fixture
def package(monkeypatch):
    helper = fixtures.CachedEpisodeRecoveryTests()
    helper.setUp()
    try:
        monkeypatch.setattr(RequestGovernor, "request", _real_governor_request)
        root = helper.root.resolve() / "system"
        root.mkdir()
        skill = SkillSpecV11(
            skill_id="synthetic-skill", version="1.0.0", question_type="object_counting",
            evidence_state="g0m0", skill_md="---\nname: synthetic-skill\n"
            "description: Synthetic test method.\n---\nSynthetic method.\n")
        helper.skills = [skill.model_dump(mode="json")]
        helper.dispatch["skills_sha256"] = evidence_sha256(helper.skills)
        deployment = SyntheticDeployment(root, helper.dispatch["run_identity"])
        variant = {
            "variant_id": helper.dispatch["variant_id"], "slot_key": helper.dispatch["slot_key"],
            "qa_id": "synthetic", "question_type": "object_counting",
            "scene_id": "synthetic-scene", "evidence_state": "g0m0", "split": "induction",
        }
        key = fitset._variant_key(variant["variant_id"])
        original = root / "variants" / key / "trace"
        original.mkdir(parents=True)
        (original / "original.jsonl").write_bytes(b"synthetic original trace\n")
        plan = {"schema_version": "synthetic-plan", "variant": variant,
                "skill_identity": helper.dispatch["skill_identity"]}
        write(original.parent / "plan.json", plan)
        write(root / "dispatch" / (key + ".json"), helper.dispatch)
        base_cfg = OnlineRunConfig(seed=138)
        context = deepcopy(helper.context)
        context.update(
            shared_input_sha256=evidence_sha256({
                k: v for k, v in plan.items() if k != "skill_identity"}),
            solver_config_sha256=evidence_sha256(fitset._json_value(asdict(base_cfg))),
            scheduler=fitset.SINGLE_PROCESS_SCHEDULER,
            skill_content_sha256=[skill.content_sha256])
        calls = []
        for index, old in enumerate(helper.calls):
            seed_record = deepcopy(old.seed_record)
            seed_record["frozen_context"] = deepcopy(context)
            seed_record["request_context"]["delivered_method_hashes"] = [skill.content_sha256]
            governor_record = deepcopy(old.governor_record)
            governor_record["request_id"] = index
            approval = deepcopy(old.approval)
            approval.update(seed_record_sha256=evidence_sha256(seed_record),
                            governor_record_sha256=evidence_sha256(governor_record))
            call = replace(old, seed_record=seed_record, governor_record=governor_record,
                           approval=approval)
            calls.append(call)
            write(root / "request_seeds" / f"{key}-{index + 1:03d}.json", seed_record)
            if call.response_record is not None:
                write(root / "request_seeds" / f"{key}-response-{index + 1:03d}.json",
                      call.response_record)
        policy = RequestGovernorConfig(max_requests=200, max_inflight=1)
        state = {**deepcopy(helper.state), "policy": asdict(policy),
                 "requests": [deepcopy(call.governor_record) for call in calls]}
        # Exercise a shared root.parent ledger rather than silently using root/request_governor.
        ledger_root = root.parent / "shared-governor"
        deployment.request_governor_root = ledger_root
        write(ledger_root / "state.json", state)
        private = deployment.path.parent / "private-recovery"
        private.mkdir()
        database = private / "replies.sqlite3"
        database.write_bytes(helper.database.read_bytes())
        custody_client = open_cached_episode_recovery(
            frozen_dispatch=helper.dispatch, resumed_dispatch=helper.dispatch,
            frozen_context=context, resumed_context=context, skills=helper.skills,
            model_identity=helper.model, calls=calls, original_trace_dir=original,
            continuation_root=private / "synthetic-custody", cache_path=database)
        custody = custody_client.recover_original_responses()
        custody_path = private / "response_recovery_receipt.json"
        write(custody_path, custody)
        value = {
            "schema_version": PREFIX_RESUME_SCHEMA, "recovery_id": "synthetic-prefix",
            "deployment_id": deployment.deployment_id,
            "original_trace_dir": deployment.logical_path(original),
            "original_plan_sha256": fixtures.sha((original.parent / "plan.json").read_bytes()),
            "frozen_dispatch": helper.dispatch, "frozen_context": context,
            "model_identity": helper.model, "calls": [asdict(call) for call in calls],
            "cache_path": str(database),
            "custody_receipt_ref": custody_path.relative_to(root).as_posix(),
        }
        value["identity_sha256"] = evidence_sha256(value)
        spec_path = private / "cached_prefix_resume.json"
        write(spec_path, value)
        for path in (spec_path, custody_path, database):
            deployment.seal(path)
        spec = CachedPrefixRecoverySpec(
            spec_path, continuation=deployment, system_root=root,
            run_identity=helper.dispatch["run_identity"])
        frozen_bytes = {path: path.read_bytes() for folder in (
            original.parent, root / "dispatch", root / "request_seeds", private)
            for path in folder.rglob("*") if path.is_file()}
        p = SimpleNamespace(
            helper=helper, root=root, original=original, spec=spec, deployment=deployment,
            plan=plan, variant=variant, base_cfg=base_cfg, context=context, calls=calls,
            state=state, ledger=ledger_root / "state.json", policy=policy, frozen=frozen_bytes,
            constructed=[], normal_calls=[], fail=False, new_count=1)
        yield p
        for path, raw in p.frozen.items():
            assert path.read_bytes() == raw
        live = json.loads(p.ledger.read_bytes())
        assert live["requests"][:4] == state["requests"]
    finally:
        helper.tearDown()
        helper.doCleanups()


class SyntheticNormalClient:
    """Pure payload codec and deterministic response, never a model constructor."""

    def __init__(self, package):
        self.package = package
        package.constructed.append(self)
        self.model = package.helper.model["model"]
        self.temperature = 0.0
        self.max_output_tokens = 256
        self.last_usage = {}
        self.last_request_identity = {}
        self.last_response_identity = {}
        self.last_generation = {}

    def build_payload(self, *args, **kwargs):
        return BailianClient.build_payload(self, *args, **kwargs)

    def audit_identity(self, *, seed=None):
        return package_identity(self.package, seed)

    def chat(self, *args, **kwargs):
        self.package.normal_calls.append(kwargs["seed"])
        if self.package.fail:
            raise RuntimeError("synthetic transport failure")
        self.last_request_identity = describe_request(
            self.build_payload(*args, **kwargs),
            model_identity=self.audit_identity(seed=kwargs["seed"]))
        self.last_usage = {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}
        self.last_response_identity = {"id": "synthetic-new-response"}
        return "synthetic continuation response"


def package_identity(package, seed):
    return {**package.helper.model, "seed": seed}


class SyntheticFactory:
    def __init__(self, package):
        self.package = package

    def __call__(self, _variant_id):
        pytest.fail("normal factory must never construct a client for cached calls")

    def resume_after_cached_prefix(self, _variant_id, *, calls):
        return resumed_seeded_gate_client(
            SyntheticNormalClient(self.package), seed=138,
            audit_path=self.package.spec.audit_path(), calls=calls)


def open_resume(package, io=None):
    spec = package.spec
    spec.output_root.parent.mkdir(exist_ok=True)
    cached = open_cached_episode_recovery(
        frozen_dispatch=package.helper.dispatch, resumed_dispatch=package.helper.dispatch,
        frozen_context=package.context, resumed_context=package.context,
        skills=package.helper.skills, model_identity=package.helper.model, calls=package.calls,
        original_trace_dir=package.original, continuation_root=spec.output_root,
        cache_path=spec.cache_path, framework_state_io=io)

    def delegate():
        normal = SyntheticFactory(package).resume_after_cached_prefix(
            spec.variant_id, calls=package.calls)
        return GovernedClient(normal, RequestGovernor(package.ledger.parent, package.policy),
                              category="solver", context={
                                  "variant_id": spec.variant_id,
                                  "run_identity": package.helper.dispatch["run_identity"]})

    client = CachedPrefixResumeClient(
        cached, spec=spec, delegate_factory=delegate, ledger_path=package.ledger)
    client.set_audit_context(package.context)
    return client


def replay(package, client, n=4):
    for index in range(n):
        client.set_request_context(package.calls[index].seed_record["request_context"])
        with request_phase("program"):
            text = client.chat(package.helper.messages[index], max_tokens=4096, seed=137)
        assert text == package.helper.results[index]["content"]
        assert not package.constructed
        assert json.loads(package.ledger.read_bytes()) == package.state


def next_call(package, client):
    context = {**deepcopy(package.calls[-1].seed_record["request_context"]), "round_index": 5}
    client.set_request_context(context)
    with request_phase("program"):
        return client.chat([{"role": "user", "content": "synthetic new request"}],
                           max_tokens=4096, seed=137)


def test_four_cached_calls_then_only_request_five_is_seeded_and_governed(package):
    client = open_resume(package)
    replay(package, client)
    assert next_call(package, client) == "synthetic continuation response"
    assert len(package.constructed) == 1
    assert package.normal_calls == [138]
    seed = json.loads(package.spec.audit_path().with_name(
        package.spec.audit_path().stem + "-005.json").read_bytes())
    assert seed["request_index"] == 5 and seed["request_seed"] == 138
    assert seed["retry_of"] is None and seed["frozen_context"] == package.context
    assert len(json.loads(package.ledger.read_bytes())["requests"]) == 5
    client.require_execution_continued()


@pytest.mark.parametrize("index", range(4))
def test_any_prefix_payload_mismatch_is_terminal_without_delegate(package, index):
    client = open_resume(package)
    replay(package, client, n=index)
    client.set_request_context(package.calls[index].seed_record["request_context"])
    with request_phase("program"), pytest.raises(RecoveryStop, match="request_hash_mismatch"):
        client.chat([{"role": "user", "content": "changed synthetic prompt"}], max_tokens=4096)
    with pytest.raises(RecoveryStop, match="prefix_resume_closed"):
        next_call(package, client)
    assert not package.constructed


@pytest.mark.parametrize("n", [0, 2, 4])
def test_incomplete_or_unexpected_early_finish_never_completes_episode(package, n):
    client = open_resume(package)
    replay(package, client, n=n)
    with pytest.raises(RecoveryStop, match=(
            "unexpected_early_finish" if n == 4 else "prefix_replay_incomplete")):
        client.require_execution_continued()
    assert not (package.spec.output_root / "continuation_receipt.json").exists()
    assert not (package.spec.output_root / "result.json").exists()
    assert not package.constructed
    with pytest.raises(RecoveryStop):
        next_call(package, client)


@pytest.mark.parametrize("filename", ["-005.json", "-response-005.json", "-006.json"])
def test_existing_future_seed_or_pending_response_refuses_before_construction(package, filename):
    path = package.spec.audit_path().with_name(package.spec.audit_path().stem + filename)
    write(path, {"synthetic_pending": True})
    client = open_resume(package)
    replay(package, client)
    with pytest.raises(RecoveryStop, match="continuation_seed_already_exists"):
        next_call(package, client)
    assert not package.constructed and path.read_bytes() == payload_bytes({"synthetic_pending": True})


def test_new_request_failure_is_terminal_never_auto_retried(package):
    package.fail = True
    client = open_resume(package)
    replay(package, client)
    with pytest.raises(RecoveryStop, match="continuation_request_failed_no_retry"):
        next_call(package, client)
    with pytest.raises(RecoveryStop, match="prefix_resume_closed"):
        next_call(package, client)
    assert package.normal_calls == [138]
    ledger = json.loads(package.ledger.read_bytes())
    assert len(ledger["requests"]) == 5 and ledger["requests"][-1]["status"] == "error"
    assert not (package.spec.output_root / "continuation_receipt.json").exists()


def test_seed_helper_itself_refuses_pending_even_without_prefix_wrapper(package):
    normal = SyntheticFactory(package).resume_after_cached_prefix(
        package.spec.variant_id, calls=package.calls)
    audit = package.spec.audit_path()
    path = audit.with_name(audit.stem + "-005.json")
    write(path, {"synthetic_pending": True})
    with pytest.raises(RecoveryStop, match="continuation_seed_already_exists"):
        normal.chat([], max_tokens=4096)
    path.unlink()
    with pytest.raises(RecoveryStop, match="continuation_request_pending_or_failed"):
        normal.chat([], max_tokens=4096)
    assert not package.normal_calls


def test_recovery_namespace_is_strict_one_use(package):
    open_resume(package)
    with pytest.raises(RecoveryStop, match="continuation_already_exists"):
        open_resume(package)


@pytest.mark.parametrize("artifact", ["spec", "custody", "cache", "sidecar"])
def test_changed_sealed_file_prevents_even_first_cached_call(package, artifact):
    path = {"spec": package.spec.path, "custody": package.spec.custody_path,
            "cache": package.spec.cache_path, "sidecar": package.deployment.path}[artifact]
    raw = path.read_bytes()
    client = open_resume(package)
    try:
        path.write_bytes(raw + b" ")
        with pytest.raises(RecoveryStop):
            replay(package, client, n=1)
        assert not package.constructed
    finally:
        path.write_bytes(raw)


def test_spec_requires_owner_only_and_sidecar_seal(package):
    package.spec.path.chmod(0o644)
    with pytest.raises(RecoveryStop, match="recovery_file_not_owner_only"):
        CachedPrefixRecoverySpec(
            package.spec.path, continuation=package.deployment, system_root=package.root,
            run_identity=package.helper.dispatch["run_identity"])
    package.spec.path.chmod(0o600)
    anchor = package.deployment.value["preserved_files"].pop(
        package.spec.path.relative_to(package.root).as_posix())
    with pytest.raises(RecoveryStop, match="recovery_file_not_sealed"):
        CachedPrefixRecoverySpec(
            package.spec.path, continuation=package.deployment, system_root=package.root,
            run_identity=package.helper.dispatch["run_identity"])
    package.deployment.value["preserved_files"][
        package.spec.path.relative_to(package.root).as_posix()] = anchor


@dataclass
class SyntheticOutcome:
    initial_skill_selection: dict
    rounds: list


def make_runner(package, monkeypatch, *, hook=True, replay_count=4, mismatch=None):
    runner = fitset.SystemVariantRunner.__new__(fitset.SystemVariantRunner)
    runner.__dict__.update(
        root=package.root, framework_state_io=None, cached_prefix_recovery=(
            package.spec.path if hook else None), _cached_prefix_spec=(
            package.spec if hook else None), _continuation=package.deployment,
        run_identity=package.helper.dispatch["run_identity"], capture_offline_case_input=False,
        evolution_config=None, episode_concurrency=1, workers=1, scheduler="slot_shards",
        request_governor_config=package.policy, _request_governor_root=package.ledger.parent,
        client_factory=SyntheticFactory(package), variants=[package.variant],
        variant_rows={package.variant["variant_id"]: package.variant},
        phase="final_system", evolution_finished=False)
    prepared = {
        "variant_id": package.variant["variant_id"],
        "item": SimpleNamespace(episode=SimpleNamespace(
            qa_id="synthetic", ground_truth="", options=[])),
        "artifact_path": Path("/synthetic/no-read.artifact"),
        "state": "g0m0", "prepared": "synthetic-prepared",
        "runtime_base_cfg": package.base_cfg,
        "skills": [SkillSpecV11.model_validate(skill) for skill in package.helper.skills],
        "skill_identity": package.helper.dispatch["skill_identity"],
        "variant_root": package.original.parent, "plan": deepcopy(package.plan),
        "artifact_relocation": None,
    }
    if mismatch == "plan":
        prepared["plan"]["unexpected"] = "synthetic change"
    elif mismatch == "config":
        prepared["runtime_base_cfg"] = replace(package.base_cfg, seed=139)
    elif mismatch == "skills":
        original_skill = prepared["skills"][0]
        prepared["skills"][0] = original_skill.model_copy(update={
            "skill_md": original_skill.skill_md + "Changed synthetic method.\n"})
    elif mismatch == "concurrency":
        runner.episode_concurrency = 2
    monkeypatch.setattr(runner, "_prepare_one", lambda _row, **_kwargs: prepared)

    def synthetic_session(items, cfg, *, llm, trace_store=None, **_kwargs):
        if hook:
            assert cfg.trace_dir == package.spec.value["original_trace_dir"]
            assert trace_store.root == package.spec.output_root / "trace"
            assert Path(cfg.work_dir).is_relative_to(package.spec.output_root)
        # A synthetic session regenerates calls from local state in order.
        replay(package, llm, n=replay_count)
        if replay_count == 4:
            for _ in range(package.new_count):
                next_call(package, llm)
        trace_store.append("synthetic_session", {"cached_calls": replay_count})
        return [SyntheticOutcome({"evidence_state": "g0m0"}, [])], SimpleNamespace(
            model_dump=lambda **_kwargs: {"synthetic": True})

    monkeypatch.setattr("skill3d.online.runner.run_split", synthetic_session)
    monkeypatch.setattr(fitset, "_system_result_row", lambda *_args, **_kwargs: {
        "variant_id": package.variant["variant_id"], "status": "synthetic_complete"})
    return runner


def test_runner_rebuilds_context_preserves_logical_trace_and_writes_separate_result(package, monkeypatch):
    runner = make_runner(package, monkeypatch)
    row = runner._run_one_locked(package.variant)
    assert not (package.root / "results").exists()
    assert (package.original.parent / "outcome.json").read_bytes() == (
        package.spec.output_root / "outcome.json").read_bytes()
    assert (package.spec.output_root / "trace" / "synthetic_session.jsonl").is_file()
    assert package.spec.completed_result() == row
    assert runner.completed() == {package.spec.variant_id: row}
    forked = copy(runner)
    restored = fitset.SystemVariantRunner.__new__(fitset.SystemVariantRunner)
    restored.__setstate__(forked.__getstate__())
    assert restored.root == runner.root and restored.root.parent == runner.root.parent
    assert restored.request_governor_root == package.ledger.parent
    assert restored.episode_concurrency == 1
    summary = runner._write_summary(
        {"identity_sha256": package.helper.dispatch["run_identity"]}, runner.completed())
    assert summary["response_recovery"]["recovered_requests"] == 1
    assert summary["requests"]["usage"]["total_tokens"] == 59
    assert summary["requests"]["usage_with_recovered_responses"]["total_tokens"] == 71
    assert summary["response_recovery"]["unsettled_started_requests"] == 0
    second = runner._write_summary(
        {"identity_sha256": package.helper.dispatch["run_identity"]}, runner.completed())
    assert second["requests"]["usage_with_recovered_responses"]["total_tokens"] == 71


def test_ordinary_runner_still_refuses_incomplete_original_trace(package, monkeypatch):
    runner = make_runner(package, monkeypatch, hook=False)
    with pytest.raises(fitset.FitSetError, match="incomplete trace exists"):
        runner._run_one_locked(package.variant)
    assert not package.constructed and not package.spec.output_root.exists()


@pytest.mark.parametrize("mismatch", ["plan", "config", "skills", "concurrency"])
def test_actual_prepared_mismatch_refuses_before_claim_and_factory(package, monkeypatch, mismatch):
    runner = make_runner(package, monkeypatch, mismatch=mismatch)
    with pytest.raises(RecoveryStop):
        runner._run_one_locked(package.variant)
    assert not package.constructed and not package.spec.output_root.exists()


@pytest.mark.parametrize("count", [2, 4])
def test_runner_cannot_write_result_for_partial_replay_or_cached_only_finish(package, monkeypatch, count):
    package.new_count = 0
    runner = make_runner(package, monkeypatch, replay_count=count)
    with pytest.raises(RecoveryStop):
        runner._run_one_locked(package.variant)
    assert not (package.spec.output_root / "result.json").exists()
    assert not (package.spec.output_root / "continuation_receipt.json").exists()
    assert not package.constructed


def test_custody_is_not_episode_completion_or_a_second_usage_settlement(package):
    assert package.spec.completed_result() is None
    client = open_resume(package)
    replay(package, client)
    next_call(package, client)
    state = json.loads(package.ledger.read_bytes())
    overlay = recovery_accounting(state, [package.spec.custody])
    assert overlay["reserved_requests"] == 5 and overlay["recovered_usage"]["total_tokens"] == 12
    with pytest.raises(RecoveryStop, match="duplicate_recovery_settlement"):
        recovery_accounting(state, [
            package.spec.custody, json.loads(
                (package.spec.output_root / "recovery_receipt.json").read_bytes())])


class SyntheticQuotaIO:
    def __init__(self, root, ceiling):
        self.root, self.ceiling = root, ceiling
        self.writes = []

    def mkdir(self, path):
        assert path.is_relative_to(self.root)
        path.mkdir(exist_ok=True)

    def write_bytes(self, path, raw, *, exclusive):
        assert exclusive and path.is_relative_to(self.root)
        self.writes.append(path)
        if len(raw) > self.ceiling:
            raise OSError("synthetic quota exhausted")
        with path.open("xb") as stream:
            stream.write(raw)


def test_recovery_quota_failure_cannot_return_cached_text_or_construct_delegate(package):
    io = SyntheticQuotaIO(package.root, 1_000_000)
    client = open_resume(package, io)
    io.ceiling = 1
    with pytest.raises(RecoveryStop):
        replay(package, client, n=1)
    assert client.cached._index == 0 and not package.constructed
    assert len(io.writes) >= 2
    assert not (package.spec.output_root / "recovery_receipt.json").exists()


def test_cli_factory_conflict_check_is_before_any_normal_constructor(package, monkeypatch):
    from scripts.run_fitset_v11 import _SeededClientFactory
    factory = _SeededClientFactory({}, 138, package.root)
    audit = package.spec.audit_path()
    write(audit.with_name(audit.stem + "-005.json"), {"synthetic_pending": True})
    with pytest.raises(RecoveryStop, match="continuation_seed_already_exists_or_changed"):
        factory.resume_after_cached_prefix(package.spec.variant_id, calls=package.calls)
    assert not package.constructed


def test_real_skill_dump_without_computed_hash_preserves_exact_dispatch_hash(package):
    skill = SkillSpecV11(
        skill_id="synthetic-skill", version="1.0.0", question_type="object_counting",
        evidence_state="g0m0",
        skill_md="---\nname: synthetic-skill\ndescription: Synthetic test method.\n---\n"
                 "Use only synthetic observations.\n")
    dump = skill.model_dump(mode="json")
    assert "content_sha256" not in dump
    dispatch = {**package.helper.dispatch, "skills_sha256": evidence_sha256([dump])}
    context = {**deepcopy(package.context), "skill_content_sha256": [skill.content_sha256]}
    calls = []
    for old in package.calls:
        seed = deepcopy(old.seed_record)
        seed["frozen_context"] = deepcopy(context)
        approval = {**deepcopy(old.approval), "seed_record_sha256": evidence_sha256(seed)}
        calls.append(replace(old, seed_record=seed, approval=approval))
    client = open_cached_episode_recovery(
        frozen_dispatch=dispatch, resumed_dispatch=dispatch,
        frozen_context=context, resumed_context=context,
        skills=[dump], model_identity=package.helper.model, calls=calls,
        original_trace_dir=package.original,
        continuation_root=package.root / "synthetic-real-skill-custody",
        cache_path=package.spec.cache_path)
    assert client.recover_original_responses()["episode_execution_reproduced"] is False
    assert evidence_sha256([dump]) == dispatch["skills_sha256"]


def test_real_skill_explicit_computed_hash_must_match_without_rewriting_dump(package):
    from skill3d.evaluation.cached_episode_recovery_v11 import _skill_content_sha256
    skill = SkillSpecV11(
        skill_id="synthetic-skill", version="1.0.0", question_type="object_counting",
        evidence_state="g0m0", skill_md="---\nname: synthetic-skill\n"
        "description: Synthetic test method.\n---\nSynthetic method.\n")
    dump = skill.model_dump(mode="json")
    assert _skill_content_sha256(dump) == skill.content_sha256
    explicit = {**dump, "content_sha256": skill.content_sha256}
    assert _skill_content_sha256(explicit) == skill.content_sha256
    explicit["content_sha256"] = "0" * 64
    with pytest.raises(RecoveryStop, match="skills_mismatch"):
        _skill_content_sha256(explicit)
    with pytest.raises(RecoveryStop, match="skills_mismatch"):
        _skill_content_sha256({"content_sha256": "0" * 64, "extra": "synthetic"})


def test_broader_selection_remains_closed_until_verified_continuation_result(package, monkeypatch):
    runner = make_runner(package, monkeypatch)
    with pytest.raises(fitset.FitSetError, match="only the explicit bound variant"):
        runner._validate_cached_prefix_selection([], 0, {})
    assert runner._validate_cached_prefix_selection([package.spec.variant_id], 1, {}) is True
    row = runner._run_one_locked(package.variant)
    assert runner._validate_cached_prefix_selection([], 0, runner.completed()) is False
    assert runner._validate_cached_prefix_selection(
        ["synthetic-other-variant"], 8, runner.completed()) is False
    assert runner._result_path(package.spec.variant_id) == package.spec.output_root / "result.json"
    assert runner.completed()[package.spec.variant_id] == row
    receipt_path = package.spec.output_root / "continuation_receipt.json"
    raw = receipt_path.read_bytes()
    try:
        receipt = json.loads(raw)
        receipt["new_completed_calls"] = 0
        write(receipt_path, receipt)
        with pytest.raises(RecoveryStop):
            runner._validate_cached_prefix_selection([], 0, runner.completed())
    finally:
        receipt_path.write_bytes(raw)


def test_initialize_auto_pins_unique_sealed_spec_without_claim_or_model(package, monkeypatch):
    runner = make_runner(package, monkeypatch)
    runner.cached_prefix_recovery = None
    runner._cached_prefix_spec = None
    runner._continuation = None
    runner.deployment_continuation = package.deployment.path
    manifest = {"identity_sha256": package.helper.dispatch["run_identity"]}
    write(package.root / "manifest.json", manifest)
    monkeypatch.setattr(runner, "_manifest", lambda: manifest)
    monkeypatch.setattr(package.deployment, "validate", lambda **_kwargs: manifest, raising=False)
    monkeypatch.setattr(
        "skill3d.evolution.deployment_continuation_v11.DeploymentContinuation",
        lambda _path: package.deployment)
    runner.initialize(resume=True)
    assert runner.cached_prefix_recovery == package.spec.path
    assert runner._cached_prefix_spec.variant_id == package.spec.variant_id
    assert not package.spec.output_root.exists()
    assert not package.constructed and not package.normal_calls


def test_completion_cannot_be_forged_from_custody_or_missing_new_ledger_call(package, monkeypatch):
    runner = make_runner(package, monkeypatch)
    runner._run_one_locked(package.variant)
    raw = package.ledger.read_bytes()
    try:
        write(package.ledger, package.state)
        with pytest.raises(RecoveryStop, match="continuation_ledger_incomplete"):
            package.spec.completed_result()
    finally:
        package.ledger.write_bytes(raw)


def test_verified_canonical_offline_copy_cannot_change_under_later_normal_run(package, monkeypatch):
    runner = make_runner(package, monkeypatch)
    runner._run_one_locked(package.variant)
    canonical = package.original.parent / "outcome.json"
    raw = canonical.read_bytes()
    try:
        canonical.write_bytes(b"synthetic changed canonical outcome")
        with pytest.raises(RecoveryStop, match="canonical_recovery_outcome_changed"):
            runner.completed()
    finally:
        canonical.write_bytes(raw)


@pytest.mark.parametrize("tamper", [None, "snapshot", "skill"])
def test_cached_prefix_uses_original_worker_skill_without_advancing_evolution(
        package, monkeypatch, tamper):
    runner = make_runner(package, monkeypatch)
    runner.evolution_config = object()
    skill = SkillSpecV11.model_validate(package.helper.skills[0])
    identity = {
        "mode": "fitset_slot_stream", "snapshot_ref": "synthetic-worker-snapshot",
        "manifest_sha256": "a" * 64, "slot_key": skill.slot_key,
    }
    runner._cached_prefix_spec = SimpleNamespace(value={"frozen_dispatch": {
        "skill_identity": identity,
        "skills_sha256": evidence_sha256([skill.model_dump(mode="json")]),
    }})
    write(package.root / "manifest.json", {"execution": {
        "slot_ownership": {package.variant["slot_key"]: 2}}})
    snapshot = {"snapshot_id": identity["snapshot_ref"],
                "manifest_sha256": identity["manifest_sha256"]}
    if tamper == "snapshot":
        snapshot["snapshot_id"] = "synthetic-other-snapshot"
    if tamper == "skill":
        skill = skill.model_copy(update={"skill_md": skill.skill_md + "Changed.\n"})
    visited = []
    def active(path):
        visited.append(path)
        return {(skill.question_type, skill.evidence_state): skill}, snapshot
    monkeypatch.setattr("skill3d.evolution.fitset_stream_v11._active", active)
    original_provider = object()
    runner.skills_for_variant = original_provider
    if tamper:
        with pytest.raises(fitset.FitSetError, match="original dispatch"):
            runner._cached_prefix_execution_runner(package.variant)
    else:
        worker = runner._cached_prefix_execution_runner(package.variant)
        skills, actual_identity = worker.skills_for_variant(package.variant)
        assert actual_identity == identity and skills == [skill]
        assert worker is not runner and worker.client_factory is runner.client_factory
    assert runner.skills_for_variant is original_provider
    assert visited == [package.root / "workers/02/library"]
    assert not package.constructed and not package.normal_calls
