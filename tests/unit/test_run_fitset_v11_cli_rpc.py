"""Published CLI wiring and RPC admission, with no labels, keys or model IO."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import urllib.request

import pytest

from skill3d.cluster.perception_client import RPC_SCHEMA, RemoteBailianClient
from skill3d.evaluation.fitset_v11 import SystemVariantRunner, _variant_key
from skill3d.evolution.fitset_stream_v11 import (
    FitSetWorkerEvolution, FrozenSkillProvider, SeededGateClient,
)
from skill3d.online import runner as online_runner
from skill3d.online.config import _FALLBACK
from skill3d.skills.registry import load_active_skills
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import payload_bytes

from test_cluster_perception_rpc import gateway_module
from test_published_fitset_v11 import _assert_safe, _published


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_fitset_v11_cli_rpc_tests", ROOT / "scripts/run_fitset_v11.py")
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)
ENDPOINT = "http://127.0.0.1:18499"
LABEL = "synthetic-offline-label-must-not-reach-executors"


def _forbidden(*_args, **_kwargs):
    pytest.fail("unexpected model, credential, dataset or network access")


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(BailianClient, "chat", _forbidden)
    monkeypatch.setattr(RemoteBailianClient, "chat", _forbidden)
    monkeypatch.setattr(BailianClient, "configured", property(lambda _self: True))
    monkeypatch.setattr(RemoteBailianClient, "configured", property(lambda _self: True))
    monkeypatch.setattr(
        urllib.request, "build_opener",
        lambda *_handlers: SimpleNamespace(open=_forbidden),
    )
    monkeypatch.setattr(online_runner, "_make_online_client", _forbidden)
    monkeypatch.setattr(online_runner, "reconstruct", _forbidden)
    monkeypatch.setattr(online_runner, "_bind_objects_best_effort", _forbidden)


@pytest.fixture
def case(tmp_path, monkeypatch):
    published = _published(tmp_path)
    variants = [
        {**row, "evidence_state": "g1m0",
         "slot_key": "object_rel_distance/g1m0"}
        for row in published.variants.values()
    ]
    metadata = str(tmp_path / "offline-metadata-must-not-be-opened.jsonl")
    raw = deepcopy(_FALLBACK)
    raw["paths"]["vsi_bench_meta"] = metadata
    raw["online_model"].update(
        model_revision="frozen-test-revision", max_tokens=3072,
        timeout_s=27.0, api_key_env="TEST_UNUSED_UPSTREAM_KEY",
    )
    panel = {"proposal_sha256": "a" * 64, "panel_file_sha256": "b" * 64}
    skills, _, snapshot = load_active_skills(ROOT / "skill_library/snapshots")
    skills_for_variant = FrozenSkillProvider(skills, {
        "snapshot_ref": snapshot, "manifest_sha256": "c" * 64,
    })
    value = SimpleNamespace(
        published=published, variants=variants, metadata=metadata, raw=raw,
        runners=[], actions=[], label_reads=[], identity_reads=[],
    )

    def labels(path):
        assert path == metadata
        value.label_reads.append(path)
        return {
            **{row["qa_id"]: {
                "scene_id": row["scene_id"], "question_type": row["question_type"],
                "options": [], "reference_answer": LABEL,
            } for row in variants},
            "unselected-qa": {"reference_answer": "excluded-synthetic-label"},
        }

    def identity(path):
        assert path == metadata
        value.identity_reads.append(path)
        return {"path": path, "sha256": "e" * 64, "size": 123}

    class CaptureRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            value.runners.append(self)

        def preflight(self, **kwargs):
            value.actions.append(("preflight", kwargs))
            return {"status": "preflight"}

        def run(self, **kwargs):
            value.actions.append(("run", kwargs))
            return {"status": "completed"}

    monkeypatch.setattr(cli, "load_fitset_panel", lambda _path: panel)
    monkeypatch.setattr(cli, "select_panel_variants", lambda *_args: variants)
    monkeypatch.setattr(cli, "load_config", lambda _path: deepcopy(raw))
    monkeypatch.setattr(cli, "load_label_metadata", labels)
    monkeypatch.setattr(cli, "file_identity", identity)
    monkeypatch.setattr(cli, "_FitSetItemProvider", _forbidden)
    monkeypatch.setattr(cli, "load_variant_bindings", _forbidden)
    monkeypatch.setattr(cli, "_active_skill_provider", lambda _root: skills_for_variant)
    monkeypatch.setattr(cli, "_fit_skill_provider", lambda _root: skills_for_variant)
    monkeypatch.setattr(cli, "SystemVariantRunner", CaptureRunner)
    value.argv = [
        "run-system", "--panel", "unused-panel.json", "--panel-name", "ready_579",
        "--metadata", metadata, "--published-inputs", str(published.manifest_path),
        "--phase", "accumulation", "--output-dir", str(tmp_path / "run"),
        "--scheduler", "episode_pool", "--workers", "32", "--skill-owners", "4",
        "--max-requests", "120", "--request-concurrency", "8",
        "--perception-endpoint", ENDPOINT, "--perception-capacity", "8",
        "--episode-concurrency", "32", "--revision-max-tokens", "1536",
    ]
    return value


def _invoke(case, *extra):
    return cli.main([*case.argv, *extra])


def _health(monkeypatch, raw):
    calls = []

    def build(*handlers):
        if isinstance(handlers[0], urllib.request.HTTPSHandler):
            return SimpleNamespace(open=_forbidden)
        assert len(handlers) in (1, 2)
        assert isinstance(handlers[0], urllib.request.ProxyHandler)
        assert handlers[0].proxies == {}
        if len(handlers) == 2:
            return SimpleNamespace(open=_forbidden)

        def open_stub(url, *, timeout):
            calls.append((url, timeout))
            if isinstance(raw, Exception):
                raise raw
            return io.BytesIO(raw)

        return SimpleNamespace(open=open_stub)

    monkeypatch.setattr(urllib.request, "build_opener", build)
    return calls


def _ready(case):
    client = BailianClient(
        model="qwen3.8-flash", model_revision="frozen-test-revision",
        base_url=case.raw["online_model"]["base_url"],
        api_key_env="TEST_UNUSED_UPSTREAM_KEY", max_output_tokens=3072,
    )
    return {
        "schema_version": RPC_SCHEMA, "status": "ready",
        "identity": client.audit_identity(), "max_concurrency": 8,
    }


def test_published_preflight_wires_provider_budget_and_all_factories(case, capsys, monkeypatch):
    monkeypatch.setattr(cli._RevisionClientFactory, "__call__", _forbidden)
    assert _invoke(case, "--preflight-only", "--resume", "--max-variants", "1",
                   "--variant-ids", "published/qa-01,") == 0
    assert json.loads(capsys.readouterr().out) == {"status": "preflight"}
    kwargs = case.runners[0].kwargs
    provider = kwargs["published_inputs"]
    assert kwargs["item_provider"] is provider
    assert kwargs["input_receipt_provider"].__self__ is provider
    assert provider.max_scenes == 2
    assert kwargs["binding_refs"] == {
        row["variant_id"]: case.published.catalog[row["qa_id"]]["reference"]
        for row in case.variants
    }
    item = provider(case.variants[0])
    assert item.episode.ground_truth == "" and len(item.pixels) == 32
    _assert_safe(kwargs["input_receipt_provider"](case.variants[0], item))
    assert kwargs["input_contract_identity"] == {
        **provider.input_contract_identity,
        "offline_metadata": {"path": case.metadata, "sha256": "e" * 64, "size": 123},
        "model_transport": {
            "kind": "perception_rpc", "endpoint": ENDPOINT, "upstream_capacity": 8,
        },
    }
    assert kwargs["workers"] == kwargs["episode_concurrency"] == 32
    assert kwargs["scheduler"] == "episode_pool" and kwargs["skill_owners"] == 4
    assert asdict(kwargs["request_governor_config"]) == {
        "max_requests": 120, "max_inflight": 8, "requests_per_second": 0,
    }
    assert kwargs["base_cfg"].seed == 137
    assert kwargs["evolution_config"].gate_seeds == (137, 138)
    assert kwargs["evolution_config"].max_tokens == 1536
    assert set(kwargs["label_metadata"]) == {"qa-00", "qa-01"}
    assert case.label_reads == case.identity_reads == [case.metadata]
    assert case.actions == [("preflight", {
        "resume": True, "max_variants": 1, "variant_ids": ["published/qa-01"],
    })]
    for name in ("client_factory", "revision_client_factory", "gate_client_factory"):
        factory = kwargs[name]
        assert factory.perception_endpoint == ENDPOINT
        assert set(factory.config) == {"online_model"}
        wire = pickle.dumps(factory)
        assert LABEL.encode() not in wire and case.metadata.encode() not in wire


def test_online_requires_matching_health_before_runner_construction(case, monkeypatch):
    calls = _health(monkeypatch, json.dumps(_ready(case)).encode())
    assert _invoke(case, "--perception-endpoint", ENDPOINT + "/") == 0
    assert calls == [(ENDPOINT + "/health", 10)]
    assert [action for action, _ in case.actions] == ["run"]


@pytest.mark.parametrize("mutation", [
    "schema", "status", "model_revision", "capacity", "missing_identity",
    "malformed", "duplicate", "array", "unavailable",
])
def test_untrusted_health_fails_before_runner_or_offline_metadata(
    case, monkeypatch, capsys, mutation,
):
    health = _ready(case)
    if mutation == "schema":
        health["schema_version"] = "future-schema"
    elif mutation == "status":
        health["status"] = "busy"
    elif mutation == "model_revision":
        health["identity"]["model_revision"] = "different"
    elif mutation == "capacity":
        health["max_concurrency"] = 7
    elif mutation == "missing_identity":
        del health["identity"]
    raw = json.dumps(health).encode()
    if mutation == "malformed":
        raw = b"{"
    elif mutation == "duplicate":
        raw = b'{"status":"ready","status":"ready"}'
    elif mutation == "array":
        raw = b"[]"
    elif mutation == "unavailable":
        raw = OSError("synthetic-health-failure")
    calls = _health(monkeypatch, raw)
    assert _invoke(case) == 2
    assert "RPC gateway" in capsys.readouterr().err
    assert calls == [(ENDPOINT + "/health", 10)]
    assert case.runners == case.actions == case.label_reads == case.identity_reads == []


def test_missing_rpc_configuration_fails_without_reading_upstream_key_or_health(
    case, monkeypatch, capsys,
):
    monkeypatch.setattr(RemoteBailianClient, "configured", property(lambda _self: False))
    assert _invoke(case) == 2
    assert "RPC endpoint/token is not configured" in capsys.readouterr().err
    assert case.runners == case.actions == case.label_reads == []


@pytest.mark.parametrize("extra,message", [
    (["--perception-capacity", "0"], "capacity must be frozen"),
    (["--perception-capacity", "9"], "capacity must be frozen"),
    (["--perception-capacity", "-1"], "capacity must be frozen"),
    (["--request-concurrency", "9"], "exceeds frozen"),
    (["--request-concurrency", "0"], "exceeds frozen"),
    (["--perception-endpoint", ""], "requires --perception-endpoint"),
    (["--perception-endpoint", ENDPOINT + "/chat"], "credential-free"),
    (["--perception-endpoint", "http://user:pass@localhost:18499"], "credential-free"),
    (["--max-requests", "0"], "explicit --max-requests"),
    (["--revision-max-tokens", "0"], "revision token budget"),
    (["--revision-max-tokens", "4096"], "revision token budget"),
])
def test_invalid_rpc_or_pool_arguments_fail_before_client_or_runner(case, capsys, extra, message):
    assert _invoke(case, "--preflight-only", *extra) == 2
    assert message in capsys.readouterr().err
    assert case.runners == case.actions == case.label_reads == []


@pytest.mark.parametrize("workers,episode_concurrency", [("16", "0"), ("1", "32")])
def test_unenforced_request_concurrency_cannot_bypass_rpc_capacity(
    case, capsys, workers, episode_concurrency,
):
    assert _invoke(
        case, "--preflight-only", "--scheduler", "slot_shards",
        "--workers", workers, "--episode-concurrency", episode_concurrency,
        "--max-requests", "0",
    ) == 2
    assert "exceeds frozen" in capsys.readouterr().err
    assert case.runners == case.actions == case.label_reads == []


@pytest.mark.parametrize("flag", [
    "--bindings", "--direct-run-root", "--video-root", "--video-fallback-root",
])
def test_published_inputs_reject_each_legacy_locator(case, capsys, flag):
    assert _invoke(case, "--preflight-only", flag, "never-opened") == 2
    assert "cannot be mixed" in capsys.readouterr().err
    assert case.runners == case.label_reads == []


def test_published_selection_rejects_non_induction_before_execution(case, capsys):
    case.variants[0]["split"] = "inner_validation"
    assert _invoke(case, "--preflight-only") == 2
    assert "induction-only" in capsys.readouterr().err
    assert case.runners == case.label_reads == []


@pytest.mark.parametrize("locators", [
    [], ["--bindings", "never-opened"], ["--direct-run-root", "never-opened"],
])
def test_legacy_mode_still_requires_both_binding_and_direct_roots(case, capsys, locators):
    assert _invoke(case, "--preflight-only", "--published-inputs", "", *locators) == 2
    assert "require --bindings and --direct-run-root" in capsys.readouterr().err
    assert case.runners == case.label_reads == []


def test_legacy_preflight_preserves_input_locators_and_direct_transport(case, monkeypatch):
    inputs = []
    bindings = {row["variant_id"]: {"key": "f" * 64} for row in case.variants}
    provider = SimpleNamespace(
        receipt=_forbidden, input_contract_identity={"legacy_input": "frozen"},
    )

    def item_provider(**kwargs):
        inputs.append(kwargs)
        return provider

    def load_bindings(path, variants):
        assert path == "frozen-bindings.json" and variants == case.variants
        return bindings

    monkeypatch.setattr(cli, "_FitSetItemProvider", item_provider)
    monkeypatch.setattr(cli, "load_variant_bindings", load_bindings)
    monkeypatch.setattr(cli, "file_identity", lambda path: {"path": path, "sha256": "f" * 64})
    assert _invoke(
        case, "--preflight-only", "--published-inputs", "",
        "--bindings", "frozen-bindings.json", "--direct-run-root", "frozen-direct",
        "--video-root", "frozen-videos", "--video-fallback-root", "fallback-one",
        "--video-fallback-root", "fallback-two", "--scene-cache-size", "3",
        "--perception-endpoint", "", "--perception-capacity", "0",
        "--scheduler", "slot_shards", "--workers", "2", "--max-requests", "0",
    ) == 0
    assert inputs == [{
        "metadata_path": case.metadata, "direct_run_root": "frozen-direct",
        "video_root": "frozen-videos", "fallback_roots": ["fallback-one", "fallback-two"],
        "max_scenes": 3,
    }]
    kwargs = case.runners[0].kwargs
    assert kwargs["published_inputs"] is None
    assert kwargs["item_provider"] is provider and kwargs["binding_refs"] == bindings
    assert kwargs["request_governor_config"] is None
    assert kwargs["input_contract_identity"] == {
        "legacy_input": "frozen",
        "prepared_bindings_manifest": {"path": "frozen-bindings.json", "sha256": "f" * 64},
        "model_transport": {"kind": "direct_bailian", "endpoint": "", "upstream_capacity": None},
    }
    assert kwargs["client_factory"].perception_endpoint == ""


@pytest.mark.parametrize("field,value,message", [
    ("enabled", False, "enabled bailian/qwen3.8-flash"),
    ("provider", "vllm", "online_model.provider must be bailian"),
    ("model", "different-model", "enabled bailian/qwen3.8-flash"),
])
def test_system_model_contract_is_checked_before_clients_or_metadata(
    case, capsys, field, value, message,
):
    case.raw["online_model"][field] = value
    assert _invoke(case, "--preflight-only") == 2
    assert message in capsys.readouterr().err
    assert case.runners == case.label_reads == []


def test_published_final_system_avoids_label_loading_and_evolution(case):
    assert _invoke(case, "--preflight-only", "--phase", "final_system",
                   "--skill-bundle", "unused-bundle") == 0
    kwargs = case.runners[0].kwargs
    assert kwargs["label_metadata"] is kwargs["evolution_config"] is None
    assert kwargs["published_inputs"] is kwargs["item_provider"]
    assert case.label_reads == []


@pytest.mark.parametrize("phase,extra,message", [
    ("accumulation", ["--skill-bundle", "unused"], "only valid for final_system"),
    ("final_system", [], "requires --skill-bundle"),
])
def test_phase_specific_skill_arguments_remain_required(case, capsys, phase, extra, message):
    assert _invoke(case, "--preflight-only", "--phase", phase, *extra) == 2
    assert message in capsys.readouterr().err
    assert case.runners == case.label_reads == []


@pytest.mark.parametrize("endpoint", ["", ENDPOINT])
def test_solver_reviser_and_paired_gate_factories_preserve_model_options_and_seeds(
    tmp_path, monkeypatch, endpoint,
):
    config = {"online_model": {
        "model": "qwen3.8-flash", "model_revision": "frozen-test-revision",
        "timeout_s": 27.0, "max_tokens": 3072, "api_key_env": "TEST_UNUSED_UPSTREAM_KEY",
    }}
    made = []

    def construct(**options):
        made.append(options)
        return SimpleNamespace()

    monkeypatch.setattr("skill3d.synthesis.bailian_client.BailianClient", construct)
    monkeypatch.setattr("skill3d.cluster.perception_client.RemoteBailianClient", construct)
    revision = pickle.loads(pickle.dumps(cli._RevisionClientFactory(config, 4096, endpoint)))
    revision()
    solver = cli._SeededClientFactory(config, 137, tmp_path / "solver", endpoint)
    gate = pickle.loads(pickle.dumps(cli._GateClientFactory(config, endpoint)))
    factories = [solver, gate(137, tmp_path / "parent"), gate(138, tmp_path / "candidate")]
    for factory in factories:
        client = factory("published/qa-00")
        assert isinstance(client, SeededGateClient)
        assert client.seed == factory.seed
        assert client.audit_path == (
            factory.output / "request_seeds" / (_variant_key("published/qa-00") + ".json"))
    for index, options in enumerate(made):
        assert options == {
            "model": "qwen3.8-flash", "model_revision": "frozen-test-revision",
            "base_url": _FALLBACK["online_model"]["base_url"],
            "api_key_env": "TEST_UNUSED_UPSTREAM_KEY", "timeout_s": 27.0,
            "temperature": 0.0,
            "max_output_tokens": 4096 if index == 0 and not endpoint else 3072,
            **({"endpoint": endpoint} if endpoint else {}),
        }


@pytest.mark.parametrize("role", ["solver", "reviser", "parent_gate", "candidate_gate"])
def test_gateway_validator_accepts_factory_identity_and_per_request_budget(tmp_path, role):
    config = {"online_model": {
        "model": "qwen3.8-flash", "model_revision": "frozen-test-revision",
        "max_tokens": 3072, "api_key_env": "TEST_UNUSED_UPSTREAM_KEY",
    }}
    seed = 138 if role == "candidate_gate" else 137
    if role == "reviser":
        client = cli._RevisionClientFactory(config, 1536, ENDPOINT)()
        budget = 1536
    elif role == "solver":
        client = cli._SeededClientFactory(config, seed, tmp_path, ENDPOINT)("qa-00")
        budget = 3072
    else:
        client = cli._GateClientFactory(config, ENDPOINT)(seed, tmp_path)("qa-00")
        budget = 3072
    # Exercise the real gateway validator without constructing a service or
    # touching authentication, the response cache, HTTP, or any model client.
    gateway = gateway_module.PerceptionGateway.__new__(gateway_module.PerceptionGateway)
    gateway._options = gateway_module.client_options(config)
    gateway._identity = BailianClient(**gateway._options).audit_identity()
    payload = client.build_payload(
        [{"role": "user", "content": "safe question only"}], max_tokens=budget, seed=seed)
    identity = client.audit_identity(seed=seed)
    request_id = "00000000-0000-0000-0000-000000000001"
    raw = payload_bytes({
        "schema_version": RPC_SCHEMA, "request_id": request_id,
        "payload": payload, "audit_identity": identity,
    })
    assert gateway._validate(raw) == (request_id, payload, identity)
    assert identity["max_tokens"] == 3072 and payload["max_tokens"] == budget


@pytest.mark.parametrize("binding_seeds", [(0, 0), (0, 1)])
def test_published_gate_copies_keep_transport_but_drop_offline_metadata_and_labels(
    case, monkeypatch, tmp_path, binding_seeds,
):
    assert _invoke(case, "--preflight-only", "--gate-size", "2") == 0
    runner = SystemVariantRunner(**case.runners[0].kwargs)
    runner.run_identity = "d" * 64
    for seed, reference in zip(binding_seeds, runner.binding_refs.values(), strict=True):
        reference["request_identity"]["seed"] = seed
    worker = FitSetWorkerEvolution(runner, 0, runner.variants)
    slot = runner.variants[0]["slot_key"]
    # The transport fixture has one scene; isolate arm construction from selection.
    runner.gate_panels[slot] = [row["variant_id"] for row in runner.variants]
    parent = next(skill for skill in runner.seed_skills if skill.slot_key == slot)
    captured = []

    class StopBeforeExecution(Exception):
        pass

    def capture(_runner, items):
        captured.extend(items)
        raise StopBeforeExecution

    monkeypatch.setattr(SystemVariantRunner, "_run_gate_episodes", capture)
    with pytest.raises(StopBeforeExecution):
        worker._gate(SimpleNamespace(slot_key=slot), parent, parent, tmp_path / "attempt")
    arms = {id(arm): arm for arm, _row in captured}.values()
    assert len(arms) == 4 and len(captured) == 8
    for arm in arms:
        assert arm.label_metadata == {} and arm.evolution_config is None
        assert arm.seed_skills == [] and arm.gate_panels == {}
        assert arm.revision_client_factory is arm.gate_client_factory is None
        assert arm.published_inputs is arm.item_provider is runner.published_inputs
        assert arm.input_receipt_provider.__self__ is arm.published_inputs
        assert "offline_metadata" not in arm.input_contract_identity
        assert arm.input_contract_identity["model_transport"] == (
            runner.input_contract_identity["model_transport"])
        assert arm.request_governor_root == runner.request_governor_root
        assert arm.request_governor_config == runner.request_governor_config
        assert arm._request_category == "gate"
        assert arm.client_factory.perception_endpoint == ENDPOINT
        assert arm.client_factory.seed in (137, 138)
        wire = pickle.dumps(arm)
        assert LABEL.encode() not in wire and case.metadata.encode() not in wire
        frozen = json.loads((arm.root / "gate_identity.json").read_text())
        assert frozen["seed"] == arm.client_factory.seed
        assert frozen["binding_seed"] == (0 if binding_seeds == (0, 0) else None)
        assert frozen["binding_seed_by_variant"] == {
            "published/qa-00": binding_seeds[0], "published/qa-01": binding_seeds[1],
        }
    assert runner.label_metadata["qa-00"]["reference_answer"] == LABEL
    assert runner.input_contract_identity["offline_metadata"]["path"] == case.metadata
