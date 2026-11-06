"""Check the changed task positioning and unchanged comparison policy offline."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import action_household_v2 as module
from action_household_v2 import OfficialHouseholdV2
from protocol import ContractError, ROOT, sha256


@pytest.mark.skipif(
    not all((ROOT / "config" / name).is_file()
            for name in ("action-compare-4090.json", "action-compare-extra-4090.json")),
    reason="requires host-local comparison configuration",
)
def test_additional_task_config_changes_selection_only():
    original = json.loads((ROOT / "config/action-compare-4090.json").read_bytes())
    extra = json.loads((ROOT / "config/action-compare-extra-4090.json").read_bytes())
    assert {k: v for k, v in extra.items() if k != "selection"} == {
        k: v for k, v in original.items() if k != "selection"}
    assert len(extra["selection"]) == 12
    for environment in ("eb_navigation", "eb_alfred", "eb_habitat"):
        assert [s["index"] for s in extra["selection"] if s["environment"] == environment] == [1, 2, 3, 4]


@pytest.mark.parametrize("kind", ("alfred", "habitat"))
def test_open_nonzero_index_keeps_official_binding(kind, monkeypatch):
    worker = OfficialHouseholdV2.__new__(OfficialHouseholdV2)
    worker.kind, worker.subset, worker.env = kind, "base", None
    worker.asset_manifest = {"subset": "base", "indexes": list(range(5))}
    worker.read_asset = lambda name: b"verified"
    monkeypatch.setattr(module, "source", lambda name: b"dataset")
    monkeypatch.setattr(module, "verify_environment", lambda name: None)
    task = {"task": "task-path", "repeat_idx": 0, "instruction": "public instruction"}
    def initialize():
        env = SimpleNamespace(_max_episode_steps=30, _max_invalid_actions=10,
                              _current_episode_num=worker.requested_index,
                              dataset=[task] * 5, episode_language_instruction="public instruction")
        env.env = SimpleNamespace(env=SimpleNamespace(env=SimpleNamespace(
            _env=SimpleNamespace(current_episode=SimpleNamespace(episode_id="4")))))
        def reset():
            env._current_episode_num += 1
            return {"frame": "public"}
        env.reset = reset
        worker.env = env
    worker.init_alfred = worker.init_habitat = initialize
    worker.packet = lambda obs: {"obs": obs, "input_sha256": worker.input_sha256}
    packet = worker.open(4, sha256(b"dataset"))
    assert worker.env._current_episode_num == 5
    assert packet["input_sha256"]
    if kind == "habitat":
        worker.env._current_episode_num = 4
        worker.env.env.env.env._env.current_episode.episode_id = "0"
        with pytest.raises(ContractError, match="identity_mismatch"):
            worker.open(4, sha256(b"dataset"))
