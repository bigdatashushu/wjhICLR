"""Task-index support for V2 only; original V1 workers and inputs stay unchanged."""
from __future__ import annotations

import json
import os

from household_worker import OfficialHousehold
from household_protocol import ROOT, SUBSETS, source, verify_environment
from protocol import ContractError, canonical, sha256


class OfficialHouseholdV2(OfficialHousehold):
    ALFRED_RENDER_PROFILE = {
        "rgb": True, "instance_segmentation": True,
        "official_depth": False, "class_segmentation": False}

    def __init__(self, environment, subset, work):
        super().__init__(environment, subset, work)
        self.requested_index = 0
        self.asset_directory = ROOT / "action-alfred-data"
        self.asset_manifest = None
        if environment == "alfred":
            self.asset_manifest = json.loads((ROOT / "action_alfred_data.json").read_bytes())
            original = json.loads((ROOT / "alfred_pilot_data.json").read_bytes())
            if (self.asset_manifest["schema"] != "embodied-alfred-assets/1"
                    or self.asset_manifest["subset"] != "base"
                    or any(self.asset_manifest[k] != original[k] for k in ("repository", "revision"))
                    or self.asset_manifest["indexes"] != list(range(5))
                    or [r["index"] for r in self.asset_manifest["files"]] != list(range(5))):
                raise ContractError("alfred_asset_manifest_shape")
            for row in self.asset_manifest["files"]:
                self.read_asset(row["path"])

    def read_asset(self, name):
        row = next((r for r in self.asset_manifest["files"] if r["path"] == name), None)
        path = (self.asset_directory / name).resolve()
        if row is None or not path.is_relative_to(self.asset_directory.resolve()):
            raise ContractError("alfred_asset_not_selected")
        raw = path.read_bytes()
        if sha256(raw) != row["sha256"] or len(raw) != row["bytes"]:
            raise ContractError("alfred_asset_identity")
        return raw

    def init_alfred(self):
        # Only V2 disables channels unused by its RGB/predicted-depth chain.
        # Keep instance segmentation for the official visibility checks.
        from embodiedbench.envs.eb_alfred.gen import constants
        constants.RENDER_DEPTH_IMAGE = False
        constants.RENDER_CLASS_IMAGE = False
        if not constants.RENDER_IMAGE or not constants.RENDER_OBJECT_IMAGE:
            raise ContractError("alfred_required_render_channel_disabled")
        # The official loader reads the exact same ann JSON from the new
        # worker-owned asset directory; shared dataset paths are read-only.
        super().init_alfred()
        from embodiedbench.envs.eb_alfred import EBAlfEnv as module
        module.utils.load_task_json = lambda task: json.loads(self.read_asset(
            task["task"] + f"/pp/ann_{task['repeat_idx']}.json"))

    def init_habitat(self):
        import habitat
        from omegaconf import OmegaConf
        original = habitat.get_config
        def config(*args, **kwargs):
            result = original(*args, **kwargs)
            OmegaConf.set_readonly(result, False)
            result.habitat.seed = 137
            result.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
            result.habitat.environment.iterator_options.shuffle = False
            result.habitat.environment.iterator_options.group_by_scene = False
            return result
        habitat.get_config = config
        os.chdir(ROOT / "upstream-household/embodiedbench/envs/eb_habitat")
        from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
        # Official start_epi_index advances the deterministic iterator using
        # resets only. Skipped episodes never enter the solver or action ledger.
        self.env = EBHabEnv(
            eval_set=self.subset, exp_name="embodiedrunner", down_sample_ratio=1.0,
            start_epi_index=self.requested_index, resolution=500, recording=False)
        self.env.log_path = str(self.work / "simulator")

    def open(self, index, dataset_sha256):
        if self.subset not in SUBSETS[self.kind] or type(index) is not int or not 0 <= index < 5:
            raise ContractError("invalid_command")
        verify_environment(self.kind)
        name = ("embodiedbench/envs/eb_alfred/data/splits/splits.json" if self.kind == "alfred"
                else f"embodiedbench/envs/eb_habitat/datasets/{self.subset}.pickle")
        if sha256(source(name)) != dataset_sha256:
            raise ContractError("official_source_changed")
        self.requested_index = index
        if self.env is None:
            self.init_alfred() if self.kind == "alfred" else self.init_habitat()
        if self.env._max_episode_steps != 30 or self.env._max_invalid_actions != 10:
            raise ContractError("official_source_changed")
        if self.kind == "alfred":
            if self.subset != self.asset_manifest["subset"] or index not in self.asset_manifest["indexes"]:
                raise ContractError("alfred_asset_not_selected")
            self.env._current_episode_num = index
            task = self.env.dataset[index]
            self.read_asset(task["task"] + f"/pp/ann_{task['repeat_idx']}.json")
        elif self.env._current_episode_num != index:
            raise ContractError("habitat_nonsequential_episode")
        obs = self.env.reset()
        if self.env._current_episode_num != index + 1:
            raise ContractError("official_episode_index_changed")
        if self.kind == "habitat":
            actual = self.env.env.env.env._env.current_episode.episode_id
            if str(actual) != str(index):
                raise ContractError("habitat_episode_identity_mismatch")
        elif self.env.episode_language_instruction != task["instruction"]:
            raise ContractError("alfred_episode_identity_mismatch")
        self.input_sha256 = sha256(canonical({
            "environment": self.kind, "subset": self.subset, "index": index,
            "dataset_sha256": dataset_sha256,
            "instruction": self.env.episode_language_instruction}))
        return self.packet(obs)
