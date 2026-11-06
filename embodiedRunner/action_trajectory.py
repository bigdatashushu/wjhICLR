"""Opt-in complete public trajectories with lossless, content-addressed assets."""
from __future__ import annotations

import base64
import os
from pathlib import Path
import time

import numpy as np

from action_contract import plain
from protocol import ContractError, canonical, sha256


def write_new(path, value):
    with Path(path).open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class Trajectory:
    def __init__(self, directory, *, episode_id, arm):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700)
        (self.directory / "assets").mkdir(mode=0o700)
        self.episode_id, self.arm = episode_id, arm
        self.events, self.assets = [], {}
        self.started = time.monotonic()
        self.secrets = tuple(v.encode() for k in
                             ("DASHSCOPE_API_KEY", "HARNESS3D_PERCEPTION_TOKEN",
                              "OPENROUTER_API_KEY", "MAKERS_MODELS_KEY")
                             if (v := os.environ.get(k, "")))

    def asset(self, raw, suffix):
        digest = sha256(raw)
        name = "assets/" + digest + suffix
        if name not in self.assets:
            with (self.directory / name).open("xb") as stream:
                os.chmod(self.directory / name, 0o600)
                stream.write(raw)
            self.assets[name] = {"path": name, "sha256": digest, "bytes": len(raw)}
        return self.assets[name]

    def encode(self, value):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {"trajectory_encoding": "ndarray", "dtype": array.dtype.str,
                    "shape": list(array.shape), "asset": self.asset(array.tobytes(), ".bin")}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return {"trajectory_encoding": "bytes", "asset": self.asset(value, ".png")}
        if isinstance(value, str) and value.startswith("data:image/"):
            prefix, encoded = value.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
            suffix = ".png" if "image/png" in prefix else ".jpg"
            return {"trajectory_encoding": "data_url", "prefix": prefix,
                    "asset": self.asset(raw, suffix)}
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "base64" and {"dtype", "shape", "sha256"} <= set(value):
                    raw = base64.b64decode(item, validate=True)
                    result[key] = {"trajectory_encoding": "base64",
                                   "asset": self.asset(raw, ".bin")}
                else:
                    result[key] = self.encode(item)
            return result
        if isinstance(value, (list, tuple)):
            return [self.encode(item) for item in value]
        return value

    def emit(self, kind, **data):
        event = {"index": len(self.events), "elapsed_ms": int(
            (time.monotonic() - self.started) * 1000), "kind": kind,
            **self.encode(plain(data))}
        wire = canonical(event)
        if any(secret in wire for secret in self.secrets):
            raise ContractError("trajectory_credential_rejected")
        with (self.directory / "events.jsonl").open("ab") as stream:
            os.chmod(self.directory / "events.jsonl", 0o600)
            stream.write(wire + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    def finish(self, report):
        self.emit("episode_result", report=report)
        result = {"schema": "embodied-complete-trajectory/1", "episode_id": self.episode_id,
                  "arm": self.arm, "events": self.events, "assets": list(self.assets.values()),
                  "reference_answer_visible": False, "automatic_retry": False}
        write_new(self.directory / "trajectory.json", result)
        return result


def restore(value, directory):
    """Reconstruct original messages/evidence for independent hash verification."""
    if isinstance(value, list):
        return [restore(item, directory) for item in value]
    if not isinstance(value, dict):
        return value
    encoding = value.get("trajectory_encoding")
    if encoding:
        asset = value["asset"]
        path = (Path(directory) / asset["path"]).resolve()
        if not path.is_relative_to(Path(directory).resolve()):
            raise ContractError("trajectory_asset_path")
        raw = path.read_bytes()
        if sha256(raw) != asset["sha256"] or len(raw) != asset["bytes"]:
            raise ContractError("trajectory_asset_identity")
        if encoding == "bytes":
            return raw
        if encoding == "ndarray":
            return np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])
        text = base64.b64encode(raw).decode("ascii")
        if encoding == "base64":
            return text
        if encoding == "data_url":
            return value["prefix"] + "," + text
        raise ContractError("trajectory_encoding")
    return {key: restore(item, directory) for key, item in value.items()}
