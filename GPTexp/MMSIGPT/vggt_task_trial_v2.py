"""Versioned compact-prompt entry; historical prompt and trial files stay intact."""
from __future__ import annotations

import json
import subprocess
import sys
from types import FunctionType, SimpleNamespace

from ..traces import file_digest
from . import vggt_pose_trial as base, vggt_task_trial as previous
from .vggt_task_prompt_v2 import PROMPT_VERSION, pose_note, task_binding

DESTINATION = base.ROOT.parent / "MMSI位姿题目适配-20261108-v2"
request = previous.request


def code_identity():
    identity = previous.code_identity()
    for name in ("vggt_task_prompt_v2.py", "vggt_task_trial_v2.py"):
        path = base.PACKAGE / name
        identity[str(path.relative_to(base.ROOT))] = file_digest(path)
    return identity


def controlled_execute(directory):
    from .depth_service_probe import all_keys
    keys = all_keys()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_task_trial_v2",
         "execute", "--directory", str(directory)],
        input=json.dumps(keys).encode(), env=base.clean_environment(), cwd=base.ROOT)
    if result.returncode:
        raise RuntimeError("online_worker_failed")


def make_engine():
    """Bind v2 context to the existing loop in a separate globals dictionary."""
    namespace = {
        **vars(base), **vars(previous), "PROMPT_VERSION": PROMPT_VERSION,
        "pose_note": pose_note, "task_binding": task_binding, "code_identity": code_identity,
        "DESTINATION": DESTINATION, "controlled_execute": controlled_execute,
    }
    names = ("prepare", "validate", "preview", "execute", "freeze",
             "validate_frozen", "score", "main")
    for name in names:
        source = getattr(previous, name)
        bound = FunctionType(source.__code__, namespace, name, source.__defaults__,
                             source.__closure__)
        bound.__kwdefaults__ = source.__kwdefaults__
        namespace[name] = bound
    return SimpleNamespace(**{name: namespace[name] for name in names})


_engine = make_engine()
prepare, validate, preview, execute, freeze, validate_frozen, score, main = (
    _engine.prepare, _engine.validate, _engine.preview, _engine.execute, _engine.freeze,
    _engine.validate_frozen, _engine.score, _engine.main)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"status":"failed","reason_code":"compact_pose_trial_failed"}', flush=True)
        raise SystemExit(2)
