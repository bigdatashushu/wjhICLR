"""One persistent request cap for M5 parser, Qwen description and binary calls."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time

from skill3d.infra.m5_identity import content_sha256

ENV = "VSI8B_PREPARATION_BUDGET"


def reserve(provider, request):
    filename = os.environ.get(ENV)
    if not filename:
        return
    path = Path(filename)
    with path.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = json.load(stream)
        if state["used"] >= state["max_requests"]:
            raise RuntimeError("M5 preparation request cap exhausted")
        state["used"] += 1
        state["requests"].append({
            "number": state["used"], "provider": provider,
            "request_sha256": content_sha256(request), "reserved_at": time.time(),
        })
        stream.seek(0)
        stream.truncate()
        json.dump(state, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
