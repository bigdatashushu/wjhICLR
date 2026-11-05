"""Offline checks for real concurrent permits and preserved policy transitions."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Barrier

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from .common import IDS, concurrency, digest, read, write
from .rerun import transition
from .reuse import accounted_requests, request_cap


def main():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        shared = root / "shared"
        RequestGovernor(shared, RequestGovernorConfig(20000, 4))
        state = read(shared / "state.json")
        state["requests"].append({"request_id": 0, "status": "started", "category": "solver",
                                 "context": {}, "slot": 0, "started_at": 1})
        write(shared / "state.json", state)
        prefix = digest(state["requests"])
        with (shared / "slot-000.lock").open("a+b") as live:
            fcntl.flock(live, fcntl.LOCK_EX)
            try:
                transition(shared, 16, root / "should-not-exist.json")
            except BlockingIOError:
                pass
            else:
                raise AssertionError("active permits allowed policy mutation")
        assert read(shared / "state.json")["policy"]["max_inflight"] == 4
        transition(shared, 30, root / "to30.json")
        gate = Barrier(30)

        def request(index):
            governor = RequestGovernor(shared, RequestGovernorConfig(20000, 30))
            with governor.request("solver", {"slot_key": index}) as receipt:
                gate.wait(timeout=20)
                receipt["usage"] = {"total_tokens": index}

        with ThreadPoolExecutor(max_workers=30) as pool:
            list(pool.map(request, range(30)))
        state = read(shared / "state.json")
        assert state["peak_inflight"] == 30
        assert len(state["requests"]) == 31 and digest(state["requests"][:1]) == prefix
        assert all(r["status"] == "completed" for r in state["requests"][1:])
        transition(shared, 4, root / "to4.json")
        restored = read(shared / "state.json")
        assert restored["requests"] == state["requests"]
        assert len(restored["policy_transitions"]) == 2
        assert request_cap(dict.fromkeys(IDS[:15])) == 125
        assert request_cap(dict.fromkeys(IDS)) == 120
        assert request_cap({}, tuple(str(i) for i in range(120, 135))) == 105
        assert accounted_requests([
            {"arm": "direct", "sample_id": "1", "request_count": 1},
            {"arm": "direct", "sample_id": "2", "request_count": 1},
            {"arm": "skill", "sample_id": "1", "request_count": 6}], {"1": {}}) == 7
        assert concurrency(20) == 20 and concurrency(40) == 40
        for invalid in (0, 41, True, 20.0):
            try:
                concurrency(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid concurrency accepted")
    print("passed: 30 live permits, active-slot rejection, immutable unknown rows, "
          "4→30→4 restoration, cap105/cap125/cap120 and reused-charge exclusion; model requests=0")


if __name__ == "__main__":
    main()
