"""Independent budget and shared OS slots, without modifying another run's ledger."""
from contextlib import contextmanager, ExitStack
import fcntl
from pathlib import Path
import time

from skill3d.synthesis.request_governor import RequestBudgetExhausted


class SharedSlots:
    def __init__(self, governor, shared_slots_root=None, *, reserve_direct=0, stop_event=None,
                 max_slots=4):
        self.governor = governor
        self.slots_root = Path(shared_slots_root) if shared_slots_root else None
        self.reserve_direct = reserve_direct
        self.stop_event = stop_event
        self.max_slots = max_slots

    @contextmanager
    def request(self, category, context):
        slot = None
        if self.slots_root:
            while slot is None:
                if self.stop_event is not None and self.stop_event.is_set():
                    raise RequestBudgetExhausted("run stopped while waiting")
                for index in range(self.max_slots):
                    stream = (self.slots_root / f"slot-{index:03d}.lock").open("a+b")
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        stream.close()
                        continue
                    slot = stream
                    break
                if slot is None:
                    time.sleep(0.05)
        try:
            if self.stop_event is not None and self.stop_event.is_set():
                raise RequestBudgetExhausted("run stopped before admission")
            # Reserve direct requests while unbounded tool sessions consume the remainder.
            # Use the same atomic governor lock that appends requests.
            with ExitStack() as stack:
                with (self.governor.root / "paired-admission.lock").open("a+b") as admission:
                    fcntl.flock(admission, fcntl.LOCK_EX)
                    with self.governor._locked():
                        state = self.governor._state()
                        direct_used = sum(r["context"].get("variant_id") == "direct"
                                          for r in state["requests"])
                        reserve = max(0, self.reserve_direct - direct_used)
                        if context.get("variant_id") != "direct" and (
                                len(state["requests"]) >= self.governor.config.max_requests - reserve):
                            raise RequestBudgetExhausted("direct baseline budget reserved")
                    receipt = stack.enter_context(self.governor.request(category, context))
                yield receipt
        finally:
            if slot is not None:
                slot.close()

    def summary(self):
        return self.governor.summary()
