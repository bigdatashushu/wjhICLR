"""Request identity for local audit wrappers; never sent as model API parameters."""

from contextlib import contextmanager
from contextvars import ContextVar

_PHASE: ContextVar[str] = ContextVar("skill3d_request_phase", default="perception")


def current_request_phase() -> str:
    return _PHASE.get()


@contextmanager
def request_phase(phase: str):
    token = _PHASE.set(phase)
    try:
        yield
    finally:
        _PHASE.reset(token)
