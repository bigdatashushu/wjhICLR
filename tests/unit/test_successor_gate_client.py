"""Gate service and effective-wire failures must stop, with fixed error codes."""
import importlib.util
from pathlib import Path

import pytest

from skill3d.synthesis.request_audit import describe_request

_path = Path(__file__).resolve().parents[2] / "scripts/run_successor_gate_v11.py"
_spec = importlib.util.spec_from_file_location("successor_gate_test", _path)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


class Client:
    transport_retries = 3

    def audit_identity(self, **kwargs):
        return {"model": "test-model", **kwargs}

    def build_payload(self, *args, **kwargs):
        return {"messages": args[0], "model": "test-model", **kwargs}

    def chat(self, *args, **kwargs):
        self.last_request_identity = describe_request(
            self.build_payload(*args, **kwargs),
            model_identity=self.audit_identity(seed=kwargs["seed"]),
            wire_encoding_known=True)
        self.last_response_identity = {"model": "test-model"}
        return "received"


def test_gate_checks_effective_wire_identity_and_has_zero_retries():
    client = Client()
    checked = gate.CheckedClient(client, "test-model", 137)
    assert checked.transport_retries == 0
    assert checked.chat([{"role": "user", "content": "public input"}], seed=137) == "received"
    with pytest.raises(gate.GateClientError, match="gate_request_identity"):
        checked.chat([], seed=138)


def test_gate_service_exception_is_replaced_by_content_free_error():
    class Failed(Client):
        def chat(self, *args, **kwargs):
            raise RuntimeError("provider text must not escape")

    checked = gate.CheckedClient(Failed(), "test-model", 137)
    with pytest.raises(gate.GateClientError) as error:
        checked.chat([], seed=137)
    assert str(error.value) == "gate_service_failure"


def test_gate_rejects_returned_model_change():
    class Changed(Client):
        def chat(self, *args, **kwargs):
            text = super().chat(*args, **kwargs)
            self.last_response_identity["model"] = "changed"
            return text

    checked = gate.CheckedClient(Changed(), "test-model", 137)
    with pytest.raises(gate.GateClientError, match="gate_response_or_wire_identity"):
        checked.chat([], seed=137)
