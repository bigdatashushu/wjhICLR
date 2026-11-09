"""Pickle-safe successor Gate execution against the selected native runtime."""
from pathlib import Path
import hashlib
import json
import time

from skill3d.evaluation.fitset_v11 import SystemVariantRunner, content_sha256
from skill3d.evolution.fitset_stream_v11 import SeededGateClient


class GateAbort(BaseException):
    """A service, identity or budget refusal escapes model failure handlers."""


class GateClientError(RuntimeError):
    """Fixed-code error recorded by the native seeded-client receipt writer."""


class CheckedClient:
    def __init__(self, client, model, seed):
        self.client, self.model, self.seed = client, model, seed
        self.calls = 0
        client.transport_retries = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, *args, **kwargs):
        from skill3d.synthesis.request_audit import describe_request
        if (self.calls >= 64 or kwargs.get("seed") != self.seed
                or self.client.transport_retries != 0
                or self.client.audit_identity(seed=self.seed).get("model") != self.model):
            raise GateClientError("gate_request_identity_or_episode_budget")
        expected = describe_request(
            self.client.build_payload(*args, **kwargs),
            model_identity=self.client.audit_identity(seed=self.seed),
            wire_encoding_known=True)
        self.calls += 1
        try:
            result = self.client.chat(*args, **kwargs)
        except Exception:
            raise GateClientError("gate_service_failure") from None
        if (self.client.last_response_identity.get("model") != self.model
                or content_sha256(self.client.last_request_identity) != content_sha256(expected)):
            raise GateClientError("gate_response_or_wire_identity")
        return result


class FailClosedSeededClient(SeededGateClient):
    def chat(self, *args, **kwargs):
        try:
            return super().chat(*args, **kwargs)
        except Exception as exc:
            code = str(exc) if isinstance(exc, GateClientError) else "gate_client_refused"
            raise GateAbort(code) from None


class VariantClientFactory:
    def __init__(self, config, endpoint, model, seed, output, framework_state_io):
        self.config, self.endpoint, self.model = config, endpoint, model
        self.seed, self.output = seed, Path(output)
        self.framework_state_io = framework_state_io

    def __call__(self, variant_id):
        from run_fitset_v11 import _RevisionClientFactory
        from skill3d.evaluation.fitset_v11 import _variant_key
        from skill3d.online.config import online_model_from
        raw = _RevisionClientFactory(
            self.config, online_model_from(self.config).max_tokens, self.endpoint)()
        if not raw.configured:
            raise GateAbort("gate_service_unconfigured")
        return FailClosedSeededClient(
            CheckedClient(raw, self.model, self.seed), self.seed,
            self.output / "request_seeds" / (_variant_key(variant_id) + ".json"),
            framework_state_io=self.framework_state_io)


class CheckedFactory:
    def __init__(self, config, endpoint, model, framework_state_io):
        self.config, self.endpoint, self.model = config, endpoint, model
        self.framework_state_io = framework_state_io

    def __call__(self, seed, output):
        return VariantClientFactory(
            self.config, self.endpoint, self.model, seed, output, self.framework_state_io)


class IsolatedRunner(SystemVariantRunner):
    def _write_create_once(self, path, value):
        if not Path(path).is_relative_to(self._output_scope):
            raise GateAbort("gate_write_scope")
        return super()._write_create_once(path, value)

    def _prepare_one(self, variant, *, preflight=False):
        prepared = super()._prepare_one(variant, preflight=preflight)
        plan = {key: value for key, value in prepared["plan"].items() if key != "skill_identity"}
        key = str(self.root.relative_to(self._output_scope)) + "/" + str(variant["variant_id"])
        expected = getattr(self, "_prepared_identity", {}).get(key)
        if expected is not None and content_sha256(plan) != expected:
            raise GateAbort("gate_prepared_input_identity")
        return prepared

    def _run_one(self, variant):
        result = super()._run_one(variant)
        if result.get("service_errors"):
            raise GateAbort("gate_episode_service_failure")
        return result

    def _run_gate_episodes(self, episodes):
        # Check every arm before starting the native bounded pool. Resume/retry
        # of a paid successor Gate requires a separate explicit audit.
        for arm, variant in episodes:
            key = str(arm.root.relative_to(self._output_scope)) + "/" + str(variant["variant_id"])
            if key not in self._prepared_identity:
                raise GateAbort("gate_unprepared_episode")
            if arm._result_path(variant["variant_id"]).exists():
                raise GateAbort("gate_existing_result_no_automatic_resume")
        super()._run_gate_episodes(episodes)
        for arm, variant in episodes:
            key = str(arm.root.relative_to(self._output_scope)) + "/" + str(variant["variant_id"])
            result = json.loads(arm._result_path(variant["variant_id"]).read_text())
            if result.get("service_errors"):
                raise GateAbort("gate_episode_service_failure")
            self.framework_state_io.write_json_once(
                arm.root / "progress" / (hashlib.sha256(key.encode()).hexdigest() + ".json"), {
                    "status": result["status"], "requests": result.get("model_request_count"),
                    "at": time.time(), "gate_request_summary": self._request_governor().summary()})
