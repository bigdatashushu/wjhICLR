import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest

from skill3d.synthesis.request_governor import (
    GovernedClient, RequestBudgetExhausted, RequestGovernor, RequestGovernorConfig,
)


def _request_worker(root, config, index, count, die=False):
    governor = RequestGovernor(root, config)
    categories = ("solver", "reviser", "gate")
    with governor.request(categories[index % 3], {"variant_id": str(index)}):
        if die:
            os._exit(17)
        marker = Path(root) / f"ready-{index}.json"
        marker.write_text("{}")
        deadline = time.monotonic() + 30
        while len(list(Path(root).glob("ready-*.json"))) < count:
            if time.monotonic() > deadline:
                raise RuntimeError("request cap prevented expected overlap")
            time.sleep(0.02)
        time.sleep(0.1)


@pytest.mark.parametrize("count", [16, 32])
def test_processes_share_exact_inflight_cap_and_global_budget(tmp_path, count):
    ctx = multiprocessing.get_context("spawn")
    config = RequestGovernorConfig(max_requests=count, max_inflight=count)
    processes = [ctx.Process(target=_request_worker, args=(tmp_path, config, i, count))
                 for i in range(count)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(40)
        assert process.exitcode == 0
    governor = RequestGovernor(tmp_path, config)
    summary = governor.summary()
    assert summary["peak_inflight"] == count
    assert summary["reserved_requests"] == summary["completed_requests"] == count
    assert set(summary["categories"]) == {"solver", "reviser", "gate"}
    with pytest.raises(RequestBudgetExhausted):
        with governor.request("solver", {}):
            pytest.fail("exhausted budget reached the API")
    assert RequestGovernor(tmp_path, config).summary() == summary


def test_death_releases_permit_but_unknown_call_stays_charged(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    config = RequestGovernorConfig(max_requests=2, max_inflight=1)
    process = ctx.Process(target=_request_worker, args=(tmp_path, config, 0, 0, True))
    process.start()
    process.join(10)
    assert process.exitcode == 17
    governor = RequestGovernor(tmp_path, config)
    with governor.request("gate", {}):
        pass
    assert governor.summary()["unknown_requests"] == 1
    assert governor.summary()["reserved_requests"] == 2
    with pytest.raises(RequestBudgetExhausted):
        with governor.request("reviser", {}):
            pass


def test_more_contenders_than_permits_finish_without_exceeding_cap(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    config = RequestGovernorConfig(max_requests=32, max_inflight=16)
    processes = [ctx.Process(target=_request_worker, args=(tmp_path, config, i, 16))
                 for i in range(32)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(40)
        assert process.exitcode == 0
    summary = RequestGovernor(tmp_path, config).summary()
    assert summary["peak_inflight"] == 16
    assert summary["completed_requests"] == 32


class Client:
    last_usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
                  "unsafe": "private", "invalid": float("inf")}

    def chat(self, *_args, **_kwargs):
        return "answer"


def test_wrapper_records_usage_only_and_policy_cannot_change(tmp_path):
    config = RequestGovernorConfig(3, 1, requests_per_second=100)
    governor = RequestGovernor(tmp_path, config)
    client = GovernedClient(Client(), governor, category="solver",
                            context={"variant_id": "qa-1", "api_key": "SECRET"})
    assert client.chat([{"content": "PRIVATE PROMPT"}]) == "answer"
    assert governor.summary()["usage"] == {
        "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    text = (tmp_path / "state.json").read_text()
    assert "SECRET" not in text and "PRIVATE PROMPT" not in text
    with pytest.raises(ValueError, match="policy changed"):
        RequestGovernor(tmp_path, RequestGovernorConfig(4, 1))


def test_summary_distinguishes_live_requests_from_unknown_completion(tmp_path):
    governor = RequestGovernor(tmp_path, RequestGovernorConfig(2, 1))
    with governor.request("solver", {}):
        summary = governor.summary()
        assert summary["active_requests"] == 1
        assert summary["unknown_requests"] == 0
    assert governor.summary()["active_requests"] == 0
    assert governor.summary()["completed_requests"] == 1
