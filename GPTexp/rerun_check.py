"""Offline checks for transient continuation and immutable partial scoring."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from .api_pool import ApiPool
from .partial_score import freeze, score
from .run import execute
from .selfcheck import Opener
from .traces import digest, file_digest, read, write
from .transport import APIError, ResponsesClient


def main():
    config = read(Path(__file__).with_name("config.json"))
    config.update(max_requests=4, max_inflight=1, max_workers=2, apis=[config["apis"][0]])
    config["apis"][0]["validated"] = True
    config["service_failure_policy"] = {
        "mode": "cooldown_continue", "cooldown_seconds": 0.01,
        "transient_http_statuses": [524], "transient_reason_codes": ["transport_outcome_unknown"]}
    passed = []
    with tempfile.TemporaryDirectory(prefix="gptexp-rerun-check-") as temp:
        root = Path(temp)
        api = config["apis"][0]
        pool = ApiPool(root / "pool", config)
        with pool.request(api, "first-operation", 1, digest("payload")) as receipt:
            receipt["http_status"] = 524
            try:
                raise APIError("http_error", 524)
            except APIError:
                # Throw through the context manager, without allowing an HTTP retry.
                pass
        try:
            with pool.request(api, "failed-operation", 1, digest("payload")):
                raise APIError("http_error", 524)
        except APIError:
            pass
        assert not pool.paused(api) and pool.cooling(api)
        state = read(root / "pool/state.json")
        assert len(state["requests"]) == 2 and state["requests"][1]["status"] == "error"
        assert state["requests"][1]["failure_policy"] == "cooldown_continue"
        until = state["cooldowns"][api["id"]]["until"]
        with patch("GPTexp.api_pool.time.time", return_value=until + 1):
            with pool.request(api, "different-operation", 1, digest("next-payload")):
                pass
        assert len(read(root / "pool/state.json")["requests"]) == 3
        passed.append("single_failed_attempt_retained_then_different_operation_admitted_after_cooldown")
        try:
            with pool.request(api, "invalid-model-operation", 1, digest("payload")):
                raise APIError("actual_model_changed")
        except APIError:
            pass
        assert pool.paused(api)
        passed.append("model_identity_deviation_permanently_pauses_even_with_cooldown_policy")

        directory = root / "scheduler"
        pairs = []
        for index in range(2):
            pair = {"dataset": "blink-multiview", "sample_id": str(index),
                    "pair_identity": digest({"synthetic_pair": index}),
                    "arms": [{"arm": arm, "problem": {"sample_id": str(index)}} for arm in ("direct", "harness")]}
            pairs.append(pair)
        manifest = {"counts": {"blink-multiview": 2}, "questions": 2, "arms": 4, "config": config,
                    "pairs": pairs, "synthetic_fixture": True}
        manifest["run_identity"] = digest(manifest)
        write(directory / "manifest.json", manifest)

        class Worker:
            pid = 12345

            def __init__(self, command, **kwargs):
                self.output = Path(command[command.index("--output") + 1])
                self.returncode = 0

            def communicate(self, data, timeout):
                job = json.loads(data)
                synthetic_pool = ApiPool(job["pool_root"], job["config"])
                failure = job["sample_id"] == "0" and job["spec"]["arm"] == "direct"
                try:
                    with synthetic_pool.request(api, job["operation_id"], 1, digest("synthetic")):
                        if failure:
                            raise APIError("http_error", 524)
                except APIError:
                    self.returncode = 2
                requests = {}
                if failure:
                    path = self.output / "requests/001.response.json"
                    write(path, {"status": "error", "reason_code": "http_error", "http_status": 524})
                    requests["requests/001.response.json"] = file_digest(path)
                episode = {"dataset": "blink-multiview", "sample_id": job["sample_id"],
                           "arm": job["spec"]["arm"], "pair_identity": job["pair_identity"],
                           "run_identity": job["run_identity"], "api_id": api["id"],
                           "job_sha256": digest(job), "operation_id": job["operation_id"],
                           "reference_answer_visible": False, "request_files": requests,
                           "failure": {"error_type": "APIError"} if failure else None,
                           "result": None if failure else {"status": "answered", "answer": "A"}}
                write(self.output / "episode.json", episode)

        with patch("GPTexp.run.verify_manifest"), patch("GPTexp.run.subprocess.Popen", side_effect=Worker), patch(
                "GPTexp.run.subprocess.run") as host_action:
            execute(directory)
        assert read(directory / "execution.json")["status"] == "complete"
        assert sum(1 for _ in (directory / "episodes").rglob("episode.json")) == 4
        assert host_action.call_args.args[0][3] == "GPTexp.score"
        scheduler_pool = ApiPool(directory / "pool", config)
        assert not scheduler_pool.paused(api)
        assert len(read(directory / "pool/state.json")["requests"]) == 4
        assert len(read(directory / "pool/state.json")["service_failures"]) == 1
        passed.append("full_pass_keeps_failure_and_dispatches_all_remaining_pairs_without_budget_growth")

        # Freeze a live partial run with one unfinished arm.
        (directory / "episodes" / pairs[1]["dataset"] / pairs[1]["pair_identity"] / "harness/episode.json").unlink()
        snapshot = root / "partial"
        assert freeze(directory, snapshot)["completed_arms"] == 3
        refs = [{"dataset": p["dataset"], "sample_id": p["sample_id"], "reference_answer": "A"} for p in pairs]
        with patch.dict(os.environ, {}, clear=True), patch("GPTexp.partial_score.references_for", return_value=refs):
            output = score(snapshot)
        item = output["statistics"][0]
        assert output["common_pairs"] == 1 and output["both_legal_pairs"] == 0
        assert item["common_pairs"]["direct"]["correct"] == 0
        assert item["common_pairs"]["harness"]["correct"] == 1
        assert item["available"]["direct"]["records"] == 2
        assert item["available"]["direct"]["accuracy"] == 0.5
        assert output["full_dataset_accuracy"] is None
        passed.append("partial_scoring_uses_observed_denominators_common_pairs_and_failure_as_wrong")
        path = snapshot / read(snapshot / "snapshot.json")["index"][0]["episode"]
        path.write_text(path.read_text() + " ")
        with patch.dict(os.environ, {}, clear=True), patch("GPTexp.partial_score.references_for") as lookup:
            try:
                score(snapshot)
            except ValueError:
                pass
            else:
                raise AssertionError("changed_snapshot_accepted")
            lookup.assert_not_called()
        passed.append("snapshot_tampering_rejected_before_any_reference_access")
    print(json.dumps({"actual_model_requests": 0, "checks": passed}))


if __name__ == "__main__":
    main()
