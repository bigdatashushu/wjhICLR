"""Exercise the additive path through the real solver, scorer and paired driver."""

import copy
import json

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.evaluation import skill_ablation_v11 as ab
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig
from skill3d.synthesis.request_context import current_request_phase
from skill3d.synthesis.vllm_client import ServiceUnavailable
from skill3d.trace.store import TraceStore

from test_tool_contract_recovery import _write_v6_artifact


class Client:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.last_usage = {"prompt_tokens": 123}

    def chat(self, messages, **kwargs):
        phase = current_request_phase()
        assert phase in {"program", "eval_visual_fallback"}
        self.calls.append((copy.deepcopy(messages), phase, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    item = load_synthetic_items(
        "inner_validation", question_types=["room_size_estimation"],
        frame_size=(120, 160), n_frames=32, seed=0)[0]
    artifact = _write_v6_artifact(tmp_path_factory.mktemp("eval-visual"), with_poses=True)
    item.episode.ground_truth = "12.5"
    return item, artifact


@pytest.fixture(autouse=True)
def no_m5_requests(monkeypatch):
    monkeypatch.setattr(runner, "_bind_objects_best_effort",
                        lambda *args, **kwargs: ([], {}, [], False))


def run_case(source, tmp_path, replies, *, split="inner_validation", enabled=True,
             max_rounds=2):
    item, artifact = source
    episode = item.episode.model_copy(deep=True, update={"split": split})
    cfg = OnlineRunConfig(
        mode="real", reuse_artifact=artifact, eval_visual_fallback=enabled,
        max_solver_rounds=max_rounds, finalization_rounds=1, seed=7,
        allow_final_test=split == "final_test", trace_dir=str(tmp_path / "traces"))
    client = Client(replies)
    out = runner.run_episode(
        episode, item.pixels, cfg, llm=client, trace_store=TraceStore(cfg.trace_dir))
    return out, client


@pytest.mark.parametrize("split", ["inner_validation", "outer_holdout", "final_test"])
def test_failure_gets_exactly_one_clean_visual_answer(source, tmp_path, split):
    out, client = run_case(
        source, tmp_path,
        ["r = euclidean_distance([0,0,0], [1,0,0])\nReturnAnswer(-1)",
         "ReturnAnswer(-1)", "12.5"], split=split)
    assert out.final_state == "answer" and out.answer == "12.5" and out.mra_value == 1.0
    assert out.agent_rounds == 2 and len(out.rounds) == 2
    assert len(client.calls) == 3 and out.finalization_used
    assert [phase for _, phase, _ in client.calls] == [
        "program", "program", "eval_visual_fallback"]
    messages = client.calls[-1][0]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert len(content) == 33
    assert content[1:] == client.calls[0][0][0]["content"][1:]
    text = content[0]["text"]
    for forbidden in ("ReturnAnswer", "AnswerPayload", "参考方法", "场景", "M11", "12.5",
                      "euclidean_distance", "上一轮"):
        assert forbidden not in text
    assert source[0].episode.question in text
    assert all(params["seed"] == 7 for _, _, params in client.calls)
    assert out.answer_basis == "visual_estimate" and out.answer_source == "direct_vlm_routed"
    assert out.used_result_ids == [] and not out.answer_untrusted
    assert out.attribution.ledger.ignored_result_ids
    assert out.attribution.ledger.verified_used_result_ids == []
    audit = out.eval_visual_fallback
    assert audit["request_count"] == 1 and audit["accepted"]
    assert audit["original_result"]["failure_code"] == "geometry_rejected"
    assert audit["original_result"]["score"] == 0.0
    assert out.failure_code is None and out.terminal_failure == {}
    assert all(not r["submission_accepted"] for r in out.rounds)
    assert out.episode_trace.eval_visual_fallback == audit
    assert out.episode_trace.derivation_replay["passed"]
    for topic in ("trace_record", "evaluation_result"):
        row = json.loads((tmp_path / f"traces/{topic}.jsonl").read_text().splitlines()[0])
        assert row["eval_visual_fallback"] == audit
    image_round = audit["image_round"]
    assert image_round["n_images"] == 32 and image_round["derived_image_ids"] == []
    assert image_round["delivered"] and image_round["observed"]
    assert out.receipts_ok


def test_crop_from_failed_solver_never_enters_visual_request(source, tmp_path):
    out, client = run_case(source, tmp_path, [
        "img = inspect_frames([0])\nreturn YieldObservations([img], 'inspect')",
        "x = 1", "12.5"])
    assert out.final_state == "answer"
    # Original solver's finalization still receives the yielded crop.
    assert len(client.calls[1][0]) == 3
    assert "裁剪图" in client.calls[1][0][0]["content"][0]["text"]
    assert client.calls[-1][0][0]["content"][1:] == client.calls[0][0][0]["content"][1:]
    assert out.eval_visual_fallback["image_round"]["derived_image_ids"] == []


@pytest.mark.parametrize(("split", "enabled"), [
    ("induction", True), ("inner_validation", False)])
def test_learning_and_disabled_eval_keep_original_failure(source, tmp_path, split, enabled):
    out, client = run_case(source, tmp_path, ["x = 1", "x = 1"],
                           split=split, enabled=enabled)
    assert len(client.calls) == 2
    assert out.final_state == "run_error" and not out.eval_visual_fallback


@pytest.mark.parametrize("answer", ["12.5", "99"])
def test_accepted_answer_never_triggers_fallback_even_when_wrong(source, tmp_path, answer):
    out, client = run_case(source, tmp_path, [f"ReturnAnswer({answer})"])
    assert len(client.calls) == 1 and out.answer == answer
    assert not out.eval_visual_fallback
    assert out.mra_value == (1.0 if answer == "12.5" else 0.0)


@pytest.mark.parametrize("reply", ["not an answer", "ReturnAnswer(12.5)", "-1",
                                  ServiceUnavailable("offline")])
def test_failed_visual_answer_has_no_retry(source, tmp_path, reply):
    out, client = run_case(source, tmp_path, ["x = 1", "x = 1", reply])
    assert len(client.calls) == 3 and out.answer is None
    assert not out.eval_visual_fallback["accepted"]
    assert out.eval_visual_fallback["original_result"]["failure_code"] == "ast_violation"
    if isinstance(reply, Exception):
        assert out.final_state == "unavailable"
        assert not out.eval_visual_fallback["image_round"]["observed"]
    else:
        assert out.final_state == "run_error" and out.mra_value == 0.0
        assert out.eval_visual_fallback["image_round"]["observed"]


def test_original_service_failure_does_not_start_visual_fallback(source, tmp_path):
    out, client = run_case(source, tmp_path, [ServiceUnavailable("offline")])
    assert out.final_state == "unavailable" and len(client.calls) == 1
    assert not out.eval_visual_fallback


def test_pair_counts_visual_request_without_skill_delivery(source, tmp_path):
    item, artifact = source
    clients = {}
    def factory(arm, qa):
        clients[arm] = Client(["x = 1", "x = 1", "12.5"])
        return clients[arm]
    root = tmp_path / "pair"
    summary = ab.run_skill_ablation_v11(
        [copy.deepcopy(item)], artifact_paths={item.episode.qa_id: artifact},
        output_dir=root, base_cfg=OnlineRunConfig(
            mode="real", eval_visual_fallback=True, max_solver_rounds=2),
        seed=7, question_types=["room_size_estimation"], client_factory=factory)
    assert summary["status"] == "completed"
    pairs = [json.loads(line) for line in (root / "paired_results.jsonl").read_text().splitlines()]
    assert pairs[0]["eval_visual_common_request_sha256"]
    for arm, row in pairs[0]["arms"].items():
        assert row["delivery_ok"]
        assert row["model_request_count"] == 3 and row["program_request_count"] == 2
        assert row["eval_visual_request_count"] == 1 and row["eval_visual_accepted"]
        assert row["original_score"] == 0 and row["score"] == 1
        assert row["requests"][-1]["phase"] == "eval_visual_fallback"
        assert row["requests"][-1]["skill_content_sha256"] == {}
        assert row["requests"][-1]["request_sha256"] == row["eval_visual_fallback"]["request_sha256"]
        assert row["eval_visual_request_sha256"] == pairs[0][
            "eval_visual_common_request_sha256"]
        stats = summary["groups"]["all"]["arms"][arm]
        assert stats["original_MRA"] == 0 and stats["MRA"] == 1
        assert stats["mean_model_requests"] == 3 and stats["eval_visual_acceptance_rate"] == 1
    assert clients["B01"].calls[-1] == clients["B11"].calls[-1]
    assert summary["groups"] == ab.summarize_pairs(pairs)["groups"]
    mismatched = copy.deepcopy(pairs[0]["arms"]["B11"])
    mismatched["eval_visual_request_sha256"] = "0" * 64
    with pytest.raises(ab.PairingError, match="visual requests differ"):
        ab.pair_results(
            [pairs[0]["arms"]["B01"]], [mismatched], [item.episode.qa_id])
