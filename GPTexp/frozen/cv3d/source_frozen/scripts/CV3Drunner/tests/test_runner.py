import base64
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from CV3Drunner.config import RunnerConfig
from CV3Drunner.runner import CV3DSolver
from CV3Drunner.program import prepare_request
from fixtures import fixture, frozen

ANSWER = """def solve(ctx):
    # Reviewed the existing sampling patches against the visible object relation.
    ReturnAnswer(AnswerPayload(value='B', unit='option', basis='visual_estimate'))
"""
MEASURE_ANSWER = """def solve(ctx):
    r = ctx.tools.point_depth_in_box(0, 'v0:red', .5, .5)
    ReturnAnswer(AnswerPayload(value='A', unit='option', basis='mixed'))
"""
YIELD = """def solve(ctx):
    r = ctx.tools.point_depth_in_box(0, 'v0:red', .5, .5)
    YieldObservations([r['result_id']], reason='Red object is supported by its neighbor; inspect sampled surface against the visual ordering.')
"""


class FakeClient:
    model = "qwen3.8-flash"
    max_output_tokens = 8192

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages, self.parameters = [], []
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
        self.last_response_identity = {"model": self.model, "model_reported": True}

    def chat(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        self.parameters.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def solve(tmp_path, responses, *, task="Depth", config=None, image=None):
    public, _, prepared = fixture(tmp_path / "input", task=task, image=image)
    client = FakeClient(responses)
    governor = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(100, 1))
    result = CV3DSolver(
        client, config=config or RunnerConfig(), governor=governor,
        official_prompt=public.official_prompt, skill_text=frozen().texts[task]
    ).solve(public.problem, prepared=prepared)
    return result, client, public, governor


def test_same_cell_measured_answer_is_deferred_then_can_change_after_review(tmp_path):
    result, client, public, governor = solve(tmp_path, [MEASURE_ANSWER, ANSWER])
    assert result.status == "answered" and result.answer == "B"
    assert result.request_count == result.solve_requests == 2
    assert result.attempted_tool_calls == result.succeeded_tool_calls == 1
    assert result.rounds[0]["status"] == "measurement_review_required"
    assert result.rounds[0]["provisional_answer"] == "A"
    rid = result.rounds[0]["observations"][0]["result_id"]
    assert result.rounds[1]["delivered_measurement_ids"] == [rid]
    display = result.rounds[1]["sampling_overlay"]
    parts = client.messages[1][0]["content"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 3
    overlay = base64.b64decode(images[-1]["image_url"]["url"].split(",", 1)[1])
    assert hashlib.sha256(overlay).hexdigest() == display["png_sha256"]
    assert Image.open(io.BytesIO(overlay)).size == (180, 120)
    assert display["samples"][0]["result_id"] == rid and not display["surface_validated"]
    assert "provisional_answer" in parts[0]["text"] and "white" in parts[-2]["text"]
    assert governor.summary()["reserved_requests"] == 2
    for message in client.messages:
        original = [p for p in message[0]["content"] if p["type"] == "image_url"][0]
        assert base64.b64decode(original["image_url"]["url"].split(",", 1)[1]) == (
            Path(public.problem.image_paths[0]).read_bytes())


def test_measurements_restart_review_even_when_model_does_not_yield_them(tmp_path):
    result, client, _, _ = solve(tmp_path, [MEASURE_ANSWER, MEASURE_ANSWER, ANSWER])
    assert result.status == "answered" and result.request_count == 3
    ids = [row["observations"][0]["result_id"] for row in result.rounds[:2]]
    assert len(set(ids)) == 2
    assert result.rounds[2]["delivered_measurement_ids"] == ids
    assert all(row["status"] == "measurement_review_required" for row in result.rounds[:2])
    assert len(client.messages) == 3


def test_five_program_slots_plus_one_finalization_review_last_measurement(tmp_path):
    result, client, _, governor = solve(
        tmp_path, [MEASURE_ANSWER] * 5 + ['{"answer":"B","reasoning":"Surface relation is clearer."}'])
    assert result.status == "answered" and result.answer == "B"
    assert result.solve_requests == 5 and result.finalization_requests == 1
    assert result.request_count == governor.summary()["reserved_requests"] == 6
    assert result.attempted_tool_calls == 5
    assert len(result.rounds[-1]["delivered_measurement_ids"]) == 5
    assert client.parameters == [dict(
        max_tokens=8192, seed=137, temperature=0.0, enable_thinking=True)] * 6
    assert "Tools are disabled" in client.messages[-1][0]["content"][0]["text"]


def test_pure_visual_and_availability_summary_can_answer_immediately(tmp_path):
    result, _, _, _ = solve(tmp_path, [ANSWER])
    assert result.status == "answered" and result.request_count == 1
    assert result.basis == "visual_estimate" and result.succeeded_tool_calls == 0
    summary = ANSWER.replace("    # Reviewed", "    r = ctx.tools.view_depth_summary()\n    # Reviewed")
    result, client, _, _ = solve(tmp_path / "summary", [summary])
    assert result.request_count == 1 and result.succeeded_tool_calls == 1
    assert result.basis == "visual_estimate"


@pytest.mark.parametrize("source", [
    MEASURE_ANSWER.replace(".5, .5", "-1, .5"),
    MEASURE_ANSWER.replace("'v0:red'", "'invented'"),
    "def solve(ctx):\n    import os\n    ReturnAnswer('B')",
    "def solve(ctx):\n    x=ctx.tools.point_depth(0, 1, 1)\n    ReturnAnswer('B')",
    "def solve(ctx):\n    ReadObservation('x')\n    ReturnAnswer('B')",
])
def test_terminal_tool_and_security_errors_do_not_reset_or_retry(tmp_path, source):
    result, client, _, _ = solve(tmp_path, [source, ANSWER])
    assert result.status in {"run_error", "program_rejected"}
    assert result.answer is None and len(client.messages) == 1


def test_timeout_is_terminal_and_service_failure_is_not_retried(tmp_path):
    source = "def solve(ctx):\n    while True:\n        pass\n    ReturnAnswer('B')"
    result, client, _, _ = solve(
        tmp_path, [source, ANSWER], config=replace(RunnerConfig(), cell_timeout_s=.02))
    assert result.status == "run_error" and len(client.messages) == 1
    assert result.rounds[0]["execution_error"]["code"] == "timeout"
    result, client, _, governor = solve(tmp_path / "service", [TimeoutError(), ANSWER])
    assert result.status == "service_error" and len(client.messages) == 1
    assert governor.summary()["error_requests"] == 1


def test_repairable_python_error_keeps_successful_measurements_for_review(tmp_path):
    source = YIELD.replace("    YieldObservations", "    x = undefined_variable\n    YieldObservations")
    result, client, _, _ = solve(tmp_path, [source, ANSWER])
    assert result.status == "answered" and result.request_count == 2
    assert result.rounds[0]["correctable"] is True
    assert len(result.rounds[1]["delivered_measurement_ids"]) == 1
    assert result.rounds[1]["sampling_overlay"] is not None
    assert "NameError" in client.messages[1][0]["content"][0]["text"]


def test_missing_annotations_allow_visual_choice_and_no_raw_measurement(tmp_path):
    result, client, _, _ = solve(tmp_path, [ANSWER], image=Image.new("RGB", (180, 120), "gray"))
    assert result.status == "answered"
    assert "point_depth_in_box(" not in client.messages[0][0]["content"][0]["text"]


def test_new_profile_rejects_other_datasets_and_old_config_unchanged(tmp_path):
    public, directory, prepared = fixture(tmp_path)
    other = replace(public.problem, benchmark="blink")
    client = FakeClient([ANSWER])
    result = CV3DSolver(client, official_prompt=public.official_prompt,
                       skill_text=frozen().texts["Depth"]).solve(other, prepared=prepared)
    assert result.status == "input_error" and not client.messages
    from skill3d.small_image.runner import SmallImageRunnerConfig
    with pytest.raises(ValueError, match="prompt_profile"):
        SmallImageRunnerConfig(prompt_profile=RunnerConfig().prompt_profile)
    _, messages, direct_tools, png, direct = prepare_request(
        public, directory, arm="direct", frozen=frozen(), config=RunnerConfig())
    assert direct_tools == [] and png is None
    assert len([p for p in messages[0]["content"] if p["type"] == "image_url"]) == 1
