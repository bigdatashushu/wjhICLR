"""Real runner + fake model: paging, old crop reuse, and early finalization."""

import json

from skill3d.online import runner
from test_structured_finalizer_loop import Client, response, source  # noqa: F401


def execute(source, tmp_path, replies, **settings):
    item, artifact = source
    client = Client(replies)
    out = runner.run_episode(
        item.episode, item.pixels, runner.OnlineRunConfig(
            mode="real", reuse_artifact=artifact, max_solver_rounds=7,
            eval_visual_fallback=False, trace_dir=str(tmp_path), **settings), llm=client)
    return out, client


def text(call):
    return "\n".join(
        msg["content"] if isinstance(msg["content"], str) else "\n".join(
            p["text"] for p in msg["content"] if p["type"] == "text")
        for msg in call[1])


def test_repeated_successful_calls_finalize_early_within_budget(source, tmp_path):
    code = 'r = inspect_frames([31], [[0, 0, 50, 40]])\nYieldObservations([r["result_id"]], "inspect")'
    # Round 1 creates crop; round 2 first completes that image; 3 and 4 add nothing.
    out, client = execute(source, tmp_path, [code, code, code, code, response()])
    assert out.answer == "3"
    assert len(client.calls) == out.agent_rounds == 5 < 7
    assert out.rounds[3]["progress"]["finalize_required"]
    assert client.calls[-1][0] == "finalization"
    assert "No new observation content" in text(client.calls[3])
    assert all(len(r["jpeg_sha256"]) == r["n_images"] for r in out.image_ledger["rounds"])
    assert all(r["n_images"] <= 32 for r in out.image_ledger["rounds"])


def test_old_crop_reference_is_resent_and_page_reaches_following_request(source, tmp_path):
    class ScriptClient(Client):
        def chat(self, messages, **kwargs):
            # Capture real IDs from the previous host feedback, never assume an ID format.
            import re
            self.calls.append(("program", messages, kwargs))
            prompt = text(self.calls[-1])
            if len(self.calls) == 1:
                return 'r = inspect_frames([31], [[0, 0, 50, 40]])\nYieldObservations([r["result_id"]], "crop")'
            result_id = re.search(r"inspect_frames.*?\[result_id=([^\]]+)\]", prompt).group(1)
            if len(self.calls) == 2:
                return f'ReadObservation({result_id!r}, fields=["frame_ids", "images"])\nYieldObservations([{result_id!r}], "read and resend old crop")'
            assert "Requested observation pages" in prompt
            page_text = prompt.split("Requested observation pages", 1)[1]
            assert '"source_tool": "inspect_frames"' in page_text
            assert '"frame_id": 31' in page_text
            return "ReturnAnswer(3)"
    item, artifact = source
    client = ScriptClient([])
    out = runner.run_episode(item.episode, item.pixels, runner.OnlineRunConfig(
        mode="real", reuse_artifact=artifact, max_solver_rounds=5,
        eval_visual_fallback=False, trace_dir=str(tmp_path)), llm=client)
    assert out.answer == "3"
    image_rounds = out.image_ledger["rounds"]
    assert image_rounds[1]["derived_image_ids"] == image_rounds[2]["derived_image_ids"]
    assert len(out.rounds[1]["observation_reads"]) == 1
    assert out.rounds[1]["progress"]["made_progress"]
    assert not out.rounds[1]["results"]  # reader never executes the original tool
