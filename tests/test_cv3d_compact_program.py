"""Meaningful fake-only coverage for the revised CV3D main program protocol."""
import base64
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from skill3d.small_image.cv3d_protocol import (
    VERSION, FrozenCV3DSkills, OfficialCV3DPrompts,
)
from skill3d.small_image.datasets import freeze_manifest
from skill3d.small_image.precompute import SmallImagePrecomputer
from skill3d.small_image.runner import SmallImageRunnerConfig, SmallImageSolver
from skill3d.small_image.storage import save_prepared
from test_small_image_runner import ANSWER, YIELD, FakeClient, problem


def config(**changes):
    values = dict(
        prompt_profile=VERSION, evaluation_role="test_development",
        max_requests=6, max_solve_rounds=5, original_image_bytes=True,
        explicit_final_answer=True,
    )
    values.update(changes)
    return SmallImageRunnerConfig(**values)


def prompt(p):
    return p.question + "\n" + "\n".join(
        f"({label}) {text}" for label, text in zip("AB", p.options))


def prepared(p):
    h, w = 8, 10
    yy, xx = np.indices((h, w))
    return SmallImagePrecomputer(depth_estimator=lambda _: {
        "depth": 1 + xx / 10, "valid_mask": np.ones((h, w), dtype=bool),
        "points": np.stack([xx, yy, 1 + xx / 10], axis=-1),
    }, model_identities={"depth_estimator": {"model": "fake-depth"}}).process(p, categories=())


def solver(client, p, **changes):
    return SmallImageSolver(
        client, config=config(**changes), skill_text="Ground boxes, sample surfaces, compare.",
        official_prompt=prompt(p))


def test_main_uses_five_program_requests_and_one_finalizer(tmp_path):
    p = problem(tmp_path)
    client = FakeClient([YIELD] * 5 + ['{"answer":"B"}', ANSWER])
    result = solver(client, p)(p, prepared=prepared(p))
    assert result.status == "answered" and result.answer == "B"
    assert result.request_count == 6 and len(client.calls) == 6
    assert result.category_requests == 0
    assert result.solve_requests == 5 and result.finalization_requests == 1
    assert result.prompt_version == VERSION
    assert result.evaluation_role == "test_development"
    for ordinal, (messages, _) in enumerate(client.calls):
        text = messages[0]["content"][0]["text"]
        assert text.startswith(prompt(p) + "\n\n")
        assert f"Remaining model requests: {6 - ordinal}" in text
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert "Input: {" not in text and "BLINK/MMSI" not in text
        images = [part for part in messages[0]["content"] if part["type"] == "image_url"]
        assert len(images) == 1
        assert base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]) == (
            Path(p.image_paths[0]).read_bytes())
    final = client.calls[-1][0][0]["content"][0]["text"]
    assert "Finalization:" in final and "Program tools:" not in final


def test_rebuilt_prompt_delivers_values_but_not_previous_program(tmp_path):
    p = problem(tmp_path)
    program = YIELD.replace("response =", "prior_only_variable =").replace(
        "response['", "prior_only_variable['")
    client = FakeClient([program, ANSWER])
    result = solver(client, p)(p, prepared=prepared(p))
    assert result.status == "answered" and result.request_count == 2
    continuation = client.calls[1][0][0]["content"][0]["text"]
    assert '"source_tool": "view_depth_summary"' in continuation
    assert '"quantiles":' in continuation
    assert "prior_only_variable" not in continuation
    assert "Previous Python variables have been cleared" in continuation
    assert not any(m["role"] == "assistant" for m in client.calls[1][0])


def test_early_answer_uses_one_request_and_does_not_force_tool_measurements(tmp_path):
    p = problem(tmp_path)
    client = FakeClient([ANSWER])
    result = solver(client, p)(p, prepared=prepared(p))
    assert result.request_count == 1 and result.finalization_requests == 0
    assert result.attempted_tool_calls == 0


def test_original_byte_mode_never_calls_legacy_opencv_encoder(tmp_path, monkeypatch):
    from skill3d.small_image import runner
    p = problem(tmp_path)
    evidence = prepared(p)

    def forbidden(*args, **kwargs):
        pytest.fail("original-byte mode invoked the legacy image re-encoder")

    monkeypatch.setattr(runner, "build_image_messages", forbidden)
    client = FakeClient([ANSWER])
    result = solver(client, p)(p, prepared=evidence)
    assert result.status == "answered" and result.request_count == 1


@pytest.mark.parametrize("changes", [
    {"max_requests": 7}, {"max_requests": 5}, {"max_solve_rounds": 6},
    {"max_solve_rounds": 2}, {"original_image_bytes": False},
    {"evaluation_role": "benchmark"},
])
def test_six_round_protocol_rejects_wrong_budget_or_independent_test_claim(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_compact_profile_rejects_missing_prompt_or_other_dataset_before_request(tmp_path):
    p = problem(tmp_path)
    client = FakeClient([])
    run = SmallImageSolver(client, config=config(), skill_text="Ground targets.")
    assert run(p, prepared=prepared(p)).status == "input_error"
    other = problem(tmp_path, benchmark="mmsi")
    assert solver(client, other)(other, prepared=prepared(other)).status == "input_error"
    assert not client.calls


def test_depth_does_not_expose_distance_or_mask_geometry_tools(tmp_path):
    p = problem(tmp_path)
    client = FakeClient([ANSWER])
    solver(client, p)(p, prepared=prepared(p))
    text = client.calls[0][0][0]["content"][0]["text"]
    interfaces = text.split("Program tools:\n", 1)[1].split("Successful returns", 1)[0]
    assert "point_depth(" in interfaces
    assert "pair_point_distance(" not in interfaces and "candidate_depth_summary(" not in interfaces


def test_invalid_tool_ends_program_without_retry_or_visual_fallback(tmp_path):
    p = problem(tmp_path)
    client = FakeClient([
        "def solve(ctx):\n"
        "    r = ctx.tools.point_depth(view_index=0, x=9999, y=3)\n"
        "    YieldObservations([r['result_id']])",
        ANSWER,
    ])
    result = solver(client, p)(p, prepared=prepared(p))
    assert result.status == "run_error" and result.request_count == 1
    assert len(client.calls) == 1 and result.answer is None


def test_official_loader_keeps_prompt_and_excludes_labels(tmp_path):
    p = problem(tmp_path)
    file = tmp_path / "official.jsonl"
    file.write_text(json.dumps({
        "type": "3D", "task": p.task_type, "question": p.question,
        "choices": list(p.options), "prompt": prompt(p), "answer": "PRIVATE_GT_SENTINEL",
    }) + "\n")
    official = OfficialCV3DPrompts(file)
    assert official.for_problem(p) == prompt(p)
    assert "PRIVATE_GT_SENTINEL" not in repr(official.__dict__)
    wrong = replace(p, question=p.question + " altered")
    with pytest.raises(ValueError):
        official.for_problem(wrong)


def frozen_skills(tmp_path):
    folder = tmp_path / "skills"
    folder.mkdir()
    tasks = {}
    for task in ("Depth", "Distance"):
        path = folder / f"{task}.md"
        text = f"# {task}\nGround the boxes, measure representative surfaces, compare.\n"
        path.write_text(text)
        tasks[task] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = folder / "frozen.json"
    manifest.write_text(json.dumps({
        "schema": "cv3d-frozen-skills/1", "prompt_profile": VERSION,
        "evaluation_role": "test_development", "tasks": tasks,
    }))
    return manifest


def test_frozen_skill_mutation_or_missing_role_is_rejected(tmp_path):
    manifest = frozen_skills(tmp_path)
    frozen = FrozenCV3DSkills(manifest)
    assert set(frozen.texts) == {"Depth", "Distance"}
    (manifest.parent / "Depth.md").write_text("changed")
    with pytest.raises(ValueError):
        FrozenCV3DSkills(manifest)


def test_compact_cli_preflight_checks_existing_cache_without_requests(tmp_path, monkeypatch, capsys):
    p = problem(tmp_path)
    cache = tmp_path / "prepared"
    save_prepared(prepared(p), cache / "one")
    manifest = tmp_path / "inputs.json"
    freeze_manifest([p], manifest)
    official = tmp_path / "official.jsonl"
    official.write_text(json.dumps({
        "type": "3D", "task": p.task_type, "question": p.question,
        "choices": list(p.options), "prompt": prompt(p), "answer": "PRIVATE_GT_SENTINEL",
    }) + "\n")
    skills = frozen_skills(tmp_path)
    path = Path(__file__).resolve().parents[1] / "scripts/run_small_image.py"
    spec = importlib.util.spec_from_file_location("cv3d_compact_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def forbidden(*args, **kwargs):
        pytest.fail("preflight issued a model request or constructed GPU backends")

    monkeypatch.setattr(cli.BailianClient, "chat", forbidden)
    monkeypatch.setattr(cli, "build_precomputer", forbidden)
    args = [
        "--benchmark", "cv3d", "--manifest", str(manifest),
        "--dataset-root", str(tmp_path), "--prepared-root", str(cache),
        "--official-annotations", str(official), "--skills-manifest", str(skills),
    ]
    code = cli.main(args)
    assert code == 0
    row = json.loads(capsys.readouterr().out)
    assert row["status"] == "preflight" and row["request_count"] == 0
    assert row["prompt_version"] == VERSION and row["evaluation_role"] == "test_development"

    class ExecutionClient(FakeClient):
        model = "qwen3.8-flash"
        max_output_tokens = 8192
        last_response_identity = {"model": model, "model_reported": True}

        def __init__(self, **kwargs):
            super().__init__([YIELD] * 5 + ['{"answer":"B"}'])

        def build_payload(self, messages, **kwargs):
            return {"model": self.model, "messages": messages, **kwargs}

    monkeypatch.setattr(cli, "BailianClient", ExecutionClient)
    output = tmp_path / "execution"
    assert cli.main(args + [
        "--execute", "--output-dir", str(output), "--max-requests", "6",
    ]) == 0
    executed = json.loads(capsys.readouterr().out)
    assert executed["solve_requests"] == 5 and executed["finalization_requests"] == 1
    assert executed["evaluation_role"] == "test_development"
    assert len(json.loads((output / "budget/state.json").read_text())["requests"]) == 6
    assert "PRIVATE_GT_SENTINEL" not in (output / "manifest.json").read_text()
