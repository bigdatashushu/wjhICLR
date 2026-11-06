"""MMSI GPT Direct/Harness entry; the shared runtime remains a dependency."""
from __future__ import annotations

from pathlib import Path

from .prompts import direct_prompt
from .tools import registry_for


def solve(spec, client, output, run_identity):
    """Run one episode after GPTexp sets up a fresh subprocess runtime."""
    from ..adapters import problem_for
    from skill3d.small_image import runner
    from skill3d.small_image.blink_protocol import explicit_answer, original_messages, response_record
    from skill3d.small_image.storage import load_prepared

    problem = problem_for(spec["problem"])
    config = runner.SmallImageRunnerConfig(**spec["solver"])
    if spec["arm"] == "direct":
        text = client.chat(
            original_messages(problem, direct_prompt(problem)),
            max_tokens=config.max_tokens,
            seed=config.seed,
            temperature=0,
            enable_thinking=config.enable_thinking,
        )
        try:
            answer, method = explicit_answer(text, problem.options)
        except (ValueError, TypeError):
            answer, method = None, None
        return {
            "status": "answered" if answer else "invalid_answer",
            "answer": answer,
            "answer_parse_method": method,
            "request_count": 1,
            "rounds": [{"phase": "direct", **response_record(client, text)}],
            "attempted_tool_calls": 0,
            "succeeded_tool_calls": 0,
        }
    prepared = load_prepared(
        Path(spec["prepared_root"]) / problem.input_sha256, problem=problem
    )
    if prepared.artifact_sha256 != spec["artifact_sha256"]:
        raise ValueError("prepared_identity_changed")
    cells = []
    from .public_input import INTERFACE_VERSION
    from .solver import MMSISolver

    solver = MMSISolver(
        client,
        config=config,
        skill_text=spec["skill"]["text"],
        registry_builder=registry_for(spec["allowed_tools"]),
        execution_trace=cells,
    )
    result = solver.solve(problem, prepared=prepared, categories=prepared.categories).to_dict()
    result["execution_trace"] = cells
    result["interface_version"] = INTERFACE_VERSION
    result["interface_adaptations"] = solver.interface_adaptations
    result["interface_adaptation_count"] = sum(
        len(row["adaptations"]) for row in solver.interface_adaptations)
    return result
