"""CPU end-to-end contract experiment for continuous buffered evolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_streaming_contract_experiment_rejects_promotes_and_flushes_tail(tmp_path):
    output = tmp_path / "streaming-contract"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT / "src"), env.get("PYTHONPATH", "")])
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_streaming_evolution_contract_v11.py"),
            "--output",
            str(output),
            "--library-source",
            str(ROOT / "skill_library"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["attempt_outcomes"] == ["rejected", "promoted", "rejected"]
    assert summary["relative_distance"]["final_version"] == "1.1.0"
    assert summary["relative_distance"]["post_publish_buffer_cases"] == 1
    assert summary["counting_tail"]["final_version"] == "1.0.0"
    assert summary["counting_tail"]["buffer_cases"] == 5
    assert summary["operational_failure_count"] == 1
    assert all(summary["checks"].values())
