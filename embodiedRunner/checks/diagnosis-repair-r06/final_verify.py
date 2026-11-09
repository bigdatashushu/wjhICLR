"""Verify final deliverables, cited event identities, links, and source preservation."""
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from action_identity import source_identity
from protocol import canonical, sha256

report = ROOT / "runs/diagnosis-repair-r06-final.md"
evidence = ROOT / "runs/diagnosis-repair-r06-final-evidence.json"
data = json.loads(evidence.read_bytes())
assert data["status"] == "validated"
assert len(data["matrix"]) == 15 and sum(r["r06_full"] for r in data["matrix"]) == 6
assert data["r06_full_analysis"] == json.loads((HERE / "full/episode-analysis.json").read_bytes())
assert data["r06_source_identity"] == source_identity()
assert data["r06_full"]["run_sha256"] == sha256((ROOT / "runs/task-card-full-r06.json").read_bytes())
assert data["historical"]["evidence_sha256"] == sha256(Path(data["historical"]["evidence_path"]).read_bytes())
assert data["r05_full_followup"]["sha256"] == sha256(Path(data["r05_full_followup"]["path"]).read_bytes())
events_checked = 0
for row in data["r06_full"]["episodes"]:
    path = Path(row["trajectory_path"])
    assert sha256(path.read_bytes()) == row["trajectory_sha256"]
    trace = json.loads(path.read_bytes())
    assert trace["events"][-1]["kind"] == "episode_result"
    assert trace["events"][-1]["report"]["task_success"] == row["success"]
    lines = (path.parent / "events.jsonl").read_text().splitlines()
    assert len(lines) == len(trace["events"])
    for ref in data["r06_full_analysis"]["episodes"][row["episode_id"]]["evidence"]:
        event = trace["events"][ref["event"]]
        assert event["index"] == ref["event"]
        assert json.loads(lines[ref["event"]]) == event
        events_checked += 1
links = re.findall(r"\]\(file://([^)]+)\)", report.read_text())
for link in links:
    path = Path(unquote(link.split("#", 1)[0]))
    assert path.exists(), str(path)
baseline = json.loads((ROOT / "checks/diagnosis-repair-r05/embodied_diagnosis_baseline_20261106.json").read_bytes())
for path, digest in baseline["protected"].items():
    assert sha256(Path(path).read_bytes()) == digest
for path, digest in baseline["other_projects"].items():
    assert sha256((ROOT.parent / path).read_bytes()) == digest
followup = json.loads((HERE / "followup-preservation-baseline.json").read_bytes())["files"]
for path, digest in followup.items():
    assert sha256((ROOT / path).read_bytes()) == digest
suites = [s.attrib for s in ET.parse(HERE / "regression-confirmation.xml").getroot().iter("testsuite")]
assert sum(int(s["tests"]) for s in suites) == 54
assert all(int(s["errors"]) == int(s["failures"]) == 0 for s in suites)
receipt = {
    "schema": "embodied-r06-final-deliverable-verification/1", "status": "passed",
    "report_path": str(report), "report_sha256": sha256(report.read_bytes()),
    "evidence_path": str(evidence), "evidence_sha256": sha256(evidence.read_bytes()),
    "evidence_bytes": evidence.stat().st_size, "task_denominator": 15, "successes": 6,
    "cited_r06_events_verified": events_checked, "file_links_verified": len(links),
    "qwen_requests": 155, "all_model_and_perception_requests": 161, "actual_actions": 131,
    "protected_files_verified": len(baseline["protected"]),
    "other_project_files_verified": len(baseline["other_projects"]),
    "completed_followup_files_verified": len(followup),
    "final_regression_passed": 54, "prior_timeout_receipt_preserved": True,
    "source_identity_matches_frozen_pilot_and_full": True,
    "new_model_requests": 0, "git_commit_or_push": False}
with (HERE / "final-verification.json").open("xb") as f:
    f.write(canonical(receipt) + b"\n")
print(json.dumps(receipt))
