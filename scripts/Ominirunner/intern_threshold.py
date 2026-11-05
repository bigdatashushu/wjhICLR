"""Isolated RGB-validity pilot: freeze cases, infer once, sweep probabilities."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import urllib.request

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from .common import digest, read, sha, write
from .intern_review import native_client
from .jev import Client, NoRedirect, Policy, host_review, request_payload, sampling_intent

VERSION = "omni-intern-threshold-pilot/1"
THRESHOLDS = (.50, .60, .70, .77, .80, .90)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def row(case):
    xy = case["points"]
    count = 1 if case["quantity"] == "camera_z" else 2
    if len(xy) != count:
        raise ValueError("pilot_point_count")
    data = {"view_index": 0, "jev_quantity": case["quantity"],
            "intended_targets": sampling_intent(case["target"], case["quantity"], count)}
    data.update({"pixel_xy": xy[0], "depth": 1.0} if count == 1 else {
        "a_xy": xy[0], "b_xy": xy[1], "distance": 1.0})
    return {"result_id": case["case_id"], "source_tool": "point_depth" if count == 1 else "pair_distance",
            "status": "ok", "data": data}


def inputs(directory, case, policy):
    path = (directory / case["image"]).resolve()
    if not path.is_relative_to(directory.resolve()) or sha(path) != case["rgb_sha256"]:
        raise ValueError("pilot_image_changed")
    rgb = path.read_bytes()
    measurement = row(case)
    request, identity = request_payload(measurement, rgb, policy)
    return rgb, measurement, request, identity


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def freeze(directory):
    manifest = read(directory / "cases.json")
    cases = manifest["cases"]
    policy = Policy.parse(manifest["policy"])
    if (manifest["schema"] != VERSION or policy.provider != "intern"
            or not 2 <= len(cases) <= 64 or policy.max_requests != len(cases)
            or set(manifest["thresholds"]) != set(THRESHOLDS)
            or len({c["case_id"] for c in cases}) != len(cases)
            or any(c["split"] not in ("selection", "validation")
                   or type(c["expected_usable"]) is not bool for c in cases)):
        raise ValueError("pilot_manifest_invalid")
    images = {split: {c["rgb_sha256"] for c in cases if c["split"] == split}
              for split in ("selection", "validation")}
    if not all(images.values()) or images["selection"] & images["validation"]:
        raise ValueError("pilot_split_overlap")
    source_root = Path(__file__).resolve().parent
    source = {p.name: sha(p) for p in sorted(source_root.glob("*.py"))}
    service = native_client(policy, opener()).identity()
    frozen = []
    for case in cases:
        _, measurement, request, identity = inputs(directory, case, policy)
        case_dir = directory / "cases" / case["case_id"]
        case_dir.mkdir(parents=True)
        for index, url in enumerate(request["images"]):
            raw = base64.b64decode(url.split(",", 1)[1], validate=True)
            (case_dir / f"input-{index}.png").write_bytes(raw)
        frozen.append({"case_id": case["case_id"], "measurement": measurement,
                       "identity": identity})
    write(directory / "preflight.json", {
        "schema": VERSION, "created_at": timestamp(), "manifest_sha256": digest(manifest),
        "source": source, "cases": frozen, "service_identity_sha256": digest(service),
        "calibration": service["calibration"], "model_revision": service["model_revision"],
        "threshold_comparison": "p_yes > threshold",
        "selection_rule": "minimize false admissions, then false rejections, then threshold",
        "total_request_cap": len(cases), "max_inflight": 1,
        "model_requests": 0, "solver_requests": 0, "benchmark_answers_read": False,
        "rgb_validity_annotations_visible_to_model": False, "automatic_retry": False,
        "measurement_values_are_placeholders": True}, exclusive=True)
    print(f"Frozen {len(cases)} cases; zero inference requests", flush=True)


def checked(directory):
    manifest, preflight = read(directory / "cases.json"), read(directory / "preflight.json")
    if preflight["schema"] != VERSION or digest(manifest) != preflight["manifest_sha256"]:
        raise ValueError("pilot_manifest_changed")
    for name, expected in preflight["source"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("pilot_source_changed")
    policy = Policy.parse(manifest["policy"])
    frozen = {r["case_id"]: r for r in preflight["cases"]}
    for case in manifest["cases"]:
        _, measurement, _, identity = inputs(directory, case, policy)
        if frozen[case["case_id"]] != {
                "case_id": case["case_id"], "measurement": measurement, "identity": identity}:
            raise ValueError("pilot_input_changed")
        for index, expected in enumerate(identity["review_image_sha256"]):
            if sha(directory / "cases" / case["case_id"] / f"input-{index}.png") != expected:
                raise ValueError("pilot_sent_image_changed")
    return manifest, preflight, policy


def run(directory, split):
    manifest, preflight, policy = checked(directory)
    if split == "validation":
        selection = read(directory / "selection.json")
        if (selection["manifest_sha256"] != preflight["manifest_sha256"]
                or selection["selection_rule"] != preflight["selection_rule"]):
            raise ValueError("pilot_selection_changed")
    service = native_client(policy, opener()).identity()
    if digest(service) != preflight["service_identity_sha256"]:
        raise ValueError("pilot_service_changed")
    selected = [c for c in manifest["cases"] if c["split"] == split]
    random.Random(137).shuffle(selected)
    started = directory / f"{split}-started.json"
    write(started, {"started_at": timestamp(), "manifest_sha256": digest(manifest),
                    "case_order": [c["case_id"] for c in selected],
                    "request_cap": len(selected), "automatic_retry": False,
                    **({"selection_sha256": sha(directory / "selection.json")}
                       if split == "validation" else {})}, exclusive=True)
    governor = RequestGovernor(directory / f"{split}-requests",
                               RequestGovernorConfig(len(selected), 1))

    class Budget:
        current = ""

        @contextmanager
        def request(self, category="gate"):
            with governor.request(category, {
                    "campaign_id": directory.name, "variant_id": VERSION,
                    "slot_key": self.current, "run_identity": digest(manifest)}) as charge:
                yield charge

    budget = Budget()
    reviewer = Client(policy, budget, opener=opener())
    for case in selected:
        budget.current = case["case_id"]
        rgb, measurement, _, _ = inputs(directory, case, policy)
        receipt = reviewer.review(measurement, rgb)
        write(directory / "cases" / case["case_id"] / "receipt.json", receipt, exclusive=True)
        host_review(measurement, receipt, rgb, policy)
        if receipt["status"] != "reviewed":
            write(directory / f"{split}-stop.json", {
                "at": timestamp(), "case_id": case["case_id"],
                "error_code": receipt.get("error_code"), "http_status": receipt.get("http_status"),
                "request_count": reviewer.count, "automatic_retry": False}, exclusive=True)
            print(f"Stopped at {case['case_id']}; no automatic retry", flush=True)
            return
        p_yes = receipt["binary_answer"]["probabilities"]["yes"]
        print(f"{case['case_id']}: p_yes={p_yes:.6f}", flush=True)
    write(directory / f"{split}-completed.json", {
        "completed_at": timestamp(), "model_requests": reviewer.count,
        "case_count": len(selected), "solver_requests": 0, "automatic_retry": False}, exclusive=True)


def scores(cases, threshold):
    good = sum(c["expected_usable"] for c in cases)
    bad = len(cases) - good
    admitted = [c for c in cases if c["p_yes"] > threshold]
    tp = sum(c["expected_usable"] for c in admitted)
    fp = len(admitted) - tp
    return {"threshold": threshold, "n": len(cases), "valid": good, "invalid": bad,
            "true_admissions": tp, "false_admissions": fp,
            "false_rejections": good-tp, "true_rejections": bad-fp,
            "precision": tp/len(admitted) if admitted else None,
            "valid_retention": tp/good if good else None,
            "false_admission_rate": fp/bad if bad else None,
            "rejection_rate": (len(cases)-len(admitted))/len(cases) if cases else None}


def analyze(directory, split):
    manifest, preflight, policy = checked(directory)
    cases = []
    for case in manifest["cases"]:
        if case["split"] not in (("selection", "validation") if split == "all" else (split,)):
            continue
        path = directory / "cases" / case["case_id"] / "receipt.json"
        if not path.is_file():
            raise ValueError("pilot_results_incomplete")
        receipt = read(path)
        if receipt["status"] != "reviewed":
            raise ValueError("pilot_operational_failure")
        rgb, measurement, _, _ = inputs(directory, case, policy)
        host_review(measurement, receipt, rgb, policy)
        cases.append({**case, "p_yes": receipt["binary_answer"]["probabilities"]["yes"],
                      "receipt_sha256": receipt["receipt_sha256"],
                      "request_sha256": receipt["request_sha256"]})
    groups = {"all": cases,
              "point": [c for c in cases if c["quantity"] == "camera_z"],
              "dimension": [c for c in cases if c["quantity"] != "camera_z"]}
    tables = {name: [scores(rows, t) for t in THRESHOLDS] for name, rows in groups.items()}
    result = {"schema": VERSION, "manifest_sha256": preflight["manifest_sha256"],
              "split": split, "cases": cases, "tables": tables,
              "selection_rule": preflight["selection_rule"]}
    if split == "selection":
        result["chosen"] = {name: min(table, key=lambda s: (
            s["false_admissions"], s["false_rejections"], s["threshold"])) for name, table in tables.items()}
        write(directory / "selection.json", result, exclusive=True)
    else:
        write(directory / f"{split}-analysis.json", result)
    for name, table in tables.items():
        print(name, [(s["threshold"], s["false_admissions"], s["false_rejections"]) for s in table])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "validation", "all"), default="selection")
    args = parser.parse_args()
    directory = args.root.resolve()
    if args.action == "freeze":
        freeze(directory)
    elif args.action == "run":
        if args.split == "all":
            parser.error("Run selection and validation separately")
        run(directory, args.split)
    else:
        analyze(directory, args.split)


if __name__ == "__main__":
    main()
