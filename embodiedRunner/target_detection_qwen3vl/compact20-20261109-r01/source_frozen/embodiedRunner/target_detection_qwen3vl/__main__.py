"""python -m target_detection_qwen3vl check|probe|detect-image|run"""
import argparse
import base64
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "target_detection_qwen3vl"

from target_detection_r05.__main__ import check
from target_detection_r05.client import ModelClient
from target_detection_r05.transport import Session, preflight
from .config import load_config
from .detector import Detector
from .perception import Perception
from .runner import Journal, save, suite
from .sources import ContractError, LIMITS, OperationalError, ROOT, Sources, canonical, digest, implementation_hashes


def detect_image(config, image_path, categories, directory):
    if not categories or len(categories) > 4 or len(set(categories)) != len(categories):
        raise ContractError("require_one_to_four_unique_categories")
    raw = image_path.read_bytes()
    directory.mkdir(exist_ok=False)
    journal = Journal(directory)
    try:
        public = {"step": 0, "instruction": "Detect " + " and ".join(categories) + ".",
                  "raw_rgb": {"sha256": digest(raw), "base64": base64.b64encode(raw).decode()},
                  "anonymous_anchors": []}
        perception = Perception(config, directory, journal)
        perception.bind(public, 0)
        targets = [{"id": f"T{i + 1}", "role": "object", "category": category,
                    "source_phrase": category, "quantity": None, "attributes": [],
                    "relation": "", "aliases": []} for i, category in enumerate(categories)]
        perception.set_targets(targets)
        result, images = perception.tool({"name": "detect_targets", "target_ids": [r["id"] for r in targets]})
        crop_records = []
        for offset in range(0, len(perception.groups), 2):
            crop_result, _ = perception.tool({
                "name": "inspect_candidates",
                "group_ids": [g["id"] for g in perception.groups[offset:offset + 2]]})
            crop_records.extend(crop_result["crops"])
        save(directory / "detections.json", {
            "status": "detection_integration_verified", "model": config["detection"]["service"]["model"],
            "input_file": str(image_path), "input_sha256": digest(raw),
            "reports": list(perception.cache.values()), "groups": result["groups"],
            "overlay": images[0]["file"], "crops": crop_records,
            "detector_requests": perception.detector_calls, "planning_requests": 0,
            "simulator_launches": 0, "reference_answer_visible": False, "automatic_retry": False,
            "implementation_hashes": implementation_hashes(),
        })
        return {"status": "detection_integration_verified", "output_directory": str(directory),
                "detector_requests": perception.detector_calls, "groups": len(perception.groups),
                "crops": len(crop_records), "planning_requests": 0, "simulator_launches": 0}
    finally:
        journal.close()


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-8B detector in the existing RGB-D action pipeline")
    parser.add_argument("action", choices=("check", "probe", "detect-image", "run"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"))
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--manipulation-catalog", type=Path,
                        default=ROOT / "official_direct/assets/manipulation_assets.json")
    parser.add_argument("--environment", choices=tuple(LIMITS))
    parser.add_argument("--run-id")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--category", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config, tasks = load_config(args.config)
        if args.action == "detect-image":
            if args.image is None or args.output is None:
                raise ContractError("detect_image_requires_image_and_output")
            print(canonical(detect_image(config, args.image, args.category, args.output)).decode())
            return 0
        if args.environment:
            tasks = [t for t in tasks if t.environment == args.environment]
            if not tasks:
                raise ContractError("no_selected_episode_for_environment")
        sources = Sources(args.source_root, args.manipulation_catalog)
        report = check(config, tasks, sources)
        if args.action == "check":
            report.update(detector_backend="qwen3vl8b", detector_model=config["detection"]["service"]["model"],
                          effective_n_shots=config["n_shots"],
                          implementation_hashes=implementation_hashes())
            print(canonical(report).decode())
            return 0
        detector_status = Detector(config).status()
        if args.action == "probe":
            reports = {t.identity: preflight(config, t, sources) for t in tasks}
            print(canonical({"status": "runtime_ready", "reports": reports, "detector": detector_status,
                             "simulator_launches": 0, "model_requests": 0}).decode())
            return 0
        run_id = args.run_id or ("qwen3vl-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                + "-" + uuid.uuid4().hex[:8])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
            raise ContractError("invalid_run_identity")
        directory = Path(__file__).parent / "runs" / run_id
        client = ModelClient(config["model"])
        client.assert_available()
        def factory(task, local_dir, journal):
            return Session(config, task, local_dir, journal, sources, run_id)
        result = suite(tasks, config, sources, client, factory, directory)
        print(canonical({"run_directory": str(directory), **result}).decode())
        return 0 if result["status"] == "finished" else 2
    except (ContractError, OperationalError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, OperationalError) else "local_configuration_or_source_error"
        print(canonical({"status": "failed", "error_code": code, "exception_type": type(exc).__name__}).decode())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
