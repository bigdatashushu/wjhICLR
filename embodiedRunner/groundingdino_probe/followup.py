"""Two single-category probes and four static Qwen density comparisons."""
import base64
import json
import os
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
from run_probe import annotate, call_service, canonical, filtered, save, sha
from rgbd_three_envs.client import ModelClient


def main():
    assert not (OUT / "followup-frozen.json").exists(), "followup_already_started"
    primary = json.loads((OUT / "detections.json").read_text())
    cases = {r["environment"]: r for r in primary["inputs"]}
    config = json.loads((ROOT / "rgbd_three_envs/experiment-20261107-r02.json").read_text())["model"]
    schedule = [("manipulation", "dense"), ("habitat", "filtered"),
                ("manipulation", "filtered"), ("habitat", "dense")]
    save(OUT / "followup-frozen.json", {
        "source_results_sha256": sha((OUT / "detections.json").read_bytes()),
        "single_category_queries": ["star.", "container."], "box_threshold": 0.15,
        "qwen_schedule": schedule, "qwen_configuration": config, "maximum_qwen_requests": 4,
        "maximum_extra_detector_requests": 2, "automatic_retry": False,
        "same_raw_rgb_and_depth_in_each_density_pair": True, "environment_actions": 0,
        "scope": "static target/candidate selection, not environment task evaluation",
    })
    png = (OUT / "inputs/manipulation-rgb.png").read_bytes()
    supplementary = []
    for name, prompt in (("star_only", "star."), ("container_only", "container.")):
        payload = {"image": base64.b64encode(png).decode(), "text_prompt": prompt, "box_threshold": 0.15}
        response = call_service({"mode": "infer", "payload": payload})
        assert response["model_sha256"] == primary["model_sha256"]
        save(OUT / "responses" / f"manipulation-{name}.json", response)
        rows = [{"id": i, **r} for i, r in enumerate(response["detections"])]
        supplementary.append({"name": name, "prompt": prompt, "result": annotate("manipulation", rows, name),
                              "request_sha256": response["request_sha256"],
                              "response_sha256": response["response_sha256"]})
        print(json.dumps({"probe": name, "boxes": len(rows), "labels": [r["label"] for r in rows]}), flush=True)
    save(OUT / "single-category-probes.json", supplementary)
    legends = {}
    for kind in ("manipulation", "habitat"):
        source = primary["results"][kind]["target_low"]
        rows = source["candidate_legend"]
        reduced = filtered(rows)
        rendered = annotate(kind, reduced, "target_filtered")
        legends[(kind, "dense")] = {"legend": rows, "image": source["image"]}
        legends[(kind, "filtered")] = {"legend": reduced, "image": rendered["image"]}
    key_file = ROOT.parent / "qwenkey.md"
    assert not key_file.stat().st_mode & 0o077
    keys = set(re.findall(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}", key_file.read_text()))
    assert len(keys) == 1
    os.environ["DASHSCOPE_API_KEY"] = keys.pop()
    os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    outputs = []
    try:
        client = ModelClient(config)
        client.assert_available()
        for index, (kind, density) in enumerate(schedule, 1):
            case = cases[kind]
            information = legends[(kind, density)]
            paths = [OUT / "inputs" / f"{kind}-rgb.png",
                     OUT / "inputs" / f"{kind}-depth.png", Path(information["image"])]
            image_hashes = [sha(p.read_bytes()) for p in paths]
            assert image_hashes[:2] == [case["rgb_sha256"], case["depth_sha256"]]
            prompt = (
                "Inspect the CURRENT screenshot for the task below. This is a static perception check; "
                "do not provide actions or claim task completion. Image 1 is original RGB. "
                "Image 2 is synchronized depth: white near 0m, black >=10m or invalid. "
                "Image 3 marks detector candidate IDs on the SAME RGB frame. The candidate list is "
                "incomplete and its labels may be wrong. Identify the task object and destination "
                "from actual RGB appearance, color, shape and spatial context. Multiple overlapping "
                "boxes can represent the same physical object. Choose candidate IDs only when their "
                "regions match what is visibly present. Use an empty ID list for uncovered or unseen targets. "
                "Report only visible evidence, not how many objects the task asks you to find. "
                "Return a JSON object with object={visible:boolean,candidate_ids:[integer],description:string}, "
                "destination={visible:boolean,candidate_ids:[integer],description:string}, "
                "visible_distinct_task_objects:integer|null, required_quantity:integer|null, "
                "uncertainties:[string].\nTask: " + case["instruction"] +
                "\nCandidates: " + canonical(information["legend"]).decode())
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                    base64.b64encode(p.read_bytes()).decode()}}
                for p in paths]
            payload = client.payload([{"role": "user", "content": content}], {})
            save(OUT / f"qwen-request-{index:02d}.json", payload)
            response = client.complete(payload)
            save(OUT / f"qwen-response-{index:02d}.json", response)
            assert response["model"] == config["name"]
            value = json.loads(response["text"])
            allowed = {r["id"] for r in information["legend"]}
            for role in ("object", "destination"):
                assert type(value[role]["visible"]) is bool
                assert isinstance(value[role]["candidate_ids"], list)
                assert all(type(x) is int and x in allowed for x in value[role]["candidate_ids"])
            row = {"environment": kind, "condition": density, "candidate_count": len(information["legend"]),
                   "image_sha256": image_hashes, "request_sha256": sha(canonical(payload)),
                   "response_sha256": response["response_sha256"], "usage": response["usage"],
                   "result": value, "candidate_legend": information["legend"]}
            outputs.append(row)
            save(OUT / "qwen-density-progress.json", outputs)
            print(json.dumps({k: row[k] for k in ("environment", "condition", "candidate_count", "result")},
                             ensure_ascii=False), flush=True)
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)
        os.environ.pop("DASHSCOPE_BASE_URL", None)
    assert len(outputs) == 4
    report = {"schema": "groundingdino-density-static-qwen/1", "status": "verified",
              "detector_requests_primary": primary["detector_inference_requests"],
              "detector_requests_supplementary": len(supplementary), "qwen_requests": len(outputs),
              "environment_actions": 0, "automatic_retry": False, "model": config,
              "same_original_rgb_depth_each_pair": True, "density_pairs": outputs,
              "scope": "two single-frame pairs; no task rollout or general performance claim"}
    save(OUT / "qwen-density-results.json", report)
    for path in OUT.glob("*.json"):
        assert not re.search(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}", path.read_bytes())


if __name__ == "__main__":
    main()
