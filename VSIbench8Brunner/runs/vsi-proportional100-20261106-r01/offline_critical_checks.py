"""Offline checks of new source/scene reuse and the 100-question request budget."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pilot40_direct as direct


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def main():
    sampling_path = Path(__file__).with_name("sampling_manifest.json")
    sampling = direct.read(sampling_path)
    rows = sampling["questions"]
    model = SimpleNamespace(
        model="qwen3.8-flash", model_revision="", temperature=0., max_tokens=8192,
        enable_thinking=True, timeout_s=900, n_frames=32, base_url="offline",
        api_key_env="OFFLINE_ABSENT_CREDENTIAL")
    scenes = sorted({(r["dataset"], r["scene_name"]) for r in rows})
    with tempfile.TemporaryDirectory(prefix="vsi100-offline-") as temporary:
        root = Path(temporary)
        official = root / "official"
        old = []
        for index in range(700):
            dataset, scene = scenes[index % len(scenes)]
            row = dict(qa_id=str(70000 + index), scene_id=scene, dataset=dataset,
                       question_type="object_counting", question="Offline test fixture",
                       options=[], source_split="induction", experiment_split="test",
                       video_path="/offline/" + dataset + "/" + scene)
            row["public_question_sha256"] = direct.digest(row)
            old.append(row)
        public = dict(schema="qwen-official32-public-inputs/1", questions=old,
                      reference_answer_visible=False, real_labels_available=False)
        input_id = direct.digest(public)
        public["identity_sha256"] = input_id
        save(official / "public_inputs.json", public)
        for dataset, scene in scenes:
            identity = dict(dataset=dataset, scene_id=scene,
                            video_path="/offline/" + dataset + "/" + scene)
            key = direct.digest([dataset, scene])[:24]
            frames = dict(images=[{"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64,b2ZmbGluZQ=="}}] * 32,
                          source_video_sha256="video",
                          jpeg_sha256=["jpeg"] * 32, source_frame_indices=list(range(32)))
            frame_path = official / "frames" / (key + ".json")
            save(frame_path, frames)
            save(official / "receipts" / (key + ".json"), dict(
                status="complete", n_images=32, reference_answer_visible=False,
                input_identity_sha256=input_id, scene_key=key, scene=identity,
                source_video_sha256="video", file_sha256=direct.file_hash(frame_path),
                frame_artifact_sha256="fixture-artifact"))
        validator = root / "validator.py"
        validator.write_text("# Offline frame-validator fixture\n")
        args = SimpleNamespace(
            sampling=sampling_path, expected_questions=100, source_questions=Path(
                sampling["source_question_file"]), official_root=official,
            validator=validator, run_dir=root / "direct", workers=16)

        class PreflightClient:
            endpoint_hash = "offline"

            def __init__(self, **kwargs):
                pass

            def build_payload(self, messages, **kwargs):
                return {"messages": messages, **kwargs}

        with patch.object(direct, "INPUT_ID", input_id), \
                patch.object(direct, "online_model_from", return_value=model), \
                patch.object(direct, "load_config", return_value=None), \
                patch.object(direct, "resolve_runtime", return_value=None), \
                patch.object(direct, "BailianClient", PreflightClient), \
                patch.object(direct, "frame_validator", return_value=lambda f, i: "fixture-artifact"):
            selected, _, manifest = direct.preflight(args)
        assert len(selected) == manifest["model_requests_cap"] == 100
        assert set(selected).isdisjoint({r["qa_id"] for r in old})
        assert len({r["preflight"]["scene_key"] for r in selected.values()}) == 69
        calls = []

        class ExecuteClient:
            last_response_identity = {"model": model.model}
            last_usage = {}
            last_generation = {}

            def __init__(self, model, root, qa, expected):
                self.qa = qa

            def chat(self, *args, **kwargs):
                calls.append(self.qa)
                return "{}"

        with patch.object(direct, "DirectClient", ExecuteClient), \
                patch.object(direct, "parse_visual_answer", return_value=(
                    SimpleNamespace(value="1"), {})), \
                patch.object(direct, "export_question"):
            assert direct.execute(args.run_dir, selected, model, manifest) == 0
        budget = direct.read(args.run_dir / "request_budget.json")
        status = direct.read(args.run_dir / "live_status.json")
        assert len(calls) == len(set(calls)) == budget["used"] == budget["max_requests"] == 100
        assert status["completed"] == 100 and status["status"] == "complete"
        assert len(list((args.run_dir / "outcomes").glob("*.json"))) == 100
    print(json.dumps({"offline_checks_passed": [
        "new question identities reuse only their authoritative scene frames",
        "100 questions each receive exactly one request under the 100 request cap"],
        "live_model_requests": 0, "reference_answers_read": False}))


if __name__ == "__main__":
    main()
