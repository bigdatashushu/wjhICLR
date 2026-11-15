#!/usr/bin/env python3
"""Read frozen RGB, reproduce request JPEG hashes, and fetch review images."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native(spec, output):
    import cv2
    import numpy as np
    from PIL import Image

    output = Path(output)
    assert output.parent.is_dir()
    output.mkdir(exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)
    manifest = {"cases": {}, "model_requests": 0}
    for row in spec:
        qa = row["qa_id"]
        source = Path(row["input_root"])
        raw_scene = (source / "scene.json").read_bytes()
        assert hashlib.sha256(raw_scene).hexdigest() == row["frozen_scene_sha256"]
        scene = json.loads(raw_scene)
        assert len(scene["frames"]) == 32 and scene["real_labels_available"] is False
        scene_dir = images / "scenes" / row["scene_id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for slot, frame in enumerate(scene["frames"]):
            target = (source / scene["files"][frame["source_path"]]).resolve()
            assert str(target).startswith(str(source.resolve()) + "/")
            png = target.read_bytes()
            assert hashlib.sha256(png).hexdigest() == frame["png_sha256"]
            with Image.open(io.BytesIO(png)) as image:
                pixels = np.asarray(image)
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            assert ok
            jpeg = encoded.tobytes()
            sha = hashlib.sha256(jpeg).hexdigest()
            image_id = row["frame_ids"][slot]
            assert sha == row["jpeg_sha256"][slot], (qa, image_id, sha)
            relative = "scenes/" + row["scene_id"] + "/" + image_id + ".jpg"
            (images / relative).write_bytes(jpeg)
            frames.append({"image_id": image_id, "path": relative, "sha256": sha,
                           "slot": slot, "source_frame_index": row["source_frame_indices"][slot]})
        overview_png = Path(row["overview_path"]).read_bytes()
        assert hashlib.sha256(overview_png).hexdigest() == row["overview_png_sha256"]
        with Image.open(io.BytesIO(overview_png)) as image:
            pixels = np.asarray(image.convert("RGB"))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        assert ok
        jpeg = encoded.tobytes()
        sha = hashlib.sha256(jpeg).hexdigest()
        assert sha == row["overview_jpeg_sha256"], (qa, "overview", sha)
        relative = "scenes/" + row["scene_id"] + "/" + row["overview_id"] + ".jpg"
        (images / relative).write_bytes(jpeg)
        manifest["cases"][qa] = {
            "scene_id": row["scene_id"], "frames": frames,
            "overview": {"image_id": row["overview_id"], "path": relative, "sha256": sha},
            "frozen_scene_sha256": row["frozen_scene_sha256"],
            "all_jpegs_equal_actual_model_request": True,
        }
    (images / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = output / "images.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(images, arcname="images")
    raw = archive.read_bytes()
    print(json.dumps({"archive": str(archive), "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "qa_count": len(spec), "model_requests": 0}), flush=True)


def collect(work, native_root):
    work = Path(work).resolve()
    project = work.parents[2]
    sys.path.insert(0, str(project))
    from scripts.control_small_comparison import remote, TITAN_PYTHON

    intake = read(work / "intake.json")
    selected = intake["cases"] + [r for r in intake["all_pairs"] if r["qa_id"] == "4325"]
    spec = []
    for row in selected:
        files = row["system_files"]
        plan, outcome = read(files["plan_file"]), read(files["outcome_file"])
        transport = plan["published_transport"]
        image_round = outcome["first_synthesis"]["image_round"]
        hashes = dict(zip(image_round["image_ids"], image_round["jpeg_sha256"]))
        frame_ids = [x for x in image_round["image_ids"] if x.startswith("frame-")]
        overview = plan["scene_overview"]
        spec.append({
            "qa_id": row["qa_id"], "scene_id": row["scene_id"],
            "input_root": str(Path(transport["frozen_scene"]["path"]).parent),
            "frozen_scene_sha256": transport["frozen_scene"]["sha256"],
            "frame_ids": frame_ids, "jpeg_sha256": [hashes[x] for x in frame_ids],
            "source_frame_indices": transport["source_frame_indices"],
            "overview_path": overview["image_path"], "overview_id": overview["image_id"],
            "overview_png_sha256": overview["png_sha256"],
            "overview_jpeg_sha256": hashes[overview["image_id"]],
        })
    assert len(spec) == 12 and all(len(r["frame_ids"]) == 32 for r in spec)
    remote("wangy@10.10.0.34", "ls -ld " + shlex.quote(str(Path(native_root).parent)))
    command = (
        "env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
        "MKL_NUM_THREADS=2 "
        "PYTHONPATH=/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat "
        + TITAN_PYTHON + " -B -c " + shlex.quote(Path(__file__).read_text())
        + " --mode native --spec-json " + shlex.quote(json.dumps(spec))
        + " --native-root " + shlex.quote(native_root))
    info = json.loads(remote("wangy@10.10.0.34", command))
    data = bytearray()
    for offset in range(0, info["bytes"], 1024 * 1024):
        length = min(1024 * 1024, info["bytes"] - offset)
        code = (
            "import sys;from pathlib import Path;"
            f"\nwith Path({info['archive']!r}).open('rb') as f:"
            f"\n f.seek({offset});sys.stdout.buffer.write(f.read({length}))")
        chunk = remote("wangy@10.10.0.34", TITAN_PYTHON + " -B -c " + shlex.quote(code))
        assert len(chunk) == length
        data.extend(chunk)
        print(json.dumps({"received_bytes": len(data), "total_bytes": info["bytes"]}), flush=True)
    assert hashlib.sha256(data).hexdigest() == info["sha256"]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            assert member.isfile() or member.isdir()
            assert (work / member.name).resolve().is_relative_to(work)
        tar.extractall(work)
    manifest = read(work / "images/manifest.json")
    for row in manifest["cases"].values():
        for image in row["frames"] + [row["overview"]]:
            assert hashlib.sha256((work / "images" / image["path"]).read_bytes()).hexdigest() == image["sha256"]
    (work / "images/transfer_receipt.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "qa_count": len(manifest["cases"]),
                      "image_request_hashes_verified": True, "model_requests": 0}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "native"), required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--spec-json")
    parser.add_argument("--native-root", required=True)
    args = parser.parse_args()
    if args.mode == "native":
        native(json.loads(args.spec_json), args.native_root)
    else:
        collect(args.work_root, args.native_root)


if __name__ == "__main__":
    main()
