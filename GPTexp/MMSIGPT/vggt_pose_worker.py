"""Offline joint RGB camera estimation; deliberately never run depth/point heads."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / "gpu-input.json").read_text())
    if (config["reference_answer_visible"] is not False or config["depth_enabled"] is not False
            or len(config["items"]) != 6 or os.environ.get("CUDA_VISIBLE_DEVICES") != "3"):
        raise ValueError("gpu_configuration_changed")
    source = Path(config["vggt_source"])
    actual_source = {str(p.relative_to(source)): sha(p)
                     for p in sorted((source / "vggt").rglob("*.py"))}
    if actual_source != config["vggt_source_identity"]:
        raise ValueError("vggt_source_changed")
    for name, expected in config["worker_identity"].items():
        if sha(directory / "code" / name) != expected:
            raise ValueError("worker_source_changed")
    save(directory / "gpu-started.json", {
        "pid": os.getpid(), "started_at": time.time(), "physical_gpu": 3,
        "reference_answer_visible": False, "depth_enabled": False,
        "config_sha256": sha(directory / "gpu-input.json"),
    })
    print("weight_hash_start", flush=True)
    weight = Path(config["weight_path"])
    if sha(weight) != config["weight_sha256"]:
        raise ValueError("vggt_weight_changed")
    print("weight_hash_verified", flush=True)
    sys.path.insert(0, str(source))
    import torch
    import torchvision
    import numpy
    import PIL
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt_pose import relative_cameras

    torch.manual_seed(137)
    torch.set_num_threads(4)
    model = VGGT()
    print("model_created", flush=True)
    state = load_file(str(weight), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    # All checkpoint keys were validated; omit unused heads from GPU memory and execution.
    model.depth_head = model.point_head = model.track_head = None
    model.eval().to("cuda")
    print("model_loaded", flush=True)
    environment = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": numpy.__version__,
        "pillow": PIL.__version__, "gpu": torch.cuda.get_device_name(0),
        "precision": "bfloat16_aggregator_float32_camera",
    }
    files = {}
    for item in config["items"]:
        sample = item["sample_id"]
        paths = [directory / receipt["path"] for receipt in item["images"]]
        if len(paths) != 2 or any(sha(p) != r["sha256"]
                                  for p, r in zip(paths, item["images"])):
            raise ValueError("original_rgb_changed")
        images = load_and_preprocess_images([str(p) for p in paths], mode="crop")
        tensor_sha = hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
        images = images.to("cuda")
        started = time.monotonic()
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens, _ = model.aggregator(images.unsqueeze(0))
            with torch.autocast("cuda", enabled=False):
                encoding = model.camera_head(tokens)[-1]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(
                    encoding, image_size_hw=images.shape[-2:])
        torch.cuda.synchronize()
        result = {
            "schema": "mmsi-joint-vggt-camera/1", "sample_id": sample,
            "input_sha256": item["input_sha256"],
            "original_image_sha256": [r["sha256"] for r in item["images"]],
            "preprocessed_tensor_sha256": tensor_sha,
            "preprocessed_shape": list(images.shape),
            "preprocess": "official_load_and_preprocess_images_crop_width518_patch14",
            "joint_forward": True, "depth_head_executed": False,
            "point_head_executed": False, "track_head_executed": False,
            "reference_answer_visible": False, "model_requests": 0,
            "weight_sha256": config["weight_sha256"],
            "world_to_camera_convention": "OpenCV x_right_y_down_z_forward",
            "pose_encoding": encoding[0].float().cpu().tolist(),
            "intrinsics_preprocessed_pixels": intrinsics[0].float().cpu().tolist(),
            "camera_evidence": relative_cameras(extrinsics[0].float().cpu().numpy()),
            "elapsed_seconds": time.monotonic() - started, "environment": environment,
        }
        path = directory / "poses" / (sample + ".json")
        save(path, result)
        files[str(path.relative_to(directory))] = sha(path)
        print(json.dumps({"sample_id": sample, "status": "pose_ready",
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        del images, encoding, extrinsics, intrinsics, tokens
    save(directory / "gpu-complete.json", {
        "status": "complete", "completed_at": time.time(), "joint_forwards": 6,
        "config_sha256": sha(directory / "gpu-input.json"), "files": files,
        "reference_answer_visible": False, "depth_enabled": False, "model_requests": 0,
    })


if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except Exception as error:
        # This worker has no service credentials, labels, or endpoint configuration.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(2)
