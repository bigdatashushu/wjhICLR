"""One newly sampled public pair, identical VGGT camera recipe and checkpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from vggt_pose_worker import save, sha


def run(directory):
    directory = Path(directory)
    config = json.loads((directory / "gpu-input.json").read_text())
    if (config["schema"] != "mmsi-pose30-replacement-gpu/1"
            or len(config["items"]) != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "3"
            or any(config[k] is not False for k in (
                "reference_answer_visible", "depth_enabled", "point_enabled", "track_enabled"))):
        raise ValueError("replacement_configuration_changed")
    source = Path(config["vggt_source"])
    identity = {str(p.relative_to(source)): sha(p)
                for p in sorted((source / "vggt").rglob("*.py"))}
    if identity != config["vggt_source_identity"]:
        raise ValueError("vggt_source_changed")
    for name, expected in config["worker_identity"].items():
        if sha(directory / "code" / name) != expected:
            raise ValueError("replacement_worker_changed")
    if sha(config["weight_path"]) != config["weight_sha256"]:
        raise ValueError("vggt_weight_changed")
    sys.path.insert(0, str(source))
    import hashlib
    import socket
    import torch
    import torchvision
    import numpy
    import PIL
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt_pose import relative_cameras
    if socket.gethostname() != "cvailab":
        raise ValueError("replacement_host_changed")
    save(directory / "gpu-started.json", {
        "pid": os.getpid(), "started_at": time.time(), "physical_gpu": 3,
        "hostname": "cvailab", "config_sha256": sha(directory / "gpu-input.json"),
        "reference_answer_visible": False, "model_requests": 0,
    })
    torch.manual_seed(137)
    torch.set_num_threads(4)
    model = VGGT()
    state = load_file(config["weight_path"], device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.depth_head = model.point_head = model.track_head = None
    model.eval().to("cuda")
    environment = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": numpy.__version__,
        "pillow": PIL.__version__, "gpu": torch.cuda.get_device_name(0),
        "precision": "bfloat16_aggregator_float32_camera",
    }
    if environment != config["environment"]:
        raise ValueError("replacement_environment_changed")
    item = config["items"][0]
    paths = [directory / r["path"] for r in item["images"]]
    if len(paths) != 2 or any(sha(p) != r["sha256"] for p, r in zip(paths, item["images"])):
        raise ValueError("replacement_rgb_changed")
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
        "schema": "mmsi-joint-vggt-camera/1", "sample_id": item["sample_id"],
        "input_sha256": item["input_sha256"],
        "original_image_sha256": [r["sha256"] for r in item["images"]],
        "preprocessed_tensor_sha256": tensor_sha, "preprocessed_shape": list(images.shape),
        "preprocess": "official_load_and_preprocess_images_crop_width518_patch14",
        "joint_forward": True, "depth_head_executed": False, "point_head_executed": False,
        "track_head_executed": False, "reference_answer_visible": False, "model_requests": 0,
        "weight_sha256": config["weight_sha256"],
        "world_to_camera_convention": "OpenCV x_right_y_down_z_forward",
        "pose_encoding": encoding[0].float().cpu().tolist(),
        "intrinsics_preprocessed_pixels": intrinsics[0].float().cpu().tolist(),
        "camera_evidence": relative_cameras(extrinsics[0].float().cpu().numpy()),
        "elapsed_seconds": time.monotonic() - started, "environment": environment,
    }
    path = "poses/" + item["sample_id"] + ".json"
    save(directory / path, result)
    save(directory / "gpu-complete.json", {
        "status": "complete", "joint_forwards": 1,
        "config_sha256": sha(directory / "gpu-input.json"), "files": {path: sha(directory / path)},
        "reference_answer_visible": False, "depth_enabled": False, "model_requests": 0,
    })
    print(json.dumps({"status": "complete", "sample_id": item["sample_id"]}), flush=True)


if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(2)
