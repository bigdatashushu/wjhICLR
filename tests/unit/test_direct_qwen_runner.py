import importlib.util
from pathlib import Path

import numpy as np

from skill3d.adapters import vsibench_loader as vl

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_direct_qwen_v11.py"
_SPEC = importlib.util.spec_from_file_location("run_direct_qwen_v11", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_scene_cache_decodes_and_encodes_same_scene_once(monkeypatch):
    calls = {"load": 0, "receipt": 0, "encode": 0}
    pixels = [np.full((4, 4, 3), index, np.uint8) for index in range(32)]

    def load(metadata, video_root, fallback_roots, n_frames):
        calls["load"] += 1
        return {
            "pixels": pixels,
            "planned": list(range(32)),
            "missing": [],
            "source": vl.VideoSource(
                Path("/videos/scannet/scene-1.mp4"), "", "primary"),
            "total": 32,
            "fps": 30.0,
        }, [{"result": "used"}]

    def receipt(scene_id, roots):
        calls["receipt"] += 1
        return {"scene_id": scene_id}

    def encode(frames, *, max_images):
        calls["encode"] += 1
        assert frames is pixels and max_images == 32
        return tuple(
            {"type": "image_url", "image_url": {"url": f"frame-{index}"}}
            for index in range(32))

    monkeypatch.setattr(runner, "_load_from_sources", load)
    monkeypatch.setattr(runner, "load_input_receipt", receipt)
    monkeypatch.setattr(runner, "encode_visual_frames", encode)
    cache = runner._SceneInputCache(
        max_scenes=2,
        input_receipt_roots=["/receipts"],
        video_root="/videos",
        fallback_roots=[],
    )
    metadata = {"dataset": "scannet", "scene_name": "scene-1"}
    first = cache.get({"qa_id": "1", "scene_id": "scene-1"}, metadata)
    second = cache.get({"qa_id": "2", "scene_id": "scene-1"}, metadata)
    assert first is second
    assert calls == {"load": 1, "receipt": 1, "encode": 1}
