"""Only RGB and robot proprioception can enter the GPT request."""
from __future__ import annotations

import base64
from io import BytesIO
import json

CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIOCEPTION = {
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_qpos": 2,
    "robot0_joint_pos": 7,
}


def public_observation(obs: dict, orientation: str) -> tuple[dict, dict[str, bytes]]:
    import numpy as np
    from PIL import Image

    images = {}
    for key in CAMERAS:
        value = np.asarray(obs[key])
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"Invalid RGB observation: {key}")
        if orientation == "flip_vertical":
            value = value[::-1]
        elif orientation == "rotate_180":
            value = value[::-1, ::-1]
        elif orientation != "raw":
            raise ValueError("Invalid image orientation.")
        stream = BytesIO()
        Image.fromarray(np.ascontiguousarray(value)).save(stream, format="PNG")
        images[key] = stream.getvalue()
    state = {}
    for key, length in PROPRIOCEPTION.items():
        if key not in obs:
            if key == "robot0_joint_pos":
                continue
            raise ValueError(f"Missing proprioception: {key}")
        value = np.asarray(obs[key], dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid proprioception: {key}")
        state[key] = value.tolist()
    return state, images


def build_messages(language: str, step: int, state: dict, images: dict,
                   last_action, system_prompt: str) -> list[dict]:
    from xcodeapi import image_url_part, text_part

    text = json.dumps({
        "task": language, "control_step": step,
        "robot_state": state, "previous_action": last_action,
    }, ensure_ascii=False, allow_nan=False)
    content = [text_part(text)]
    for key in CAMERAS:
        content += [text_part(key + ":"),
                    image_url_part("data:image/png;base64," +
                                   base64.b64encode(images[key]).decode("ascii"),
                                   detail="high")]
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": content}]
