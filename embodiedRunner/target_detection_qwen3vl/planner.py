"""Official planning messages with two additional current-observation images."""
import json

from official_direct.planner import OfficialPlanner
from target_detection_r05.planner import TargetPlanner as NativePlanner
from .perception import PlannerError


IMAGE_PROMPT = (
    "After the official RGB, the two additional images are synchronized depth "
    "(0-10 m: brighter is nearer; black is far or invalid) and RGB with detection boxes. "
    "Detection boxes may be incorrect."
)

TARGET_EXTRACTION_PROMPT = """Extract 1-4 detection targets from the public task instruction.
Return only a JSON object: {"targets":[{"id":"T1","role":"object","category":"star",
"source_phrase":"star","quantity":null,"attributes":[],"relation":"","aliases":[]}]}.
Each target has exactly these eight fields. IDs are unique T1-T4; role is
object/destination/support. Include task-relevant objects, destinations and supports.
source_phrase must be verbatim from the instruction. category is a short English
category of 1-6 words; separate color, quantity, attributes and spatial relation.
quantity is null or an integer 1-100; attributes is a list of strings; relation is
a string; aliases is a list of at most three short English category phrases.
Use only the supplied instruction."""


def extraction_messages(instruction):
    """Only public task text enters this independent preprocessing request."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise PlannerError("target_extraction_requires_public_instruction")
    return [{"role": "system", "content": TARGET_EXTRACTION_PROMPT},
            {"role": "user", "content": instruction}]


def extract_targets(text, perception):
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or set(value) != {"targets"}:
            raise PlannerError("invalid_target_extraction_json")
        perception.set_targets(value["targets"])
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, PlannerError):
            raise
        raise PlannerError("invalid_target_extraction_json") from None


class TargetPlanner(OfficialPlanner):
    # Retain the existing accepted integer/pair forms without changing any IDs.
    action_text = NativePlanner.action_text

    def messages(self, public, perception):
        if not perception.detected or perception.overlay is None:
            raise PlannerError("current_detection_image_required_before_planning")
        if perception.current != (perception.serial, public["step"], public["raw_rgb"]["sha256"]):
            raise PlannerError("detection_image_observation_mismatch")
        messages = super().messages(public)
        messages.insert(0, {"role": "system", "content": IMAGE_PROMPT})
        current = [m for m in messages if m["role"] == "user"][-1]["content"]
        for image in (public["depth"]["image"], perception.overlay):
            current.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + image["base64"]}})
        return messages

    def decision(self, text, perception=None):
        # Parsing follows the official contract; no custom description/review fields.
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                text = self.action_text(value)
        except (ValueError, TypeError, KeyError):
            pass
        return super().parse(text)
