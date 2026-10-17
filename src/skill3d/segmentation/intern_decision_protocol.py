"""Frozen indoor inventory protocol, independent of any current question/label.

Categories are a priori indoor furniture, fixtures, electronics and containers.
This experiment does not learn a vocabulary or thresholds from scored examples.
"""
from __future__ import annotations
from skill3d.infra.m5_identity import content_sha256

PROTOCOL_VERSION = "m5-indoor-choice-observation-v1"
# Ordered domain inventory: fixed before any Intern experiment, <=62 symbols.
CATEGORIES = (
    "chair", "sofa", "stool", "bench", "table", "desk", "bed", "cabinet",
    "shelf", "dresser", "wardrobe", "refrigerator", "oven", "microwave",
    "dishwasher", "washing machine", "sink", "toilet", "bathtub", "shower",
    "door", "window", "wall", "floor", "ceiling", "curtain", "rug",
    "lamp", "radiator", "portable heater", "air conditioner", "fan",
    "computer tower", "computer monitor", "television", "laptop", "printer",
    "keyboard", "computer mouse", "mousepad", "speaker", "telephone",
    "paper bag", "plastic bag", "fabric bag", "cardboard box", "plastic box",
    "basket", "trash bin", "bottle", "cup", "plate", "plant", "book",
    "loose paper", "clipboard", "tissue or paper towel", "stuffed toy",
    "other", "unknown",
)
STATE = """The three images depict one candidate region, in this exact order:
1. full scene with an orange bounding rectangle;
2. original unmasked crop;
3. the same crop with a translucent turquoise segmentation overlay.
Identify the dominant physical object actually covered by the mask. No requested
target is supplied. Judge visible evidence only. Preserve material and form.
Other means a recognizable object outside the fixed inventory; unknown means
identity cannot be supported. Completeness refers ONLY to the visible contour,
never to an imagined shape behind occlusion. Do not infer a detector's intent."""
QUESTIONS = {
    "observed_object": {"type": "choice", "instructions":
        "Which physical category is actually under the turquoise mask?",
        "criteria": {c: c for c in CATEGORIES}},
    "visibility": {"type": "choice", "instructions":
        "Is the masked object's identity visually discernible?", "criteria": {
        "clear": "Identity is clearly discernible", "partial":
        "Partly visible but identity is discernible", "insufficient":
        "Blur, occlusion or low visibility prevents reliable identification"}},
    "mask_purity": {"type": "choice", "instructions":
        "Does the turquoise region belong to one physical object?", "criteria": {
        "clean": "Region covers one object with little background leakage",
        "mixed": "Region mixes multiple objects or substantial background",
        "background": "Primarily wall, floor or other background",
        "unknown": "Purity cannot be assessed"}},
    "mask_completeness": {"type": "choice", "instructions":
        "How much of the object's visible silhouette does the mask cover?", "criteria": {
        "whole": "Nearly all of the visible silhouette", "major_part":
        "Most of the visible silhouette", "fragment": "Only a small fragment",
        "unknown": "Visible contour coverage cannot be assessed"}},
}
PROTOCOL = {"version": PROTOCOL_VERSION, "state": STATE, "questions": QUESTIONS,
            "category_source": "a priori indoor furniture/fixture/electronic/container inventory",
            "admission": "match + clear/partial + clean + whole/major_part; no probability threshold"}
PROTOCOL_SHA256 = content_sha256(PROTOCOL)
assert len(CATEGORIES) <= 62

def visual_request(images):
    # No target, IDs, paths, detection metadata, geometry or labels in model evidence.
    return {"state": STATE, "questions": QUESTIONS, "images": list(images)}

ALIASES = {
    "radiator": "heater", "portable heater": "heater", "computer monitor": "monitor",
    "desktop computer": "computer tower", "pc tower": "computer tower",
    "filing cabinet": "cabinet", "stuffed toy": "plush toy", "chairs": "chair",
    "paper bags": "paper bag", "fridge": "refrigerator",
}
AMBIGUOUS = {"unknown", "other", "object", "container", "furniture", "device",
             "bag", "box", "paper", "tissue or paper towel"}
def fixed_relation(observed, target):
    o, t = str(observed).strip().lower(), str(target).strip().lower()
    if o in AMBIGUOUS or t in AMBIGUOUS:
        return "uncertain"
    o, t = ALIASES.get(o, o), ALIASES.get(t, t)
    if o == t:
        return "match"
    # A closed domain rule only asserts mismatch for known concrete categories.
    concrete = set(CATEGORIES) | set(ALIASES) | set(ALIASES.values())
    return "mismatch" if o in concrete and t in concrete else "uncertain"
RELATION_VERSION = "m5-frozen-indoor-relations-v1"
RELATION_IDENTITY = {"version": RELATION_VERSION, "aliases": ALIASES,
                     "ambiguous": sorted(AMBIGUOUS), "categories": CATEGORIES}
