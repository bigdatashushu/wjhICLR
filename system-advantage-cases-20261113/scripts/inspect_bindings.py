"""Recover displayed object rows and make three-anchor visual review sheets."""
import json
import pathlib
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
cases = json.loads((ROOT / "build/cases-VSIbench.json").read_text())
assets = json.loads((ROOT / "build/assets.json").read_text())
decoder = json.JSONDecoder()
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)

def rows(value):
    if isinstance(value, dict):
        if "obj_id" in value and "category_name" in value:
            yield value
        for v in value.values():
            yield from rows(v)
    elif isinstance(value, list):
        for v in value:
            yield from rows(v)
    elif isinstance(value, str) and '"obj_id"' in value:
        for i, char in enumerate(value):
            if char in "{[":
                try:
                    obj, end = decoder.raw_decode(value[i:])
                except ValueError:
                    continue
                if isinstance(obj, (dict, list)):
                    yield from rows(obj)

allrows = {}
for c in cases:
    used = set(c["evidence"]["derivation_replay"]["input_result_ids"])
    result = next(t for t in c["evidence"]["tool_results"] if t["result_id"] in used)
    ids = [result["args"][k] for k in ("observer_id", "facing_at_id", "target_id")]
    records = {}
    for r in c["arms"]["system"]["rounds"]:
        for row in rows(r["messages"]):
            if row["obj_id"] in ids:
                old = records.get(row["obj_id"], {})
                if len(json.dumps(row)) > len(json.dumps(old)):
                    records[row["obj_id"]] = row
    allrows[c["id"]] = {"ids": ids, "rows": records, "direction": result["payload"]["direction"]}
    print(json.dumps({"id": c["id"], "ids": ids, "rows": {
        key: {k: v for k, v in val.items() if k in ("obj_id", "category_name", "visible_frames", "bbox", "box", "box_xyxy", "binding_flags", "binding_validation")}
        for key, val in records.items()}}, ensure_ascii=False))
    # Original frozen frames, two views per deciding anchor. No generated evidence.
    sheet = Image.new("RGB", (1260, 560), "#e8edf1")
    draw = ImageDraw.Draw(sheet)
    for col, obj_id in enumerate(ids):
        row = records.get(obj_id, {})
        frames = row.get("visible_frames", [])
        if isinstance(frames, dict):
            frames = [i for a, b in frames.get("inclusive_ranges", []) for i in range(a, b+1)]
        frames = frames[:2]
        draw.text((col*420+8, 6), obj_id + " " + row.get("category_name", "?"), font=font, fill="#123")
        for n, frame in enumerate(frames):
            if not isinstance(frame, int) or not 0 <= frame < 32:
                continue
            g = c["gallery"][frame]
            im = Image.open(ROOT / assets[g["asset"]]["file"]).convert("RGB")
            im.thumbnail((410, 235))
            sheet.paste(im, (col*420+5, 32+n*260))
            draw.text((col*420+8, 272+n*260), f"Frame {frame} / {g['label']}", font=font, fill="#123")
    sheet.save(ROOT / "review" / f"anchors-VSI-{c['id']}.jpg", quality=94)
(ROOT / "review/vsi-anchor-records.json").write_text(json.dumps(allrows, ensure_ascii=False, indent=2))
