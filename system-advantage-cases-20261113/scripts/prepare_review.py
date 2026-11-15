"""Create visual review sheets and readable excerpts from normalized contexts."""
import json
import pathlib
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[1]
cases = json.loads((ROOT / "build/cases.json").read_text())
assets = json.loads((ROOT / "build/assets.json").read_text())
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
for dataset in dict.fromkeys(c["dataset"] for c in cases):
    selected = [c for c in cases if c["dataset"] == dataset]
    rows = []
    for c in selected:
        ims = c["gallery"]
        if dataset == "VSIbench":
            ims = [g for g in ims if g["kind"] == "overview"] or ims[::8]
        rows.append((c, ims))
    cols = max(len(ims) for c, ims in rows)
    w, h = min(cols, 8) * 340, len(rows) * 270
    sheet = Image.new("RGB", (w, h), "#e8edf1")
    draw = ImageDraw.Draw(sheet)
    for row, (c, ims) in enumerate(rows):
        draw.text((8, row*270+6), c["id"][-60:], fill="#152a40", font=font)
        for col, g in enumerate(ims[:8]):
            im = Image.open(ROOT / assets[g["asset"]]["file"]).convert("RGB")
            im.thumbnail((330, 224))
            sheet.paste(im, (col*340+(330-im.width)//2+5, row*270+30+(224-im.height)//2))
            draw.text((col*340+8, row*270+253), g["label"][:40], fill="#152a40", font=font)
    sheet.save(ROOT / "review" / f"contact-{dataset}.jpg", quality=94)
    out = []
    for c in selected:
        out += [f"\n{'='*60}\n{c['id']}\n{c['question']}\nREF {c['reference']}"]
        for name, arm in c["arms"].items():
            out += [f"\n{name.upper()} {arm['answer']}"]
            for i, r in enumerate(arm["rounds"]):
                out += [f"ROUND {i+1}\nRESPONSE\n{r['response']}\nTHINKING END\n{r['reasoning'][-6000:]}"]
                tooltext = json.dumps(r["tools"], ensure_ascii=False, indent=2)
                out += ["TOOLS\n" + tooltext[:15000]]
        if dataset == "MMSI":
            out += ["RELATION\n" + c["evidence"]["relation_text"]]
        if dataset == "VSIbench":
            out += ["DERIVATION\n" + json.dumps(c["evidence"]["derivation_replay"], ensure_ascii=False, indent=2)[:20000]]
    (ROOT / "review" / f"excerpts-{dataset}.txt").write_text("\n".join(out))
print("Prepared six contact sheets and paired context excerpts.")
