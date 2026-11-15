"""Verify archived request round trips and both final share packages."""
import base64
from collections import Counter
import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
cases = json.loads((ROOT / "build/embodied-reviewed-cases.json").read_text())
assets = json.loads((ROOT / "build/embodied-assets.json").read_text())

def restore(value):
    if isinstance(value, list):
        return [restore(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "original_data_url_prefix" in value:
        return value["original_data_url_prefix"] + "," + base64.b64encode(
            (ROOT / assets[value["asset"]]["file"]).read_bytes()).decode()
    return {k: restore(v) for k, v in value.items()}

counts = Counter(c["environment"] for c in cases)
assert len(cases) == 20 and sorted(counts.values()) == [5, 5, 5, 5]
records = []
for c in cases:
    directory = ROOT / c["source"]["archived_directory"]
    for r in c["requests"]:
        original = json.loads((directory / r["request_file"]).read_text())
        assert restore(r["payload"]) == original, (c["id"], r["request_file"])
        original_response = json.loads((directory / f'response-{r["request_index"]:03d}.json').read_text())
        assert restore(r["response"]) == original_response
    assert c["result"]["official"]["done"] and c["result"]["official"]["task_success"] == 1
    assert c["observations"][0]["official"]["task_success"] == 0
    assert c["actions"][-1]["official"]["done"] and c["actions"][-1]["official"]["task_success"] == 1
    records.append({"id": c["id"], "request_roundtrip_equal": True,
                    "response_roundtrip_equal": True, "official_success_verified": True,
                    "requests": len(c["requests"]), "actions": len(c["actions"]),
                    "observations": len(c["observations"])})
qa = json.loads((ROOT / "build/reviewed-cases.json").read_text())
assert len(qa) == 30 and sorted(Counter(c["dataset"] for c in qa).values()) == [5] * 6
for c in qa:
    assert c["arms"]["system"]["answer"] == c["reference"]
    assert c["arms"]["direct"]["answer"] != c["reference"]
    assert c["arms"]["system"]["rounds"] and c["arms"]["direct"]["rounds"]
manifest = json.loads((ROOT / "build/delivery-manifest.json").read_text())
packages = []
for name, expected in [
    ("空间智能系统优势案例-50条-具身优先-离线分享.zip", [m["file"] for m in manifest]),
    ("EmbodiedBench-20条成功轨迹-离线分享.zip",
     [m["file"] for m in manifest if m["file"] == "EmbodiedBench.html" or m["file"].startswith("EB-")]),
]:
    with zipfile.ZipFile(ROOT / name) as z:
        assert z.testzip() is None
        assert sorted(z.namelist()) == sorted(expected + ["README.txt"])
        for m in manifest:
            if m["file"] in expected:
                assert hashlib.sha256(z.read(m["file"])).hexdigest() == m["sha256"]
    packages.append({"file": name, "bytes": (ROOT / name).stat().st_size,
                     "html_files": expected, "all_hashes_verified": True})
report = {"embodied_counts": dict(counts), "embodied": records, "qa_pairs_preserved": 30,
          "qa_original_dataset_counts": dict(Counter(c["dataset"] for c in qa)),
          "unique_embedded_embodied_images": len(assets), "packages": packages}
(ROOT / "checks/embodied-integrity-and-package-validation.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ("embodied", "packages")}, ensure_ascii=False))
