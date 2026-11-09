"""Public-only sampling, exact RGB transfer and metadata-only API qualification."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from ..traces import digest, file_digest, read, write
from .depth_comparison.pool import qualify
from .depth_service_probe import all_keys
from .prototype.inputs import extract_public, load_images, select

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT.parent / "MMSI六类五题位姿实验-20261108-v2"
SOURCE = ROOT / "GPTexp/frozen/mmsi/manifest.json"
SOURCE_SHA = "f93d55238af0208d972b96f8a38f7513329c5e51fba3e2490506c55065b28f21"
IMAGE_ROOT = Path("/tmp/h3d-four-smallimage-public-20261031/dataset/MMSI_images")
SSH = [
    "ssh", "-F", str(ROOT.parent / ".trae/skills/harness3d-cluster-ops/assets/ssh_config"),
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-T", "h3d-titan",
]
REMOTE_PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
REFERENCE = "/home/wangy/wangjh/harness3d/four-datasets-20261031/smallimage/host_evaluation/answers.json"
REFERENCE_SHA = "cbd06fa24683a0739e67119ba8d318721bad9ca7c919927cb5a2caabe7144aed"


def remote_json(code):
    result = subprocess.run(
        SSH + [shlex.quote(REMOTE_PYTHON) + " -B -c " + shlex.quote(code)],
        capture_output=True, timeout=60,
    )
    if result.returncode:
        raise ValueError("titan_public_query_failed")
    return json.loads(result.stdout)


def prior_ids():
    remote = remote_json(
        "import pathlib,json,socket; "
        "assert socket.gethostname()=='gpu34'; "
        "root=pathlib.Path('/home/wangy/wangjh/harness3d/GPTexp/runs'); out=[]; "
        "\nfor run in sorted(root.glob('mmsi*')):"
        "\n m=json.loads((run/'manifest.json').read_text());"
        "\n out.extend({'run':run.name,'sample_id':p['sample_id']} for p in m.get('pairs',[]) "
        "if (run/'episodes/mmsi'/p['pair_identity']).exists());"
        "\nprint(json.dumps(out))"
    )
    excluded = {row["sample_id"] for row in remote}
    receipts = [{"source": "server_episode_directory", "rows": remote}]
    diagnostic = read(ROOT.parent / "MMSIGPT轨迹诊断-20261106/diagnostic-traces.json")
    # Previous development selections remain development material, even when unrun.
    campaign = diagnostic["campaign"]["runs"][0]["selection"]
    excluded.update(campaign["all_selected_sample_ids"])
    excluded.update(campaign["excluded_dispatched_sample_ids"])
    receipts.append({
        "source": "prior_gpt_development_selection",
        "sample_ids": campaign["all_selected_sample_ids"],
    })
    local = []
    for directory in sorted(ROOT.parent.glob("MMSI*")):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.dispatch.json"):
            value = read(path)
            sample = value.get("sample_id")
            if sample:
                excluded.add(sample)
                local.append({"path": str(path.relative_to(ROOT.parent)), "sample_id": sample})
    receipts.append({"source": "local_gpt_dispatch_records", "rows": local})
    return sorted(excluded), receipts


def transfer(rows):
    expected = {}
    for row in rows:
        for path, receipt in zip(row["problem"]["image_paths"], row["images"]):
            path = Path(path)
            if path.parent != IMAGE_ROOT or path.suffix != ".png":
                raise ValueError("unexpected_original_path")
            if path.name in expected and expected[path.name] != receipt["sha256"]:
                raise ValueError("original_hash_binding_disagrees")
            expected[path.name] = receipt["sha256"]
    command = "tar -czf - -C " + shlex.quote(str(IMAGE_ROOT)) + " " + " ".join(
        shlex.quote(name) for name in sorted(expected))
    result = subprocess.run(SSH + [command], capture_output=True, timeout=180)
    if result.returncode:
        raise ValueError("original_rgb_transfer_failed")
    originals = {}
    import hashlib
    with tarfile.open(fileobj=BytesIO(result.stdout)) as archive:
        for member in archive.getmembers():
            if member.name not in expected or not member.isfile() or member.name in originals:
                raise ValueError("unexpected_original_archive_member")
            raw = archive.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != expected[member.name]:
                raise ValueError("original_rgb_hash_changed")
            originals[member.name] = raw
    if originals.keys() != expected.keys():
        raise ValueError("missing_original_rgb")
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for name, raw in originals.items():
        path = IMAGE_ROOT / name
        if path.exists():
            if file_digest(path) != expected[name]:
                raise ValueError("existing_original_rgb_changed")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
    for row in rows:
        load_images(row["problem"], row["images"])
    return {"images": len(originals), "bytes": sum(map(len, originals.values())),
            "sha256": expected, "reference_files_copied": False, "model_requests": 0}


def capture(directory=DESTINATION):
    directory = Path(directory)
    if not directory.parent.is_dir() or directory.exists():
        raise ValueError("requires_new_directory")
    if file_digest(SOURCE) != SOURCE_SHA:
        raise ValueError("public_frozen_source_changed")
    source = read(SOURCE)
    if source["reference_answer_visible"] is not False:
        raise ValueError("requires_public_source")
    rows = extract_public(source)
    if len(rows) != 365:
        raise ValueError("requires_365_public_position_questions")
    excluded, exclusion_receipts = prior_ids()
    selected = select(rows, excluded=excluded, per_type=5, seed=137)
    keys = all_keys()
    qualification = qualify(keys)
    apis = [row["api_id"] for row in qualification
            if row.get("terra_listed") is True and row.get("http_status") == 200][:6]
    if not apis:
        raise ValueError("no_current_qualified_api")
    transfer_receipt = transfer(selected)
    directory.mkdir()
    items = []
    for index, row in enumerate(selected):
        rank, task_index = divmod(index, 6)
        items.append({
            "index": index, **row, "api_id": apis[(rank + task_index) % len(apis)],
            "arm_order": ["direct", "pose"] if (rank + task_index) % 2 == 0 else ["pose", "direct"],
        })
    selection = {
        "schema": "mmsi-vggt30-public-selection/1", "questions": 30, "per_type": 5,
        "seed": 137, "max_requests": 60, "api_ids": apis, "items": items,
        "source_manifest_sha256": SOURCE_SHA, "excluded_sample_ids": excluded,
        "exclusion_receipts": exclusion_receipts,
        "reference_answer_visible": False, "selection_reads_labels": False,
        "source_split_preserved": True, "sample_role": "stratified_development_diagnostic",
        "selected_at": time.time(),
    }
    selection["selection_identity"] = digest(selection)
    write(directory / "selection.json", selection, exclusive=True)
    write(directory / "transfer.json", transfer_receipt, exclusive=True)
    write(directory / "api_qualification.json", {
        "status": "qualified", "active_api_ids": apis, "results": qualification,
        "model_requests": 0, "qualification_request": "GET_models",
        "credentials_persisted": False,
    }, exclusive=True)
    print(json.dumps({
        "status": "public_inputs_ready", "questions": 30, "model_requests": 0,
        "api_ids": apis, "original_images": transfer_receipt["images"],
        "image_counts": [len(row["problem"]["image_paths"]) for row in items],
        "excluded_questions": len(excluded), "selection_identity": selection["selection_identity"],
    }), flush=True)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=str(DESTINATION))
    try:
        args = parser.parse_args()
        capture(args.directory)
    except Exception:
        print('{"status":"failed","reason_code":"pose30_public_capture_failed"}', flush=True)
        raise SystemExit(2)
