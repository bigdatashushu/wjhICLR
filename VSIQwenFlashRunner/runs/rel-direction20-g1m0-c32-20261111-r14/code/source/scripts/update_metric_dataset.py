#!/usr/bin/env python3
"""Apply an audited replacement list to a new, immutable VSI1000 package."""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = "vsibench-m4-success-g1m1-1000-seed137-r01"
PROPOSAL_HASH = "54d19014b14f85dcb0d6b67731b966b0f2c85d054f356a8e3b716ed96ae46379"
PUBLIC_FIELDS = {
    "id", "qa_id", "dataset", "scene_name", "question_type", "question",
    "options", "source_split", "split", "benchmark_split", "experiment_split",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def read(path, expected=None):
    require(expected is None or digest(path) == expected, "input hash changed")
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def lines_by_id(path):
    return {str(json.loads(line)["qa_id"]): line + b"\n"
            for line in Path(path).read_bytes().splitlines()}


def task(raw):
    return "object_rel_direction" if raw.startswith("object_rel_direction_") else raw


def quotas(rows, fields):
    return Counter(tuple(row[field] for field in fields) for row in rows)


def file_hashes(root):
    return {str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def proposal_context(original, proposal_file):
    proposal = read(proposal_file, PROPOSAL_HASH)
    audit_path = proposal_file.parent / "native_report.json"
    audit = read(audit_path, proposal["native_report_file_sha256"])
    require(proposal["status"] == "complete_proposal" and audit["status"] == "complete",
            "complete audit and proposal required")
    anchors = proposal["anchors"]
    for filename, key in (
        ("sampling_manifest.json", "sampling_manifest_sha256"),
        ("questions.jsonl", "questions_sha256"), ("panel.json", "panel_sha256"),
        ("source_m4_receipts.json", "m4_receipts_sha256"),
        ("source_catalog.json", "catalog_sha256"),
    ):
        require(digest(original / filename) == anchors[key], "original anchor changed")
    require(audit["anchors"] == anchors, "audit and proposal anchors differ")
    return proposal, audit


def build(original, proposal_file, destination):
    require(destination.parent == ROOT / "prepared_cache", "destination must be runner cache")
    require(not destination.exists(), "immutable destination already exists")
    proposal, audit = proposal_context(original, proposal_file)
    old_hashes = file_hashes(original)
    old = jsonl(original / "questions.jsonl")
    manifest = read(original / "sampling_manifest.json")
    source = Path(manifest["source_metadata"]["path"])
    require(digest(source) == manifest["source_metadata"]["sha256"], "official metadata changed")
    official = {str(r["id"]): r for r in jsonl(source)}
    catalog = read(original / "source_catalog.json")
    require(catalog["identity_hash"] == proposal["anchors"]["catalog_identity_hash"]
            and catalog["no_labels"] is True, "source catalog identity changed")
    selection = proposal["proposed_public_selection"]
    replacements = {p["old"]["qa_id"]: p["new"] for p in proposal["replacement_pairs"]}
    require(len(replacements) == 20 and len(selection) == 1000, "unexpected replacement scope")
    old_by_id = {r["qa_id"]: r for r in old}
    selected_by_id = {r["qa_id"]: r for r in selection}
    require(len(selected_by_id) == 1000, "duplicate proposed QA")
    old_pub = read(original / "published_inputs.json")
    old_panel = read(original / "panel.json")
    rows, source_rows = [], {}
    for old_row in old:
        qa = replacements.get(old_row["qa_id"], {}).get("qa_id", old_row["qa_id"])
        identity = selected_by_id[qa]
        original_row = official[qa]
        require(original_row["question_type"] == identity["raw_type"]
                and str(original_row["scene_name"]) == identity["scene_id"],
                "replacement public identity differs from official source")
        source_rows[qa] = original_row
        if qa in old_by_id:
            row = copy.deepcopy(old_by_id[qa])
        else:
            row = {k: copy.deepcopy(original_row[k]) for k in (
                "id", "dataset", "scene_name", "question_type", "question", "options")}
            row.update(qa_id=qa, source_split=identity["source_split"],
                       split=identity["source_split"], benchmark_split="test",
                       experiment_split=identity["experiment_split"])
        require(set(row) == PUBLIC_FIELDS, "public question schema changed")
        require(all(row[k] == identity[v] for k, v in (
            ("qa_id", "qa_id"), ("scene_name", "scene_id"), ("question_type", "raw_type"),
            ("source_split", "source_split"), ("experiment_split", "experiment_split"))),
            "proposed selection differs from constructed row")
        rows.append(row)
    for fields in (("question_type",), ("question_type", "source_split"),
                   ("question_type", "source_split", "experiment_split")):
        require(quotas(rows, fields) == quotas(old, fields), "joint quota changed")
    scenes = sorted({r["scene_name"] for r in rows})
    require(len(scenes) == 160, "unexpected scene denominator")
    scene_checks = {r["scene_id"]: r for r in audit["scene_checks"]}
    require(all(scene_checks[s]["status"] == "pass" for s in scenes),
            "selected scene is not metric qualified")

    destination.mkdir(mode=0o700)
    (destination / "host_evaluation").mkdir(mode=0o700)
    (destination / "scene_overviews").mkdir()
    (destination / "eligibility").mkdir()
    public_lines = lines_by_id(original / "questions.jsonl")
    metadata_lines = lines_by_id(original / "host_evaluation/metadata.jsonl")
    label_lines = lines_by_id(original / "host_evaluation/labels.jsonl")

    def encoded(value):
        return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()

    payloads = {}
    for row in rows:
        qa = row["qa_id"]
        payloads[qa] = public_lines.get(qa, encoded(row))
        if qa not in metadata_lines:
            gt = source_rows[qa]["ground_truth"]
            metadata_lines[qa] = encoded({**row, "ground_truth": gt})
            label_lines[qa] = encoded({k: v for k, v in {
                "id": row["id"], "qa_id": qa, "source_split": row["source_split"],
                "ground_truth": gt}.items()})
    for filename, selected_rows, data in (
        ("questions.jsonl", rows, payloads),
        ("train.jsonl", [r for r in rows if r["experiment_split"] == "train"], payloads),
        ("test.jsonl", [r for r in rows if r["experiment_split"] == "test"], payloads),
        ("host_evaluation/metadata.jsonl", rows, metadata_lines),
        ("host_evaluation/labels.jsonl", rows, label_lines),
    ):
        (destination / filename).write_bytes(b"".join(data[r["qa_id"]] for r in selected_rows))
        if filename.startswith("host_evaluation/"):
            (destination / filename).chmod(0o600)
    for filename in ("source_catalog.json", "source_m4_receipts.json", "online.yaml"):
        shutil.copyfile(original / filename, destination / filename)
    for src, name in ((proposal_file, "replacement_proposal.json"),
                      (proposal_file.parent / "native_report.json", "native_report.json")):
        shutil.copyfile(src, destination / "eligibility" / name)
    split = read(original / "split.json")
    split.update(split_id=NAME + "-train300-test700", scene_names=scenes,
                 questions_sha256=digest(destination / "questions.jsonl"),
                 metadata_sha256=digest(destination / "host_evaluation/metadata.jsonl"))
    for role in ("train", "test"):
        split["files"][role]["sha256"] = digest(destination / f"{role}.jsonl")
    write(destination / "split.json", split)
    published = copy.deepcopy(old_pub)
    published["source_catalog_file"] = str(destination / "source_catalog.json")
    published["scene_scopes"] = {s: old_pub["scene_scopes"][s] for s in scenes}
    entries = {}
    for row in rows:
        qa, scene = row["qa_id"], row["scene_name"]
        scope = published["scene_scopes"][scene]
        require(qa in scope["qa_ids"] and scope["split"] == row["source_split"],
                "replacement not in existing immutable scene scope")
        entries[qa] = {**copy.deepcopy(catalog["variants"]["base"][qa]),
                       **{k: scope[k] for k in ("input_root", "frozen_scene_sha256")}}
    published["entries"] = entries
    published["selection_policy"].update(
        selection="1000 metric-qualified questions; train300/test700; each covers all160 selected scenes",
        train_test_manifest_sha256=digest(destination / "split.json"))
    write(destination / "published_inputs.json", published)
    references = read(original / "prepared_references.json")
    references["source_catalog_file"] = str(destination / "source_catalog.json")
    references["variants"] = {
        variant: {r["qa_id"]: catalog["variants"][variant][r["qa_id"]] for r in rows}
        for variant in ("base", "metric")}
    write(destination / "prepared_references.json", references)
    variant_templates = {r["qa_id"]: r for r in old_panel["system_variants"]}
    direct_templates = {r["qa_id"]: r for r in old_panel["direct_qwen_rows"]}
    panel = copy.deepcopy(old_panel)
    panel["experiment_id"] = NAME
    panel["system_variants"], panel["direct_qwen_rows"] = [], []
    for old_row, row in zip(old, rows):
        qa, entry = row["qa_id"], entries[row["qa_id"]]
        variant = copy.deepcopy(variant_templates[old_row["qa_id"]])
        variant.update(qa_id=qa, variant_id=qa + "@g1m0", dataset=row["dataset"],
                       question_type=task(row["question_type"]), scene_id=row["scene_name"],
                       split=row["source_split"], source_split=row["source_split"],
                       artifact_ref=entry["reference"]["scene_artifact_ref"],
                       binding_key=entry["reference"]["key"],
                       binding_manifest_sha256=entry["reference"]["manifest_sha256"])
        direct = copy.deepcopy(direct_templates[old_row["qa_id"]])
        direct.update(qa_id=qa, question_type=variant["question_type"],
                      scene_id=row["scene_name"], split=row["source_split"],
                      system_variant_ids=[variant["variant_id"]])
        panel["system_variants"].append(variant)
        panel["direct_qwen_rows"].append(direct)
    panel["proposal_sha256"] = content_hash({k: v for k, v in panel.items()
                                           if k != "proposal_sha256"})
    write(destination / "panel.json", panel)
    overview = read(original / "scene_overviews/manifest.json")
    overview.update(n_scenes=160, n_questions=1000,
                    local_image_root=str(destination / "scene_overviews"),
                    source_published_manifest_sha256=digest(destination / "published_inputs.json"))
    overview["images"] = [r for r in overview["images"] if r["scene_id"] in scenes]
    overview["local_verification"] = {
        "copied_pngs_verified": 160, "source_png_and_rgb_identities_preserved": True,
    }
    for image in overview["images"]:
        src = original / "scene_overviews" / image["file"]
        require(digest(src) == image["png_sha256"], "original overview changed")
        shutil.copyfile(src, destination / "scene_overviews" / image["file"])
    write(destination / "scene_overviews/manifest.json", overview)
    manifest.update(selection_id=NAME, selected_qa_ids=[r["qa_id"] for r in rows],
                    n_selected_scenes=160, experiment_split_scene_counts={"train": 160, "test": 160})
    manifest["sampling"].update(
        all_successful_scenes_in_each_role=False,
        all_selected_metric_qualified_scenes_in_each_role=True,
        selection_method="apply hash-pinned 20-pair proposal with exact type/source/role joint quotas")
    manifest["verification"].update(scene_coverage_each=160, all_selected_scenes_g1m1_qualified=True)
    for key in ("online_questions", "train", "test", "host_only_labels", "host_only_metadata",
                "prepared_inputs"):
        manifest[key]["sha256"] = digest(destination / manifest[key]["file"])
    manifest["prepared_inputs"].update(n_scenes=160, package_root=str(destination),
                                      frozen_input_roots_reused=True)
    manifest["replacement_provenance"] = {
        "original_selection_id": proposal["selection_id"],
        "proposal_file": "eligibility/replacement_proposal.json",
        "proposal_sha256": PROPOSAL_HASH,
        "native_report_file": "eligibility/native_report.json",
        "native_report_sha256": proposal["native_report_file_sha256"],
        "n_replaced": 20, "n_retained": 980, "selection_uses_answers_or_scores": False,
        "runtime_variant": "base", "metric_references_available": 1000,
        "original_files_sha256": old_hashes,
    }
    write(destination / "sampling_manifest.json", manifest)
    (destination / "README.md").write_text(
        "# VSI1000 metric-qualified selection\n\n"
        "20 audited replacements; 980 retained; 1000 unique questions; train300/test700; "
        "160 scenes in both roles; seed137. Exact original type/source/role quotas.\n\n"
        "Original questions, options and host-only labels come from hash-pinned official metadata. "
        "Frozen 32-frame scene inputs and base/metric bindings retain their original identities. "
        "The active provider/panel continues to use base/g1m0. Metric qualification of the "
        "selection does not itself enable metric tools.\n")
    require(file_hashes(original) == old_hashes, "original package was modified")
    return validate(destination, original, proposal_file)


def validate(package, original=None, proposal_file=None, native=False, replay=False):
    manifest = read(package / "sampling_manifest.json")
    provenance = manifest["replacement_provenance"]
    proposal = read(package / provenance["proposal_file"], provenance["proposal_sha256"])
    audit = read(package / provenance["native_report_file"], provenance["native_report_sha256"])
    rows = jsonl(package / "questions.jsonl")
    ids = {r["qa_id"] for r in rows}
    require(len(ids) == len(rows) == 1000, "question denominator changed")
    require(ids == set(manifest["selected_qa_ids"]), "manifest QA union differs")
    scenes = {r["scene_name"] for r in rows}
    require(len(scenes) == 160, "scene denominator changed")
    proposal_rows = {r["qa_id"]: r for r in proposal["proposed_public_selection"]}
    require(ids == set(proposal_rows), "selection differs from exact proposal")
    require(all(set(r) == PUBLIC_FIELDS for r in rows), "public question label or schema violation")
    for row in rows:
        identity = proposal_rows[row["qa_id"]]
        require(all(row[k] == identity[v] for k, v in (
            ("scene_name", "scene_id"), ("source_split", "source_split"),
            ("question_type", "raw_type"), ("experiment_split", "experiment_split"))),
            "public identity changed")
    old_identities = copy.deepcopy(proposal_rows)
    for pair in proposal["replacement_pairs"]:
        require(pair["new"] == old_identities.pop(pair["new"]["qa_id"]),
                "replacement pair differs from proposed union")
        old_identities[pair["old"]["qa_id"]] = pair["old"]
    observed = [{"raw_type": r["question_type"], "source_split": r["source_split"],
                 "experiment_split": r["experiment_split"]} for r in rows]
    for fields in (("raw_type",), ("raw_type", "source_split"),
                   ("raw_type", "source_split", "experiment_split")):
        require(quotas(observed, fields) == quotas(old_identities.values(), fields),
                "original joint quota changed")
    for key in ("online_questions", "train", "test", "host_only_labels", "host_only_metadata",
                "prepared_inputs"):
        require(digest(package / manifest[key]["file"]) == manifest[key]["sha256"],
                "manifest file anchor changed")
    for role, count in (("train", 300), ("test", 700)):
        role_rows = jsonl(package / f"{role}.jsonl")
        require(len(role_rows) == count
                and role_rows == [r for r in rows if r["experiment_split"] == role]
                and {r["scene_name"] for r in role_rows} == scenes, "role partition changed")
        require(dict(Counter(r["question_type"] for r in role_rows))
                == manifest[f"{role}_raw_type_quotas"], "role raw quota changed")
    require(dict(Counter(r["question_type"] for r in rows)) == manifest["raw_type_quotas"],
            "raw quota changed")
    require(dict(Counter(task(r["question_type"]) for r in rows)) == manifest["task_quotas"],
            "canonical quota changed")
    require(dict(Counter(r["source_split"] for r in rows)) == manifest["source_split_counts"],
            "source quota changed")
    scenes_checked = {r["scene_id"]: r for r in audit["scene_checks"]}
    require(all(scenes_checked[s]["status"] == "pass"
                and all(scenes_checked[s]["scale_subconditions"].values()) for s in scenes),
            "selected scene not metric-qualified")
    catalog = read(package / "source_catalog.json", proposal["anchors"]["catalog_sha256"])
    refs = read(package / "prepared_references.json")
    published = read(package / "published_inputs.json")
    require(published["selection_policy"]["train_test_manifest_sha256"]
            == digest(package / "split.json"), "published split anchor differs")
    for variant in ("base", "metric"):
        require(set(refs["variants"][variant]) == ids, "reference QA set differs")
        require(all(refs["variants"][variant][qa] == catalog["variants"][variant][qa]
                    for qa in ids), "original binding reference changed")
    require(set(published["entries"]) == ids and set(published["scene_scopes"]) == scenes,
            "published scope differs")
    for row in rows:
        qa, scene = row["qa_id"], row["scene_name"]
        scope = published["scene_scopes"][scene]
        require(qa in scope["qa_ids"]
                and published["entries"][qa] == {
                    **catalog["variants"]["base"][qa],
                    **{k: scope[k] for k in ("input_root", "frozen_scene_sha256")}},
                "published entry identity changed")
    metadata, labels = jsonl(package / "host_evaluation/metadata.jsonl"), jsonl(
        package / "host_evaluation/labels.jsonl")
    require(len(metadata) == len(labels) == 1000, "host-only denominator changed")
    for row, meta, label in zip(rows, metadata, labels):
        require({k: v for k, v in meta.items() if k != "ground_truth"} == row
                and label == {"id": row["id"], "qa_id": row["qa_id"],
                              "source_split": row["source_split"],
                              "ground_truth": meta["ground_truth"]},
                "host metadata/label alignment changed")
    if original is not None:
        proposal_context(original, proposal_file)
        require(file_hashes(original) == provenance["original_files_sha256"],
                "original package changed")
        old = jsonl(original / "questions.jsonl")
        for fields in (("question_type",), ("question_type", "source_split"),
                       ("question_type", "source_split", "experiment_split")):
            require(quotas(rows, fields) == quotas(old, fields), "joint quota changed")
        for filename in ("questions.jsonl", "host_evaluation/metadata.jsonl",
                         "host_evaluation/labels.jsonl"):
            before, after = lines_by_id(original / filename), lines_by_id(package / filename)
            retained = set(before) & ids
            require(len(retained) == 980 and all(before[q] == after[q] for q in retained),
                    "retained bytes changed")
        source = Path(manifest["source_metadata"]["path"])
        require(digest(source) == manifest["source_metadata"]["sha256"], "official source changed")
        official = {str(r["id"]): r for r in jsonl(source)}
        for row, meta in zip(rows, metadata):
            src = official[row["qa_id"]]
            require(all(row[k] == src[k] for k in (
                "id", "dataset", "scene_name", "question_type", "question", "options"))
                    and meta["ground_truth"] == src["ground_truth"], "official content changed")
        require(digest(package / "online.yaml") == digest(original / "online.yaml"),
                "model configuration changed")
    total = sum(manifest["full_task_counts"].values())
    types = [{
        "question_type": k, "count": n, "percentage": n / 10,
        "official_count": manifest["full_task_counts"][k],
        "official_percentage": manifest["full_task_counts"][k] / total * 100,
        "difference_percentage_points": n / 10 - manifest["full_task_counts"][k] / total * 100,
    } for k, n in manifest["task_quotas"].items()]
    maximum = max(abs(r["difference_percentage_points"]) for r in types)
    require(abs(maximum - manifest["max_task_percentage_point_error"]) < 1e-12,
            "official proportion deviation changed")
    result = {
        "schema": "vsi1000-metric-selection-validation/1", "status": "complete",
        "package_root": str(package), "n_questions": 1000, "n_retained": 980,
        "n_replacements": 20, "train": 300, "test": 700, "n_scenes": 160,
        "role_scenes": {"train": 160, "test": 160}, "metric_qualified": 1000,
        "exact_raw_type_quotas_preserved": True, "exact_canonical_type_quotas_preserved": True,
        "exact_source_and_role_joint_quotas_preserved": True,
        "max_official_deviation_percentage_points": maximum, "task_types": types,
        "raw_type_counts": manifest["raw_type_quotas"],
        "runtime_evidence_states": {"g1m0": 1000}, "model_requests": 0, "gpu_requests": 0,
        "reference_answers_visible_to_provider": False,
        "proposal_sha256": PROPOSAL_HASH, "sampling_manifest_sha256": digest(
            package / "sampling_manifest.json"), "native_loaders_checked": False,
    }
    if native or replay:
        sys.path.insert(0, str(ROOT / "src"))
        from skill3d.evaluation.fitset_v11 import load_fitset_panel, select_panel_variants
        from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider
        from skill3d.evaluation.qa_partition_v11 import load_qa_partition_manifest
        from skill3d.online.scene_overview import SceneOverviewStore
        panel = load_fitset_panel(package / "panel.json")
        for name in panel["panels"]:
            require(len(select_panel_variants(panel, name)) == 1000, "native panel scope changed")
        partition = load_qa_partition_manifest(
            package / "split.json", metadata_path=package / "host_evaluation/metadata.jsonl")
        require(set(partition.questions) == ids, "native partition scope changed")
        provider = PublishedFitSetInputProvider(package / "published_inputs.json", max_scenes=1)
        overview = SceneOverviewStore(package / "scene_overviews/manifest.json")
        result["overview_validation"] = overview.validate_entries(published["entries"])
        result["native_loaders_checked"] = True
        if replay:
            replacement_ids = {p["new"]["qa_id"] for p in proposal["replacement_pairs"]}
            variants = [r for r in panel["system_variants"] if r["qa_id"] in replacement_ids]
            variants.sort(key=lambda r: (r["scene_id"], r["qa_id"]))
            receipts = []
            for variant in variants:
                item = provider(variant)
                require(item.episode.ground_truth == "", "provider exposed reference")
                require(len(item.pixels) == 32, "replacement frame count changed")
                receipt = provider.receipt(variant, item)
                require(receipt["reference_answer_visible"] is False
                        and receipt["real_labels_available"] is False,
                        "transport receipt reference visibility changed")
                receipts.append({"qa_id": variant["qa_id"], "scene_id": variant["scene_id"],
                                 "n_frames": 32, "reference_answer_visible": False,
                                 "transport_receipt_sha256": content_hash(receipt)})
                print(json.dumps({"replay_progress": len(receipts), "total": 20}), file=sys.stderr)
            require(len(receipts) == 20, "incomplete replacement replay")
            result["replacement_replay"] = receipts
    return result


def rebase(package):
    published = read(package / "published_inputs.json")
    published["source_catalog_file"] = str(package / "source_catalog.json")
    write(package / "published_inputs.json", published)
    refs = read(package / "prepared_references.json")
    refs["source_catalog_file"] = str(package / "source_catalog.json")
    write(package / "prepared_references.json", refs)
    overview = read(package / "scene_overviews/manifest.json")
    overview.update(local_image_root=str(package / "scene_overviews"),
                    remote_image_root=str(package / "scene_overviews"),
                    source_published_manifest_sha256=digest(package / "published_inputs.json"))
    write(package / "scene_overviews/manifest.json", overview)
    manifest = read(package / "sampling_manifest.json")
    manifest["prepared_inputs"].update(package_root=str(package),
                                      sha256=digest(package / "published_inputs.json"))
    write(package / "sampling_manifest.json", manifest)


def main():
    for key in list(os.environ):
        if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "ENDPOINT", "PROXY")):
            os.environ.pop(key)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "validate", "rebase"))
    parser.add_argument("--package", type=Path, default=ROOT / "prepared_cache" / NAME)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.operation == "build":
        require(args.original is not None and args.proposal is not None, "build requires source/proposal")
        result = build(args.original, args.proposal, args.package)
    else:
        if args.operation == "rebase":
            rebase(args.package)
        result = validate(args.package, args.original, args.proposal, args.native, args.replay)
    if args.report:
        write(args.report, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
