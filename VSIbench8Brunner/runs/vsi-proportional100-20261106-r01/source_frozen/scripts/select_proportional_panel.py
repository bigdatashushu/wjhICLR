#!/usr/bin/env python3
"""Select a label-free panel after inventorying prior Direct/Harness requests."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skill3d.routing.task_classifier import canonical_task
from ssh_from_server import remote_run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def inventory(host, python, code):
    status, output, _ = remote_run(
        host, python + " -c " + shlex.quote(code), return_output=True)
    if status:
        raise RuntimeError("prior experiment inventory is unavailable: " + host)
    return json.loads(output)


def apportion(counts, total):
    denominator = sum(counts.values())
    if total > denominator:
        raise ValueError("not enough unused questions for this panel")
    quotas, remainders = {}, {}
    for key, count in counts.items():
        quotas[key], remainders[key] = divmod(total * count, denominator)
    for key in sorted(counts, key=lambda key: (-remainders[key], key))[
            :total - sum(quotas.values())]:
        quotas[key] += 1
    if sum(quotas.values()) != total or any(
            quotas[key] > counts[key] for key in quotas):
        raise ValueError("proportional quota calculation failed")
    return quotas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("panel size must be positive")
    if args.output_dir.exists():
        raise ValueError("panel already exists; preserve its frozen selection")
    questions = [json.loads(line) for line in args.questions.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if any(row.get(key) not in (None, "") for row in questions for key in (
            "ground_truth", "answer", "reference_answer", "label")):
        raise ValueError("sampling inputs contain reference labels")
    by_id = {str(row["qa_id"]): row for row in questions}
    if len(by_id) != len(questions):
        raise ValueError("public question identifiers are not unique")
    titan = inventory("wangy@10.10.0.34", "python3", """
import json
from pathlib import Path
b=Path('/home/wangy/wangjh/VSIbench8Brunner')
sources=[]
for r in sorted((b/'runs').iterdir()):
    if not r.is_dir():
        continue
    outcomes={p.stem for p in (r/'outcomes').glob('*.json')}
    issued={p.parent.name for p in (r/'solver_requests').glob('*/*.json')}
    if outcomes or issued:
        sources.append({'source':str(r),'qa_ids':sorted(outcomes|issued),
                        'outcomes':len(outcomes),'issued_questions':len(issued)})
stages=Path('/home/wangy/wangjh/harness3d/four-datasets-20261031/vsi1000/stages')
for stage in sorted(stages.iterdir()):
    files=list((stage/'system/results').glob('*.json'))
    if not files:
        continue
    ids=set()
    for p in files:
        value=json.loads(p.read_text())
        qa=value.get('qa_id')
        if qa is None:
            raise ValueError('historical result lacks a question identifier')
        ids.add(str(qa))
    sources.append({'source':str(stage/'system/results'),'qa_ids':sorted(ids),
                    'result_files':len(files)})
print(json.dumps({'sources':sources}))
""")
    old_direct = inventory(
        "cvailab@10.10.0.198", "/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", """
import json
from pathlib import Path
r=Path('/home/cvailab/experiments/v11-direct-qwen-full-20261020-r02/run')
if not r.is_dir():
    raise ValueError('historical Direct run is unavailable')
ids={p.stem for p in (r/'responses').glob('*.json')}
print(json.dumps({'source':str(r),'qa_ids':sorted(ids),'responses':len(ids)}))
""")
    sources = [*titan["sources"], old_direct]
    excluded = set().union(*(set(source["qa_ids"]) for source in sources))
    eligible = [row for row in questions if str(row["qa_id"]) not in excluded]
    counts = Counter(canonical_task(row["question_type"]) for row in eligible)
    quotas = apportion(counts, args.n)
    groups = defaultdict(list)
    for row in eligible:
        groups[canonical_task(row["question_type"])].append(row)
    rng = random.Random(args.seed)
    selected = []
    raw_quotas = {}
    for task in sorted(groups):
        strata = defaultdict(list)
        for row in groups[task]:
            strata[row["question_type"]].append(row)
        subquotas = apportion(
            {key: len(rows) for key, rows in strata.items()}, quotas[task])
        raw_quotas.update(subquotas)
        for raw_type in sorted(strata):
            rows = sorted(strata[raw_type], key=lambda row: int(row["qa_id"]))
            selected.extend(rng.sample(rows, subquotas[raw_type]))
    if len(selected) != args.n or any(str(row["qa_id"]) in excluded for row in selected):
        raise ValueError("selected panel violates the unseen-question contract")
    selected.sort(key=lambda row: (canonical_task(row["question_type"]),
                                    row["question_type"], int(row["qa_id"])))
    manifest = {
        "schema": "vsi-proportional-unrun-panel/1",
        "n_questions": args.n, "seed": args.seed,
        "sampling": "largest remainder over unused canonical task counts, then over raw difficulty strata; seeded sampling without replacement",
        "source_question_file": str(args.questions),
        "source_question_sha256": sha(args.questions),
        "source_questions": len(questions), "eligible_unused_questions": len(eligible),
        "excluded_prior_question_count_in_source": len(set(by_id) & excluded),
        "proportion_basis": "remaining questions with no discovered prior Direct/Harness result or issued solver request",
        "eligible_type_counts": dict(sorted(counts.items())),
        "type_counts": dict(sorted(quotas.items())),
        "raw_type_counts": dict(sorted(raw_quotas.items())),
        "source_split_counts": dict(Counter(row["source_split"] for row in selected)),
        "experiment_split_counts": dict(Counter(row["experiment_split"] for row in selected)),
        "source_split_preserved": True, "reference_answer_visible": False,
        "real_labels_available": False, "questions": selected,
    }
    args.output_dir.mkdir()
    write(args.output_dir / "prior_run_inventory.json", {
        "source_question_sha256": sha(args.questions), "sources": sources,
        "excluded_qa_ids_in_source": sorted(set(by_id) & excluded),
        "reference_answers_read": False,
    })
    write(args.output_dir / "sampling_manifest.json", manifest)
    write(args.output_dir / "experiment_spec.json", {
        "schema": "vsi-proportional-paired-experiment/1",
        "n_questions": args.n, "sampling_sha256": sha(
            args.output_dir / "sampling_manifest.json"),
        "arms": ["qwen_direct_official32", "qwen8b_description_harness"],
        "solver_model": "qwen3.8-flash", "enable_thinking": True,
        "seed": 137, "temperature": 0., "max_tokens": 8192, "timeout_s": 900,
        "direct_requests_per_question": 1, "direct_request_cap": args.n,
        "harness_requests_per_question_cap": 12,
        "automatic_model_retry": False, "skill_evolution": False,
        "reference_answer_visible": False,
        "scoring": "MCA exact match and official-equivalent 10-threshold numeric MRA; report all answers and mutually legal common pairs",
        "trajectory_retention": "full native requests/responses, reasoning, program and tool records, preparation/description/review records, source hashes",
        "status": "selected",
    })
    print(json.dumps({key: manifest[key] for key in (
        "n_questions", "eligible_unused_questions", "excluded_prior_question_count_in_source",
        "type_counts", "raw_type_counts", "source_split_counts", "experiment_split_counts")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
