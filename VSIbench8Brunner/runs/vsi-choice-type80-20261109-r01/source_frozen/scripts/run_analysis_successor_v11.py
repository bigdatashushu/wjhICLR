"""Isolated source-linked trajectory analysis and final anchored revision."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time

script_dir = Path(__file__).resolve().parent
# Keep frozen run_fitset_v11 and operators ahead of this full checkout.
sys.path[:] = [p for p in sys.path if Path(p).resolve() != script_dir]
spec = importlib.util.spec_from_file_location(
    "analysis_successor_receipt_tools", script_dir / "run_reviser_successor_v11.py")
tools = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tools
spec.loader.exec_module(tools)

PILOT = ("596e418df035604a134c", "5e44b53713c0750a314c", "052616638f71d731e97c")


def setup(args):
    helper, config, client, history, editor_type, budget_type = tools.setup(args)
    analysis = tools.load_module("skill3d.evolution.trajectory_analysis_v11",
        args.source_root / "src/skill3d/evolution/trajectory_analysis_v11.py")
    tools.load_module("skill3d.evolution.trajectory_fragments_v11",
        args.source_root / "src/skill3d/evolution/trajectory_fragments_v11.py")
    return helper, config, client, history, editor_type, budget_type, analysis


def source_identity(args):
    return {**tools.source_identity(args), **{
        str(p): tools.identity(p) for p in (
            Path(__file__), args.source_root / "scripts/run_reviser_successor_v11.py",
            args.source_root / "src/skill3d/evolution/trajectory_analysis_v11.py",
            args.source_root / "src/skill3d/evolution/trajectory_fragments_v11.py")}}


def selections(args, helper):
    if args.selection == "pilot":
        by_id = {row["campaign_id"]: row for row in tools.catalog_specs(args, helper)}
        selected = [by_id[name] for name in PILOT]
    else:
        raw = tools.read(args.campaign_index)["campaigns"]
        by_slot = {}
        for row in raw:
            by_slot.setdefault(row["slot_key"], []).append(row)
        names = []
        for slot, rows in sorted(by_slot.items()):
            rows = sorted(rows, key=lambda row: (
                row["refusal_facts"].get("needed_batches",
                    row["refusal_facts"].get("completed_batch_count", 0) + 1), row["campaign_id"]))
            names.append(rows[(len(rows)-1)//2]["campaign_id"])
        by_id = {row["campaign_id"]: row for row in tools.catalog_specs(args, helper)}
        selected = [by_id[name] for name in names]
    slots = getattr(args, "slots", None)
    if slots:
        if len(slots) != len(set(slots)) or set(slots) - {r["slot_key"] for r in selected}:
            raise RuntimeError("requested slots are not unique members of the fixed selection")
        selected = [r for r in selected if r["slot_key"] in slots]
    return selected


def compact_feedback(feedback):
    from skill3d.evolution.trajectory_analysis_v11 import compact_analysis_feedback
    return compact_analysis_feedback(feedback)


def prepare_one(args, helper, config, history, client, analysis, selection):
    paths, experience, parent, feedback, identities = tools.inputs(
        args, helper, history, selection)
    from skill3d.evolution.revision_input_v11 import render_case_evidence
    render_case_evidence(experience, experiment_split=config["manifest"]["experiment_split"])
    if (experience.parent_content_sha256 != parent.content_sha256
            or experience.question_type != parent.question_type
            or experience.evidence_state != parent.evidence_state
            or len(experience.cases) != 12):
        raise RuntimeError("learning bundle parent slot or case denominator mismatch")
    feedback = compact_feedback(feedback)
    analyst = analysis.TrajectoryAnalyst(output_root=args.output / selection["campaign_id"] / "analysis",
        client=client, budget=analysis.RevisionBudget(max_requests=260, repair_reserve=0),
        method_context_max_chars=config["base"].retrieval_policy.method_context_max_chars,
        expected_model="qwen3.8-flash")
    native, _ = helper.make_reviser(args.state_root, {**selection, "paths": paths},
                                   args.output / "unused-native", client, config)

    def planning_refusal(row, error):
        complete = row["complete_case"]
        round_ = complete["rounds"][0]
        tools.seal(args.output / selection["campaign_id"] / "planning_refusal.json", {
            "status": "indivisible_input", "slot": parent.slot_key, "model_requests": 0,
            "case_index": row["case_index"], "round_index": round_.get("index"),
            "multi_field_refusal": error["code"], "multi_field_facts": error["facts"],
            "round_field_chars": {k: len(json.dumps(v, ensure_ascii=False))
                                  for k, v in round_.items()},
            "header_field_chars": {k: len(json.dumps(v, ensure_ascii=False))
                                   for k, v in complete.items() if k != "rounds"},
        })

    batches, preparation = analysis.prepare_analysis_batches(
        native, analyst=analyst, parent=parent, feedback=feedback,
        experience=experience, on_fragment_refusal=planning_refusal)
    metadata = {
        **selection, "input_identities": identities,
        "input_refs": {k:str(v) for k,v in paths.items()},
        "parent_sha256": parent.content_sha256,
        **preparation, "revision_request_cap": 3,
        "total_request_cap": 2 * len(batches) + 3,
    }
    return metadata, parent, feedback, experience, batches


def preflight(args):
    before = tools.protected(args.state_root, args.frozen_root)
    helper, config, client, history, _, _, analysis = setup(args)
    rows = []
    for selection in selections(args, helper):
        tools.seal(args.output / "planning" / (selection["campaign_id"] + ".started.json"),
                   {"slot": selection["slot_key"], "at": time.time(), "model_requests": 0})
        row, *_ = prepare_one(args, helper, config, history, client, analysis, selection)
        rows.append(row)
        tools.seal(args.output / "planning" / (selection["campaign_id"] + ".ready.json"),
                   {"slot": selection["slot_key"], "batch_count": row["batch_count"],
                    "at": time.time(), "model_requests": 0})
    cap = sum(r["total_request_cap"] for r in rows)
    required_caps = {}
    for row in rows:
        required_caps[row["slot_key"]] = required_caps.get(row["slot_key"], 0) + row["total_request_cap"]
    budget = {
        "status": "ready" if cap <= args.reviser_request_cap else "budget_refused",
        "model_requests": 0, "required_total": cap,
        "authorized_total": args.reviser_request_cap,
        "required_revision_request_caps": required_caps,
        "gate_allowance_unchanged": args.gate_request_cap,
    }
    tools.seal(args.output / "required_budget.json", budget)
    if budget["status"] != "ready":
        return budget
    if before != tools.protected(args.state_root, args.frozen_root):
        raise RuntimeError("protected inputs changed")
    tools.seal(args.output / "preflight.json", {
        "status":"ready", "selected":rows, "source_identity":source_identity(args),
        "protected_inputs":before, "model_identity":client.audit_identity(seed=137),
        "model_requests":0, "total_reviser_request_cap":cap,
        "gate_request_cap":args.gate_request_cap, "selection":args.selection,
        "analysis_concurrency": args.analysis_concurrency,
        "required_revision_request_caps": required_caps,
        "selection_slots": getattr(args, "slots", None),
        "gate_feedback_visible_to_reviser":False, "publication_authorized":False})
    return {"status":"ready", "model_requests":0, "reviser_cap":cap,
            "required_revision_request_caps":required_caps,
            "cohort":[{"slot":r["slot_key"],"campaign":r["campaign_id"],
                      "batches":r["batch_count"]} for r in rows]}


def run(args):
    from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig, GovernedClient
    from skill3d.evolution.fitset_revision_v11 import RevisionOperationalError
    ready = tools.read(args.output / "preflight.json")
    if ready["source_identity"] != source_identity(args):
        raise RuntimeError("source changed after preflight")
    if ready["protected_inputs"] != tools.protected(args.state_root, args.frozen_root):
        raise RuntimeError("protected input changed after preflight")
    if args.analysis_concurrency != ready["analysis_concurrency"]:
        raise RuntimeError("analysis concurrency changed after preflight")
    tools.seal(args.output / "startup.json", {"at":time.time(),
                                             "preflight":tools.identity(args.output/"preflight.json")})
    __import__("storage_scaleup_service_ops").facade().load_service_environment()
    helper, config, raw, history, editor_type, budget_type, analysis = setup(args)
    if not raw.configured:
        raise RuntimeError("service environment unavailable")
    governor = RequestGovernor(args.output / "request_governor", RequestGovernorConfig(
        max_requests=ready["total_reviser_request_cap"],
        max_inflight=ready["analysis_concurrency"], requests_per_second=0.0))
    results, stopped = [], False
    for selection in ready["selected"]:
        metadata, parent, feedback, experience, batches = prepare_one(
            args, helper, config, history, raw, analysis, selection)
        if metadata != selection:
            raise RuntimeError("prepared analysis input changed")
        root = args.output / selection["campaign_id"]
        context = {
            "campaign_id":selection["campaign_id"],"slot_key":parent.slot_key,
            "run_identity":tools.identity(args.output/"preflight.json")["sha256"]}
        client = GovernedClient(raw, governor, category="reviser", context=context)
        def fresh_client():
            fresh = config["factory_type"](
                config["settings"], config["evolution"]["max_tokens"],
                config["transport"]["endpoint"])()
            fresh.transport_retries = 0
            return GovernedClient(fresh, governor, category="reviser", context=context)
        forbidden = {str(row[k]) for row in config["panel"]["system_variants"]
                     for k in ("qa_id","scene_id","variant_id")}
        analyst = analysis.TrajectoryAnalyst(output_root=root/"analysis",client=client,
            budget=budget_type(max_requests=selection["analysis_request_cap"],repair_reserve=0),
            method_context_max_chars=config["base"].retrieval_policy.method_context_max_chars,
            expected_model="qwen3.8-flash", analysis_concurrency=args.analysis_concurrency,
            client_factory=fresh_client)
        caps = selection["analysis_chars_per_batch"]
        if len(caps) != len(batches):
            raise RuntimeError("analysis batch budget count changed")
        analyst.analysis_batch_chars = {
            analysis.digest(batch): cap for batch, cap in zip(batches, caps)}
        try:
            summaries = analyst.analyze(parent=parent,feedback=feedback,batches=batches,
                cases=experience.cases,forbidden_ids=forbidden)
            if summaries is None:
                analysis_final = tools.read(root/"analysis/analysis_final.json")
                final = {"status":"analysis_rejected",
                         "all_evidence_delivered":analysis_final["all_evidence_delivered"],
                         "all_batches_resolved":False}
            else:
                evidence = analysis.synthesis_evidence(
                    summaries, question_type=parent.question_type)
                editor = editor_type(output_root=root/"revision",client=client,
                    budget=budget_type(max_requests=3,repair_reserve=2),
                    method_context_max_chars=config["base"].retrieval_policy.method_context_max_chars,
                    expected_model="qwen3.8-flash")
                editor.run(parent=parent,feedback=feedback,batches=[evidence],
                    delivery_plan={"analysis_input_sha256":analysis.digest(evidence),
                                   "original_batch_count":len(batches),
                                   "analysis_complete":True},
                    forbidden_ids=forbidden,cases=experience.cases)
                final = tools.read(root/"revision/final.json")
                final["all_batches_resolved"] = True
            final.update(source_campaign=selection["campaign_id"],source_kind=selection["kind"],
                slot_key=parent.slot_key,
                candidate_ref=selection["campaign_id"]+"/revision/SKILL.md",
                analyst_requests=analyst.requests)
            tools.seal(root/"result.json",final)
            results.append(final)
            tools.seal(args.output/"progress"/(selection["campaign_id"]+".json"), {
                "at":time.time(),"completed":len(results),"total":len(ready["selected"]),
                "status":final["status"],"requests":governor.summary()["completed_requests"]})
        except RevisionOperationalError:
            results.append({"status":"operational_failure","source_campaign":selection["campaign_id"],
                            "candidate_failure":False,"analyst_requests":analyst.requests})
            stopped=True
            break
    unchanged = ready["protected_inputs"] == tools.protected(args.state_root,args.frozen_root)
    final = {"status":"stopped_on_operational_failure" if stopped else "revision_complete",
             "results":results,"request_governor":governor.summary(),
             "static_valid_candidates":sum(r["status"]=="static_valid" for r in results),
             "protected_inputs_unchanged":unchanged,"published":False,"at":time.time()}
    tools.seal(args.output/"revision_final.json",final)
    if not unchanged:
        raise RuntimeError("original assets changed")
    return {k:final[k] for k in ("status","static_valid_candidates","request_governor",
                                "protected_inputs_unchanged")}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("preflight","run"))
    for name in ("frozen-root","state-root","source-root","output","campaign-index"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--selection",choices=("pilot","cohort"),default="pilot")
    p.add_argument("--slots", nargs="+")
    p.add_argument("--reviser-request-cap",type=int,default=1536)
    p.add_argument("--gate-request-cap",type=int,default=4096)
    p.add_argument("--analysis-concurrency",type=int,default=4)
    args=p.parse_args()
    if not 1 <= args.analysis_concurrency <= 64:
        raise SystemExit("analysis concurrency must be within 1..64")
    if min(args.reviser_request_cap, args.gate_request_cap) < 1:
        raise SystemExit("shared request allowances must be positive")
    if args.output.is_relative_to(args.frozen_root) or args.output.is_relative_to(args.state_root):
        raise SystemExit("new analysis output must be outside frozen assets")
    try:
        print(json.dumps(preflight(args) if args.action=="preflight" else run(args),sort_keys=True))
    except Exception as exc:
        import traceback
        print(json.dumps({"status":"operational_failure","exception_type":type(exc).__name__,
            "frames":[{"file":Path(f.filename).name,"line":f.lineno}
                      for f in traceback.extract_tb(exc.__traceback__)]}))
        raise SystemExit(2) from None


if __name__=="__main__":
    main()
