"""Isolated controlled trial using read-only native evidence and successor code.

Run with the frozen runtime's src/scripts/ops on PYTHONPATH. This command never
installs code into the frozen runtime or publishes a Skill.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def identity(path):
    raw = Path(path).read_bytes()
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seal(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def protected(state, frozen):
    paths = set()
    for root in (state / "request_governor", state / "final_bundle",
                 state / "accumulation/results", state / "accumulation/workers",
                 frozen / "posthoc-reviser-rescue-successor-20261029"):
        paths.update(p for p in root.rglob("*") if p.is_file() and
                     p.suffix in {".json", ".jsonl", ".md"})
    paths.update(p for p in (state / "accumulation/manifest.json",
                            state / "accumulation/summary.json",
                            frozen / "receipts/final_bundle_frozen.json") if p.is_file())
    return {str(path): identity(path) for path in sorted(paths)}


def source_identity(args):
    native = args.frozen_root / "runtime-parallel-reviser-v11-20261028"
    paths = [Path(__file__), args.source_root / "src/skill3d/evolution/revision_edits_v11.py",
             native / "code_identity.json"]
    paths.extend(p for p in (native / "source").rglob("*")
                 if p.is_file() and p.suffix in {".py", ".yaml", ".toml"})
    # Native operators are part of the imported runtime, even when no Git checkout
    # contains them. Hash their identities without exporting their contents.
    paths.extend((args.frozen_root / "ops").glob("*.py"))
    return {str(path): identity(path) for path in sorted(set(paths))}


def setup(args):
    # Add only the task's new editor module to the native package search path.
    import skill3d.evolution
    skill3d.evolution.__path__.insert(0, str(args.source_root / "src/skill3d/evolution"))
    from skill3d.evolution.revision_edits_v11 import AnchoredSlotReviser, RevisionBudget
    helper = load_module("successor_native_readonly_helper",
                         args.frozen_root / "ops/posthoc_reviser_rescue_v5_ops.py")
    if helper.state_root() != args.state_root:
        raise RuntimeError("native state identity mismatch")
    config = helper.configuration(args.state_root)
    client = config["factory_type"](config["settings"], config["evolution"]["max_tokens"],
                                   config["transport"]["endpoint"])()
    client.transport_retries = 0
    history = helper.history_reader(args.state_root)
    return helper, config, client, history, AnchoredSlotReviser, RevisionBudget


def inputs(args, helper, history, selection):
    from skill3d.schemas import SkillSpecV11, V11ExperienceBundle
    _, paths = helper.campaign_inputs(args.state_root, selection)
    raw = history.read_history_bytes(paths["experience"])
    experience = V11ExperienceBundle.model_validate_json(raw)
    request = read(paths["request"])
    parent = SkillSpecV11.model_validate(request["parent"])
    return paths, experience, parent, read(paths["feedback"]), {
        name: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if name == "experience" else identity(path) for name, path in paths.items()}


def plan_batches(native, editor, *, parent, feedback, experience):
    """Measure the successor prompt, retaining native reconstruction checks."""
    def successor_prompt(**kwargs):
        context = kwargs.get("batch_context") or {}
        return editor._prompt(
            parent=kwargs["parent"], working=context.get("working_skill_md", parent.skill_md),
            feedback=kwargs["feedback"], evidence=kwargs.get("evidence_text", ""),
            coverage={key: value for key, value in context.items() if key != "working_skill_md"},
            repair=kwargs.get("repair_context"),
        )
    native._prompt = successor_prompt
    native.evidence_request_budget = editor.budget.max_requests - editor.budget.repair_reserve - 1
    return native._evidence_batches(parent=parent, feedback=feedback,
                                    experience=experience, diagnosis=None)


def preflight(args):
    before = protected(args.state_root, args.frozen_root)
    helper, config, client, history, reviser_type, budget_type = setup(args)
    budget = budget_type(max_requests=12, repair_reserve=2,
                        max_tokens=config["evolution"]["max_tokens"])
    selected = []
    for selection in helper.SELECTION:
        paths, experience, parent, feedback, identities = inputs(args, helper, history, selection)
        native, _ = helper.make_reviser(args.state_root, {**selection, "paths": paths},
                                       args.output / "unused_native_output", client, config)
        # Use the native planner only. Its revision execution/repair is not called.
        editor = reviser_type(output_root=args.output / selection["kind"],
                    client=client, budget=budget,
                    method_context_max_chars=config["base"].retrieval_policy.method_context_max_chars,
                    expected_model=client.audit_identity()["model"])
        batches, plan = plan_batches(native, editor, parent=parent, feedback=feedback,
                                    experience=experience)
        checked = editor.preflight(parent=parent, feedback=feedback, batches=batches, delivery_plan=plan)
        selected.append({**selection, "input_identities": identities,
                         "input_refs": {name: str(path) for name, path in paths.items()},
                         "editor_preflight": checked, "delivery_plan": plan})
    if protected(args.state_root, args.frozen_root) != before:
        raise RuntimeError("protected assets changed during read-only preparation")
    seal(args.output / "preflight.json", {
        "status": "ready", "at": time.time(), "selected": selected,
        "protected_inputs": before, "source_identity": source_identity(args),
        "model_identity": client.audit_identity(seed=budget.seed),
        "budget": asdict(budget), "transport_retries": 0,
        "model_requests": 0, "gate_seeds": [137, 138],
        "gate_request_cap": 512, "total_reviser_request_cap": 24,
        "publication_authorized": False, "gate_feedback_visible_to_reviser": False})
    return {"status": "ready", "model_requests": 0, "selected": [
        {"campaign": item["campaign_id"],
         "batches": item["editor_preflight"]["budget"]["batch_count"],
         "max_wire_bytes": max(row["wire_bytes"] for row in item["editor_preflight"]["request_sizes"])}
        for item in selected]}


def run(args):
    from skill3d.evolution.fitset_revision_v11 import RevisionOperationalError
    from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig, GovernedClient
    ready = read(args.output / "preflight.json")
    if ready["source_identity"] != source_identity(args):
        raise RuntimeError("source identity changed after zero-request preflight")
    if ready["protected_inputs"] != protected(args.state_root, args.frozen_root):
        raise RuntimeError("original asset identity changed")
    seal(args.output / "startup.json", {"at": time.time(),
                                     "preflight_identity": identity(args.output / "preflight.json")})
    service = __import__("storage_scaleup_service_ops").facade()
    service.load_service_environment()
    helper, config, raw_client, history, reviser_type, budget_type = setup(args)
    if not raw_client.configured:
        raise RuntimeError("authorized service environment unavailable")
    governor = RequestGovernor(args.output / "request_governor",
                  RequestGovernorConfig(max_requests=24, max_inflight=1, requests_per_second=0.0))
    results = []
    stopped = False
    for selection in ready["selected"]:
        paths, experience, parent, feedback, observed = inputs(args, helper, history, selection)
        if observed != selection["input_identities"]:
            raise RuntimeError("retained campaign input identity changed")
        native, _ = helper.make_reviser(args.state_root, {**selection, "paths": paths},
                                       args.output / "unused_native_output", raw_client, config)
        client = GovernedClient(raw_client, governor, category="reviser", context={
                  "slot_key": selection["slot_key"], "campaign_id": selection["campaign_id"],
                  "run_identity": identity(args.output / "preflight.json")["sha256"]})
        editor = reviser_type(output_root=args.output / selection["kind"], client=client,
                    budget=budget_type(**ready["budget"]),
                    method_context_max_chars=config["base"].retrieval_policy.method_context_max_chars,
                    expected_model=ready["model_identity"]["model"])
        batches, plan = plan_batches(native, editor, parent=parent, feedback=feedback,
                                    experience=experience)
        checked = editor.preflight(parent=parent, feedback=feedback, batches=batches, delivery_plan=plan)
        if checked != selection["editor_preflight"]:
            raise RuntimeError("assembled evidence or request plan changed")
        forbidden = {str(row[key]) for row in config["panel"]["system_variants"]
                     for key in ("qa_id", "scene_id", "variant_id")}
        try:
            candidate = editor.run(parent=parent, feedback=feedback, batches=batches,
                                  delivery_plan=plan, forbidden_ids=forbidden, cases=experience.cases)
            result = read(args.output / selection["kind"] / "final.json")
            result.update(source_campaign=selection["campaign_id"],
                          source_kind=selection["kind"], slot_key=selection["slot_key"])
            results.append(result)
        except RevisionOperationalError:
            results.append({"source_campaign": selection["campaign_id"], "status": "operational_failure",
                            "candidate_failure": False, "requests": editor.requests})
            stopped = True
            break
    unchanged = ready["protected_inputs"] == protected(args.state_root, args.frozen_root)
    final = {"status": "stopped_on_operational_failure" if stopped else "revision_complete",
             "at": time.time(), "results": results, "request_governor": governor.summary(),
             "static_valid_candidates": sum(row["status"] == "static_valid" for row in results),
             "protected_inputs_unchanged": unchanged, "gate_executed": False, "published": False}
    seal(args.output / "revision_final.json", final)
    if not unchanged:
        raise RuntimeError("protected asset identity changed; execution stopped")
    return {key: final[key] for key in (
        "status", "results", "request_governor", "static_valid_candidates", "protected_inputs_unchanged")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.is_relative_to(args.frozen_root) or args.output.is_relative_to(args.state_root):
        raise SystemExit("successor must be outside frozen assets")
    try:
        result = preflight(args) if args.action == "preflight" else run(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        # No provider text, raw evidence or credential-bearing environment is logged.
        import traceback
        print(json.dumps({"status": "operational_failure", "exception_type": type(exc).__name__,
                          "frames": [{"file": Path(frame.filename).name, "line": frame.lineno}
                                     for frame in traceback.extract_tb(exc.__traceback__)]}))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
