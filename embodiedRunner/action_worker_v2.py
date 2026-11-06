"""Separate opt-in V2 endpoint; never changes V1 default CLI or experiment paths."""
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import sys

from action_adapters import FakeAdapter, HouseholdAdapter, NavigationAdapter
from mutation_protocol import MAX_BYTES, MutationSession
from protocol import ContractError, OperationalError, ROOT, canonical, json_object, sha256, within_root


def serve(session, input_stream, wire):
    try:
        while True:
            line = input_stream.readline(MAX_BYTES + 1)
            if not line:
                break
            sequence = None
            try:
                if len(line) > MAX_BYTES or not line.endswith("\n"):
                    raise ContractError("mutation_command_cap")
                command = json_object(line)
                sequence = command.get("transport_sequence")
                if command == {"op": "handshake", "transport_sequence": 0} and type(sequence) is int:
                    data = session.handshake()
                else:
                    data = session.execute(command)
                response = {"transport_sequence": sequence, "ok": True, "data": data}
            except Exception as exc:
                response = {"transport_sequence": sequence, "ok": False,
                            "error_code": "action_outcome_unknown" if isinstance(exc, OperationalError)
                            else "mutation_contract_rejected"}
            wire.write(canonical(response).decode("utf-8") + "\n")
            wire.flush()
            if response["ok"] is False:
                break
    finally:
        session.adapter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("eb_navigation", "eb_alfred", "eb_habitat", "eb_manipulation"), required=True)
    parser.add_argument("--subset", default="base")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--official-runtime-verified", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--gpu-device", type=int)
    args = parser.parse_args()
    work = within_root(args.work_dir)
    if not work.is_dir():
        parser.error("work-dir must already exist")
    if args.fake:
        adapter = FakeAdapter(args.environment)
        with contextlib.redirect_stdout(sys.stderr):
            serve(MutationSession(adapter), sys.stdin, sys.__stdout__)
        return
    if not args.official_runtime_verified:
        parser.error("official runtime deployment preflight is required")
    # Credential-free process; dependency/native output is discarded rather
    # than persisted. Native log and auth stripping reuse the trusted V1 glue.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as wire, \
            open(os.devnull, "w") as log:
        from household_worker import native_log
        with native_log(log):
            if args.environment == "eb_navigation":
                from bridge import simulator_environment
                environment = simulator_environment(ROOT)
                environment = {k: v for k, v in environment.items() if "PROXY" not in k.upper()}
                environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                environment.pop("CUDA_VISIBLE_DEVICES", None)
                environment["PATH"] = str(ROOT / "runtime/vulkan-tools/usr/bin") + os.pathsep + environment.get("PATH", "")
                os.environ.clear()
                os.environ.update(environment)
                from simulator_worker import OfficialNavigation, probe
                receipt = probe()
                adapter = NavigationAdapter(OfficialNavigation(
                    ROOT / "upstream", work, resolution=500, gpu_device=args.gpu_device))
            elif args.environment == "eb_manipulation":
                from manipulation_worker import runtime, probe, OfficialManipulation
                runtime()
                receipt = probe()
                adapter = OfficialManipulation(args.subset, work)
            else:
                from household_worker import probe, runtime
                from action_household_v2 import OfficialHouseholdV2
                runtime()
                kind = args.environment[3:]
                receipt = probe(kind)
                adapter = HouseholdAdapter(OfficialHouseholdV2(kind, args.subset, work), args.environment)
                if kind == "alfred":
                    receipt["configured_render_profile"] = OfficialHouseholdV2.ALFRED_RENDER_PROFILE
            if args.probe:
                names = ["action_worker_v2.py", "action_adapters.py", "mutation_protocol.py", "protocol.py"]
                if args.environment in {"eb_alfred", "eb_habitat"}:
                    names.append("action_household_v2.py")
                if args.environment == "eb_alfred":
                    names.append("action_alfred_data.json")
                if args.environment == "eb_manipulation":
                    names += ["manipulation_worker.py", "manipulation_protocol.py",
                              "manipulation_sources.json", "manipulation_assets.json"]
                receipt.update(schema="embodied-v2-runtime-check/1", environment=args.environment,
                               source_sha256={name: sha256((ROOT / name).read_bytes()) for name in names},
                               automatic_retry=False, reference_answer_visible=False)
                wire.write(canonical(receipt).decode() + "\n")
                wire.flush()
            else:
                if not receipt["runtime_ready"]:
                    raise OperationalError("missing_simulator_dependencies")
                serve(MutationSession(adapter), sys.stdin, wire)


if __name__ == "__main__":
    main()
