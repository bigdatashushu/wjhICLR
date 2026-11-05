"""One image-protocol diagnostic, with a separate one-request ledger."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from .common import digest, write
from .jev import Client, Policy, host_review


def probe(image, target, xy, directory, provider="intern", endpoint="", identity="",
          yes_threshold=None):
    if not directory.parent.is_dir():
        raise ValueError("probe_parent_missing")
    directory.mkdir()
    budget_root = directory / "requests"
    budget_root.mkdir()
    policy = Policy(provider=provider, max_requests=1, endpoint=endpoint,
                    service_identity_sha256=identity, yes_threshold=yes_threshold)
    rgb = image.read_bytes()
    row = {"result_id": "image-protocol-probe", "source_tool": "point_depth", "status": "ok",
           "data": {"pixel_xy": xy, "view_index": 0, "depth": 1.0,
                    "jev_quantity": "camera_z", "intended_targets": [
                        {"target": target, "surface": "visible_surface"}]}}
    governor = RequestGovernor(budget_root, RequestGovernorConfig(1, 1))

    class Budget:
        @contextmanager
        def request(self, category="gate"):
            with governor.request(category, {
                    "campaign_id": directory.name, "variant_id": "jev-image-protocol-diagnostic",
                    "slot_key": "one", "run_identity": digest({"row": row, "policy": policy.public()})}) as charge:
                yield charge

    receipt = Client(policy, Budget()).review(row, rgb)
    review = host_review(row, receipt, rgb, policy)
    write(directory / "receipt.json", receipt, exclusive=True)
    write(directory / "summary.json", {
        "status": receipt["status"], "decision": receipt["decision"],
        "http_status": receipt.get("http_status"), "error_code": receipt.get("error_code"),
        "model_requests": int(receipt["request_attempted"]), "solver_requests": 0,
        "formal_campaign": False, "budget_scope": "separate one-request protocol diagnostic",
        "reference_answer_visible": False, "automatic_retry": False,
        "measurement_value_is_placeholder": True, "quantitative_validation": False,
        "host_admission": review["decision"]}, exclusive=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--xy", type=int, nargs=2, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--provider", choices=("intern", "makers", "typesafe"), default="intern")
    parser.add_argument("--review-endpoint", default="")
    parser.add_argument("--review-identity", default="")
    parser.add_argument("--yes-threshold", type=float, default=.5,
                        help="Intern calibrated p(yes) must strictly exceed this value (default .5)")
    args = parser.parse_args()
    receipt = probe(args.image, args.target, args.xy, args.root.resolve(), args.provider,
                    args.review_endpoint, args.review_identity,
                    args.yes_threshold if args.provider == "intern" else None)
    print(f'{receipt["status"]}; HTTP={receipt.get("http_status")}; decision={receipt["decision"]}')


if __name__ == "__main__":
    main()
