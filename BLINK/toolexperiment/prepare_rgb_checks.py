#!/usr/bin/env python3
"""Freeze RGB-only SIFT correspondences before inspecting cached geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from run_tools import PROTOCOL
    cv2.setNumThreads(1)
    results = {}
    for split in ("train", "test"):
        rows = [json.loads(s) for s in (args.dataset_root/"skill3d_splits"/
                f"blink_multiview_{split}.jsonl").read_text().splitlines() if s.strip()]
        for row in rows:
            cv2.setRNGSeed(PROTOCOL["seed"])
            sift = cv2.SIFT_create(nfeatures=PROTOCOL["sift_features"])
            paths = [args.dataset_root/p for p in row["image"]]
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            gray = [np.asarray(Image.open(p).convert("L")) for p in paths]
            (k0, d0), (k1, d1) = [sift.detectAndCompute(image, None) for image in gray]
            base = {"image_sha256": hashes}
            if d0 is None or d1 is None:
                results[row["id"]] = {**base, "status": "no_descriptors"}
                continue
            matcher = cv2.BFMatcher()
            def ratio(a, b):
                return {m.queryIdx: m for pair in matcher.knnMatch(a, b, k=2)
                        if len(pair) == 2 for m, n in [pair]
                        if m.distance < PROTOCOL["sift_ratio"]*n.distance}
            fwd, rev = ratio(d0, d1), ratio(d1, d0)
            mutual = sorted([m for i, m in fwd.items()
                             if m.trainIdx in rev and rev[m.trainIdx].trainIdx == i],
                            key=lambda m: (m.distance, m.queryIdx, m.trainIdx))
            base["mutual_matches"] = len(mutual)
            if len(mutual) < PROTOCOL["min_inlier_matches"]:
                results[row["id"]] = {**base, "status": "insufficient_matches"}
                continue
            xy0 = np.array([k0[m.queryIdx].pt for m in mutual], dtype=np.float32)
            xy1 = np.array([k1[m.trainIdx].pt for m in mutual], dtype=np.float32)
            _, mask = cv2.findFundamentalMat(xy0, xy1, cv2.FM_RANSAC,
                                           PROTOCOL["fundamental_ransac_px"], .999)
            good = np.flatnonzero(mask.ravel()) if mask is not None else []
            base["ransac_inliers"] = len(good)
            if len(good) < PROTOCOL["min_inlier_matches"]:
                results[row["id"]] = {**base, "status": "insufficient_inliers"}
                continue
            good = good[:PROTOCOL["max_matches"]]
            results[row["id"]] = {**base, "status": "matched",
                                   "xy0": xy0[good].tolist(), "xy1": xy1[good].tolist()}
    record = {"schema": "blink-rgb-only-correspondences/1", "opencv": cv2.__version__,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "geometry_access": False, "answer_access": False, "samples": results}
    args.output.write_text(json.dumps(record, separators=(",", ":"))+"\n")
    print(json.dumps({"samples": len(results), "matched": sum(
        r["status"] == "matched" for r in results.values()), "output": str(args.output)}))


if __name__ == "__main__":
    main()
