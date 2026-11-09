"""Offline camera arithmetic and unknown-depth object-bearing audit; no API/GPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import time

from ..traces import digest, file_digest, read, write
from .prototype.inputs import load_images
from .vggt_pose import relative_cameras, same_evidence

DEFAULT = Path("/Users/bytedance/wangjh/MMSI六类五题位姿实验-20261108-v2/continuation")
# Visible RGB rectangles selected for the named objects, before comparison.
# They express image localization uncertainty, not validated 3D surfaces.
REGIONS = {
    "MMSI_837": {"view": 1, "xyxy": [350, 650, 595, 925], "target": "orange box"},
    "MMSI_502": {"view": 1, "xyxy": [900, 170, 1295, 650], "target": "black shelf"},
}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def angle(a, b):
    cosine = dot(a, b) / math.sqrt(dot(a, a) * dot(b, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def cardinal(vector):
    x, _, z = vector
    if abs(x) < 1e-12 and abs(z) < 1e-12:
        return "undefined"
    if abs(x) > abs(z):
        return "right" if x > 0 else "left"
    return "front" if z > 0 else "back"


def quadrant(vector):
    x, _, z = vector
    if abs(x) < 1e-12 or abs(z) < 1e-12:
        return "boundary"
    return ("front" if z > 0 else "rear") + " " + ("right" if x > 0 else "left")


def option(problem, label):
    aliases = {"in front": "front", "to the left": "left", "to the right": "right"}
    choices = [aliases.get(value.lower(), value.lower()) for value in problem["options"]]
    return chr(65 + choices.index(label)) if label in choices else None


def camera_row(pose, camera, reference):
    return next(row for row in pose["camera_evidence"]["relative_cameras"]
                if row["camera_image"] == camera and row["reference_image"] == reference)


def camera_result(problem, pose, sent_text):
    question = problem["question"]
    result = {
        "sample_id": problem["sample_id"], "question": question, "options": problem["options"],
        "answer": None, "status": "computed_in_camera_axes",
        "rotation_sent_to_model": "_rotation_rows" in sent_text,
    }
    if "projections on the ground" in question.lower():
        forward = camera_row(pose, 2, 1)["camera_forward_in_reference"]
        full = angle([0, 0, 1], forward)
        projected = angle([0, 0, 1], [forward[0], 0, forward[2]])
        label = "acute angle" if projected < 90 else "obtuse angle"
        result.update(
            relation="ground_projected_camera_angle", status="ground_normal_missing",
            angle_3d_degrees=full, angle_assuming_camera1_y_is_vertical=projected,
            conditional_answer=option(problem, label), conditional_label=label,
            condition="camera1 y axis is the ground normal; not verified",
            reason="Camera poses alone do not identify the physical ground plane.",
        )
    elif "+Y up" in question and "-Z forward" in question:
        center = camera_row(pose, 2, 1)["camera_center_xyz"]
        converted = [center[0], -center[1], -center[2]]
        xsign, zsign = ("positive" if converted[axis] > 0 else "negative" for axis in (0, 2))
        choices = []
        for index, text in enumerate(problem["options"]):
            if re.search(xsign + r"\s+X direction", text) and re.search(
                    zsign + r"\s+Z direction", text):
                choices.append(chr(65 + index))
        if len(choices) != 1:
            raise ValueError("unresolved_question_coordinate_options")
        result.update(relation="signed_camera_displacement", center_opencv=center,
                      center_question_axes=converted, conversion="diag(1,-1,-1)",
                      answer=choices[0], label=f"{xsign} X, {zsign} Z")
    elif re.search(r"perspective.*first image.*transformed.*second image", question, re.I):
        forward = camera_row(pose, 2, 1)["camera_forward_in_reference"]
        yaw = math.degrees(math.atan2(forward[0], forward[2]))
        pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
        label = (("turn right" if yaw > 0 else "turn left") if abs(yaw) >= abs(pitch)
                 else ("turn up" if pitch > 0 else "turn down"))
        result.update(relation="camera_turn", forward_camera2_in_camera1=forward,
                      yaw_degrees=yaw, pitch_degrees=pitch,
                      rule="classify the dominant yaw/pitch component",
                      answer=option(problem, label), label=label)
    else:
        if re.search(r"relative to the shooting location of the second photo", question, re.I):
            camera, reference = 1, 2
        elif re.search(r"from the location where the first photo was taken", question, re.I):
            camera, reference = 2, 1
        else:
            raise ValueError("unsupported_public_camera_question")
        center = camera_row(pose, camera, reference)["camera_center_xyz"]
        if center is None:
            result.update(status="degenerate_camera_baseline")
            return result
        diagonal = any("rear left" == text.lower() for text in problem["options"])
        label = quadrant(center) if diagonal else cardinal(center)
        result.update(relation="camera_location", camera=camera, reference=reference,
                      center=center, label=label, answer=option(problem, label),
                      bearing_degrees=math.degrees(math.atan2(center[0], center[2])),
                      rule="quadrant signs" if diagonal else "dominant horizontal x/z component")
    return result


def depth_intervals(center, ray, diagonal):
    """Enumerate every sector over lambda>0 with analytic boundary roots."""
    expressions = [(center[0], ray[0]), (center[2], ray[2])]
    if not diagonal:
        expressions += [(center[0] - center[2], ray[0] - ray[2]),
                        (center[0] + center[2], ray[0] + ray[2])]
    roots = sorted({-constant / slope for constant, slope in expressions
                    if abs(slope) > 1e-12 and -constant / slope > 0})
    boundaries = [0.0, *roots]
    intervals = []
    classify = quadrant if diagonal else cardinal
    for index, lower in enumerate(boundaries):
        upper = boundaries[index + 1] if index + 1 < len(boundaries) else None
        sample = (lower + upper) / 2 if upper is not None else max(1.0, lower * 2 + 1.0)
        point = [c + sample * a for c, a in zip(center, ray)]
        label = classify(point)
        if intervals and intervals[-1]["label"] == label:
            intervals[-1]["upper"] = upper
        else:
            intervals.append({"lower": lower, "upper": upper, "label": label})
    return intervals


def original_pixel_ray(pose, receipt, view, xy):
    width, height = receipt["width"], receipt["height"]
    resized_width = 518
    resized_height = round(height * (resized_width / width) / 14) * 14
    cropped_height = min(518, resized_height)
    crop_top = max(0, (resized_height - 518) // 2)
    if pose["preprocessed_shape"][2:] != [cropped_height, resized_width]:
        raise ValueError("requires_same_shape_official_crop_without_batch_padding")
    # Preserve pixel centers under PIL's resize grid.
    u = (xy[0] + 0.5) * resized_width / width - 0.5
    v = (xy[1] + 0.5) * resized_height / height - 0.5 - crop_top
    intrinsic = pose["intrinsics_preprocessed_pixels"][view - 1]
    if intrinsic[0][1] != 0 or intrinsic[1][0] != 0 or intrinsic[2] != [0.0, 0.0, 1.0]:
        raise ValueError("requires_vggt_diagonal_pinhole_intrinsic")
    ray = [(u - intrinsic[0][2]) / intrinsic[0][0],
           (v - intrinsic[1][2]) / intrinsic[1][1], 1.0]
    return ray, [u, v]


def object_result(problem, receipts, pose, region):
    source, reference = region["view"], 2
    transform = camera_row(pose, source, reference)
    center = transform["camera_center_xyz"]
    x1, y1, x2, y2 = region["xyxy"]
    pixels = [[(x1 + x2) / 2, (y1 + y2) / 2],
              [x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    diagonal = any("rear right" == text.lower() for text in problem["options"])
    rays = []
    for pixel in pixels:
        original, preprocessed = original_pixel_ray(pose, receipts[source - 1], source, pixel)
        aligned = matvec(transform["camera_axes_to_reference_rotation"], original)
        intervals = depth_intervals(center, aligned, diagonal)
        rays.append({
            "original_pixel_xy": pixel, "preprocessed_pixel_xy": preprocessed,
            "ray_in_source_camera": original, "ray_in_reference_camera": aligned,
            "positive_depth_intervals": intervals,
            "rotation_only_label": quadrant(aligned) if diagonal else cardinal(aligned),
        })
    labels = sorted({interval["label"] for row in rays for interval in row["positive_depth_intervals"]})
    undefined = option(problem, "cannot be determined")
    center_labels = {interval["label"] for interval in rays[0]["positive_depth_intervals"]}
    result = {
        "sample_id": problem["sample_id"], "question": problem["question"], "options": problem["options"],
        "relation": "object_bearing_with_unknown_depth", "source_view": source,
        "reference_view": reference, "region": region, "center_reference": center,
        "formula": "p_reference(lambda) = center_source_in_reference + lambda * R_source_to_reference * K_source^-1[u,v,1]",
        "lambda_unit": "source-camera z depth in the same normalized scale as camera centers",
        "lambda_constraint": "lambda>0; no object depth supplied",
        "extra_evidence": "manual RGB localization and saved VGGT-predicted intrinsics; intrinsics were not in GPT prompt",
        "rays": rays, "possible_labels": labels,
        "unique_direction": len(center_labels) == 1,
        "answer": (option(problem, next(iter(center_labels))) if len(center_labels) == 1 else undefined),
        "status": ("unique_given_predicted_geometry" if len(center_labels) == 1 else
                   "ambiguous_with_undetermined_option" if undefined is not None else
                   "ambiguous_without_undetermined_option"),
        "rotation_only_answer": option(problem, rays[0]["rotation_only_label"]),
        "rotation_only_assumption": "translation is negligible relative to object distance",
    }
    return result


def selfcheck():
    assert depth_intervals([-1, 0, 0], [1, 0, 0.5], False) == [
        {"lower": 0.0, "upper": 2 / 3, "label": "left"},
        {"lower": 2 / 3, "upper": 2.0, "label": "front"},
        {"lower": 2.0, "upper": None, "label": "right"},
    ]
    sectors = depth_intervals([1, 0, -1], [-1, 0, -0.2], True)
    assert [row["label"] for row in sectors] == ["rear right", "rear left"]
    assert cardinal([1, 0, 0]) == "right" and quadrant([-1, 0, 1]) == "front left"
    assert matvec([[1, 0, 0], [0, -1, 0], [0, 0, -1]], [-2, 1, -3]) == [-2, -1, 3]


def calculate(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    selected = [item for item in manifest["items"]
                if item["problem"]["task_type"] == "Positional Relationship (Cam.–Cam.)"
                or item["problem"]["sample_id"] in REGIONS]
    rows = []
    for item in selected:
        problem = item["problem"]
        load_images(problem, item["images"])
        pose = read(directory / item["pose_path"])
        evidence = pose["camera_evidence"]
        if (not same_evidence(evidence, relative_cameras(evidence["world_to_camera_extrinsics"]))
                or file_digest(directory / item["pose_path"]) != item["pose_sha256"]):
            raise ValueError("frozen_camera_evidence_changed")
        if problem["sample_id"] in REGIONS:
            row = object_result(problem, item["images"], pose, REGIONS[problem["sample_id"]])
        else:
            row = camera_result(problem, pose, item["pose_text"])
        row["pose_sha256"] = item["pose_sha256"]
        rows.append(row)
    result = {
        "schema": "mmsi-offline-vggt-math-check/1", "source_run_identity": manifest["run_identity"],
        "rows": rows, "source_identity": {"vggt_math_check.py": file_digest(__file__)},
        "model_requests": 0, "gpu_forwards": 0, "ground_truth_geometry_used": False,
        "reference_answers_read_in_calculation": False, "calculated_at": time.time(),
    }
    result["calculation_identity"] = digest(result)
    write(directory / "位姿数学计算冻结.json", result, exclusive=True)
    return result


def compare(directory):
    directory = Path(directory)
    calculation = read(directory / "位姿数学计算冻结.json")
    if digest({k: v for k, v in calculation.items() if k != "calculation_identity"}) != calculation["calculation_identity"]:
        raise ValueError("math_calculation_changed")
    scoring = read(directory / "score.json")
    if (digest({k: v for k, v in scoring.items() if k != "score_identity"}) != scoring["score_identity"]
            or scoring["run_identity"] != calculation["source_run_identity"]):
        raise ValueError("host_reference_score_changed")
    pairs = {p["sample_id"]: p for p in scoring["pairs"]}
    rows = []
    for source in calculation["rows"]:
        pair = pairs[source["sample_id"]]
        reference = pair["reference_answer"]
        index = ord(reference) - 65
        reference_text = source["options"][index]
        aliases = {"in front": "front", "to the left": "left", "to the right": "right"}
        canonical = aliases.get(reference_text.lower(), reference_text.lower())
        row = {
            **source, "reference_answer": reference, "reference_text": reference_text,
            "math_answer_matches_gt": source["answer"] == reference if source["answer"] else None,
            "direct_answer": pair["direct"]["answer"], "pose_model_answer": pair["pose"]["answer"],
            "direct_correct": pair["direct"]["correct"], "pose_model_correct": pair["pose"]["correct"],
        }
        if source["relation"] == "object_bearing_with_unknown_depth":
            row["gt_direction_feasible"] = (
                None if canonical == "cannot be determined" else canonical in source["possible_labels"])
            row["gt_matches_underdetermined_assessment"] = (
                canonical == "cannot be determined" and not source["unique_direction"])
            model_label = source["options"][ord(pair["pose"]["answer"]) - 65].lower()
            row["pose_model_direction_feasible"] = model_label in source["possible_labels"]
        if source.get("conditional_answer"):
            row["conditional_answer_matches_gt"] = source["conditional_answer"] == reference
        rows.append(row)
    cameras = [row for row in rows if row["relation"] != "object_bearing_with_unknown_depth"]
    direct_computable = [row for row in cameras if row["answer"] is not None]
    result = {
        "schema": "mmsi-offline-vggt-math-comparison/1",
        "calculation_identity": calculation["calculation_identity"],
        "source_score_identity": scoring["score_identity"], "rows": rows,
        "direct_computable_camera_questions": len(direct_computable),
        "direct_computable_camera_correct": sum(row["math_answer_matches_gt"] for row in direct_computable),
        "conditional_camera_questions": len(cameras) - len(direct_computable),
        "model_requests": 0, "gpu_forwards": 0, "compared_at": time.time(),
    }
    result["comparison_identity"] = digest(result)
    write(directory / "位姿数学核对.json", result, exclusive=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("selfcheck", "calculate", "compare"))
    parser.add_argument("--directory", default=str(DEFAULT))
    args = parser.parse_args()
    if args.phase == "selfcheck":
        selfcheck()
        print('{"status":"passed","synthetic_only":true,"model_requests":0}', flush=True)
    else:
        result = globals()[args.phase](args.directory)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
