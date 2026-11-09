"""Question-bound categories, lossless candidate grouping, current-frame review."""
import base64
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

from .sources import OperationalError, canonical, digest
from .transport import child_environment


class PlannerError(ValueError):
    """A model-authored tool/target error ends this episode without a retry."""


def normalized(text):
    return " ".join(text.casefold().split())


def validate_targets(value, instruction):
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        raise PlannerError("targets_require_one_to_four_entries")
    fields = {"id", "role", "category", "source_phrase", "quantity", "attributes", "relation", "aliases"}
    ids = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != fields:
            raise PlannerError("invalid_target_fields")
        if (not isinstance(row["id"], str) or not re.fullmatch(r"T[1-4]", row["id"])
                or row["id"] in ids or row["role"] not in ("object", "destination", "support")):
            raise PlannerError("invalid_target_identity")
        ids.add(row["id"])
        if not isinstance(row["aliases"], list) or len(row["aliases"]) > 3:
            raise PlannerError("invalid_target_aliases")
        for category in [row["category"]] + row["aliases"]:
            if (not isinstance(category, str) or not re.fullmatch(r"[A-Za-z][A-Za-z -]{0,59}", category)
                    or not 1 <= len(category.split()) <= 6):
                raise PlannerError("invalid_english_category")
        if (not isinstance(row["source_phrase"], str) or not row["source_phrase"].strip()
                or normalized(row["source_phrase"]) not in normalized(instruction)):
            raise PlannerError("target_not_grounded_in_public_instruction")
        if (row["quantity"] is not None and
                (type(row["quantity"]) is not int or not 1 <= row["quantity"] <= 100)):
            raise PlannerError("invalid_target_quantity")
        if (not isinstance(row["attributes"], list) or not isinstance(row["aliases"], list)
                or len(row["attributes"]) > 8 or len(row["aliases"]) > 3
                or any(not isinstance(v, str) or not v.strip() or len(v) > 80 for v in row["attributes"])
                or not isinstance(row["relation"], str) or len(row["relation"]) > 200):
            raise PlannerError("invalid_target_qualifiers")
    return json.loads(canonical(value))


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0


class Detector:
    def __init__(self, config):
        self.config = config
        self.script = Path(__file__).with_name("detector_worker.py").read_text()

    def request(self, value):
        value = dict(value, model_sha256=self.config["detection"]["model_sha256"])
        args = ["/usr/bin/python3", "-B", "-"]
        if self.config["runtime"]["transport"] == "ssh":
            args = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10",
                    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                    self.config["runtime"]["ssh_target"], shlex.join(args)]
        code = ("INPUT=" + repr(value) + "\n" + self.script).encode()
        try:
            result = subprocess.run(args, input=code, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    env=child_environment(), timeout=240, check=False)
            report = json.loads(result.stdout)
            if result.returncode or report.get("status") != "ok":
                raise OperationalError(report.get("code", "detector_process_error"))
            if (report.get("model_sha256") != value["model_sha256"]
                    or [r["physical_gpu"] for r in report["physical_bindings"]] != [1]):
                raise OperationalError("detector_identity_mismatch")
            if value["mode"] != "status" and (
                    report.get("request_sha256") != digest(canonical(value["payload"]))
                    or report.get("detector_requests") != 1):
                raise OperationalError("detector_request_identity_mismatch")
            return report
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            raise OperationalError("detector_transport_or_protocol_error") from None

    def status(self):
        return self.request({"mode": "status"})

    def detect(self, image, category, threshold):
        return self.request({"mode": "detect", "payload": {
            "image": image["base64"], "text_prompt": category + ".", "box_threshold": threshold}})


COLORS = ("#DC2626", "#2563EB", "#059669", "#9333EA", "#D97706", "#0891B2",
          "#DB2777", "#4F46E5", "#65A30D", "#7C3AED", "#B45309", "#0F766E")


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def page_image(raw_png, groups, page, page_size=12):
    """All filled label rectangles lie in a separate sidebar, outside source pixels."""
    from PIL import Image, ImageDraw
    rgb = Image.open(io.BytesIO(raw_png)).convert("RGB")
    canvas = Image.new("RGB", (810, 500), "#F8FAFC")
    canvas.paste(rgb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    count = max(1, math.ceil(len(groups) / page_size))
    draw.text((512, 10), f"Candidates page {page + 1}/{count}", fill="#0F172A")
    placements = []
    for index, group in enumerate(groups[page * page_size:(page + 1) * page_size]):
        color, y = COLORS[index], 42 + index * 37
        box = group["bbox"]
        draw.rectangle(tuple(box), outline=color, width=2)
        rect = [510, y, 804, y + 31]
        draw.line([(box[2], (box[1] + box[3]) / 2), (505, y + 15), (510, y + 15)],
                  fill=color, width=1)
        draw.rectangle(rect, fill="white", outline=color, width=2)
        target_ids = ",".join(sorted({m["target_id"] for m in group["members"]}))
        draw.text((518, y + 5), group["id"] + "  " + target_ids, fill=color)
        draw.text((598, y + 17), f'{len(group["members"])} proposal(s)', fill="#475569")
        placements.append({"group_id": group["id"], "label_rect": rect,
                           "source_image_rect": [0, 0, 500, 500]})
    return png_bytes(canvas), placements


def crop_bounds(box, size=(500, 500)):
    width, height = box[2] - box[0], box[3] - box[1]
    margin = max(8, math.ceil(max(width, height) * 0.12))
    return [max(0, math.floor(box[0]) - margin), max(0, math.floor(box[1]) - margin),
            min(size[0], math.ceil(box[2]) + margin), min(size[1], math.ceil(box[3]) + margin)]


class Perception:
    def __init__(self, config, directory, journal, detector=None):
        self.config, self.directory, self.journal = config, Path(directory), journal
        self.detector = detector or Detector(config)
        self.targets, self.current, self.next_id = None, None, 1
        self.tool_calls, self.detector_calls, self.crop_count = 0, 0, 0
        self.groups, self.cache, self.inspected = [], {}, set()

    def bind(self, public, serial):
        identity = (serial, public["step"], public["raw_rgb"]["sha256"])
        if identity != self.current:
            expired = [g["id"] for g in self.groups]
            self.current, self.public, self.serial = identity, public, serial
            self.groups, self.cache, self.inspected = [], {}, set()
            self.turns, self.detected = [], False
            self.completed_review = None
            self.journal.append({"type": "perception_bind", "observation_serial": serial,
                                 "environment_step": public["step"], "raw_rgb_sha256": identity[2],
                                 "expired_group_ids": expired})

    def set_targets(self, value):
        validated = validate_targets(value, self.public["instruction"])
        if self.targets is not None and validated != self.targets:
            raise PlannerError("frozen_targets_changed")
        if self.targets is None:
            self.targets = validated
            self.journal.append({"type": "targets_extracted", "targets": validated,
                                 "source": "same_planning_model_public_instruction",
                                 "reference_answer_visible": False})

    def group(self, members):
        threshold = self.config["detection"]["group_iou"]
        for member in members:
            matched = next((g for g in self.groups if all(
                iou(member["bbox"], old["bbox"]) >= threshold for old in g["members"])), None)
            if matched is None:
                matched = {"id": f"G{self.next_id:03d}", "bbox": list(member["bbox"]), "members": []}
                self.next_id += 1
                self.groups.append(matched)
            matched["members"].append(member)
            boxes = [r["bbox"] for r in matched["members"]]
            matched["bbox"] = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                               max(b[2] for b in boxes), max(b[3] for b in boxes)]
            # A newly added proposal changes what a prior crop review covers.
            self.inspected.discard(matched["id"])
            self.completed_review = None

    def exported(self):
        rows = []
        for group in self.groups:
            row = dict(group)
            row["possible_anonymous_objects"] = [
                anchor for anchor in self.public["anonymous_anchors"]
                if group["bbox"][0] <= anchor["pixel"][0] <= group["bbox"][2]
                and group["bbox"][1] <= anchor["pixel"][1] <= group["bbox"][3]]
            rows.append(row)
        return rows

    def image(self, png, name, caption):
        path = self.directory / name
        path.write_bytes(png)
        return {"file": name, "sha256": digest(png), "caption": caption,
                "base64": base64.b64encode(png).decode()}

    def view(self, page):
        page_size = self.config["detection"]["page_size"]
        pages = max(1, math.ceil(len(self.groups) / page_size))
        if type(page) is not int or not 0 <= page < pages:
            raise PlannerError("invalid_candidate_page")
        png, placement = page_image(base64.b64decode(self.public["raw_rgb"]["base64"]),
                                    self.groups, page, page_size)
        name = f"perception-{self.serial:03d}-tool-{self.tool_calls:03d}-page-{page + 1}.png"
        return self.image(png, name, "Current raw RGB candidate overlay; labels in separate sidebar."), placement

    def tool(self, request):
        if self.targets is None or not isinstance(request, dict):
            raise PlannerError("targets_required_before_tools")
        name = request.get("name")
        images, extra = [], {}
        self.tool_calls += 1
        if name != "detect_targets" and not self.detected:
            raise PlannerError("detection_required_before_candidate_tools")
        if name == "detect_targets":
            if not {"name", "target_ids"} <= set(request) or not set(request) <= {"name", "target_ids", "query_aliases"}:
                raise PlannerError("invalid_detect_request_fields")
            ids = request["target_ids"]
            if not isinstance(ids, list) or not 1 <= len(ids) <= 4 or len(ids) != len(set(ids)):
                raise PlannerError("invalid_detect_targets")
            aliases = request.get("query_aliases", {})
            if not isinstance(aliases, dict) or not set(aliases) <= set(ids):
                raise PlannerError("invalid_alias_selection")
            catalog = {t["id"]: t for t in self.targets}
            queries = []
            for tid in ids:
                if tid not in catalog:
                    raise PlannerError("unknown_target_id")
                target = catalog[tid]
                query = aliases.get(tid, target["category"])
                if query not in [target["category"]] + target["aliases"]:
                    raise PlannerError("query_not_in_frozen_targets")
                queries.append((tid, query))
            # Validate the entire request before any network action.
            for tid, query in queries:
                if (tid, query) in self.cache:
                    continue
                self.journal.append({"type": "detector_request_sent", "target_id": tid, "query": query,
                                     "observation_serial": self.serial, "environment_step": self.public["step"],
                                     "raw_rgb_sha256": self.public["raw_rgb"]["sha256"], "automatic_retry": False})
                report = self.detector.detect(self.public["raw_rgb"], query,
                                              self.config["detection"]["box_threshold"])
                self.detector_calls += 1
                self.cache[(tid, query)] = report
                members = [dict(r, target_id=tid, query=query) for r in report["detections"]]
                self.group(members)
                self.journal.append({"type": "detector_result", "target_id": tid, "query": query,
                                     "observation_serial": self.serial, "report": report})
            self.detected = True
            image, placement = self.view(0)
            images.append(image)
            extra.update(label_placements=placement, page=0)
        elif name == "view_candidates":
            if set(request) != {"name", "page"}:
                raise PlannerError("invalid_view_request")
            image, placement = self.view(request["page"])
            images.append(image)
            extra.update(label_placements=placement, page=request["page"])
        elif name == "inspect_candidates":
            if set(request) != {"name", "group_ids"}:
                raise PlannerError("invalid_inspect_request")
            ids = request["group_ids"]
            catalog = {g["id"]: g for g in self.groups}
            if (not isinstance(ids, list) or not 1 <= len(ids) <= self.config["detection"]["max_crop_groups"]
                    or len(ids) != len(set(ids)) or any(gid not in catalog for gid in ids)):
                raise PlannerError("unknown_expired_or_excess_crop_groups")
            from PIL import Image
            raw = Image.open(io.BytesIO(base64.b64decode(self.public["raw_rgb"]["base64"]))).convert("RGB")
            crops = []
            for gid in ids:
                bounds = crop_bounds(catalog[gid]["bbox"])
                png = png_bytes(raw.crop(bounds))
                name = f"perception-{self.serial:03d}-tool-{self.tool_calls:03d}-{gid}-crop.png"
                images.append(self.image(png, name, f"{gid} unannotated RGB crop; original bounds {bounds}."))
                crops.append({"group_id": gid, "bounds": bounds, "sha256": digest(png)})
                self.inspected.add(gid)
                self.crop_count += 1
            extra["crops"] = crops
            self.completed_review = None
        elif name == "review_candidates":
            if set(request) != {"name", "perception_review", "selected_targets"}:
                raise PlannerError("invalid_review_request")
            self.approve_action(request)
            self.completed_review = {
                "perception_review": request["perception_review"],
                "selected_targets": request["selected_targets"]}
            extra["accepted_visual_review"] = self.completed_review
        else:
            raise PlannerError("unknown_perception_tool")
        result = {"tool": name, "observation_serial": self.serial,
                  "environment_step": self.public["step"], "raw_rgb_sha256": self.public["raw_rgb"]["sha256"],
                  "targets": self.targets, "groups": self.exported(),
                  "group_count": len(self.groups),
                  "proposal_count": sum(len(g["members"]) for g in self.groups),
                  "pages": max(1, math.ceil(len(self.groups) / self.config["detection"]["page_size"])),
                  "inspected_group_ids": sorted(self.inspected),
                  "note": "Groups are overlapping proposals, not confirmed object counts. Scores and projected "
                          "anonymous anchors do not validate semantics. All proposals are retained.",
                  **extra}
        self.journal.append({"type": "perception_tool_result", "tool_request": request,
                             "result": result, "images": [{k: v for k, v in i.items() if k != "base64"}
                                                          for i in images],
                             "simulator_steps": 0, "simulator_observations": 0})
        return result, images

    def approve_action(self, value):
        if not self.detected:
            raise PlannerError("detection_required_before_action")
        if self.groups and not self.inspected:
            raise PlannerError("crop_review_required_before_action")
        stored = self.completed_review or {}
        reviews = value.get("perception_review", stored.get("perception_review", []))
        selections = value.get("selected_targets", stored.get("selected_targets", []))
        if not isinstance(reviews, list) or not isinstance(selections, list):
            raise PlannerError("invalid_perception_review")
        reviewed = set()
        for row in reviews:
            if (not isinstance(row, dict) or set(row) != {"group_id", "verdict", "reason"}
                    or row["group_id"] not in self.inspected or row["group_id"] in reviewed
                    or row["verdict"] not in ("match", "not_match", "uncertain")
                    or not isinstance(row["reason"], str) or not row["reason"].strip()):
                raise PlannerError("invalid_or_expired_group_review")
            reviewed.add(row["group_id"])
        if self.groups and not reviews:
            raise PlannerError("explicit_crop_verdict_required")
        verdicts = {r["group_id"]: r["verdict"] for r in reviews}
        target_ids = {t["id"] for t in self.targets}
        catalog = {g["id"]: g for g in self.groups}
        for row in selections:
            if (not isinstance(row, dict) or set(row) != {"target_id", "group_id"}
                    or row["target_id"] not in target_ids or row["group_id"] not in reviewed
                    or verdicts[row["group_id"]] != "match"
                    or not any(m["target_id"] == row["target_id"] for m in catalog[row["group_id"]]["members"])):
                raise PlannerError("selection_requires_current_inspected_match")
        self.journal.append({"type": "perception_action_review", "observation_serial": self.serial,
                             "perception_review": reviews, "selected_targets": selections,
                             "source": "model_visual_judgment_not_ground_truth"})
