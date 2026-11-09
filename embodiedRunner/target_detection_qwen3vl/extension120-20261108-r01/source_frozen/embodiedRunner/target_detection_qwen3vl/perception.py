"""Keep R05 target/group/review semantics, using real image dimensions throughout."""
import base64
import io
import math

from target_detection_r05.perception import (
    COLORS, Perception as ParentPerception, PlannerError, png_bytes,
)
from .detector import Detector
from .sources import digest


def crop_bounds(box, size):
    """Round outwards, add the original context margin, and clip to original RGB."""
    width, height = box[2] - box[0], box[3] - box[1]
    margin = max(8, math.ceil(max(width, height) * 0.12))
    return [max(0, math.floor(box[0]) - margin), max(0, math.floor(box[1]) - margin),
            min(size[0], math.ceil(box[2]) + margin), min(size[1], math.ceil(box[3]) + margin)]


def page_image(raw_png, groups, page, page_size=12):
    from PIL import Image, ImageDraw
    rgb = Image.open(io.BytesIO(raw_png)).convert("RGB")
    width, height = rgb.size
    canvas = Image.new("RGB", (width + 310, max(height, 500)), "#F8FAFC")
    canvas.paste(rgb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    count = max(1, math.ceil(len(groups) / page_size))
    draw.text((width + 12, 10), f"Candidates page {page + 1}/{count}", fill="#0F172A")
    placements = []
    for index, group in enumerate(groups[page * page_size:(page + 1) * page_size]):
        color, y = COLORS[index], 42 + index * 37
        box = group["bbox"]
        # Logical pixel edges include width/height; drawing addresses pixels up to W-1/H-1.
        display = [min(max(v, 0), limit - 1)
                   for v, limit in zip(box, (width, height, width, height))]
        draw.rectangle(tuple(display), outline=color, width=2)
        rect = [width + 10, y, width + 304, y + 31]
        draw.line([(display[2], (display[1] + display[3]) / 2),
                   (width + 5, y + 15), (width + 10, y + 15)], fill=color, width=1)
        draw.rectangle(rect, fill="white", outline=color, width=2)
        target_ids = ",".join(sorted({m["target_id"] for m in group["members"]}))
        draw.text((width + 18, y + 5), group["id"] + "  " + target_ids, fill=color)
        draw.text((width + 98, y + 17), f'{len(group["members"])} proposal(s)', fill="#475569")
        placements.append({"group_id": group["id"], "label_rect": rect,
                           "source_image_rect": [0, 0, width, height]})
    return png_bytes(canvas), placements


class Perception(ParentPerception):
    def __init__(self, config, directory, journal, detector=None):
        super().__init__(config, directory, journal, detector or Detector(config))

    def view(self, page):
        size = self.config["detection"]["page_size"]
        if type(page) is not int or not 0 <= page < max(1, math.ceil(len(self.groups) / size)):
            raise PlannerError("invalid_candidate_page")
        png, placement = page_image(base64.b64decode(self.public["raw_rgb"]["base64"]),
                                    self.groups, page, size)
        name = f"perception-{self.serial:03d}-tool-{self.tool_calls:03d}-page-{page + 1}.png"
        return self.image(png, name, "Current raw RGB candidate overlay; labels in separate sidebar."), placement

    def tool(self, request):
        # View/review branches are independent of the detector and can be reused verbatim.
        if isinstance(request, dict) and request.get("name") in ("view_candidates", "review_candidates"):
            return super().tool(request)
        if self.targets is None or not isinstance(request, dict):
            raise PlannerError("targets_required_before_tools")
        name, images, extra = request.get("name"), [], {}
        self.tool_calls += 1
        if name != "detect_targets" and not self.detected:
            raise PlannerError("detection_required_before_candidate_tools")
        if name == "detect_targets":
            if (not {"name", "target_ids"} <= set(request)
                    or not set(request) <= {"name", "target_ids", "query_aliases"}):
                raise PlannerError("invalid_detect_request_fields")
            ids = request["target_ids"]
            if not isinstance(ids, list) or not 1 <= len(ids) <= 4 or len(ids) != len(set(ids)):
                raise PlannerError("invalid_detect_targets")
            aliases = request.get("query_aliases", {})
            if not isinstance(aliases, dict) or not set(aliases) <= set(ids):
                raise PlannerError("invalid_alias_selection")
            catalog, queries = {t["id"]: t for t in self.targets}, []
            for tid in ids:
                if tid not in catalog:
                    raise PlannerError("unknown_target_id")
                target = catalog[tid]
                query = aliases.get(tid, target["category"])
                if query not in [target["category"]] + target["aliases"]:
                    raise PlannerError("query_not_in_frozen_targets")
                queries.append((tid, query))
            for tid, query in queries:
                if (tid, query) in self.cache:
                    continue
                self.journal.append({
                    "type": "detector_request_sent", "backend": "qwen3vl8b", "target_id": tid,
                    "query": query, "observation_serial": self.serial,
                    "environment_step": self.public["step"],
                    "raw_rgb_sha256": self.public["raw_rgb"]["sha256"], "automatic_retry": False})
                report = self.detector.detect(self.public["raw_rgb"], query)
                self.detector_calls += 1
                self.cache[(tid, query)] = report
                self.group([dict(r, target_id=tid, query=query) for r in report["detections"]])
                self.journal.append({"type": "detector_result", "target_id": tid, "query": query,
                                     "observation_serial": self.serial, "report": report})
            self.detected = True
            image, placement = self.view(0)
            images.append(image)
            extra.update(label_placements=placement, page=0)
        elif name == "inspect_candidates":
            if set(request) != {"name", "group_ids"}:
                raise PlannerError("invalid_inspect_request")
            ids, catalog = request["group_ids"], {g["id"]: g for g in self.groups}
            if (not isinstance(ids, list) or not 1 <= len(ids) <= self.config["detection"]["max_crop_groups"]
                    or len(ids) != len(set(ids)) or any(gid not in catalog for gid in ids)):
                raise PlannerError("unknown_expired_or_excess_crop_groups")
            from PIL import Image
            raw = Image.open(io.BytesIO(base64.b64decode(self.public["raw_rgb"]["base64"]))).convert("RGB")
            crops = []
            for gid in ids:
                bounds = crop_bounds(catalog[gid]["bbox"], raw.size)
                png = png_bytes(raw.crop(bounds))
                filename = f"perception-{self.serial:03d}-tool-{self.tool_calls:03d}-{gid}-crop.png"
                images.append(self.image(png, filename, f"{gid} unannotated RGB crop; original bounds {bounds}."))
                crops.append({"group_id": gid, "bounds": bounds, "sha256": digest(png)})
                self.inspected.add(gid)
                self.crop_count += 1
            extra["crops"] = crops
            self.completed_review = None
        else:
            raise PlannerError("unknown_perception_tool")
        result = {
            "tool": name, "observation_serial": self.serial, "environment_step": self.public["step"],
            "raw_rgb_sha256": self.public["raw_rgb"]["sha256"], "targets": self.targets,
            "groups": self.exported(), "group_count": len(self.groups),
            "proposal_count": sum(len(g["members"]) for g in self.groups),
            "pages": max(1, math.ceil(len(self.groups) / self.config["detection"]["page_size"])),
            "inspected_group_ids": sorted(self.inspected),
            "note": "Boxes are Qwen3-VL proposals in original-image pixels. No calibrated confidence "
                    "is available. Groups and anonymous anchors do not validate semantics. "
                    "All proposals are retained; inspect original RGB crops.", **extra,
        }
        self.journal.append({
            "type": "perception_tool_result", "tool_request": request, "result": result,
            "images": [{k: v for k, v in image.items() if k != "base64"} for image in images],
            "simulator_steps": 0, "simulator_observations": 0})
        return result, images
