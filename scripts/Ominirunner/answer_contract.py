"""Public labels and requested quantities; never reads benchmark answers."""
from __future__ import annotations

import math
import re

VERSION = "omni-public-answer-contract/4"
LENGTH_UNIT_METERS = {"m": 1.0, "ft": 0.3048, "in": 0.0254,
                      "yd": 0.9144, "mi": 1609.344}
UNIT_ALIASES = {
    "m": "m", "meter": "m", "meters": "m",
    "ft": "ft", "foot": "ft", "feet": "ft",
    "in": "in", "inch": "in", "inches": "in",
    "yd": "yd", "yard": "yd", "yards": "yd",
    "mi": "mi", "mile": "mi", "miles": "mi",
}
UNIT_PATTERN = r"(?:meters?|miles?|inches|inch|yards?|feet|foot|ft|yd|mi|in|m)"
DIMENSION_PATTERN = r"(?:width|length|height|diameter|volume)"
ADJECTIVE_QUANTITIES = {"wide": "width", "long": "length", "tall": "height",
                       "high": "height"}


def requested_length_unit(question):
    """Resolve an explicit output unit in the final question clause."""
    asks = list(re.finditer(r"\b(?:how|what)\b", question))
    clause = question[asks[-1].start():] if asks else question
    pattern = (r"(?:\b(?:in|into)\s+|\bhow\s+(?:many|much)\s+|\(\s*)("
               + UNIT_PATTERN + r")\b")
    matches = list(re.finditer(pattern, clause))
    return UNIT_ALIASES[matches[-1][1]] if matches else None


def converted_span(value, source_unit, answer_unit):
    """Preserve native values when no conversion is required."""
    if source_unit == answer_unit:
        return value
    return value * (LENGTH_UNIT_METERS[source_unit] / LENGTH_UNIT_METERS[answer_unit])


def normalize_object_name(value):
    """Normalize spelling separators only; do not infer object synonyms."""
    return " ".join(value.casefold().replace("_", " ").split())


def scale_quantity(before, after):
    """Bind the span to its adjacent adjective or nearest preceding dimension."""
    adjective = re.match(r"\s*(wide|long|tall|high)\b", after)
    if adjective:
        return ADJECTIVE_QUANTITIES[adjective[1]]
    dimension = re.match(r"\s*(?:in\s+)?(" + DIMENSION_PATTERN + r")\b", after)
    if dimension:
        return dimension[1]
    clause = re.split(r"[.,;:?!\n]", before)[-1]
    dimensions = list(re.finditer(
        r"\b(width|length|height|diameter|volume|walk|walking|moves?|moving)\b", clause))
    if dimensions:
        word = dimensions[-1][1]
        if word in ("walk", "walking"):
            return "walking_distance"
        return "motion" if word in ("move", "moves", "moving") else word
    return "motion"


def request_clause(question):
    """Separate the requested expression from a leading condition and instructions."""
    text = re.split(r"\bOptions\s*:|\bChoose from\b", question, maxsplit=1, flags=re.I)[0]
    asks = list(re.finditer(
        r"(?:^|[,;]\s*)(what|how|which|would|can|is|does|are)\b", text, re.I))
    if asks:
        text = text[asks[-1].start(1):]
    return text.split("?", 1)[0].strip()


def target_name(text, antecedent=None):
    text = re.sub(r"\s+in\s+(?:3d(?:\s+in\s+)?)?" + UNIT_PATTERN + r"\b.*$", "", text)
    text = re.sub(r"\s+in\s+3d\s*$", "", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text.strip(" .,:;?!"))
    text = normalize_object_name(text)
    return antecedent if text in ("it", "its", "they", "them", "their") else text or None


def choice_targets(text):
    """Split enumerated alternatives, preserving qualified object names and order."""
    return [target_name(v) for v in re.split(r"\s*,\s*(?:or\s+)?|\s+or\s+", text)
            if v.strip()]


def implicit_choices(clause):
    def labels(text):
        return [re.sub(r"^(?:the|a|an)\s+", "", v.strip(), flags=re.I)
                for v in re.split(r"\s*,\s*(?:or\s+)?|\s+or\s+", text, flags=re.I)
                if v.strip()]

    if ":" in clause and re.search(r"\bor\b", clause, re.I):
        return labels(clause.split(":", 1)[1])
    more = re.search(r"\bmore\s+.+?\s+(?:at|on|in)\s+(.+?\s+or\s+.+)$", clause, re.I)
    if more:
        return labels(more[1])
    closer = re.search(
        r"\b(?:closer to|farther from|further from)\s+(.+?\s+or\s+.+)$", clause, re.I)
    return labels(closer[1]) if closer else []


def public_labels(item, clause):
    options = item.get("options", [])
    if not isinstance(options, list) or any(not isinstance(v, str) for v in options):
        raise ValueError("public_options_invalid")
    if options:
        labels = list(options)
    else:
        match = re.search(r"Options:\s*\{([^{}]+)\}", item["question"], re.I)
        if not match:
            match = re.search(r"Choose from\s+([^.\n]+)", item["question"], re.I)
        if match:
            labels = [v.strip() for v in match[1].split(",")]
        elif item.get("task_type") == "str":
            labels = implicit_choices(clause)
            if not labels and re.match(r"^(can|would|is|does|are)\b", clause, re.I):
                labels = ["yes", "no"]
        else:
            labels = []
    if (len(labels) > 16 or any(not s or len(s) > 80 for s in labels)
            or len({s.casefold() for s in labels}) != len(labels)):
        raise ValueError("public_options_invalid")
    return labels


def operand(role, quantity, target):
    return {"role": role, "quantity": quantity, "target": target}


def dimension_operand(text, role, inherited=None, antecedent=None):
    match = re.search(r"\b(" + DIMENSION_PATTERN + r")\s+of\s+(.+)$", text)
    if match:
        return operand(role, match[1], target_name(match[2], antecedent))
    match = re.search(r"(?:^|\bof\s+)(.+?)'s\s+(?:3d\s+)?(" + DIMENSION_PATTERN + r")\b", text)
    if match:
        return operand(role, match[2], target_name(match[1], antecedent))
    match = re.search(r"\b(?:its|their)\s+(?:3d\s+)?(" + DIMENSION_PATTERN + r")\b", text)
    if match:
        return operand(role, match[1], antecedent)
    match = re.search(
        r"\bhow\s+(?:(?:many|much)\s+" + UNIT_PATTERN + r"\s+)?"
        r"(wide|long|tall|high)\s+(?:is|are)\s+(.+)$", text)
    if match:
        return operand(role, ADJECTIVE_QUANTITIES[match[1]], target_name(match[2], antecedent))
    if re.search(r"\bhow\s+(?:many|much)\s+" + UNIT_PATTERN + r"\b", text) and re.search(
            r"\b(?:raised|raise|rise|lifted|lift)\b", text):
        return operand(role, "height", antecedent)
    if inherited:
        return operand(role, inherited, target_name(text, antecedent))
    return None


def scale_target(question, match, quantity):
    """Resolve the object from the local scale predicate, not a nearby noun."""
    before = re.split(r"[,;?!\n]|\bif\b", question[:match.start()])[-1].strip()
    after = re.split(r"[,;?!\n]", question[match.end():])[0]
    if quantity == "walking_distance":
        target = re.search(r"\b(?:to get to|to reach|to)\s+(.+)$", after)
        return target_name(target[1]) if target else None
    if quantity == "motion":
        target = re.search(r"(.+?)\s+(?:moves?|moving)\s*$", before)
        return target_name(target[1]) if target else None
    target = re.search(r"\b" + quantity + r"\s+of\s+(.+)\s+(?:is|are|measures)\s*$", before)
    if not target:
        target = re.search(
            r"(.+?)\s+has\s+(?:(?:a|an|the|3d)\s+)*" + quantity + r"\s+of\s*$", before)
    if not target:
        before = re.split(r"\s+and\s+", before)[-1]
        target = re.search(r"(.+?)\s+(?:is|are|measures)\s*$", before)
    return target_name(target[1]) if target else None


def ratio_quantity(operands):
    quantities = {row["quantity"] for row in operands}
    return next(iter(quantities)) + "_ratio" if len(quantities) == 1 else "dimension_ratio"


def ratio_operands(text, role, inherited=None, antecedent=None):
    row = dimension_operand(text, role, inherited, antecedent)
    if row is None:
        return [operand(role, "count", target_name(text))]
    if re.search(r"\bcombined\b", text) and row["target"]:
        return [operand(role, row["quantity"], target_name(v))
                for v in re.split(r"\s+and\s+", row["target"])]
    return [row]


def distance_semantics(clause):
    relation = r"(closer to|farther away from|further away from|farther from|further from)"
    if ":" in clause:
        prefix, suffix = clause.split(":", 1)
        match = re.search(r"\b" + relation + r"\s+(.+)$", prefix)
        if not match:
            return None
        reference, candidates = target_name(match[2]), choice_targets(suffix)
        operator, roles, kind = ("argmin" if match[1] == "closer to" else "argmax"), (
            ["candidate"] * len(candidates)), "common_reference"
    else:
        match = re.search(r"^(?:is|are)\s+(.+?)\s+" + relation + r"\s+(.+)$", clause)
        if not match:
            return None
        subject, direction, tail = target_name(match[1]), match[2], match[3]
        if re.search(r"\bor\b", tail):
            candidates, reference, kind = choice_targets(tail), subject, "fixed_origin"
            operator, roles = ("argmin" if direction == "closer to" else "argmax"), (
                ["candidate"] * len(candidates))
        elif " than " in tail:
            anchor, other = tail.split(" than ", 1)
            reference, candidates, kind = target_name(anchor), [subject, target_name(other)], "common_reference"
            operator, roles = ("<" if direction == "closer to" else ">"), ["left", "right"]
        else:
            return None
    camera = kind == "common_reference" and reference == "camera"
    quantity = "camera_order" if camera else "object_distance_order"
    measured = "camera_z" if camera else "object_distance"
    operands = [operand(role, measured, target) for role, target in zip(roles, candidates)]
    anchor_role = "reference" if kind == "common_reference" else "origin"
    if kind == "fixed_origin":
        operands.insert(0, operand(anchor_role, measured, reference))
    else:
        operands.append(operand(anchor_role, measured, reference))
    return {"quantity": quantity, "operation": "comparison", "operands": operands,
            "reference": reference, "comparison": {"operator": operator},
            "distance_relation": {"kind": kind, anchor_role: reference, "candidates": candidates}}


def task_semantics(clause, labels, antecedent):
    """Parse operand expressions before deriving the compatibility quantity."""
    result = {"operation": "measure", "operands": [], "reference": None,
              "distance_relation": None}
    ratio = re.search(r"\b(?:(width|length|height|diameter|volume)\s+)?ratio\s+of\s+(.+)$", clause)
    if ratio:
        parts = re.split(
            r"\s+to\s+(?=(?:(?:the|a|an|combined|total|3d)\s+)*" + DIMENSION_PATTERN + r"\b)",
            ratio[2], maxsplit=1)
        if len(parts) == 1:
            parts = re.split(r"\s+to\s+", ratio[2], maxsplit=1)
        if len(parts) == 1 and ratio[1]:
            parts = re.split(r"\s+and\s+", ratio[2], maxsplit=1)
        if len(parts) == 2:
            operands = [row for text, role in zip(parts, ("numerator", "denominator"))
                        for row in ratio_operands(text, role, ratio[1], antecedent)]
            return {**result, "quantity": ratio_quantity(operands), "operation": "ratio",
                    "operands": operands}
    multiply = re.search(r"\bmultiply\s+(.+?)\s+(?:such that (?:it )?matches|to match)\s+(.+)$", clause)
    if multiply:
        operands = [dimension_operand(text, role, antecedent=antecedent)
                    for text, role in zip(reversed(multiply.groups()), ("numerator", "denominator"))]
        if all(operands):
            return {**result, "quantity": ratio_quantity(operands), "operation": "ratio",
                    "operands": operands}
    distance = distance_semantics(clause)
    if distance:
        return {**result, **distance}
    more = re.search(r"\bmore\s+(.+?)\s+(at|on|in)\s+(.+?\s+or\s+.+)$", clause)
    if more:
        operands = [operand("candidate", "count", target_name(more[1]) + " " + more[2] + " " + v)
                    for v in choice_targets(more[3])]
        return {**result, "quantity": "count", "operation": "comparison", "operands": operands,
                "comparison": {"operator": "argmax"}}
    comparison = re.search(r"\s+(larger|greater|smaller|less|taller|shorter|wider|longer)\s+than\s+", clause)
    if comparison:
        left = dimension_operand(clause[:comparison.start()], "left", antecedent=antecedent)
        right = dimension_operand(clause[comparison.end():], "right", antecedent=antecedent)
        if left and right:
            multiplier = re.match(r"(one|two|three|four|five|\d+(?:\.\d+)?)\s+stacked\s+(.+)$",
                                  right["target"] or "")
            details = {"operator": "<" if comparison[1] in ("smaller", "less", "shorter") else ">"}
            if multiplier:
                numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
                word = multiplier[1]
                details["factor"] = numbers[word] if word in numbers else float(word)
                right["target"] = multiplier[2]
            return {**result, "quantity": left["quantity"] if left["quantity"] == right["quantity"]
                    else "dimension_comparison", "operation": "comparison",
                    "operands": [left, right], "comparison": details}
    extreme = re.search(r"\b(largest|smallest|tallest|shortest|widest|longest)\b", clause)
    dimension = re.search(r"\b(" + DIMENSION_PATTERN + r")\b", clause)
    if extreme and labels and dimension:
        targets = choice_targets(clause.split(":", 1)[1]) if ":" in clause else labels
        return {**result, "quantity": dimension[1], "operation": "comparison",
                "operands": [operand("candidate", dimension[1], v) for v in targets],
                "comparison": {"operator": "argmin" if extreme[1] in ("smallest", "shortest") else "argmax"}}
    if re.search(r"\bwalk(?:ing)?\b", clause):
        target = re.search(r"\bwalk(?:ing)?\b.*?\b(?:to get to|to reach|to)\s+(.+)$", clause)
        return {**result, "quantity": "walking_distance", "operands": [
            operand("target", "walking_distance", target_name(target[1]) if target else None)]}
    if re.search(r"\bhit\b|\bcollid", clause):
        return {**result, "quantity": "collision"}
    change = re.search(r"\b(?:rise|raise|raised|lift|lifted)\b.*?\bto match\s+(.+)$", clause)
    if change and antecedent:
        target = dimension_operand(change[1], "target", antecedent=antecedent)
        if target:
            return {**result, "quantity": target["quantity"], "operation": "difference",
                    "operands": [target, operand("subtrahend", target["quantity"], antecedent)]}
    requested = dimension_operand(clause, "target", antecedent=antecedent)
    if requested:
        return {**result, "quantity": requested["quantity"], "operands": [requested]}
    if dimension:
        return {**result, "quantity": dimension[1], "operands": [operand("target", dimension[1], None)]}
    if re.search(r"\bdistance\b", clause):
        pair = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", clause)
        operands = [operand(role, "object_distance", target_name(v))
                    for role, v in zip(("origin", "target"), pair.groups())] if pair else []
        return {**result, "quantity": "object_distance", "operands": operands}
    count = re.search(r"\bhow many\s+(.+?)(?:\s+(?:are|is|do|does)\b|$)", clause)
    if count:
        return {**result, "quantity": "count", "operands": [
            operand("target", "count", target_name(count[1]))]}
    if "direction" in clause and labels:
        return {**result, "quantity": "direction"}
    return {**result, "quantity": "visual_judgment"}


def specification(item):
    """Return the public task contract without I/O.

    Operands are ordered {role, quantity, target} records. Measures list target
    then scale_anchor; ratios list numerator then denominator (repeated roles
    are summed); comparisons list left/right or candidates and a reference/origin.
    comparison.factor multiplies the right side. Unknown targets remain None.
    """
    question = item["question"]
    clause = request_clause(question)
    labels = public_labels(item, clause)
    lower = question.lower()
    scales = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(" + UNIT_PATTERN + r")\b", lower):
        before, after = lower[max(0, m.start()-65):m.start()], lower[m.end():m.end()+16]
        role = scale_quantity(before, after)
        # A spaced "in" without a dimension can be the English preposition.
        if m[2] == "in" and role == "motion" and re.search(r"\s+in$", m[0]):
            continue
        scales.append({"value": float(m[1]), "unit": UNIT_ALIASES[m[2]], "quantity": role,
                       "source": before+m[0]+after, "target": scale_target(lower, m, role)})
    antecedent = scales[-1]["target"] if scales else None
    semantics = task_semantics(clause.lower(), labels, antecedent)
    quantity = semantics["quantity"]
    for span in scales:
        span["operand_index"] = len(semantics["operands"])
        semantics["operands"].append(operand("scale_anchor", span["quantity"], span["target"]))
    ratio = quantity.endswith("_ratio")
    unit = "ratio" if ratio else "count" if quantity == "count" and not labels else "label"
    if not ratio and quantity in (
            "height", "width", "length", "diameter", "walking_distance", "object_distance", "volume") and not labels:
        unit = requested_length_unit(lower) or (
            "m" if any(span["unit"] == "m" for span in scales)
            else scales[0]["unit"] if scales else "label")
    if unit in LENGTH_UNIT_METERS:
        for span in scales:
            if span["unit"] != unit:
                span.update(answer_unit=unit, answer_unit_value=converted_span(
                    span["value"], span["unit"], unit))
    return {"version": VERSION, **semantics, "unit": unit,
            "allowed_answers": [{"option_id": f"O{n}", "label": v}
                                for n, v in enumerate(labels, 1)],
            "public_metric_values": scales,
            "geometry_verification": "same_model_surface_declaration"}


def canonical(value, task):
    options = task["allowed_answers"]
    if options:
        if not isinstance(value, str):
            raise ValueError("answer_option_required")
        for option in options:
            if value == option["option_id"] or value.casefold() == option["label"].casefold():
                return option["label"]
        raise ValueError("answer_option_invalid")
    if type(value) not in (str, int, float) or (
            type(value) is float and not math.isfinite(value)):
        raise ValueError("native_finite_scalar_required")
    return value
