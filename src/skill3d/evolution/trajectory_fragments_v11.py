"""Typed multi-field fragments with independent, complete reconstruction."""
from copy import deepcopy
import hashlib
import json

SCHEMA = "revision-multiple-field-fragments/1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def at(value, path):
    for key in path:
        value = value[key]
    return value


def put(value, path, replacement):
    at(value, path[:-1])[path[-1]] = replacement


def missing(path):
    return {"fragmented_value": {"path": path, "status": "carried_in_other_field_fragments",
                                "missing_is_not_empty_or_negative_evidence": True}}


def slice_value(value, start, end):
    if isinstance(value, dict):
        return {key: value[key] for key in list(value)[start:end]}
    return value[start:end]


def split_fields(row, fits):
    """Retain the full program and split enough other fields to fit together."""
    candidates = []

    def visit(value, path):
        if any(key in ("program_source", "index", "recorded_read_ref") for key in path):
            return
        size = len(canonical(value))
        structural = (path == ["complete_case"] or
                      path == ["complete_case", "rounds", 0] or path[-1] == "rounds")
        protected_page_list = path[-1] == "delivered_observation_pages"
        atomic_children_fit = (not isinstance(value, (list, dict)) or all(
            len(canonical(child)) <= 8192 for child in
            (value.values() if isinstance(value, dict) else value)))
        if (not structural and not protected_page_list and size >= 2048
                and isinstance(value, (list, dict, str)) and len(value) > 1
                and atomic_children_fit):
            is_header = path[:2] != ["complete_case", "rounds"]
            candidates.append((is_header, -size, path, value))
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, [*path, key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*path, index])

    visit(row["complete_case"], ["complete_case"])
    template, selected = deepcopy(row), []

    def piece(field_index, start, end):
        path, value = selected[field_index]
        result = deepcopy(template)
        result["round_fragment"] = {
            "schema": SCHEMA, "source_unit_sha256": sha(row),
            "fields": [{"path": p, "kind": type(v).__name__, "total": len(v)}
                       for p, v in selected],
            "field_index": field_index, "start": start, "end": end,
            "value": slice_value(value, start, end),
            "fragment_index": 99999, "fragment_count": 99999,
        }
        return result

    for _, _, path, value in sorted(candidates, key=lambda item: item[:2]):
        if any(path[:len(old)] == old for old, _ in selected):
            continue
        selected.append((path, value))
        put(template, path, missing(path))
        # Empty placeholders fitting is insufficient: an indivisible item must
        # also fit with the shared program, context and full fragment manifest.
        if all(fits([piece(i, offset, offset + 1)])
               for i, (_, field) in enumerate(selected)
               for offset in ([max(range(len(field)), key=lambda j:
                                      len(canonical(slice_value(field, j, j + 1))))]
                              if not isinstance(field, str) else [0])):
            break
    else:
        raise ValueError("complete program/context or atomic field exceeds multi-field input budget")

    pieces = []
    for i, (_, value) in enumerate(selected):
        start = 0
        while start < len(value):
            low, high = 1, len(value) - start
            while low < high:
                count = (low + high + 1) // 2
                if fits([piece(i, start, start + count)]):
                    low = count
                else:
                    high = count - 1
            current = piece(i, start, start + low)
            if not fits([current]):
                raise ValueError("atomic fragment does not fit the complete request")
            pieces.append(current)
            start += low
    for index, current in enumerate(pieces):
        current["round_fragment"].update(fragment_index=index, fragment_count=len(pieces))
    if restore_fields(pieces) != row:
        raise ValueError("multi-field fragmentation changed the source unit")
    return pieces


def restore_fields(rows):
    if not rows:
        raise ValueError("missing field fragments")
    first = rows[0]["round_fragment"]
    if first.get("schema") != SCHEMA or first["fragment_count"] != len(rows):
        raise ValueError("field fragment schema or denominator changed")
    fields = first["fields"]
    if not isinstance(fields, list) or not fields:
        raise ValueError("field fragment manifest is empty")
    template = deepcopy(rows[0])
    template.pop("round_fragment")
    paths, outputs, counts = [], [], []
    for field in fields:
        path, kind, total = field["path"], field["kind"], field["total"]
        if (not isinstance(path, list) or not path or kind not in ("list", "dict", "str")
                or type(total) is not int or total <= 0
                or any(path[:len(old)] == old or old[:len(path)] == path for old in paths)
                or at(template, path) != missing(path)):
            raise ValueError("field fragment manifest or placeholder changed")
        paths.append(path)
        outputs.append({"list": [], "dict": {}, "str": ""}[kind])
        counts.append(0)
    active_field = 0
    for index, row in enumerate(rows):
        fragment = row.get("round_fragment", {})
        clean = deepcopy(row)
        clean.pop("round_fragment", None)
        field_index = fragment.get("field_index")
        if (clean != template or fragment.get("schema") != SCHEMA
                or fragment.get("fields") != fields
                or fragment.get("source_unit_sha256") != first["source_unit_sha256"]
                or fragment.get("fragment_count") != len(rows)
                or fragment.get("fragment_index") != index
                or type(field_index) is not int or not 0 <= field_index < len(fields)
                or field_index != active_field):
            raise ValueError("field fragment context, order or identity changed")
        field, value = fields[field_index], fragment.get("value")
        start, end = fragment.get("start"), fragment.get("end")
        if (type(start) is not int or type(end) is not int
                or start != counts[field_index] or end <= start or end > field["total"]
                or type(value).__name__ != field["kind"] or len(value) != end - start):
            raise ValueError("field fragment value or coverage changed")
        if field["kind"] == "dict":
            if set(outputs[field_index]) & set(value):
                raise ValueError("field fragment dictionary key repeated")
            outputs[field_index].update(value)
        elif field["kind"] == "list":
            outputs[field_index].extend(value)
        else:
            outputs[field_index] += value
        counts[field_index] = end
        if end == field["total"]:
            active_field += 1
    if active_field != len(fields):
        raise ValueError("field fragment sequence is incomplete")
    result = deepcopy(template)
    for path, value in zip(paths, outputs):
        put(result, path, value)
    if sha(result) != first["source_unit_sha256"]:
        raise ValueError("reassembled field values changed")
    return result


def reassemble_fields(rows, native_reassemble):
    """Handle new fragments and retain native array-fragment verification."""
    output, native_rows = [], []
    index = 0
    while index < len(rows):
        fragment = rows[index].get("round_fragment", {})
        if fragment.get("schema") != SCHEMA:
            native_rows.append(rows[index])
            index += 1
            continue
        if native_rows:
            output.extend(native_reassemble(native_rows))
            native_rows = []
        count = fragment.get("fragment_count")
        if type(count) is not int or count <= 0 or index + count > len(rows):
            raise ValueError("field fragment sequence is incomplete")
        output.append(restore_fields(rows[index:index + count]))
        index += count
    if native_rows:
        output.extend(native_reassemble(native_rows))
    return output
