"""Lossless, JSON-only tables for pending and historical tool observations."""
from __future__ import annotations

from copy import deepcopy
import json

from .common import digest, encoded


SCHEMA = "mindcube-observation-context/1"
ALLOWED_TOOLS = frozenset({
    "view_candidates", "view_depth_summary", "point_depth", "pair_point_distance",
})
MEASUREMENT_TOOLS = frozenset({"point_depth", "pair_point_distance"})
_RECORD_KEYS = frozenset({"result_id", "source_tool", "status", "data"})
_GROUP_KEYS = frozenset({"source_tool", "status", "shared_data", "data_keys", "rows"})


def _plain_json(value):
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("json_object_keys_must_be_strings")
        for child in value.values():
            _plain_json(child)
    elif type(value) is list:
        for child in value:
            _plain_json(child)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("plain_json_values_required")


def _snapshot(value):
    # The shared encoder rejects nonfinite numbers and circular containers.
    try:
        payload = encoded(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("invalid_json_payload") from exc
    _plain_json(value)
    return json.loads(payload)


def _nonempty_string(value):
    return type(value) is str and bool(value.strip())


def _validate_header(source_tool, status):
    if type(source_tool) is not str or source_tool not in ALLOWED_TOOLS:
        raise ValueError("unknown_source_tool")
    if not _nonempty_string(status):
        raise ValueError("invalid_observation_status")


def _validate_records(records):
    by_id = {}
    for record in records:
        if type(record) is not dict or record.keys() != _RECORD_KEYS:
            raise ValueError("observation_requires_exact_root_keys")
        result_id = record["result_id"]
        if not _nonempty_string(result_id):
            raise ValueError("invalid_result_id")
        if result_id in by_id:
            raise ValueError("duplicate_result_id")
        _validate_header(record["source_tool"], record["status"])
        if type(record["data"]) is not dict:
            raise ValueError("observation_data_must_be_dict")
        by_id[result_id] = record
    return by_id


def _pending_ids(values, by_id, *, kind, allowed_tools):
    if isinstance(values, (str, bytes, dict)):
        raise ValueError(f"invalid_{kind}_ids")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"invalid_{kind}_ids") from exc
    selected = set()
    for result_id in values:
        if not _nonempty_string(result_id):
            raise ValueError(f"invalid_{kind}_id")
        if result_id not in by_id:
            raise ValueError(f"unknown_{kind}_id: {result_id}")
        if by_id[result_id]["source_tool"] not in allowed_tools:
            raise ValueError(f"invalid_{kind}_observation_kind")
        selected.add(result_id)
    return selected


def _tables(indexed_records):
    buckets = {}
    for sequence_index, record in indexed_records:
        data = record["data"]
        view = encoded(data["view_index"]) if "view_index" in data else None
        key = (record["source_tool"], record["status"], view, tuple(sorted(data)))
        buckets.setdefault(key, []).append((sequence_index, record))

    groups = []
    for records in buckets.values():
        first = records[0][1]
        shared_data, data_keys = {}, []
        for key in sorted(first["data"]):
            canonical = encoded(first["data"][key])
            if all(encoded(record["data"][key]) == canonical for _, record in records[1:]):
                shared_data[key] = first["data"][key]
            else:
                data_keys.append(key)
        groups.append({
            "source_tool": first["source_tool"],
            "status": first["status"],
            "shared_data": shared_data,
            "data_keys": data_keys,
            "rows": [[index, record["result_id"], *(
                record["data"][key] for key in data_keys)] for index, record in records],
        })
    return groups


def build_observation_context(observations, *, review_ids=(), hint_ids=()):
    """Return ``(packet, receipt)`` without changing or dropping observations.

    Review IDs must reference measurements; hint IDs must reference candidate
    or depth-summary results. ID selections are sets, ordered by the original
    observations in the receipt. Tables are ordered by first occurrence.

    Each group's ``data_keys`` names the varying data columns. A row contains
    ``[sequence_index, result_id, *data_values]``; other data fields are stored
    in ``shared_data``. The zero-based sequence index spans both partitions.
    """
    try:
        records = list(observations)
    except TypeError as exc:
        raise ValueError("observations_must_be_iterable") from exc
    records = _snapshot(records)
    by_id = _validate_records(records)
    review = _pending_ids(
        review_ids, by_id, kind="review", allowed_tools=MEASUREMENT_TOOLS)
    hints = _pending_ids(
        hint_ids, by_id, kind="hint", allowed_tools=ALLOWED_TOOLS - MEASUREMENT_TOOLS)
    pending_ids = review | hints
    pending, history = [], []
    for index, record in enumerate(records):
        target = pending if record["result_id"] in pending_ids else history
        target.append((index, record))
    packet = {"schema": SCHEMA, "pending": _tables(pending), "history": _tables(history)}
    receipt = {
        "record_count": len(records),
        "pending_count": len(pending),
        "history_count": len(history),
        "raw_sha256": digest(records),
        "compact_sha256": digest(packet),
        "review_ids": [record["result_id"] for record in records if record["result_id"] in review],
        "hint_ids": [record["result_id"] for record in records if record["result_id"] in hints],
        "pending_ids": [record["result_id"] for _, record in pending],
        "history_ids": [record["result_id"] for _, record in history],
    }
    return packet, receipt


def restore_observations(packet):
    """Restore independent record dictionaries in their original global order.

    Reject malformed tables, duplicate IDs, and missing or repeated sequence
    indices so that a successful restoration is meaningful for digest checks.
    """
    packet = _snapshot(packet)
    if (type(packet) is not dict
            or packet.keys() != {"schema", "pending", "history"}
            or packet["schema"] != SCHEMA):
        raise ValueError("invalid_observation_context_schema")
    indexed = {}
    for partition in ("pending", "history"):
        if type(packet[partition]) is not list:
            raise ValueError("observation_groups_must_be_lists")
        for group in packet[partition]:
            if type(group) is not dict or group.keys() != _GROUP_KEYS:
                raise ValueError("invalid_observation_group")
            _validate_header(group["source_tool"], group["status"])
            shared, keys, rows = group["shared_data"], group["data_keys"], group["rows"]
            if (type(shared) is not dict or type(keys) is not list
                    or any(type(key) is not str for key in keys)
                    or len(set(keys)) != len(keys)
                    or set(keys) & shared.keys() or "view_index" in keys):
                raise ValueError("invalid_observation_data_columns")
            if type(rows) is not list or not rows:
                raise ValueError("invalid_observation_rows")
            for row in rows:
                if type(row) is not list or len(row) != len(keys) + 2:
                    raise ValueError("invalid_observation_row")
                index, result_id = row[:2]
                if type(index) is not int or index < 0 or index in indexed:
                    raise ValueError("invalid_observation_sequence_index")
                data = deepcopy(shared)
                data.update(zip(keys, row[2:]))
                indexed[index] = {
                    "result_id": result_id, "source_tool": group["source_tool"],
                    "status": group["status"], "data": data,
                }
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError("noncontiguous_observation_sequence")
    records = [indexed[index] for index in range(len(indexed))]
    _validate_records(records)
    return records
