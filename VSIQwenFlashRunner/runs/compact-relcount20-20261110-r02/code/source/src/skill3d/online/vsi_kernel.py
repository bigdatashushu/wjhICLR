"""Explicit VSI kernel extension over the read-only shared sandbox."""

from __future__ import annotations

import copy
import json
import math

from skill3d.online.observation_reader import OBSERVATION_READER_VERSION, payload_hash
from skill3d.routing.task_classifier import canonical_task
from skill3d.sandbox.kernel import RestrictedNamespaceKernel as SharedKernel
from skill3d.schemas import AnswerPayload
from skill3d.schemas.tool_response import TOOL_RESPONSE_VERSION
from skill3d.verifier.derivation import replay_derivation

from .vsi_execution_contract import (
    MEASUREMENT_FIELDS, VSI_INTERFACE_NAMES, VSI_INTERFACE_VERSION,
)


def _result_id(response_or_id):
    value = response_or_id.get("result_id") if isinstance(response_or_id, dict) else response_or_id
    if not isinstance(value, str) or not value:
        raise ValueError("Use an actual nonempty host result_id")
    return value


def _nonfinite(_value):
    raise ValueError("Measurement result contains nonfinite JSON")


def _merged_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


class RestrictedNamespaceKernel(SharedKernel):
    """Keep helper state in the host, never in generated Python variables."""

    def __init__(self, *args, **kwargs):
        self.vsi_interface_calls = []
        self._cell_vsi_mark = 0
        self._vsi_ranges = {}
        self._vsi_last_fields = {}
        self.vsi_reader_progress = []
        self._cell_vsi_page_mark = 0
        super().__init__(*args, **kwargs)
        self._install_vsi_interfaces()

    def _install_vsi_interfaces(self):
        self._ns.update(
            ToolData=self._tool_data, ReadNextObservation=self._read_next,
            MeasurementAnswer=self._measurement_answer)

    def inject(self, **kwargs):
        if VSI_INTERFACE_NAMES & kwargs.keys():
            raise ValueError("Overwriting reserved VSI interfaces is forbidden")
        super().inject(**kwargs)

    def reset_user_namespace(self):
        super().reset_user_namespace()
        self._install_vsi_interfaces()

    def reset_episode_ledger(self):
        self.vsi_interface_calls = []
        self._vsi_ranges = {}
        self._vsi_last_fields = {}
        self.vsi_reader_progress = []
        super().reset_episode_ledger()

    def mark_cell_start(self):
        super().mark_cell_start()
        self._cell_vsi_mark = len(self.vsi_interface_calls)
        self._cell_vsi_page_mark = len(self.vsi_reader_progress)

    @property
    def cell_vsi_interface_calls(self):
        return copy.deepcopy(self.vsi_interface_calls[self._cell_vsi_mark:])

    @property
    def cell_vsi_reader_progress(self):
        return copy.deepcopy(self.vsi_reader_progress[self._cell_vsi_page_mark:])

    def _interface(self, name, response_or_id, operation):
        receipt = {"name": name, "interface_version": VSI_INTERFACE_VERSION, "allowed": False}
        try:
            if not self._tools_enabled or self.answer_slot.given or self.yield_slot.given:
                raise ValueError("VSI interfaces are available only before ordinary-phase termination")
            receipt["result_id"] = _result_id(response_or_id)
            value = operation(receipt["result_id"])
        except ValueError:
            self.answer_untrusted = True
            receipt["reason_code"] = "vsi_interface_rejected"
            self.vsi_interface_calls.append(receipt)
            raise
        receipt.update(allowed=True, reason_code="valid_episode_interface")
        self.vsi_interface_calls.append(receipt)
        return value

    def _current_result(self, result_id):
        matches = [r for r in self.tool_results if r.result_id == result_id]
        if len(matches) != 1:
            raise ValueError("Unknown or duplicate host result_id")
        result = matches[0]
        if (result.status != "ok" or result.error is not None or result.error_code
                or result.authorization.get("allowed") is False or result.invalidated_by
                or result_id in self.unresolved_result_ids):
            raise ValueError("Failed, invalidated or unresolved result_id")
        try:
            # value is also the unchanged derivation replay's authoritative source.
            data = json.loads(result.value, parse_constant=_nonfinite)
        except (TypeError, ValueError) as exc:
            raise ValueError("Host result is not finite replayable JSON") from exc
        if isinstance(data, dict) and (
                data.get("status") in ("fault", "failed", "error")
                or data.get("service_healthy") is False or data.get("error")):
            raise ValueError("Fault or unhealthy data cannot support this interface")
        return result, data

    def _tool_data(self, response):
        def extract(result_id):
            if not isinstance(response, dict):
                raise ValueError("ToolData requires a tool envelope or reader page")
            result, data = self._current_result(result_id)
            version = response.get("schema_version")
            if version == TOOL_RESPONSE_VERSION:
                if response.get("source_tool") != (result.source_tool or result.tool):
                    raise ValueError("Tool envelope source does not match the ledger")
                # Extraction must not turn a forged ID-only envelope into an
                # unbounded read of historical observations.
                if "data" not in response or response["data"] != data:
                    raise ValueError("Tool envelope data is missing or differs from the host result")
                return copy.deepcopy(data)
            if version == OBSERVATION_READER_VERSION:
                # Page data remains a list. Use the immutable host copy, not local edits.
                keys = ("result_id", "source_tool", "payload_hash", "returned_range",
                        "data_path", "fields_omitted", "source_item_paths")
                page = next((p for p in reversed(self.observation_reads)
                             if all(p.get(k) == response.get(k) for k in keys)), None)
                if page is None or page["payload_hash"] != payload_hash(
                        data, tool=result.source_tool or result.tool):
                    raise ValueError("Reader page does not match current host evidence")
                return copy.deepcopy(page["data"])
            raise ValueError("Unsupported tool envelope or reader page schema")
        return self._interface("ToolData", response, extract)

    @staticmethod
    def _fields_key(fields):
        return tuple(sorted(set(fields))) if fields is not None else None

    def _read_observation(self, result_id, fields=None, offset=0, limit=8):
        # Shared reader still enforces public fields, phase, receipts and exact budget.
        page = super()._read_observation(result_id, fields, offset, limit)
        selection = self._fields_key(fields)
        key = (result_id, page["payload_hash"], tuple(page["data_path"]), selection)
        ranges = _merged_ranges([*self._vsi_ranges.get(key, []), page["returned_range"]])
        self._vsi_ranges[key] = ranges
        self._vsi_last_fields[result_id] = copy.deepcopy(fields)
        cursor = ranges[0][1] if ranges and ranges[0][0] == 0 else 0
        complete = cursor >= page["total_count"]
        page["vsi_read_progress"] = {
            "interface_version": VSI_INTERFACE_VERSION,
            "covered_ranges": copy.deepcopy(ranges),
            "next_cursor": None if complete else cursor,
            "coverage_complete": complete,
        }
        # Keep the underlying reader page/receipt byte budget unchanged.
        # Cumulative progress is separate supplementary host metadata.
        self.vsi_reader_progress.append({
            "result_id": result_id, "payload_hash": page["payload_hash"],
            "fields": copy.deepcopy(fields), "data_path": page["data_path"],
            **copy.deepcopy(page["vsi_read_progress"]),
        })
        return page

    def _read_next(self, response_or_id, fields=None, limit=8):
        def read(result_id):
            result, data = self._current_result(result_id)
            selected = (copy.deepcopy(self._vsi_last_fields.get(result_id))
                        if fields is None else fields)
            if selected is not None and (
                    not isinstance(selected, list) or not selected
                    or any(not isinstance(k, str) for k in selected)):
                raise ValueError("fields must be a nonempty list of public field names")
            collection = (selected[0] if selected is not None and len(selected) == 1
                          and isinstance(data, dict) and isinstance(data.get(selected[0]), list)
                          else None)
            key = (result_id, payload_hash(data, tool=result.source_tool or result.tool),
                   (collection,) if collection else (), self._fields_key(selected))
            ranges = self._vsi_ranges.get(key, [])
            cursor = ranges[0][1] if ranges and ranges[0][0] == 0 else 0
            return self._read_observation(result_id, selected, offset=cursor, limit=limit)
        return self._interface("ReadNextObservation", response_or_id, read)

    def _measurement_answer(self, response_or_id):
        def build(result_id):
            task = canonical_task(str(self._scene.question_type or ""))
            specification = MEASUREMENT_FIELDS.get(task)
            if specification is None:
                raise ValueError("MeasurementAnswer has no registered field for this task")
            tool, field, from_unit, unit = specification
            result, data = self._current_result(result_id)
            if result.tool != tool or (result.source_tool and result.source_tool != tool):
                raise ValueError("Measurement tool does not match the current task")
            if not isinstance(data, dict) or field not in data:
                raise ValueError("Required measurement field is absent; do not substitute zero")
            if tool == "object_surface_distance" and data.get("unit") != "m":
                raise ValueError("Absolute distance requires an actual metric measurement in m")
            number = data[field]
            try:
                finite = type(number) in (int, float) and math.isfinite(number) and number >= 0
            except OverflowError:
                finite = False
            if not finite or (unit == "count" and number != int(number)):
                raise ValueError("Measurement must be finite, nonnegative and count must be integral")
            parameters = {"field": field}
            op = "field"
            value = number
            if from_unit != unit:
                op = "convert"
                parameters.update(from_unit=from_unit, to_unit=unit)
                value = number * 100.0
            payload = AnswerPayload(
                value=value, unit=unit, basis="tool_derived", used_result_ids=[result_id],
                derivation={"op": op, "input_result_ids": [result_id], "parameters": parameters})
            replay = replay_derivation(payload, self.tool_results, question_type=task)
            if not replay.passed:
                raise ValueError("Measurement derivation rejected: "
                                 + ", ".join(i.code for i in replay.issues))
            return payload
        return self._interface("MeasurementAnswer", response_or_id, build)
