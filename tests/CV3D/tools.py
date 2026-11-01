"""Native function schemas and host dispatch to the existing numerical tools."""
from __future__ import annotations

import json
import math
import time

from skill3d.schemas.tool_response import model_tool_response
from skill3d.small_image.tools import build_registry
from skill3d.tools.contract import ToolContractError

from .common import digest

INSTRUCTIONS = (
    "Tools are optional native function calls. Coordinates use Image 1's original "
    "pixels: x increases rightwards and y downwards; view_index is 0. "
    "There is no tool-call count limit. Successful returns have status='ok'; "
    "numerical values are in data. Failed calls have status='error' and no measurement. "
    "The tools measure predicted, per-view relative geometry."
)

_INTEGER = {"type": "integer"}
_NUMBER = {"type": "number"}
_XY = {"type": "array", "items": _NUMBER, "minItems": 2, "maxItems": 2}
_SPECS = {
    "view_candidates": (
        "Inspect existing detections: category and box_xyxy in original pixels. "
        "Optional view_index=-1 selects all views; category='' selects all categories. "
        "Category confirmation does not validate mask quality or geometry.",
        {"view_index": _INTEGER, "category": {"type": "string"}}, []),
    "view_depth_summary": (
        "Get depth availability and quantiles for Image 1. Unit: per_view_relative.",
        {}, []),
    "point_depth": (
        "Read camera Z depth at (x,y) in Image 1: round to a pixel and return the "
        "median of valid, finite, positive values in its 5x5 neighborhood, requiring "
        "at least 3 valid pixels. Smaller is nearer. No SAM mask filtering: the "
        "measurement belongs to the sampled surface, which may be background.",
        {"view_index": _INTEGER, "x": _NUMBER, "y": _NUMBER},
        ["view_index", "x", "y"]),
    "pair_point_distance": (
        "Return Euclidean distance between local median predicted 3D points at a_xy "
        "and b_xy in Image 1. Compare targets with the same reference pixel. "
        "Unit: per_view_relative, not calibrated meters, object-center ground truth "
        "or whole-object minimum surface distance.",
        {"view_index": _INTEGER, "a_xy": _XY, "b_xy": _XY},
        ["view_index", "a_xy", "b_xy"]),
}


def tool_names(task):
    if task not in {"Depth", "Distance"}:
        raise ValueError("unknown CV3D task")
    names = ["view_candidates", "view_depth_summary", "point_depth"]
    return names + (["pair_point_distance"] if task == "Distance" else [])


def tool_schemas(task):
    return [{
        "type": "function", "function": {
            "name": name, "description": _SPECS[name][0],
            "parameters": {
                "type": "object", "properties": _SPECS[name][1],
                "required": _SPECS[name][2], "additionalProperties": False,
            },
        },
    } for name in tool_names(task)]


def tool_instructions(task):
    return INSTRUCTIONS + "\n\n" + "\n".join(
        name + ": " + _SPECS[name][0] for name in tool_names(task))


def _arguments(text):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate argument")
            output[key] = value
        return output

    def invalid_constant(_):
        raise ValueError("nonfinite argument")

    args = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    return args


def _validate(name, args):
    _, properties, required = _SPECS[name]
    if set(args) - properties.keys() or set(required) - args.keys():
        raise ValueError("unexpected or missing arguments")
    for key, value in args.items():
        kind = properties[key]["type"]
        if kind == "integer" and type(value) is not int:
            raise ValueError("view index must be an integer")
        if kind == "string" and not isinstance(value, str):
            raise ValueError("category must be text")
        numbers = [value] if kind == "number" else value if kind == "array" else []
        if kind == "array" and (not isinstance(value, list) or len(value) != 2):
            raise ValueError("coordinate pair required")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers):
            raise ValueError("finite numerical coordinates required")
    if args.get("view_index", 0) not in (
            {-1, 0} if name == "view_candidates" else {0}):
        raise ValueError("only Image 1 has a geometry view")


class ToolDispatcher:
    def __init__(self, prepared, problem):
        self.registry, self.scene = build_registry(prepared, problem)
        self.allowed = set(tool_names(problem.task_type))
        self.episode_id = problem.sample_id

    def call(self, call):
        started = time.monotonic()
        name = call["function"]["name"]
        arguments = call["function"]["arguments"]
        trace = {
            "tool_call_id": call["id"], "name": name,
            "arguments_sha256": digest(arguments), "authorization": None,
        }
        args = None
        response = None
        try:
            if name not in self.allowed:
                raise ValueError("tool unavailable for this task")
            args = _arguments(arguments)
            _validate(name, args)
            result = self.registry.call_tool(
                name, args, self.scene, episode_id=self.episode_id)
            trace["authorization"] = result.authorization
            if result.status != "ok":
                response = {
                    "status": "error", "source_tool": name, "data": None,
                    "error_code": result.error_code or "tool_execution_error",
                }
            else:
                response = model_tool_response(result)
        except (ValueError, TypeError, ToolContractError):
            # Invalid floats such as JSON 1e999 must not enter strict JSON traces.
            # Their exact model-supplied text is already in the saved response.
            args = None
            response = {
                "status": "error", "source_tool": name, "data": None,
                "error_code": "invalid_arguments_or_unavailable_geometry",
            }
        trace.update(
            arguments=args, response=response,
            elapsed_seconds=round(time.monotonic() - started, 6))
        return response, trace
