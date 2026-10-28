"""Source-linked revision view of decisions and observations, not audit logs.

Original bundles stay immutable. Full programs, all recorded observation pages,
error/recovery facts and supervision survive. Image bookkeeping is grouped;
unused candidate catalogs and unread detection arrays are explicitly summarized.
"""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re

from skill3d.evolution.evidence_graph_v11 import evidence_sha256

VIEW_SCHEMA = "revision-decision-view/1.3"
_CANDIDATE_FIELDS = (
    "obj_id", "category_name", "source_label", "track_id", "visible_frames",
    "bbox", "centroid_world", "det_conf", "det_conf_source", "mask_support_ratio",
    "mask_available", "mask_source_present", "grounding_status", "binding_flags",
    "binding_validation", "question_roles", "duplicate_suspect",
)
_CASE_FIELDS = (
    "case_id", "qa_id", "scene_id", "question_type", "evidence_state",
    "question_text", "options", "source_split", "split", "label_access",
    "learning_authorization", "episode_status", "initial_evidence_states",
    "evidence_states", "selected_object_ids", "model_answer", "reference_answer",
    "score", "outcome", "failure_categories", "error_direction", "frame_set_hash",
    "trace_ref", "delivered_skill_version", "delivered_content_sha256",
)
_ROUND_FIELDS = (
    "index", "trigger", "synthesis_source", "program_source", "executed",
    "error_code", "runtime_error_code", "submission_accepted", "tool_calls",
    "observed_result_ids", "used_result_ids", "invalidated_result_ids",
    "selected_object_ids", "verification_result_ids", "recovery_observation_ids",
    "service_errors", "finalization", "observation_adapter",
)
_RESULT_FIELDS = (
    "result_id", "tool", "source_tool", "source", "args", "status",
    "error", "error_code", "failure_cause", "failure_guidance", "invalidated_by",
    "degraded_evidence", "available_artifacts", "missing_artifacts",
    "resolved_object_ids", "object_dependencies", "result_dependencies",
    "evidence_version",
)
_STATIC_FEEDBACK_PREFIXES = (
    "Still-valid observations (", "No valid prior observations are available;",
    "These are raw data values,", "For omitted candidates/fields use ReadObservation(",
    "Required action this round:", "Call only currently documented tools.",
    "First define answer in code from the question and valid",
    "For an answer genuinely based entirely on original imag",
    "For combined visual and valid tool evidence,",
    "Invalidated results and rejected answers cannot serve a",
)


def _fields(row, fields):
    return {key: deepcopy(row[key]) for key in fields if key in row}


def _candidate_facts(candidate):
    facts = _fields(candidate, _CANDIDATE_FIELDS)
    validation = facts.get("binding_validation")
    if isinstance(validation, dict):
        facts["binding_validation"] = {
            key: value for key, value in validation.items()
            if key not in {"files", "generation_sha256"}}
    # A recorded reference is not proof that the file exists or is usable.
    refs = {name: bool(candidate[name]) for name in (
        "mask_per_frame", "pointcloud_world", "pointconf_world") if name in candidate}
    if refs:
        facts["artifact_reference_present"] = refs
    return facts


def _stdout_view(text):
    """Project only literal object lists; keep all other printed text verbatim."""
    if not isinstance(text, str) or not (1 <= len(text) <= 8_000_000):
        return text
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return text
    try:
        objects = ast.literal_eval(stripped)
    except (ValueError, SyntaxError, RecursionError, MemoryError):
        return text
    if not isinstance(objects, list) or not objects or not all(
            isinstance(obj, dict) and isinstance(obj.get("obj_id"), str)
            and obj["obj_id"] for obj in objects):
        return text
    metadata = {
        "binding_provenance", "semantic_request_identity", "candidate_ids",
        "mask_per_frame", "pointcloud_world", "pointconf_world",
    }
    data = []
    for obj in objects:
        facts = _candidate_facts(obj)
        # Unknown printed attributes may be useful measurements. Omit only
        # the explicit audit/storage fields, not arbitrary object values.
        facts.update({key: deepcopy(value) for key, value in obj.items()
                      if key not in _CANDIDATE_FIELDS and key not in metadata})
        data.append(facts)
    return {
        "encoding": "literal_object_list_projection",
        "data": data,
        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_text_chars": len(text),
        "projection": "object decision values retained; storage/provenance metadata omitted",
        "raw_record_retained": True,
    }


def _referenced_candidates(case, catalog):
    """Conservative recorded-ID matching; never infer referents from labels."""
    known = {obj["obj_id"] for obj in catalog
             if isinstance(obj.get("obj_id"), str) and obj["obj_id"]}
    if not known:
        return set()
    pattern = re.compile(
        r"(?<![\w-])(?:" + "|".join(re.escape(key) for key in sorted(known, key=len, reverse=True))
        + r")(?![\w-])")
    found = set()

    def scan(value):
        if isinstance(value, str):
            found.update(pattern.findall(value))
        elif isinstance(value, dict):
            for key, child in value.items():
                if key in known:
                    found.add(key)
                scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    scan(case.get("selected_object_ids"))
    scan(case.get("program_source"))
    for row in case.get("rounds", []):
        for key in ("selected_object_ids", "program_source", "response_text",
                    "feedback", "observation_reads"):
            scan(row.get(key))
        for result in row.get("results", []):
            for key in ("args", "resolved_object_ids", "object_dependencies"):
                scan(result.get(key))
            # A program may consume a returned payload without a later read page.
            if _reads_raw_detection_array(row.get("program_source")):
                scan(result.get("payload"))
                scan(result.get("value"))
    return found


def _feedback(text, read_index):
    """Keep actual feedback; remove only exact redundant page JSON/control text."""
    kept = []
    delivered_reads = []
    delivered_summaries = []
    skip_page = False
    for line in (text or "").splitlines():
        if line.startswith(_STATIC_FEEDBACK_PREFIXES):
            continue
        if line == "Requested observation pages (raw data, original references):":
            skip_page = True
            continue
        if skip_page:
            skip_page = False
            try:
                pages = json.loads(line)
            except ValueError:
                kept.extend(["Requested observation pages (raw data, original references):", line])
            else:
                if isinstance(pages, list) and all(isinstance(page, dict) for page in pages):
                    for page in pages:
                        digest = evidence_sha256(page)
                        if digest in read_index:
                            delivered_reads.append({"recorded_read_ref": read_index[digest]})
                        else:
                            delivered_reads.append({"original_delivered_page": deepcopy(page)})
                else:
                    kept.extend(["Requested observation pages (raw data, original references):", line])
            continue
        if line.startswith("- ") and " [result_id=" in line:
            delivered_summaries.append({"header": line, "summary_lines": []})
            continue
        if delivered_summaries and line.startswith((
                "  data/raw payload=", "  summary view of raw list payload=", "Summary omissions:")):
            delivered_summaries[-1]["summary_lines"].append(line)
            continue
        kept.append(line)
    if skip_page:
        kept.append("Requested observation pages (raw data, original references):")
    return "\n".join(kept).strip(), delivered_reads, delivered_summaries


def _image_groups(ledger):
    groups = {}
    for image in (ledger or {}).get("images", []):
        common = _fields(image, (
            "kind", "produced_by", "produced_round", "result_id",
            "delivered_rounds", "observed_rounds", "unobserved_reason",
            "source_hw", "sent_hw", "scale", "transform_version",
        ))
        key = json.dumps(common, sort_keys=True)
        group = groups.setdefault(key, {**common, "frames": []})
        frame = image.get("source_frame_id", image.get("source_frame_index"))
        if image.get("kind") not in {"original", "original_frame", "frame"}:
            frame = {**_fields(image, ("source_frame_id", "source_frame_index")),
                     **_fields(image, ("image_id", "box_xyxy"))}
        elif image.get("source_frame_index") not in (None, frame):
            frame = _fields(image, ("source_frame_id", "source_frame_index"))
        group["frames"].append(frame)
    return list(groups.values())

def _reads_raw_detection_array(program):
    """Retain full detection arrays when recorded code may consume raw data."""
    try:
        tree = ast.parse(program or "")
    except SyntaxError:
        return True
    fields = {"data", "payload", "value", "detections"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in fields:
            return True
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value in fields:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant):
                if node.args[0].value in fields:
                    return True
    return False


def decision_case_view(case):
    view = _fields(case, _CASE_FIELDS)
    view["source_case_sha256"] = evidence_sha256(case)
    if not case.get("rounds") and "program_source" in case:
        view["program_source"] = case["program_source"]
    view["initial_images"] = _image_groups(case.get("image_observations"))
    history = []
    for state in case.get("evidence_history", []):
        facts = _fields(state, (
            "trigger", "evidence_states", "evidence_version", "available_artifacts",
            "local_facts", "question_tool_scope", "state_reasons",
        ))
        candidates = state.get("candidates") or []
        facts["candidate_catalog"] = {
            "count": len(candidates),
            "categories": dict(Counter(str(c.get("category_name", "")) for c in candidates)),
            "binding_statuses": dict(Counter(str(c.get("grounding_status", "")) for c in candidates)),
            "detail_status": "unused catalog summarized; source retained",
        }
        history.append(facts)
    view["evidence_history"] = history
    initial_catalog = (case.get("evidence_history") or [{}])[0].get("candidates") or []
    referenced = _referenced_candidates(
        case, [*initial_catalog, *(case.get("candidate_objects") or [])])
    if initial_catalog:
        view["initial_candidate_facts"] = {
            "delivery_status": "initial runtime catalog; model delivery is not asserted",
            "retention_basis": "recorded object IDs in programs, arguments, reads, feedback or dependencies",
            "catalog_count": len(initial_catalog),
            "omitted_count": sum(obj.get("obj_id") not in referenced for obj in initial_catalog),
            "categories": dict(Counter(str(obj.get("category_name", "")) for obj in initial_catalog)),
            "objects": [_candidate_facts(obj) for obj in initial_catalog
                        if obj.get("obj_id") in referenced],
        }
    selected = set(case.get("selected_object_ids") or [])
    for row in case.get("rounds", []):
        selected.update(row.get("selected_object_ids") or [])
        for result in row.get("results", []):
            selected.update(result.get("resolved_object_ids") or [])
            selected.update(result.get("object_dependencies") or [])
    view["selected_candidates"] = [
        _candidate_facts(c)
        for c in case.get("candidate_objects", []) if c.get("obj_id") in selected]
    rounds = []
    read_index = {
        evidence_sha256(read): [position, offset]
        for position, row in enumerate(case.get("rounds", []))
        for offset, read in enumerate(row.get("observation_reads") or [])
    }
    for position, original in enumerate(case.get("rounds", [])):
        row = _fields(original, _ROUND_FIELDS)
        row["source_ref"] = original.get("experience_provenance", {}).get(
            "source_ref", f"{case.get('trace_ref', '')}#/rounds/{position}")
        response = original.get("response_text") or ""
        program = original.get("program_source") or ""
        if program and program in response:
            response = response.replace(program, "[complete program_source retained]")
        response = re.sub(r"```(?:python|py)?\s*\[complete program_source retained\]\s*```",
                          "[complete program_source retained]", response)
        if response.strip() not in {"", "[complete program_source retained]"}:
            row["response_outside_program"] = response
        diagnostics = _fields(original.get("execution_diagnostics") or {}, (
            "answer_untrusted", "error_line", "error_message", "error_source_line",
            "error_type", "stdout", "traceback_text",
        ))
        if "stdout" in diagnostics:
            diagnostics["stdout"] = _stdout_view(diagnostics["stdout"])
        if diagnostics:
            row["execution_diagnostics"] = diagnostics
        reads = original.get("observation_reads") or []
        row["observation_reads"] = [
            _fields(read, (
                "result_id", "source_tool", "data", "data_path", "source_item_paths",
                "returned_range", "total_count", "coverage_complete", "next_cursor",
                "fields_omitted", "unknown_fields_omitted_count", "evidence_version",
            )) for read in reads]
        results = []
        for result in original.get("results", []):
            record = _fields(result, _RESULT_FIELDS)
            record["authorization"] = _fields(result.get("authorization") or {}, (
                "allowed", "reason_codes", "question_tool_scope", "scene_route",
                "metric_gate_result", "invalidated_by", "dependency_refs",
            ))
            payload = result.get("payload")
            if payload is None and isinstance(result.get("value"), str):
                try:
                    payload = json.loads(result["value"])
                except ValueError:
                    pass
                else:
                    record["source_value_encoding"] = "original JSON values; whitespace omitted"
            if isinstance(payload, dict) and (
                result.get("source_tool") or result.get("tool")) == "detect_objects" and (
                    not _reads_raw_detection_array(program)):
                record["payload"] = {
                    key: deepcopy(value) for key, value in payload.items()
                    if key not in {"detections", "detector_endpoint_hash"}
                }
                record["detection_array"] = {
                    "count": len(payload.get("detections") or []),
                    "detail_status": "original read pages and delivered summaries retained below",
                }
            elif isinstance(payload, list) and (
                    result.get("source_tool") or result.get("tool")) == "list_objects":
                record["payload"] = [_fields(obj, (
                    "obj_id", "category_name", "visible_frames", "track_id",
                    "grounding_status", "binding_flags", "binding_validation",
                    "bbox", "det_conf", "det_conf_source", "mask_support_ratio",
                    "mask_available", "mask_source_present", "duplicate_suspect",
                    "question_roles", "centroid_world",
                )) if isinstance(obj, dict) else deepcopy(obj) for obj in payload]
                record["payload_projection"] = "object values retained; storage/provenance references omitted"
            elif payload is not None:
                record["payload"] = deepcopy(payload)
            else:
                record["value"] = deepcopy(result.get("value"))
            results.append(record)
        row["results"] = results
        feedback, delivered_reads, delivered_summaries = _feedback(
            original.get("feedback"), read_index)
        if feedback:
            row["feedback"] = feedback
        if delivered_reads:
            row["delivered_observation_pages"] = delivered_reads
        if delivered_summaries:
            row["delivered_tool_summaries"] = delivered_summaries
        verification = original.get("geometry_verify")
        if isinstance(verification, dict):
            row["geometry_verify"] = _fields(verification, (
                "passed", "world_frame_available", "metric_evidence_authorized",
                "checks", "check_status", "issues", "violations", "notes",
            ))
        progress = original.get("progress")
        if isinstance(progress, dict):
            row["progress"] = {key: deepcopy(value) for key, value in progress.items()
                               if not key.endswith(("hash", "sha256", "fingerprint"))}
        rounds.append(row)
    view["rounds"] = rounds
    view["projection"] = {
        "schema": VIEW_SCHEMA,
        "retained": "all programs, rounds, read pages, delivered feedback facts, supervision and validity",
        "summarized": ["image bookkeeping", "unused candidate catalogs", "unread detection arrays"],
        "omitted": ["duplicate responses/results", "repeated framework text", "generation/audit metadata",
                    "selected_candidates.binding_provenance",
                    "unreferenced initial candidate details", "candidate storage paths and audit file identities"],
        "raw_record_retained": True,
    }
    return view


def decision_bundle_view(bundle):
    common = _fields(bundle, (
        "schema_version", "campaign_id", "parent_skill_version", "parent_content_sha256",
        "parent_snapshot_id", "question_type", "evidence_state", "source_split", "split",
        "label_access", "learning_authorization", "source_run_ref",
    ))
    accesses = bundle.get("label_access_records") or [bundle.get("label_access_record")]
    common["label_access_summary"] = [
        _fields(record, ("split", "purpose", "component_role", "n_items", "label_access",
                         "input_manifest_sha256", "learning_authorization"))
        for record in accesses if isinstance(record, dict)
    ]
    return {**common, "cases": [decision_case_view(c) for c in bundle["cases"]]}


def pack_case_components(rows, source_cases):
    """Share headers within this request; carry any cross-round read references."""
    contexts, components = {}, []
    for row in rows:
        key = str(row["case_index"])
        case = row["complete_case"]
        context = {
            **_fields(row, ("case_index", "case_id", "source_case_sha256", "view_case_sha256")),
            "case_header": {name: value for name, value in case.items() if name != "rounds"},
        }
        if key not in contexts:
            contexts[key] = {**context, "referenced_reads": {}}
        elif any(contexts[key][name] != value for name, value in context.items()):
            raise ValueError("case context changed within request")
        for round_row in case.get("rounds", []):
            for page in round_row.get("delivered_observation_pages", []):
                ref = page.get("recorded_read_ref")
                if ref is not None:
                    original = source_cases[row["case_index"]]["rounds"][ref[0]][
                        "observation_reads"][ref[1]]
                    contexts[key]["referenced_reads"][f"{ref[0]}:{ref[1]}"] = original
        components.append({
            "case_context_ref": key, "rounds": case.get("rounds", []),
            **_fields(row, ("round_range", "total_rounds", "case_part", "round_fragment")),
        })
    return {"case_contexts": contexts, "case_components": components}


def unpack_case_components(value):
    """Expand request-local case references for independent coverage checking."""
    rows = []
    for component in value["case_components"]:
        context = value["case_contexts"][component["case_context_ref"]]
        rows.append({
            **_fields(context, ("case_index", "case_id", "source_case_sha256", "view_case_sha256")),
            **_fields(component, ("round_range", "total_rounds", "case_part", "round_fragment")),
            "complete_case": {**context["case_header"], "rounds": component["rounds"]},
        })
    return rows


def _at_path(value, path):
    for key in path:
        value = value[key]
    return value


def _set_path(value, path, replacement):
    _at_path(value, path[:-1])[path[-1]] = replacement


def split_round_components(row, fits):
    """Split one result/read array, keeping the full program and other facts."""
    original = row["complete_case"]["rounds"][0]
    arrays = []

    def visit(value, path):
        if isinstance(value, list):
            if len(value) > 1:
                arrays.append((len(json.dumps(value, ensure_ascii=False)), path, value))
            for index, child in enumerate(value):
                visit(child, [*path, index])
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, [*path, key])

    for field in ("results", "observation_reads", "execution_diagnostics"):
        if field in original:
            visit(original[field], [field])
    for _, path, values in sorted(arrays, key=lambda item: item[0], reverse=True):
        size = (len(values) + 1) // 2
        while True:
            pieces = []
            for start in range(0, len(values), size):
                piece = deepcopy(row)
                end = min(start + size, len(values))
                _set_path(piece["complete_case"]["rounds"][0], path, values[start:end])
                piece["round_fragment"] = {
                    "path": path, "start": start, "end": end, "total": len(values)}
                if not fits([piece]):
                    break
                pieces.append(piece)
            else:
                return pieces
            if size == 1:
                break
            size = (size + 1) // 2
    raise ValueError("one complete program/context or indivisible result exceeds revision input budget")


def reassemble_round_components(rows):
    """Check contiguous array coverage and identical surrounding facts."""
    combined = []
    active, template, fragment_path, total = None, None, None, None
    for row in rows:
        fragment = row.get("round_fragment")
        if fragment is None:
            if active is not None:
                raise ValueError("round fragment sequence is incomplete")
            combined.append(row)
            continue
        path = fragment["path"]
        values = _at_path(row["complete_case"]["rounds"][0], path)
        if not isinstance(values, list) or not values or len(values) != fragment["end"] - fragment["start"]:
            raise ValueError("round fragment array length changed")
        clean = deepcopy(row)
        clean.pop("round_fragment")
        _set_path(clean["complete_case"]["rounds"][0], path, [])
        if active is None:
            if fragment["start"] != 0:
                raise ValueError("round fragment first slice is missing")
            template, active = clean, deepcopy(clean)
            fragment_path, total = path, fragment["total"]
        if clean != template or path != fragment_path or fragment["total"] != total:
            raise ValueError("round fragment context changed")
        output = _at_path(active["complete_case"]["rounds"][0], path)
        if fragment["start"] != len(output) or fragment["end"] > total:
            raise ValueError("round fragment order or coverage changed")
        output.extend(values)
        if len(output) == total:
            combined.append(active)
            active = None
    if active is not None:
        raise ValueError("round fragment sequence is incomplete")
    return combined


def compact_slot_feedback(feedback):
    """Factor invariant process fields without dropping unknowns, zeros or errors."""
    value = deepcopy(dict(feedback))
    signatures = value.get("signatures") or []
    if not signatures or not all(isinstance(row.get("process"), dict) for row in signatures):
        return value

    def common_dict(items):
        common = {}
        for key in items[0]:
            if not all(key in item for item in items):
                continue
            values = [item[key] for item in items]
            if all(child == values[0] for child in values[1:]):
                common[key] = values[0]
            elif all(isinstance(child, dict) for child in values):
                nested = common_dict(values)
                if nested:
                    common[key] = nested
        return common

    def difference(item, shared):
        result = {}
        for key, child in item.items():
            if key not in shared:
                result[key] = child
            elif child != shared[key]:
                result[key] = difference(child, shared[key])
        return result

    shared = common_dict([row["process"] for row in signatures])
    value["shared_process"] = shared
    value["process_encoding"] = (
        "Recursively overlay each signature.process on shared_process; arrays replace as a whole. "
        "No field is omitted: null stays unknown, zero/false stay recorded values. "
        "signature_ref and identity_sha256 refer to the original expanded feedback.")
    for row in signatures:
        row["process"] = difference(row["process"], shared)
    return value
