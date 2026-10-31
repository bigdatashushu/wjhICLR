from types import SimpleNamespace

import numpy as np

from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.small_image.tools import VERSION, build_registry
from skill3d.tools.registry import REGISTRY, ToolRegistry


def prepared(*, depth=True):
    yy, xx = np.indices((12, 16))
    z = (1 + xx / 10).astype(float)
    return SimpleNamespace(
        frames=(np.zeros((12, 16, 3), dtype=np.uint8),),
        candidates=({
            "candidate_id": "v0-c0", "view_index": 0, "category": "chair",
            "box_xyxy": [1, 1, 14, 10], "category_confirmed": True,
            "mask_quality_assessed": False, "geometry_usable": False,
            "mask": np.ones((12, 16), dtype=bool),
        },),
        geometries={0: {"depth": z, "valid_mask": np.ones_like(z, dtype=bool),
                        "points": np.stack([xx, yy, z], axis=-1)}} if depth else {},
        provenance={"model": "injected-test"},
    )


def problem():
    return SimpleNamespace(input_sha256="a" * 64, benchmark="cv3d")


def test_instance_registry_keeps_vsi_catalog_and_authorizer():
    before = REGISTRY.model_names(), REGISTRY.names()
    registry, scene = build_registry(prepared(), problem())
    assert "point_depth" in registry.names_for_scope(
        scene.question_tool_scope, scene.available_artifacts)
    assert (REGISTRY.model_names(), REGISTRY.names()) == before
    assert ToolRegistry()._authorizer is None


def test_real_kernel_depth_result_and_bounded_public_read():
    registry, scene = build_registry(prepared(), problem())
    kernel = RestrictedNamespaceKernel(registry, scene, list(scene.frames))
    source = """
r = tools.point_depth(view_index=0, x=5, y=5)
page = ReadObservation(r["result_id"], fields=["depth"])
ReturnAnswer(AnswerPayload(value="A", unit="option", basis="mixed",
                          used_result_ids=[r["result_id"]]))
"""
    assert ast_guard(source, allowed_tools=set(registry.names())).ok
    result = kernel.run_cell(source)
    assert result.answer == "A"
    assert not result.answer_untrusted
    assert kernel.tool_results[0].payload["depth"] == 1.5
    assert kernel.authorization_receipts[0]["decision_version"] == VERSION
    assert kernel.observation_reads[0]["data"] == [{"depth": 1.5}]


def test_category_confirmation_does_not_authorize_geometry_and_denial_same_version():
    registry, scene = build_registry(prepared(), problem())
    kernel = RestrictedNamespaceKernel(registry, scene, list(scene.frames))
    result = kernel.run_cell('tools.candidate_depth_summary(candidate_id="v0-c0")')
    assert result.error_code == "tool_contract"
    assert result.answer_untrusted
    receipt = kernel.authorization_receipts[0]
    assert not receipt["allowed"]
    assert receipt["decision_version"] == VERSION
    assert "independently_validated_candidate_geometry" in kernel.tool_results[0].missing_artifacts
    assert kernel.tool_results[0].status == "failed"


def test_forged_geometry_flag_cannot_create_admission():
    artifact = prepared()
    artifact.candidates[0].update(geometry_usable=True, mask_quality_assessed=True)
    registry, scene = build_registry(artifact, problem())
    assert "candidate_depth_summary" not in registry.names_for_scope(
        scene.question_tool_scope, scene.available_artifacts)
    kernel = RestrictedNamespaceKernel(registry, scene, list(scene.frames))
    result = kernel.run_cell('tools.candidate_depth_summary(candidate_id="v0-c0")')
    assert result.answer_untrusted and not kernel.authorization_receipts[0]["allowed"]


def test_caught_custom_authorizer_fault_still_records_denial():
    registry, scene = build_registry(prepared(), problem())
    def broken_authorizer(*args, **kwargs):
        raise ValueError("fault injected without public exception detail")
    registry._authorizer = broken_authorizer
    kernel = RestrictedNamespaceKernel(registry, scene, list(scene.frames))
    result = kernel.run_cell("""
try:
    tools.view_candidates()
except Exception:
    pass
ReturnAnswer(AnswerPayload(value="A", unit="option", basis="visual_estimate"))
""")
    assert result.answer_untrusted
    assert len(kernel.authorization_receipts) == 1
    assert not kernel.authorization_receipts[0]["allowed"]
    assert "authorization_decision_invalid" in kernel.tool_results[0].missing_artifacts


def test_depth_unavailable_retains_visual_tools():
    registry, scene = build_registry(prepared(depth=False), problem())
    names = registry.names_for_scope(scene.question_tool_scope, scene.available_artifacts)
    assert "view_candidates" in names and "view_depth_summary" in names
    assert "point_depth" not in names
    kernel = RestrictedNamespaceKernel(registry, scene, list(scene.frames))
    result = kernel.run_cell(
        'ReturnAnswer(AnswerPayload(value="B", unit="option", basis="visual_estimate"))')
    assert result.answer == "B" and not result.answer_untrusted


def test_out_of_view_or_invalid_geometry_fails_closed():
    registry, scene = build_registry(prepared(), problem())
    kernel = RestrictedNamespaceKernel(registry, scene, list(scene.frames))
    result = kernel.run_cell("tools.point_depth(view_index=1, x=4, y=4)")
    assert result.answer_untrusted and not kernel.authorization_receipts[0]["allowed"]
    bad = prepared()
    bad.geometries[0]["valid_mask"][:] = False
    registry, scene = build_registry(bad, problem())
    kernel = RestrictedNamespaceKernel(registry, scene, list(scene.frames))
    result = kernel.run_cell("tools.point_depth(view_index=0, x=4, y=4)")
    assert result.error_code == "tool_contract"


def test_no_cross_view_distance_and_no_host_paths_in_kernel():
    registry, scene = build_registry(prepared(), problem())
    kernel = RestrictedNamespaceKernel(registry, scene, list(scene.frames))
    result = kernel.run_cell(
        "r = tools.pair_point_distance(view_index=0, a_xy=[3,4], b_xy=[8,4])\n"
        'YieldObservations([r["result_id"]])')
    assert result.terminated == "yield" and not result.error
    assert np.isclose(kernel.tool_results[0].payload["distance"], np.sqrt(25.25))
    assert not ast_guard(
        "import os\nos.listdir('.')", allowed_tools=set(registry.names())).ok
    assert not ast_guard(
        "show(scene.geometries)", allowed_tools=set(registry.names())).ok
