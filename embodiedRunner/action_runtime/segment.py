"""Task-local segment execution; no episode/scorer/global-runner dependencies."""
from __future__ import annotations

from dataclasses import dataclass

from skill3d.schemas import ProgramExecutionTrace
from action_runtime.kernel import CellResult, RestrictedNamespaceKernel


@dataclass(frozen=True)
class SegmentExecutionResult:
    kernel: RestrictedNamespaceKernel
    cell: CellResult
    trace: ProgramExecutionTrace


def collect_segment_trace(kernel, cell, *, program_id, deterministic_replay=False):
    results = list(kernel.cell_results)
    if deterministic_replay:
        results = [r.model_copy(update={"latency_ms": 0.0}) for r in results]
    return ProgramExecutionTrace(
        program_id=program_id, calls=list(kernel.cell_calls), results=results,
        stdout_tail=cell.stdout_tail, stdout=cell.stdout, traceback_text=cell.traceback_text,
        error_type=cell.error_type, error_message=cell.error_message,
        error_line=cell.error_line, error_source_line=cell.error_source_line,
        program_sha256=cell.program_sha256, source_filename=cell.source_filename,
        error_code=cell.error_code, steps=len(results) + 1, wallclock_s=0.0)


def execute_segment(kernel, source, *, program_id, reset_namespace=False,
                    deterministic_replay=False, yield_on_fallthrough=False):
    if reset_namespace:
        kernel.reset_user_namespace()
    cell = kernel.run_cell(source, yield_on_fallthrough=yield_on_fallthrough)
    return SegmentExecutionResult(kernel, cell, collect_segment_trace(
        kernel, cell, program_id=program_id, deterministic_replay=deterministic_replay))
