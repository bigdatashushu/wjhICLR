# Current-view ALFRED manipulation

Method ID: embodied.alfred.current-view. Version: 1.0.0.
Environment: EB-ALFRED. Control profile: embodied-actions/1.

Complete the household instruction as a sequence of observable subgoals: locate the relevant object, acquire or transform it when required, then place or interact with it at the intended destination. The host applies one official action from the current table before returning a new RGB and official feedback.

## Decision procedure

1. Parse the instruction into the next unfinished subgoal using the current image, official feedback and sourced memory. Distinguish finding an object, acquiring it, processing it and placing it. Carry state is inferred only from explicit public feedback or visible evidence.
2. Inspect the action table at this observation. Some actions combine navigation and interaction and can change the camera unpredictably. Select the exact current action ID; do not infer a camera transform, object pose or persistent distance from its action name.
3. If useful, call `view_candidates(0)` to inspect current image regions. Use category confirmation to support identification, while treating masks and boxes as unvalidated surfaces. Prefer an interaction supported by current visible evidence or by an explicit available official action.
4. Use `point_depth(0,x,y)` only for visible pixels when current relative depth is available. This can help compare nearby visible regions within the image. It does not establish reachability or the metric distance to an object. `pair_point_distance` similarly remains within one view.
5. Read the last official action feedback before choosing the next operation. On failure, revise the approach or choose another legal target/action. Do not repeat a failed interaction solely because old geometry predicted proximity. On success, advance the subgoal only as far as that feedback supports.
6. Submit one legal action. If the instruction appears finished or there is no productive next action, submit an empty list; the host retains official success/progress privately.

## Program, repair and memory

Return Python defining `solve(ctx)`. Tools are top-level functions or `ctx.tools.<name>`. Replies use `r["data"]` and `r["result_id"]`; include all consumed IDs in payload `evidence_refs`. Retained results may be read through bounded `ReadObservation`; `YieldObservations` ends the segment and lets the same coding agent inspect selected results. Every segment starts with fresh variables.

Use `SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision, ctx.snapshot_sha256, ctx.action_table_sha256, actions, evidence_refs, memory_update))`. Submission terminates the program. The host validates the entire list and applies only its first action. A malformed household plan invokes official invalid-action accounting without a real step, then creates a new observation revision.

Memory contains a short subgoal and at most 16 sourced records, each with text, source (`visual` or `official_feedback`), episode_id, observation_revision, decision_id and confidence (`observed` or `uncertain`). Keep factual feedback such as an acquisition attempt succeeding; keep uncertainty when evidence is ambiguous. Historical facts are not current object poses. After every action or invalid mutation, discard current geometry and result IDs and read the new snapshot.

Required capability: current_rgb. Optional: candidate_category, relative_depth, relative_points. Tools: view_candidates, view_depth_summary, point_depth, pair_point_distance, candidate_depth_summary. Candidate detections are incomplete; nearby instances are not automatically duplicates. Independent mask quality is absent, so `candidate_depth_summary` remains denied. No metric/world geometry, cross-view registration, oracle object state, environment reset/step, file/network/process access or budget expansion is available. Missing evidence is unavailable, never a fabricated numeric measurement.
