# Current-view navigation

Method ID: embodied.navigation.current-view. Version: 1.0.0.
Environment: EB-Navigation. Control profile: embodied-actions/1.

Navigate toward the object named by the instruction through short, observable decisions. The host executes one official action and then gives a new current RGB. Choose action IDs from the current action table; names and IDs can only be interpreted through that table.

## Decision procedure

1. Read the instruction, current action table, official feedback and sourced memory. Identify the target category and the current navigation subgoal. Describe evidence from the current image without assuming hidden coordinates or simulator state.
2. Use `view_candidates(view_index=0)` when detection evidence is available. Candidate boxes give image regions; category confirmation helps identify the target. Detections are not exhaustive, neighboring objects are not automatically duplicates, and boxes do not establish object surfaces.
3. If the target is visible, compare its image position with the forward direction. Prefer a small turn when it is off center. Once aligned, choose a forward action when the current image indicates open space. When useful and available, sample `point_depth(0,x,y)` at explicitly selected visible pixels. Values are 5x5 local medians in relative camera units and may compare pixels only within this view.
4. If the target is absent, choose a turn or a short move that exposes another region. Use bounded sourced memory to avoid repeatedly choosing the same unsuccessful action. A failed official action is evidence to change the next choice, even when the RGB is unchanged.
5. Check the chosen action against the table and submit it through `SubmitActions(ActionPayload(...))`. If the task appears complete or no productive legal action remains, submit an empty list. Only the host's official evaluation determines success.

## Program and evidence

Define `solve(ctx)`. Tool functions are available as top-level names or `ctx.tools.<name>`. Tool replies are envelopes: inspect `r["data"]`, and add `r["result_id"]` to `evidence_refs`. `ReadObservation(result_id, fields, offset, limit)` can retrieve current-decision results retained from a previous segment. `YieldObservations(result_ids, reason)` ends a segment so the same coding agent can inspect bounded observations. It does not execute an action.

Construct the typed payload with `ctx.decision_id`, `ctx.observation_revision`, `ctx.snapshot_sha256`, `ctx.action_table_sha256`, the action list and current result IDs. Submission terminates immediately. No program variables persist into the next segment; after a mutation no old geometry or tool result is current, even if the image repeats.

Memory is a short subgoal and at most 16 records. Each record has text, source (`visual` or `official_feedback`), episode_id, observation_revision, decision_id and confidence (`observed` or `uncertain`). Historical records retain their original source identity. Record tested actions and task progress as public observations, never as a fabricated world map.

Required capability: current_rgb. Optional capabilities: candidate_category, relative_depth, relative_points. Tools: view_candidates, view_depth_summary, point_depth, pair_point_distance, candidate_depth_summary. `pair_point_distance` is restricted to two visible pixels in the same view and relative units. `candidate_depth_summary` requires independent mask validation and is unavailable in this pipeline. Category review cannot authorize surfaces, metric depth, cross-view distances or world coordinates. Missing evidence means unavailable, never a zero distance. The method cannot expand budgets or call environment step/reset, files, network or processes.
