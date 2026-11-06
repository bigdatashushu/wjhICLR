# Current-view Habitat rearrangement

Method ID: embodied.habitat.current-view. Version: 1.0.0.
Environment: EB-Habitat. Control profile: embodied-actions/1.

Solve the rearrangement instruction through current observable subgoals: identify the required object and destination, acquire the object, then place or interact according to the instruction. The host executes one official action before supplying a fresh current RGB and feedback.

## Decision procedure

1. Identify the next task subgoal from the instruction and sourced memory. Read official feedback before inferring whether navigation, grasping, carrying or placement succeeded. Visual resemblance alone does not prove task completion.
2. Read the current action table. High-level navigation or manipulation actions may reposition the camera without a known rigid transform. Choose the exact listed ID and plan to inspect the next image rather than continuing with predicted old coordinates.
3. Use `view_candidates(0)` when useful for locating visible categories or distinguishing destination regions. A candidate box is a detected image region; category confirmation supports category identity and does not validate its surface, physical instance count or reachability.
4. When current relative depth exists, use `point_depth(0,x,y)` on visible pixels to compare image regions in this view. The tool uses a finite 5x5 local median. Use `pair_point_distance` only between visible pixels of this same camera view. Never treat these values as meters or compare scales across actions.
5. Choose an available action that advances the subgoal under the current evidence. When official feedback reports failure, revise the selection or seek another view. When feedback confirms acquisition or placement, preserve that public fact with its observation identity and move to the next supported subgoal.
6. Submit one legal action, or an empty list when the instruction appears complete or no productive action remains. Official evaluator state belongs to the host.

## Execution and state

Define `solve(ctx)`. Tool functions are available at top level or as `ctx.tools.<name>`. Read tool envelopes through `r["data"]` and cite `r["result_id"]`. Use bounded `ReadObservation` only for IDs retained in this decision; `YieldObservations` ends a segment for the same coding agent to inspect selected observations. Each new segment clears program variables.

Submit with `SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision, ctx.snapshot_sha256, ctx.action_table_sha256, actions, evidence_refs, memory_update))`. The host validates the full list, applies its first action, and revokes all prior current evidence. Submission terminates immediately. Official invalid-action accounting also produces a new observation revision even if no step or new RGB occurs.

Memory is a short subgoal and at most 16 records with text, source (`visual` or `official_feedback`), episode_id, observation_revision, decision_id and confidence (`observed` or `uncertain`). Preserve source attribution and uncertainty. Carry historical action outcomes forward without converting them into current object coordinates, distances or world identities.

Required capability: current_rgb. Optional: candidate_category, relative_depth, relative_points. Tools: view_candidates, view_depth_summary, point_depth, pair_point_distance, candidate_depth_summary. Detection coverage is incomplete, and suspected duplicates may be distinct neighbors. No independent mask quality receipt exists, so `candidate_depth_summary` is denied. Missing geometry is unavailable, not zero. The method cannot access metric/world geometry, oracle poses, files, network, processes, simulator reset/step or expanded budgets.
