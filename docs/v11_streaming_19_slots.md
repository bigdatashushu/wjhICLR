# v11 19-Slot Streaming Contract

## Reachable slots

The production plan contains 19 reachable `(question_type, evidence_state)`
slots:

- all 8 tasks in `g0m0`;
- all 8 tasks in `g1m0`;
- `object_abs_distance`, `object_size_estimation`, and
  `room_size_estimation` in `g1m1`.

Non-metric `g1m1` combinations are rejected before data loading. The canonical
machine-readable list is `configs/v11/reachable_slots_v11.json`.

## Frozen quotas

Each slot receives:

- 12 induction episodes;
- 4 fresh post-publish induction episodes;
- 8 inner-validation pairs;
- 8 outer-holdout pairs.

This produces 304 induction/post questions, 152 inner questions, and 152 outer
questions. `final_test` is not loaded by this workflow.

Selection is deterministic scene round-robin. A slot uses at most one question
from a scene in each panel. Post-publish questions cannot reuse a scene from
their slot's current induction buffer or prior post-publish history. Across
different task slots, a scene may contribute at most eight selected questions.

## Prepared bindings

`PreparedBindingStore` stores question-specific masks, object point clouds, and
receipts under a content-addressed key. It references the immutable scene
reconstruction by its validated input identity and does not copy reconstruction
arrays.

Questions selected from the same scene are prepared as one batch, with at most
eight questions. Cross-question semantic-mask conflict auditing runs before any
binding in that scene batch is published. A later single-slot evaluation loads
the sealed full-scene audit group and must not narrow or rewrite it.

E11 creates hard links for immutable scene arrays and frame payloads. It fails
if hard links cannot be created; it does not silently duplicate geometry.

M5 scans all readable frames with a joint category prompt. Unassignable labels
trigger single-category rescans of that frame. Geometry-derived complete-link
groups select up to two separated views for semantic review. Unreviewed masks
are excluded from final object geometry, so observed visibility is conservative.
Appearance-order tasks retain full per-view review to preserve temporal evidence.
This is a new preparation version and cannot reuse the old binding cache.

Large asset SHA256 values are reused within a process only while the resolved
file's device, inode, size, modification time, and change time match. File edits
or replacement invalidate the entry; first reads still hash complete content.
`binding_progress.json` records the current stage and completed stage timings.

Native Intern review can use a bounded `mask_review.endpoints` list. Every
replica pins its full health identity; all identity fields except the recorded
device placement must match. Ordered round-robin batches allow one in-flight
request per replica. Original request/image identities, probabilities, threshold
and admission rules remain in each receipt. Failed reviews stay unavailable.
The pool does not retry semantic decisions on another model.

`scripts/prepare_question_pool_v11.py` prepares an input pool by complete scene
groups, selecting one question per canonical task and retaining original split
assignments. It reuses verified scene geometry, seals cross-question audits,
checks payload replay, and writes a cumulative manifest after each scene. This
pool is an input supply; exact-slot 12/4/8/8 panels still need separate selection
and evidence-state coverage checks. Outer preparation runs separately and
`final_test` cannot be selected.

Before each scene, the pool runner requires the 80 GiB reserve plus a working
allowance (8 GiB by default, increased from observed scene storage when needed).
It records a capacity stop without modifying existing bindings. Resume requires
the identical source, config, metadata and selection identities.

## Inner and outer roles

`inner_validation` is the only split allowed to produce a promotion receipt.
Its parent and candidate arms share the same prepared input and are compared by
the campaign admission rules.

`outer_holdout` runs only after the final active snapshot ID is frozen. It
compares `S0-v11-task-tools` with that final snapshot and writes
`outer_report.json`. The report declares `feeds_revision=false` and
`can_promote=false`; no outer result enters revision, admission, or publication.

## Production entry

The GPU launcher reconstructs `induction`, `inner_validation`, and
`outer_holdout` with up to eight questions per scene. For evolution it invokes
the streaming CLI with:

- the 19-slot manifest;
- binding precomputation and mandatory reuse;
- 12/4/8/8 quotas;
- report-only outer evaluation after streaming.

The runtime keeps the configured 80 GiB free-space reserve. Binding preparation
should run in rolling scene batches when the remaining capacity cannot cover
the complete frozen panel.
