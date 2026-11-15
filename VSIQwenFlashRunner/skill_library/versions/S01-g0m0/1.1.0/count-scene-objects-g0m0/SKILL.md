---
name: count-scene-objects-g0m0
description: Method for object_counting in initial evidence state g0m0.
---

Initial geometry_3d=unavailable and metric_scale=unavailable.

# Count whole instances across views

1. Fix the category, scope and exclusions. Before consulting candidate totals, enumerate whole instances from delivered images. For each keep a brief representative frame/location anchor and appearance or neighboring structure. Do not infer hidden instances without evidence.
2. Keep objects coexisting in one frame separate. Match reappearances using spatial continuity plus appearance/neighbors; similar appearance alone is insufficient. Check tracks and geometric overlap against images. Seams, legs, attachments and nested components alone do not establish additional whole objects.
3. Revisit missed regions, occlusions, ambiguous category membership and uncertain exclusions that could change the total. Explain inclusion/exclusion using the best view. Distinguish unseen coverage from confirmed absence; a technical scan or usable binding does not certify complete real-instance coverage.
4. If current images settle identity and coverage, answer with zero tools. Otherwise identify the exact doubt and choose the relevant authorized evidence: an existing candidate page, representative crop, or target-category detection on unresolved views. A full-frame scan is justified only by a coverage gap requiring it, not as a default.
5. When reading inventory, keep fields fixed (obj_id,category_name,visible_frames,track_id,duplicate_suspect as needed); follow returned cursors/cumulative coverage, not offset arithmetic. Read existing IDs instead of relisting all objects or removing fields after completion. If mapping the inventory matters, map each relevant row to a whole instance or an explained duplicate/exclusion.
6. count_objects groups tracks and consolidates geometry; count, list length, track totals and detection totals remain diagnostics until checked against whole instances. Do not sum per-frame boxes or add new boxes directly to an old count. New 2D boxes have no 3D cloud/track; mask visibility may be unavailable. Grounded geometry can aid identity when authorized; counting requires no metric scale.
7. A fault, unhealthy detector or partial frame error provides no negative evidence. It cannot resolve an omission or disagreement with an old count. Do not repeat a failed service/binding premise unchanged; use a different available view or valid evidence. If uncertainty remains, submit the best image-supported estimate with honest basis.

Before submission, leave brief English comments identifying instances, coverage checked and material exclusions. Count each whole once; recheck possible overmerging, fragmentation and whole/part splits. Use the host's actual evidence basis and valid IDs. MeasurementAnswer only reads/replays a count; it does not establish category, identity or completeness. The method stays fixed; current tool authorization and evidence withdrawals always apply.
