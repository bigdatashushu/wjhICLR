# harness3D v9 Skill Library

This directory is the repository-local Skill library defined by the v9 architecture.

## Source of truth

Edit only `skills/<skill-name>/SKILL.md`. Each source has a YAML front matter and the fixed Markdown sections required by source format `1.0`. A source version is immutable after it has entered a released snapshot. A method change creates a new semantic version and a candidate record; it must not overwrite the S0 source.

`imports/` contains the verified transport bundle and its receipt. The bundle keeps the original document-level export identity (`schema_version=8.0`); it is not the runtime schema.

## Derived files

Run from the repository root:

```bash
EXP=/home/cvailab/anaconda3/envs/skill3d-exp/bin/python   # 冻结实验环境，必须用它
PYTHONPATH=src $EXP scripts/build_skill_library.py --check
```

Do not run these commands with the base conda env: its numpy/scipy pair is incompatible, and the failure there is silent rather than loud (M4 sub-items fail closed and the scene route degrades to `fallback_2d_only`). The online entry point and the test suite therefore refuse to start when the dependency probes fail.

The compiler parses and validates the sources, then maps the document-level S0 format into the current strict runtime `skill3d.schemas.SkillSpec` contract. Generated specs are written to `generated/<skill_id>/<version>.json`, with a digest index in `generated/index.json`. Do not hand-edit generated files.

The compiler intentionally keeps `image_2d` as the only whole-Skill requirement. Geometry, scale, world-frame, binding and temporal requirements remain local method conditions and are enforced by the runtime ToolSpec/evidence authorization path. This preserves the v9 visual fallback when an optional capability is unavailable.

## Snapshots and candidates

`snapshots/active_snapshot.json` is the only online pointer. The online loader reads only the inline `spec_content` of the pointed immutable snapshot. It never scans `generated/`, `candidates/`, or `candidates/future/`.

- `snapshots/snapshot_<id>.json`: immutable promoted snapshot.
- `candidates/`: immutable records for real, trace-backed revisions that may be validated.
- `candidates/future/`: drafts and ideas; never active and never eligible for online retrieval.
- `manifests/`: source, generated, compiler, snapshot and validation identities.
- `validation/`: static import/compile receipts. They do not claim real model/tool execution.

Promotion must use the existing atomic snapshot writer. A rejected candidate leaves the active pointer unchanged; a promoted candidate is loaded only by new episodes. S0 is `S0-seed-20260925-v1` and is a non-empty document/static seed, not a claim of runtime verification, benchmark score, real evolution or E-S0 gain.

## Data and provenance rules

Candidate records must retain the canonical question type, parent Skill version, source split, experience relation, trace references, patch/diff, expected scope, known risks and static-check receipt. Learning traces may inform a future candidate only when the Skill was actually retrieved and delivered; inner/outer/final evaluation material must not be copied into online prompts or candidate drafts.

The runtime snapshot stores only public method content and audit digests. Ground-truth labels, private credentials, offline provider output and answer-model secrets do not belong here.
