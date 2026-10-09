# harness3D Skill Library

This directory is the repository-local Skill library. The active snapshot is
`S0-v11-contract-repair`; the v9 source/compiler artifacts remain available for
historical checks and rollback.

## Source of truth

Legacy v9 sources live in `skills/<skill-name>/SKILL.md`, with YAML front matter
and the fixed Markdown sections required by source format `1.0`. Active v11 sources
live in `versions/` as described below. A released source version is immutable;
a method change creates a new semantic version and candidate record.

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

`snapshots/active_snapshot.json` is the only online pointer. The online loader reads
the pointed immutable snapshot: inline `spec_content` for v9, or verified
`source_ref` content for v11. It never scans `generated/`, `candidates/`, or
`candidates/future/`.

- `snapshots/snapshot_<id>.json`: immutable promoted snapshot.
- `candidates/`: immutable records for real, trace-backed revisions that may be validated.
- `candidates/future/`: drafts and ideas; never active and never eligible for online retrieval.
- `manifests/`: source, generated, compiler, snapshot and validation identities.
- `validation/`: static import/compile receipts. They do not claim real model/tool execution.

Promotion must use the existing atomic snapshot writer. A rejected candidate leaves the active pointer unchanged; a promoted candidate is loaded only by new episodes. The legacy S0 `S0-seed-20260925-v1` remains available for rollback. Activating the v11 format migration does not claim a benchmark score or Skill evolution gain.

## v11 active snapshot

The v11 path keeps each immutable complete source under
`versions/<skill_id>/<version>/<name>/SKILL.md`. Runtime identity and the source
digest live in a schema `runtime-skill-snapshot/2.0` entry; the loader verifies
the manifest, source path, UTF-8/LF form, and full-source hash before returning
`SkillSpecV11`. Delivery uses that exact `skill_md` string without recompiling
or reconstructing sections.

`S0-v11-format-migration` contains the eight v11 `1.1.0` methods, with exactly
one active method for each canonical question type. The earlier
`S0-v11-s03-format-migration` snapshot is retained as the S03 vertical migration
preview. The repository active pointer references `S0-v11-contract-repair`,
whose parent is the format-migration snapshot. S01/S08 are now 1.2.0 with
explicit parent-version references; the other six sources remain 1.1.0.
All previous snapshots and source bytes are preserved. This is a contract-repair
baseline (generation 0), not an evidence-backed evolution gain.

Build or verify the snapshot with:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_skill_library_v11.py
PYTHONPATH=src .venv/bin/python scripts/build_skill_library_v11.py --check
PYTHONPATH=src .venv/bin/python scripts/build_skill_library_v11.py --profile contract-repair --check
```

`--profile contract-repair --activate` selects the repaired snapshot through the atomic pointer writer
after validation. Normal benchmark commands do not change the pointer.

## v11 public prompt

The online runtime has one contract: `program_synth_v11_2`,
`solver-v11.2-m11-acceptance`, and `tool-docs-v11.1`. These identities are
recorded in every run but are not selectable through YAML, CLI, or
`OnlineRunConfig`. B01 and B11 therefore share the same public task definitions,
unit rules, execution contract, and authorized Tool documentation. B11 appends
the complete selected Skill source as a method block; B01 appends none.

The control contract documents `AnswerPayload`, real result IDs, pending
submissions, and M11 rejection/revision. S06 no longer promises a nonexistent
angle field; visibility docs describe stored frame slots. S01 groups tracks
before geometric merging; S08 uses horizontal projections and valid
world-up/handedness metadata from `object_centroid`.

Historical prompt and Tool-document behavior is available only by checking out
the corresponding Git commit. Current tool behavior remains identified as
`tool-face-v11.1`.

## v11 paired Skill ablation

`scripts/run_skill_ablation_v11.py` runs B01 (empty delivery) and B11 (the explicit
`S0-v11-contract-repair` snapshot by default) with the same v11 public prompt and an always-on
quality gate. It freezes inputs and M5 results, gives each arm private runtime
and artifact copies, and writes qa_id-paired scores and delivery hashes.
See [运行命令、隔离合同与输出说明](../docs/skill_ablation_v11.md).
This entry does not change the active pointer or use v10 fixed injection.

## Data and provenance rules

Candidate records must retain the canonical question type, parent Skill version, source split, experience relation, trace references, patch/diff, expected scope, known risks and static-check receipt. Learning traces may inform a future candidate only when the Skill was actually retrieved and delivered; inner/outer/final evaluation material must not be copied into online prompts or candidate drafts.

The runtime snapshot stores only public method content and audit digests. Ground-truth labels, private credentials, offline provider output and answer-model secrets do not belong here.
