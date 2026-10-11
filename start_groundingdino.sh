#!/usr/bin/env bash
# Use the detector environment rather than the online solver's Python packages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTOR_PYTHON="${SKILL3D_DETECTOR_PYTHON:-/home/cvailab/anaconda3/envs/skill3d/bin/python}"
if [[ ! -x "$DETECTOR_PYTHON" ]]; then
  printf 'GroundingDINO Python not found: %s; set SKILL3D_DETECTOR_PYTHON.\n' \
    "$DETECTOR_PYTHON" >&2
  exit 1
fi
exec "$DETECTOR_PYTHON" "$ROOT/scripts/serve_groundingdino.py" "$@"
