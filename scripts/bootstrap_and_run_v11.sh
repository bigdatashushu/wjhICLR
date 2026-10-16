#!/usr/bin/env bash
# Freeze development source or clone a formal revision, then run v11.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/bigdatashushu/wjhICLR.git}"
CODE_REF="${CODE_REF:-}"
WORK_DIR="${WORK_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=""
DEV=0
SMOKE=0
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  CODE_REF=<40-char-git-commit> bash scripts/bootstrap_and_run_v11.sh \
    --work-dir /data/skill3d-v11 \
    [--config /absolute/path/gpu_experiment_v11.yaml] \
    [--run-id NAME] [--dev] [--smoke] [--download-only] [--dry-run]

The work directory contains source/, venv/, weights/, runs/, and environment
receipts. Development runs freeze this working tree, including uncommitted
changes. Formal source checkouts are reused only when HEAD equals CODE_REF.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --dev)
      DEV=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      EXTRA_ARGS+=(--dry-run)
      shift
      ;;
    --download-only)
      EXTRA_ARGS+=(--download-only)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$WORK_DIR" ]]; then
  echo "--work-dir is required." >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

WORK_DIR="$(mkdir -p "$WORK_DIR" && cd "$WORK_DIR" && pwd)"
SOURCE_DIR="$WORK_DIR/source"
VENV_DIR="$WORK_DIR/venv"
ENV_RECEIPT_DIR="$WORK_DIR/environment"
RUN_ID="${RUN_ID:-v11-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run ID." >&2; exit 2; }
MODE_ARGS=()
[[ "$DEV" -eq 1 ]] && MODE_ARGS+=(--dev)
[[ "$SMOKE" -eq 1 ]] && MODE_ARGS+=(--smoke)
[[ "$DRY_RUN" -eq 1 ]] && MODE_ARGS+=(--dry-run)
if [[ -z "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$ROOT/configs/gpu_experiment_v11.yaml"
else
  CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
fi
MODE="$("$PYTHON_BIN" "$ROOT/scripts/prepare_gpu_run_v11.py" \
  --action mode --config "$CONFIG_PATH" ${MODE_ARGS[@]+"${MODE_ARGS[@]}"})"

if [[ "$MODE" == "development" ]]; then
  SOURCE_DIR="$ROOT"
  CODE_REF="${CODE_REF:-$(git -C "$ROOT" rev-parse HEAD)}"
else
  [[ "$CODE_REF" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "CODE_REF must be an immutable 40-character lowercase commit." >&2; exit 2; }
  if [[ -e "$SOURCE_DIR" && ! -e "$SOURCE_DIR/.git" ]]; then
    echo "Refusing to overwrite non-Git source directory: $SOURCE_DIR" >&2
    exit 2
  fi
  if [[ ! -e "$SOURCE_DIR/.git" ]]; then
    mkdir -p "$SOURCE_DIR"
    git -C "$SOURCE_DIR" init
    git -C "$SOURCE_DIR" remote add origin "$REPO_URL"
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$CODE_REF"
    git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
  fi
fi

PREPARED="$("$PYTHON_BIN" "$ROOT/scripts/prepare_gpu_run_v11.py" \
  --source-root "$SOURCE_DIR" --code-ref "$CODE_REF" --config "$CONFIG_PATH" \
  --work-dir "$WORK_DIR" --run-id "$RUN_ID" ${MODE_ARGS[@]+"${MODE_ARGS[@]}"})"
SOURCE_DIR="$(printf '%s\n' "$PREPARED" | sed -n '1p')"
CONFIG_PATH="$(printf '%s\n' "$PREPARED" | sed -n '2p')"
CODE_RECEIPT="$(printf '%s\n' "$PREPARED" | sed -n '3p')"
MODE="$(printf '%s\n' "$PREPARED" | sed -n '4p')"
export SKILL3D_RUN_MODE="$MODE"
export PYTHONPATH="$SOURCE_DIR/src:$SOURCE_DIR/third_party/vggt"
export PYTHONDONTWRITEBYTECODE=1
EXTRA_ARGS+=(--run-id "$RUN_ID")
if [[ "$CODE_RECEIPT" != "-" ]]; then
  export SKILL3D_CODE_RECEIPT="$CODE_RECEIPT"
  EXTRA_ARGS+=(--code-receipt "$CODE_RECEIPT")
else
  unset SKILL3D_CODE_RECEIPT
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  # No venv creation, pip, model imports, downloads or GPU operations.
  exec "$PYTHON_BIN" "$SOURCE_DIR/scripts/run_gpu_experiment_v11.py" \
    --config "$CONFIG_PATH" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
fi

if [[ -n "${HARNESS3D_EXISTING_PYTHON:-}" ]]; then
  # Opt-in reuse of the pinned server environment; never pip-modify it.
  PYTHON="$HARNESS3D_EXISTING_PYTHON"
  [[ -x "$PYTHON" ]] || { echo "Existing Python is not executable: $PYTHON" >&2; exit 2; }
  "$PYTHON" - "$SOURCE_DIR" "$CONFIG_PATH" "$ENV_RECEIPT_DIR/$RUN_ID-existing.json" <<'PY'
import pathlib, sys, yaml
from skill3d.infra.experiment_startup import validate_existing_environment, write_json
receipt = validate_existing_environment(
    pathlib.Path(sys.argv[1]), yaml.safe_load(pathlib.Path(sys.argv[2]).read_text()))
write_json(pathlib.Path(sys.argv[3]), receipt)
PY
else
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
MARKER="$ENV_RECEIPT_DIR/installed_contract_sha256.txt"
INSTALL_ID="$("$PYTHON_BIN" - "$SOURCE_DIR" "$CONFIG_PATH" <<'PY'
import pathlib, sys, yaml
from skill3d.infra.experiment_startup import environment_contract
print(environment_contract(pathlib.Path(sys.argv[1]), yaml.safe_load(
    pathlib.Path(sys.argv[2]).read_text()))["sha256"])
PY
)"

if [[ ! -f "$MARKER" || "$(cat "$MARKER")" != "$INSTALL_ID" ]]; then
  mkdir -p "$ENV_RECEIPT_DIR"
  "$PIP" install --upgrade pip setuptools wheel
  "$PIP" install "pyyaml>=6,<7"

  SAM2_REVISION="$("$PYTHON" - "$CONFIG_PATH" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["environment"]["sam2_code_revision"])
PY
)"
  MOGE_ENABLED="$("$PYTHON" - "$CONFIG_PATH" <<'PY'
import sys, yaml
print("1" if yaml.safe_load(open(sys.argv[1]))["weights"]["moge2"]["enabled"] else "0")
PY
)"
  MOGE_REVISION="$("$PYTHON" - "$CONFIG_PATH" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["environment"]["moge_code_revision"])
PY
)"

  # Install declarations, not an editable pointer to an old run's snapshot.
  DEPENDENCIES=()
  while IFS= read -r dependency; do DEPENDENCIES+=("$dependency"); done < <(
    "$PYTHON" - "$SOURCE_DIR/pyproject.toml" <<'PY'
import pathlib, sys, tomllib
for requirement in tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]["dependencies"]:
    print(requirement)
PY
  )
  PROJECT_PIP_ARGS=("${DEPENDENCIES[@]}")
  for requirements in "$SOURCE_DIR"/requirements*.txt; do
    [[ ! -f "$requirements" ]] || PROJECT_PIP_ARGS+=(-r "$requirements")
  done
  for constraints in "$SOURCE_DIR"/constraints*.txt; do
    [[ ! -f "$constraints" ]] || PROJECT_PIP_ARGS+=(-c "$constraints")
  done
  "$PIP" install "${PROJECT_PIP_ARGS[@]}"
  # The vendored VGGT metadata still declares numpy<2, while this repository
  # intentionally freezes numpy>=2 with a compatible scipy. Import it via
  # PYTHONPATH and install only its missing runtime dependencies.
  "$PIP" install Pillow einops safetensors huggingface_hub
  "$PIP" install \
    "git+https://github.com/facebookresearch/sam2.git@$SAM2_REVISION"
  if [[ "$MOGE_ENABLED" == "1" ]]; then
    "$PIP" install \
      "git+https://github.com/microsoft/MoGe.git@$MOGE_REVISION"
  fi

  "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch installed, but CUDA is not available")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
  printf '%s\n' "$INSTALL_ID" > "$MARKER"
  "$PIP" freeze | LC_ALL=C sort > "$ENV_RECEIPT_DIR/pip_freeze.txt"
  "$PYTHON" - "$ENV_RECEIPT_DIR/pip_freeze.txt" \
    > "$ENV_RECEIPT_DIR/pip_freeze.sha256" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
fi
fi  # HARNESS3D_EXISTING_PYTHON

# Always preflight the reused environment; an install marker is not a health check.
"$PYTHON" - <<'PY'
import torch
from skill3d.env_preflight import assert_runtime_dependencies
assert_runtime_dependencies(context="gpu_bootstrap")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the reused GPU environment")
PY
exec "$PYTHON" "$SOURCE_DIR/scripts/run_gpu_experiment_v11.py" \
  --config "$CONFIG_PATH" \
  --work-dir "$WORK_DIR" \
  "${EXTRA_ARGS[@]}"
