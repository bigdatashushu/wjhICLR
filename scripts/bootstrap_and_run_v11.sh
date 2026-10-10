#!/usr/bin/env bash
# Clone one immutable code revision, build a GPU environment, and run v11.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/bigdatashushu/wjhICLR.git}"
CODE_REF="${CODE_REF:-}"
WORK_DIR="${WORK_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  CODE_REF=<40-char-git-commit> bash scripts/bootstrap_and_run_v11.sh \
    --work-dir /data/skill3d-v11 \
    [--config /absolute/path/gpu_experiment_v11.yaml] \
    [--run-id NAME] [--download-only] [--dry-run]

The work directory contains source/, venv/, weights/, runs/, and environment
receipts. Existing source checkouts are reused only when HEAD equals CODE_REF.
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
    --run-id|--download-only|--dry-run)
      EXTRA_ARGS+=("$1")
      if [[ "$1" == "--run-id" ]]; then
        EXTRA_ARGS+=("$2")
        shift 2
      else
        shift
      fi
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

if [[ ! "$CODE_REF" =~ ^[0-9a-f]{40}$ ]]; then
  echo "CODE_REF must be an immutable 40-character lowercase Git commit." >&2
  exit 2
fi
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

if [[ -e "$SOURCE_DIR" && ! -d "$SOURCE_DIR/.git" ]]; then
  echo "Refusing to overwrite non-Git source directory: $SOURCE_DIR" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  mkdir -p "$SOURCE_DIR"
  git -C "$SOURCE_DIR" init
  git -C "$SOURCE_DIR" remote add origin "$REPO_URL"
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$CODE_REF"
  git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
fi

ACTUAL_REF="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_REF" != "$CODE_REF" ]]; then
  echo "Existing source HEAD $ACTUAL_REF differs from requested $CODE_REF." >&2
  exit 2
fi
if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
  echo "Source checkout is dirty; formal bootstrap requires an unmodified tree." >&2
  exit 2
fi

if [[ -z "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$SOURCE_DIR/configs/gpu_experiment_v11.yaml"
elif [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
MARKER="$ENV_RECEIPT_DIR/installed_contract_sha256.txt"
INSTALL_ID="$("$PYTHON" - "$CODE_REF" "$CONFIG_PATH" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(sys.argv[1].encode() + b"\n" +
                     pathlib.Path(sys.argv[2]).read_bytes()).hexdigest())
PY
)"

if [[ ! -f "$MARKER" || "$(cat "$MARKER")" != "$INSTALL_ID" ]]; then
  mkdir -p "$ENV_RECEIPT_DIR"
  "$PIP" install --upgrade pip setuptools wheel
  "$PIP" install "pyyaml>=6,<7"

  VLLM_VERSION="$("$PYTHON" - "$CONFIG_PATH" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["environment"]["vllm_version"])
PY
)"
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

  "$PIP" install "vllm==$VLLM_VERSION"
  "$PIP" install -e "$SOURCE_DIR"
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
import vllm
if not torch.cuda.is_available():
    raise SystemExit("torch installed, but CUDA is not available")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
PY
  printf '%s\n' "$INSTALL_ID" > "$MARKER"
  "$PIP" freeze | LC_ALL=C sort > "$ENV_RECEIPT_DIR/pip_freeze.txt"
  "$PYTHON" - "$ENV_RECEIPT_DIR/pip_freeze.txt" \
    > "$ENV_RECEIPT_DIR/pip_freeze.sha256" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
fi

export PYTHONPATH="$SOURCE_DIR/src:$SOURCE_DIR/third_party/vggt${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "$SOURCE_DIR/scripts/run_gpu_experiment_v11.py" \
  --config "$CONFIG_PATH" \
  --work-dir "$WORK_DIR" \
  "${EXTRA_ARGS[@]}"
