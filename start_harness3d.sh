#!/usr/bin/env bash
# One-command launcher for the pinned harness3D GPU evaluation pipeline.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEFAULT_CONFIG="$ROOT/configs/v11/gpu_experiment_v11.local.yaml"
DEFAULT_WORK_DIR="/home/cvailab/experiments/harness3d-v11"

CONFIG_SOURCE="${HARNESS3D_CONFIG:-$DEFAULT_CONFIG}"
WORK_DIR="${HARNESS3D_WORK_DIR:-$DEFAULT_WORK_DIR}"
RUN_ID="${HARNESS3D_RUN_ID:-harness3d-v11-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CODE_REF="${CODE_REF:-}"
REPO_URL="${REPO_URL:-}"

QWEN_GPU="${HARNESS3D_QWEN_GPU:-2}"
GEOMETRY_GPU="${HARNESS3D_GEOMETRY_GPU:-3}"
RESERVE_GPU="${HARNESS3D_RESERVE_GPU:-4}"
MIN_FREE_GPU_MIB="${HARNESS3D_MIN_FREE_GPU_MIB:-18000}"
DETECTOR_ENDPOINT="${HARNESS3D_DETECTOR_ENDPOINT:-http://127.0.0.1:20022}"
REQUIRE_DETECTOR="${HARNESS3D_REQUIRE_DETECTOR:-1}"

DRY_RUN=0
DOWNLOAD_ONLY=0
SKIP_DETECTOR_CHECK=0
SMOKE=0

usage() {
  cat <<'EOF'
Usage:
  ./start_harness3d.sh [options]

GPU plan:
  GPU 2  Qwen3-VL / vLLM
  GPU 3  VGGT reconstruction + SAM2
  GPU 4  reserved for GroundingDINO or failover

Options:
  --config PATH          GPU experiment YAML
  --work-dir PATH        weights, environment and run output root
  --run-id NAME          unique run identity
  --smoke                non-formal 2+2 sample end-to-end GPU check
  --dry-run              build receipts and print the pipeline without execution
  --download-only        prepare pinned weights without running the experiment
  --skip-detector-check  continue when the detector endpoint is not listening
  -h, --help             show this help

Environment overrides:
  HARNESS3D_QWEN_GPU, HARNESS3D_GEOMETRY_GPU, HARNESS3D_RESERVE_GPU
  HARNESS3D_DETECTOR_ENDPOINT, HARNESS3D_REQUIRE_DETECTOR
  HARNESS3D_MIN_FREE_GPU_MIB, HARNESS3D_CONFIG, HARNESS3D_WORK_DIR
  QWEN_EXISTING_PATH, VGGT_EXISTING_PATH, SAM2_EXISTING_PATH, SAM2_CONFIG
  CODE_REF, REPO_URL, PYTHON_BIN

Formal runs require a committed and pushed CODE_REF, licensed VSI-Bench videos,
and a quality confirmation matching the current code contract.
EOF
}

die() {
  printf 'start_harness3d.sh: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || die "--config requires a path"
      CONFIG_SOURCE="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "--work-dir requires a path"
      WORK_DIR="$2"
      shift 2
      ;;
    --run-id)
      [[ $# -ge 2 ]] || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    --skip-detector-check)
      SKIP_DETECTOR_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -d "$ROOT/.git" ]] || die "repository metadata not found at $ROOT"
command -v git >/dev/null 2>&1 || die "git is required"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
[[ -f "$CONFIG_SOURCE" ]] || die "GPU config not found: $CONFIG_SOURCE"
[[ "$QWEN_GPU" =~ ^[0-9]+$ ]] || die "HARNESS3D_QWEN_GPU must be an integer"
[[ "$GEOMETRY_GPU" =~ ^[0-9]+$ ]] || die "HARNESS3D_GEOMETRY_GPU must be an integer"
[[ "$RESERVE_GPU" =~ ^[0-9]+$ ]] || die "HARNESS3D_RESERVE_GPU must be an integer"
[[ "$QWEN_GPU" != "$GEOMETRY_GPU" ]] || die "Qwen and geometry must use different GPUs"
[[ "$RESERVE_GPU" != "$QWEN_GPU" && "$RESERVE_GPU" != "$GEOMETRY_GPU" ]] \
  || die "reserved GPU must differ from Qwen and geometry GPUs"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "run ID may contain only letters, digits, dot, underscore and hyphen"

if [[ -z "$CODE_REF" ]]; then
  CODE_REF="$(git rev-parse HEAD)"
fi
[[ "$CODE_REF" =~ ^[0-9a-f]{40}$ ]] || die "CODE_REF must be a 40-character lowercase commit"
if [[ -n "$(git status --porcelain)" ]]; then
  die "working tree is dirty; commit and push all changes before a pinned GPU run"
fi

if [[ -z "$REPO_URL" ]]; then
  REPO_URL="$(git remote get-url origin 2>/dev/null || true)"
fi
[[ -n "$REPO_URL" ]] || die "REPO_URL is empty and origin is not configured"

WORK_DIR="$(mkdir -p "$WORK_DIR" && cd "$WORK_DIR" && pwd)"
CONFIG_SOURCE="$(cd "$(dirname "$CONFIG_SOURCE")" && pwd)/$(basename "$CONFIG_SOURCE")"
GENERATED_DIR="$WORK_DIR/launcher_configs"
LOG_DIR="$WORK_DIR/launcher_logs"
mkdir -p "$GENERATED_DIR" "$LOG_DIR"
GENERATED_CONFIG="$GENERATED_DIR/${RUN_ID}.yaml"
LOG_PATH="$LOG_DIR/${RUN_ID}.log"

use_existing_dir() {
  local variable="$1"
  local candidate="$2"
  if [[ -z "${!variable:-}" && -d "$candidate" ]]; then
    printf -v "$variable" '%s' "$candidate"
    export "$variable"
  fi
}

use_existing_dir QWEN_EXISTING_PATH \
  "/home/cvailab/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct-FP8"
use_existing_dir VGGT_EXISTING_PATH \
  "/home/cvailab/.cache/huggingface/geothinker/VGGT-1B"
use_existing_dir SAM2_EXISTING_PATH \
  "/home/cvailab/.cache/huggingface/hub/models--facebook--sam2.1-hiera-large/snapshots/227a114a2f535cd147f82442e7d2038cdd2e5d68"
export SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

"$PYTHON_BIN" - "$CONFIG_SOURCE" "$GENERATED_CONFIG" \
  "$QWEN_GPU" "$GEOMETRY_GPU" "$DETECTOR_ENDPOINT" "$SMOKE" <<'PY'
import os
import pathlib
import re
import sys

source, target = map(pathlib.Path, sys.argv[1:3])
qwen_gpu, geometry_gpu, detector, smoke = sys.argv[3:7]
text = source.read_text(encoding="utf-8")

pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
text = pattern.sub(
    lambda match: os.environ.get(match.group(1), match.group(2) or ""),
    text,
)

def replace_scalar(name: str, value: str) -> None:
    global text
    updated, count = re.subn(
        rf"(?m)^(\s*{re.escape(name)}:)\s*.*$",
        lambda match: f"{match.group(1)} {value}",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"missing or duplicate YAML key: {name}")
    text = updated

replace_scalar("qwen_gpu", qwen_gpu)
replace_scalar("geometry_gpu", geometry_gpu)
replace_scalar("detector_endpoint", detector)
if smoke == "1":
    replace_scalar("formal", "false")
    replace_scalar("quality_confirmation", '""')
    replace_scalar("induction_limit", "2")
    replace_scalar("inner_limit", "2")
    replace_scalar("post_publish_limit", "1")
    replace_scalar("reconstruction_sampling_per_task", "4")
    replace_scalar("run_evolution", "false")
target.write_text(text, encoding="utf-8")
PY

if [[ "$SMOKE" -eq 0 && "$DRY_RUN" -eq 0 && "$DOWNLOAD_ONLY" -eq 0 ]]; then
  "$PYTHON_BIN" - "$GENERATED_CONFIG" "$ROOT" "$CODE_REF" <<'PY'
import json
import pathlib
import re
import sys

config, root, expected_commit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
text = config.read_text(encoding="utf-8")
match = re.search(r"(?m)^\s*quality_confirmation:\s*(.*?)\s*$", text)
if match is None:
    raise SystemExit("formal config has no quality_confirmation key")
value = match.group(1).strip().strip("'\"")
if not value:
    raise SystemExit("formal run requires quality_confirmation")
path = pathlib.Path(value).expanduser()
if not path.is_absolute():
    path = root / path
if not path.is_file():
    raise SystemExit(f"quality confirmation not found: {path}")
try:
    receipt = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"invalid quality confirmation JSON: {exc}") from exc
confirmed_commit = str(receipt.get("code_commit") or "")
if confirmed_commit and confirmed_commit != expected_commit:
    raise SystemExit(
        "quality confirmation belongs to code commit "
        f"{confirmed_commit}, current CODE_REF is {expected_commit}; regenerate confirmation")
PY
fi

if [[ "$DRY_RUN" -eq 0 && "$DOWNLOAD_ONLY" -eq 0 ]]; then
  [[ "$(uname -s)" == "Linux" ]] || die "real GPU runs require Linux"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

  GPU_TABLE="$(nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader,nounits)"
  printf '%s\n' "$GPU_TABLE" | awk -F, \
    -v qwen="$QWEN_GPU" -v geometry="$GEOMETRY_GPU" -v reserve="$RESERVE_GPU" \
    -v minimum="$MIN_FREE_GPU_MIB" '
      {
        index=$1; free=$4
        gsub(/^[ \t]+|[ \t]+$/, "", index)
        gsub(/^[ \t]+|[ \t]+$/, "", free)
        seen[index]=1
        free_mib[index]=free+0
      }
      END {
        if (!seen[qwen] || !seen[geometry] || !seen[reserve]) {
          printf "required GPUs %s,%s,%s are not all visible\n", qwen, geometry, reserve > "/dev/stderr"
          exit 2
        }
        if (free_mib[qwen] < minimum || free_mib[geometry] < minimum) {
          printf "GPU %s/%s need at least %s MiB free; observed %s/%s MiB\n",
                 qwen, geometry, minimum, free_mib[qwen], free_mib[geometry] > "/dev/stderr"
          exit 2
        }
      }' || die "GPU availability check failed"

  if [[ "$SKIP_DETECTOR_CHECK" -eq 0 && -n "$DETECTOR_ENDPOINT" ]]; then
    if ! "$PYTHON_BIN" - "$DETECTOR_ENDPOINT" <<'PY'
import socket
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
host = parsed.hostname
port = parsed.port or (443 if parsed.scheme == "https" else 80)
if not host:
    raise SystemExit(1)
with socket.create_connection((host, port), timeout=3):
    pass
PY
    then
      if [[ "$REQUIRE_DETECTOR" == "1" ]]; then
        die "detector is not listening at $DETECTOR_ENDPOINT; start it on reserved GPU $RESERVE_GPU or pass --skip-detector-check"
      fi
      printf 'warning: detector unavailable at %s; M5 may be incomplete\n' \
        "$DETECTOR_ENDPOINT" >&2
    fi
  fi
fi

exec > >(tee -a "$LOG_PATH") 2>&1
printf 'harness3D GPU launch\n'
printf '  code_ref:      %s\n' "$CODE_REF"
printf '  config:        %s\n' "$GENERATED_CONFIG"
printf '  work_dir:      %s\n' "$WORK_DIR"
printf '  run_id:        %s\n' "$RUN_ID"
printf '  mode:          %s\n' "$([[ "$SMOKE" -eq 1 ]] && echo smoke || echo formal)"
printf '  qwen_gpu:      %s\n' "$QWEN_GPU"
printf '  geometry_gpu:  %s\n' "$GEOMETRY_GPU"
printf '  reserve_gpu:   %s\n' "$RESERVE_GPU"
printf '  detector:      %s\n' "$DETECTOR_ENDPOINT"
printf '  log:           %s\n' "$LOG_PATH"

ARGS=(--work-dir "$WORK_DIR" --config "$GENERATED_CONFIG" --run-id "$RUN_ID")
[[ "$DRY_RUN" -eq 1 ]] && ARGS+=(--dry-run)
[[ "$DOWNLOAD_ONLY" -eq 1 ]] && ARGS+=(--download-only)

export CODE_REF REPO_URL PYTHON_BIN
exec bash "$ROOT/scripts/bootstrap_and_run_v11.sh" "${ARGS[@]}"
