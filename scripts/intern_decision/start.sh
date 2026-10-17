#!/usr/bin/env bash
set -euo pipefail
# Usage: start.sh ISOLATED_ROOT GPU [PORT]; environment/model stay outside repo.
root="${1:?isolated deployment root required}"
gpu="${2:?GPU index required}"
port="${3:-18445}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$root/runtime"
if [ -f "$root/runtime/service.pid" ] && kill -0 "$(cat "$root/runtime/service.pid")" 2>/dev/null; then
  echo 'Recorded service still running; inspect before starting.' >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup "$root/venv/bin/python" "$script_dir/serve.py" \
  --model-dir "$root/model" --revision 1e65a988dacbcf00b84aee7366f9e47563407000 \
  --port "$port" --startup "$root/runtime/startup.json" --temp-root "$root/runtime/uploads" \
  > "$root/runtime/service.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$root/runtime/service.pid"
printf 'PID=%s port=%s GPU=%s\n' "$pid" "$port" "$gpu"
printf 'Stop after verifying PID ownership: kill -TERM %s\n' "$pid"
