#!/usr/bin/env bash
set -euo pipefail

launcher_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${H3D_PYTHON:-}" ]]; then
  launcher_python="$H3D_PYTHON"
elif [[ -x "$launcher_root/.venv/bin/python" ]]; then
  launcher_python="$launcher_root/.venv/bin/python"
elif [[ -x "$launcher_root/venv/bin/python" ]]; then
  launcher_python="$launcher_root/venv/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  launcher_python="$(command -v python3.11)"
else
  launcher_python="$(command -v python3)"
fi

exec "$launcher_python" -B "$launcher_root/scripts/run_experiment.py" "$@"
