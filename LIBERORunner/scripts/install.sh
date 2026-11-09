#!/usr/bin/env bash
set -euo pipefail

RUNNER_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
UPSTREAM_ROOT="${LIBERO_SOURCE_ROOT:-$RUNNER_ROOT/third_party/LIBERO}"
ENV_ROOT="${LIBERO_VENV_ROOT:-$RUNNER_ROOT/.venv}"

if [[ "$(uname -s)" != "Linux" ]]; then
    printf '%s\n' "The simulator installation target is Linux. Mac supports offline selfcheck."
    exit 2
fi
mkdir -p "$(dirname "$UPSTREAM_ROOT")"
if [[ ! -d "$UPSTREAM_ROOT" ]]; then
    if [[ -n "${LIBERO_SOURCE_ARCHIVE:-}" ]]; then
        "${LIBERO_PYTHON_BIN:-python3.10}" "$RUNNER_ROOT/scripts/stage_official.py" \
            --archive "$LIBERO_SOURCE_ARCHIVE" --output "$UPSTREAM_ROOT"
    else
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$UPSTREAM_ROOT"
        if [[ -n "${LIBERO_REF:-}" ]]; then
            git -C "$UPSTREAM_ROOT" checkout --detach "$LIBERO_REF"
        fi
    fi
fi
if [[ -d "$UPSTREAM_ROOT/.git" ]]; then
ORIGIN_URL="$(git -C "$UPSTREAM_ROOT" remote get-url origin)"
case "$ORIGIN_URL" in
    https://github.com/Lifelong-Robot-Learning/LIBERO.git|https://github.com/Lifelong-Robot-Learning/LIBERO|git@github.com:Lifelong-Robot-Learning/LIBERO.git) ;;
    *) printf '%s\n' "Expected the original official LIBERO checkout."; exit 2 ;;
esac
if [[ -n "$(git -C "$UPSTREAM_ROOT" status --porcelain --untracked-files=no)" ]]; then
    printf '%s\n' "LIBERO has local tracked changes; use a clean separate checkout."
    exit 2
fi
if [[ -n "${LIBERO_REF:-}" ]] && [[ \
    "$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)" != \
    "$(git -C "$UPSTREAM_ROOT" rev-parse "${LIBERO_REF}^{commit}")" ]]; then
    printf '%s\n' "Existing checkout differs from LIBERO_REF; choose another source directory."
    exit 2
fi
elif [[ ! -f "$UPSTREAM_ROOT/.official-archive.json" ]]; then
    printf '%s\n' "Expected a clean official Git checkout or verified archive receipt."
    exit 2
fi
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    if command -v uv >/dev/null && [[ -z "${LIBERO_PYTHON_BIN:-}" ]]; then
        uv venv --python 3.10 --seed "$ENV_ROOT"
    else
        "${LIBERO_PYTHON_BIN:-python3.10}" -m venv "$ENV_ROOT"
    fi
fi
"$ENV_ROOT/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use isolated Python 3.10."'
"$ENV_ROOT/bin/python" "$RUNNER_ROOT/scripts/network.py" pip install 'pip==24.3.1' 'setuptools<70' wheel
"$ENV_ROOT/bin/python" "$RUNNER_ROOT/scripts/network.py" pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu
"$ENV_ROOT/bin/python" "$RUNNER_ROOT/scripts/network.py" pip install \
    --no-build-isolation \
    -c "$RUNNER_ROOT/constraints-sim.txt" \
    -r "$RUNNER_ROOT/requirements-sim.txt" -r "$RUNNER_ROOT/requirements.txt"
# Runtime imports the exact source root directly; installation would add generated
# egg-info files to the verified official tree.
"$ENV_ROOT/bin/python" -m pip check
printf 'Python: %s/bin/python\n' "$ENV_ROOT"
printf '%s\n' "Next: run the zero-request prepare command from README.md with MUJOCO_GL=egl."
