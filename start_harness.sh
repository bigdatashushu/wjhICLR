#!/usr/bin/env bash
# Public entry; keep one implementation of the GPU launch contract.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/start_harness3d.sh" "$@"
