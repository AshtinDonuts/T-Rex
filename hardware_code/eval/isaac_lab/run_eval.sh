#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/khw/IsaacLab}"

# isaaclab.sh prefers CONDA_PREFIX when set, but T-Rex's training env is often
# Python 3.10 while Isaac Lab needs 3.11+ (tomllib). Use Isaac Sim's bundled
# interpreter instead of the active conda/venv.
exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV \
  "$ISAACLAB_ROOT/isaaclab.sh" -p "$SCRIPT_DIR/eval_trex_isaac.py" \
  --config "$SCRIPT_DIR/trex_isaac.yaml" --enable_cameras "$@"
