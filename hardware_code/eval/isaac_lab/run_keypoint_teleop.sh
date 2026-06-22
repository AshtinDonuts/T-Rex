#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/khw/IsaacLab}"

exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV \
  "$ISAACLAB_ROOT/isaaclab.sh" -p "$SCRIPT_DIR/teleop_sharpa_keypoints.py" \
  --config "$SCRIPT_DIR/trex_isaac.yaml" --enable_cameras "$@"
