#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/khw/IsaacLab}"

exec "$ISAACLAB_ROOT/isaaclab.sh" -p "$SCRIPT_DIR/eval_trex_isaac.py" \
  --config "$SCRIPT_DIR/trex_isaac.yaml" --enable_cameras "$@"
