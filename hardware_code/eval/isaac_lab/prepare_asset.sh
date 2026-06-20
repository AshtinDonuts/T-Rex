#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/khw/IsaacLab}"
URDF="$SCRIPT_DIR/generated/vega_1_sharpa_wave.urdf"
USD_DIR="$SCRIPT_DIR/generated/usd"

python3 "$SCRIPT_DIR/build_wave_vega_urdf.py" --output "$URDF"
mkdir -p "$USD_DIR"
"$ISAACLAB_ROOT/isaaclab.sh" -p "$ISAACLAB_ROOT/scripts/tools/convert_urdf.py" \
  "$URDF" "$USD_DIR" --fix-base --joint-stiffness 400 --joint-damping 40 --viz none

echo "Prepared Isaac asset under: $USD_DIR"
