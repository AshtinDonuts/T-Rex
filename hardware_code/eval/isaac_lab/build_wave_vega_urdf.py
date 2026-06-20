#!/usr/bin/env python3
"""Assemble the bundled Vega-1 and two Sharpa Wave URDFs into one robot."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEXMATE = REPO / "dataset_quickstart/third_party/dexmate-urdf/src/dexmate_urdf"
SHARPA = REPO / "dataset_quickstart/third_party/sharpa-urdf-usd-xml/wave_01"


def _absolute_mesh_paths(root: ET.Element, source: Path, package_dir: Path | None = None) -> None:
    for mesh in root.findall(".//mesh"):
        name = mesh.get("filename", "")
        if name.startswith("package://"):
            if package_dir is None:
                raise ValueError(f"Cannot resolve package URI {name}")
            # package://left_sharpa_wave/meshes/foo.STL -> <package_dir>/meshes/foo.STL
            relative = Path(name.removeprefix("package://")).parts[1:]
            mesh.set("filename", str((package_dir.joinpath(*relative)).resolve()))
        elif name and not Path(name).is_absolute():
            mesh.set("filename", str((source.parent / name).resolve()))


def _append_hand(robot: ET.Element, hand_path: Path, side: str) -> None:
    hand = ET.parse(hand_path).getroot()
    _absolute_mesh_paths(hand, hand_path, hand_path.parent)
    for mujoco in hand.findall("mujoco"):
        hand.remove(mujoco)
    for child in hand:
        robot.append(copy.deepcopy(child))

    mount = ET.Element("joint", {"name": f"{side}_wave_mount", "type": "fixed"})
    # Matches the flange orientation used by the combined Vega hand variants.
    ET.SubElement(mount, "origin", {"xyz": "0.00019 0.00390 0.00014", "rpy": "-1.57079 1.57079 0"})
    ET.SubElement(mount, "parent", {"link": f"{side}_arm_l8"})
    ET.SubElement(mount, "child", {"link": "left_hand_flange" if side == "L" else "right_hand_flange"})
    robot.append(mount)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "generated/vega_1_sharpa_wave.urdf")
    args = parser.parse_args()

    vega_path = DEXMATE / "robots/humanoid/vega_1/vega_1.urdf"
    robot = ET.parse(vega_path).getroot()
    robot.set("name", "vega_1_sharpa_wave")
    _absolute_mesh_paths(robot, vega_path)
    _append_hand(robot, SHARPA / "left_sharpa_wave/left_sharpa_wave_with_flange.urdf", "L")
    _append_hand(robot, SHARPA / "right_sharpa_wave/right_sharpa_wave_with_flange.urdf", "R")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
