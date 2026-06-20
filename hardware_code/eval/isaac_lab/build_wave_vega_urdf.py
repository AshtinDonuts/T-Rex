#!/usr/bin/env python3
"""Assemble the bundled Vega-1 and two Sharpa Wave URDFs into one robot."""

from __future__ import annotations

import argparse
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEXMATE = REPO / "dataset_quickstart/third_party/dexmate-urdf/src/dexmate_urdf"
SHARPA = REPO / "dataset_quickstart/third_party/sharpa-urdf-usd-xml/wave_01"
WAVE_MOUNTS = {
    # Match the L_ee/R_ee fixed-joint transforms used by the policy model. We
    # attach to arm_l8 directly to avoid a massless intermediate PhysX body.
    "L": ("L_arm_l8", "0 0 -1.57079", "left_hand_flange"),
    "R": ("R_arm_l8", "0 0 1.57079", "right_hand_flange"),
}


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


def _make_renderable_visual_meshes(root: ET.Element, robot_dir: Path, visual_out_dir: Path) -> None:
    """URDF importer 3.0 skips Vega .glb visuals and dedupes identical visual/collision meshes."""
    collision_dir = robot_dir / "meshes/collision"
    visual_out_dir.mkdir(parents=True, exist_ok=True)
    for mesh in root.findall(".//visual/geometry/mesh"):
        filename = mesh.get("filename", "")
        if not filename.endswith(".glb"):
            continue
        src = collision_dir / f"{Path(filename).stem}.obj"
        if not src.exists():
            continue
        dst = visual_out_dir / f"{src.stem}_visual.obj"
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
        mesh.set("filename", str(dst.resolve()))


def _append_hand(robot: ET.Element, hand_path: Path, side: str) -> None:
    if side not in WAVE_MOUNTS:
        raise ValueError(f"Unsupported hand side {side!r}; expected one of {tuple(WAVE_MOUNTS)}")

    hand = ET.parse(hand_path).getroot()
    _absolute_mesh_paths(hand, hand_path, hand_path.parent)
    for mujoco in hand.findall("mujoco"):
        hand.remove(mujoco)
    for child in hand:
        robot.append(copy.deepcopy(child))

    mount = ET.Element("joint", {"name": f"{side}_wave_mount", "type": "fixed"})
    parent_link, rpy, child_link = WAVE_MOUNTS[side]
    ET.SubElement(mount, "origin", {"xyz": "0 0 0", "rpy": rpy})
    ET.SubElement(mount, "parent", {"link": parent_link})
    ET.SubElement(mount, "child", {"link": child_link})
    robot.append(mount)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "generated/vega_1_sharpa_wave.urdf")
    args = parser.parse_args()

    vega_path = DEXMATE / "robots/humanoid/vega_1/vega_1.urdf"
    robot = ET.parse(vega_path).getroot()
    robot.set("name", "vega_1_sharpa_wave")
    _absolute_mesh_paths(robot, vega_path)
    _make_renderable_visual_meshes(robot, vega_path.parent, args.output.parent / "meshes/visual")
    _append_hand(robot, SHARPA / "left_sharpa_wave/left_sharpa_wave_with_flange.urdf", "L")
    _append_hand(robot, SHARPA / "right_sharpa_wave/right_sharpa_wave_with_flange.urdf", "R")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
