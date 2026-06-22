#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Some lightweight CI environments have an OpenCV/NumPy ABI mismatch. These
# tests do not exercise Rodrigues, so avoid making metadata/stride tests depend
# on a working OpenCV wheel.
cv2 = types.ModuleType("cv2")
cv2.Rodrigues = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unused"))
sys.modules.setdefault("cv2", cv2)

try:
    import torch  # noqa: F401
except ImportError:
    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.Dataset = object
    torch_utils.data = torch_utils_data
    torch.utils = torch_utils
    torch_nn = types.ModuleType("torch.nn")
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_nn.functional = torch_nn_functional
    sys.modules.update({
        "torch": torch,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torch.nn": torch_nn,
        "torch.nn.functional": torch_nn_functional,
    })

converter = load_file("trex_public_converter", ROOT / "utils" / "convert_public_trex_to_lerobot.py")
selector = load_file("trex_pickup_selector", Path(__file__).with_name("select_episodes.py"))
loader = load_file("trex_lerobot_loader", ROOT / "qwen_vla" / "lerobot_dataset.py")


class ConverterTests(unittest.TestCase):
    def test_manifest_csv_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"episode_index": [3, 7]}).to_csv(root / "x.csv", index=False)
            (root / "x.json").write_text(json.dumps({"episode_indices": [4, 8]}))
            self.assertEqual(converter._read_manifest_episode_ids(str(root / "x.csv")), [3, 7])
            self.assertEqual(converter._read_manifest_episode_ids(str(root / "x.json")), [4, 8])

    def test_manifest_filters_and_instruction_override(self):
        rows = pd.DataFrame({
            "episode_index": [1, 2, 3],
            "motor_primitive": ["grasp and lift"] * 3,
            "object": ["box", "cup", "block"],
            "target": [None] * 3,
            "caption": ["a", "b", "c"],
        })
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "selection.json"
            manifest.write_text(json.dumps({"episode_indices": [1, 3]}))
            args = SimpleNamespace(
                episode_index=[], episode_manifest=str(manifest), motor_primitive=[],
                object=[], target=[], caption_regex=None, num_episodes=0, seed=42,
            )
            self.assertEqual(converter._select_episodes(rows, args).episode_index.tolist(), [1, 3])
        self.assertEqual(converter._caption(rows.iloc[0], "Pick up the object."),
                         "Pick up the object.")

    def test_58d_to_62d_record_and_action_chunk(self):
        class FakeFK:
            def poses(self, left, right):
                left_pose = np.eye(4)
                right_pose = np.eye(4)
                left_pose[0, 3] = left[0]
                right_pose[0, 3] = right[0]
                return left_pose, right_pose

        item = {
            converter.PUBLIC_STATE: np.arange(58, dtype=np.float32),
            converter.PUBLIC_ACTION: np.arange(58, dtype=np.float32) + 1,
            converter.PUBLIC_HEAD: None,
            converter.PUBLIC_WRIST_L: None,
            converter.PUBLIC_WRIST_R: None,
            converter.PUBLIC_TACF6: np.zeros(60, dtype=np.float32),
        }
        for side in ("left", "right"):
            for finger in converter.FINGERS:
                item[converter._public_deform_key(side, finger)] = None
        record = converter._processed_record(item, FakeFK())
        self.assertEqual(record["state_62"].shape, (62,))
        chunk = converter._action_chunk([record])
        self.assertEqual(chunk.shape, (16, 62))
        self.assertTrue(np.allclose(chunk[0], chunk[-1]))


class SelectorTests(unittest.TestCase):
    def test_deterministic_balanced_selection(self):
        rows = []
        for index in range(80):
            rows.append({
                "episode_index": index,
                "motor_primitive": "grasp_and_lift",
                "object": f"object_{index % 20}",
                "target": None,
                "caption": f"Lift object {index} with the right hand",
            })
        rows.append({
            "episode_index": 100, "motor_primitive": "grasp and lift",
            "object": "cloth", "target": None, "caption": "lift the cloth",
        })
        frame = pd.DataFrame(rows)
        first = selector.select_pickup_rows(frame, 64, 42, 4)
        second = selector.select_pickup_rows(frame, 64, 42, 4)
        self.assertEqual(first.episode_index.tolist(), second.episode_index.tolist())
        self.assertNotIn(100, first.episode_index.tolist())
        self.assertLessEqual(first.groupby("selection_object").size().max(), 4)


class LoaderStrideTests(unittest.TestCase):
    def test_stride_preserves_native_indices(self):
        dataset = loader.TRexLeRobotDataset.__new__(loader.TRexLeRobotDataset)
        dataset.ds = list(range(10))
        dataset.sample_stride = 4
        self.assertEqual(len(dataset), 3)
        self.assertEqual([dataset[i] for i in range(len(dataset))], [0, 4, 8])
        with self.assertRaises(IndexError):
            _ = dataset[3]


if __name__ == "__main__":
    unittest.main()
