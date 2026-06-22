"""Convert the public T-Rex LeRobot dataset to T-Rex's post-train schema.

The public release stores 58-D joint states and one-step absolute joint targets.
Post-training expects 62-D EEF/hand states and [16, 62] delta-base action
chunks.  This script selects public episodes, runs Vega-1 arm forward
kinematics, remaps the camera/tactile fields, and writes the q01/q99 statistics
sidecar consumed by :class:`qwen_vla.lerobot_dataset.TRexLeRobotDataset`.

Only files referenced by selected episodes are downloaded for a Hub source.

Example:
    python utils/convert_public_trex_to_lerobot.py \
        --source zekaiwang/trex_dataset \
        --output_root /data/trex/lift_and_place \
        --repo_id local/trex_lift_and_place \
        --motor_primitive lift_and_place \
        --num_episodes 50 --max_download_gb 200
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make ``python utils/convert_public_trex_to_lerobot.py`` work without requiring
# callers to set PYTHONPATH first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.lerobot_common import (
    ACTION_CHUNK,
    DEFORM_KEYS,
    KEY_ACTION,
    KEY_ACTION_ABS,
    KEY_HEAD,
    KEY_STATE,
    KEY_TACF6,
    KEY_WRIST_L,
    KEY_WRIST_R,
    NormStatsAccumulator,
    build_trex_features,
    compute_chunk_delta_pose,
    pose_matrix_to_9d,
)

PUBLIC_STATE = "observation.state"
PUBLIC_ACTION = "action"
PUBLIC_TACF6 = "observation.tactile_force"
PUBLIC_HEAD = "observation.images.head_left"
PUBLIC_WRIST_L = "observation.images.left_wrist"
PUBLIC_WRIST_R = "observation.images.right_wrist"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

LEFT_ARM = slice(0, 7)
LEFT_HAND = slice(7, 29)
RIGHT_ARM = slice(29, 36)
RIGHT_HAND = slice(36, 58)

DEFAULT_HEAD = np.array([0.28, 0.0, 0.0], dtype=np.float64)
DEFAULT_TORSO = np.array([0.9, 1.57, 0.1], dtype=np.float64)


def _public_deform_key(side: str, finger: str) -> str:
    return f"observation.images.tactile_{side}_deform_{finger}"


def _load_episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta/episodes'}")
    rows = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if "episode_index" not in rows:
        raise ValueError("Episode metadata has no episode_index column")
    return rows.sort_values("episode_index").drop_duplicates("episode_index")


def _download_meta(repo_id: str, root: Path, revision: str | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["meta/**"],
        local_dir=str(root),
    )


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, value in info["features"].items() if value.get("dtype") == "video"]


def _selected_files(info: dict[str, Any], rows: pd.DataFrame) -> list[str]:
    files: set[str] = set()
    for _, row in rows.iterrows():
        files.add(
            info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
        )
    for key in _video_keys(info):
        for _, row in rows.iterrows():
            files.add(
                info["video_path"].format(
                    video_key=key,
                    chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]),
                )
            )
    return sorted(files)


def _download_selected(
    repo_id: str,
    root: Path,
    rows: pd.DataFrame,
    revision: str | None,
    max_download_gb: float,
) -> None:
    from huggingface_hub import hf_hub_download

    info = json.loads((root / "meta" / "info.json").read_text())
    files = _selected_files(info, rows)
    size_gb = _estimate_selected_gb(repo_id, root, files, revision)
    print(f"Selected episodes reference {len(files)} unique files ({size_gb:.2f} GB).")
    if size_gb > max_download_gb:
        raise RuntimeError(
            f"Required download is {size_gb:.2f} GB, above --max_download_gb "
            f"{max_download_gb:.2f}. Reduce the subset or raise the cap."
        )
    for index, filename in enumerate(files, 1):
        print(f"  download [{index}/{len(files)}] {filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=str(root),
        )


def _estimate_selected_gb(
    repo_id: str, root: Path, files: list[str], revision: str | None, remote: bool = True
) -> float:
    """Estimate unique selected shard size without downloading those shards."""
    if remote:
        from huggingface_hub import HfApi

        repo = HfApi().repo_info(
            repo_id=repo_id, repo_type="dataset", revision=revision, files_metadata=True
        )
        sizes = {entry.rfilename: (entry.size or 0) for entry in repo.siblings}
        return sum(sizes.get(path, 0) for path in files) / 1e9
    return sum((root / path).stat().st_size for path in files if (root / path).exists()) / 1e9


def _prepare_source(source: str, cache_dir: str | None, revision: str | None) -> tuple[str, Path, bool]:
    local = Path(source).expanduser()
    if (local / "meta" / "info.json").exists():
        info = json.loads((local / "meta" / "info.json").read_text())
        repo_id = info.get("repo_id") or local.name
        return repo_id, local.resolve(), False

    repo_id = source
    root = (
        Path(cache_dir).expanduser()
        if cache_dir
        else Path.home() / ".cache" / "trex_public_conversion" / repo_id.replace("/", "__")
    )
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "meta" / "info.json").exists():
        _download_meta(repo_id, root, revision)
    return repo_id, root.resolve(), True


def _matches(value: Any, requested: list[str]) -> bool:
    if not requested:
        return True
    actual = "" if pd.isna(value) else str(value).casefold()
    return actual in {item.casefold() for item in requested}


def _read_manifest_episode_ids(path: str | None) -> list[int]:
    if not path:
        return []
    manifest = Path(path).expanduser()
    if manifest.suffix.lower() == ".csv":
        frame = pd.read_csv(manifest)
        if "episode_index" not in frame:
            raise ValueError(f"Manifest {manifest} has no episode_index column")
        return frame["episode_index"].astype(int).tolist()
    payload = json.loads(manifest.read_text())
    values = payload if isinstance(payload, list) else payload.get("episode_indices")
    if values is None:
        raise ValueError(f"Manifest {manifest} has no episode_indices field")
    return [int(value) for value in values]


def _select_episodes(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = rows
    requested_ids = [
        *args.episode_index,
        *_read_manifest_episode_ids(getattr(args, "episode_manifest", None)),
    ]
    if requested_ids:
        wanted = set(requested_ids)
        selected = selected[selected["episode_index"].isin(wanted)]
        found = set(selected["episode_index"].astype(int).tolist())
        missing = sorted(wanted.difference(found))
        if missing:
            raise ValueError(f"Requested episode indices are absent from metadata: {missing[:10]}")
    for column, requested in (
        ("motor_primitive", args.motor_primitive),
        ("object", args.object),
        ("target", args.target),
    ):
        if requested:
            if column not in selected:
                raise ValueError(f"Episode metadata has no {column!r} column")
            selected = selected[selected[column].map(lambda value: _matches(value, requested))]
    if args.caption_regex:
        if "caption" not in selected:
            raise ValueError("Episode metadata has no 'caption' column")
        pattern = re.compile(args.caption_regex, re.IGNORECASE)
        selected = selected[
            selected["caption"].fillna("").astype(str).map(lambda text: bool(pattern.search(text)))
        ]
    if args.num_episodes and len(selected) > args.num_episodes:
        indices = list(selected.index)
        random.Random(args.seed).shuffle(indices)
        selected = selected.loc[sorted(indices[: args.num_episodes])]
    if selected.empty:
        raise ValueError("No episodes matched the requested filters")
    return selected.sort_values("episode_index")


def _write_selection_report(
    selected: pd.DataFrame,
    path: str,
    source: str,
    revision: str | None,
    required_files: list[str],
    estimated_gb: float,
) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        name
        for name in ("episode_index", "motor_primitive", "object", "target", "caption")
        if name in selected
    ]
    if destination.suffix.lower() == ".csv":
        selected[columns].to_csv(destination, index=False)
        return
    episode_rows = selected[columns].copy()
    episode_rows = episode_rows.astype(object).where(pd.notna(episode_rows), None)
    payload = {
        "source": source,
        "revision": revision,
        "episode_indices": selected["episode_index"].astype(int).tolist(),
        "num_episodes": int(len(selected)),
        "num_unique_files": len(required_files),
        "estimated_download_gb": estimated_gb,
        "episodes": json.loads(episode_rows.to_json(orient="records")),
    }
    destination.write_text(json.dumps(payload, indent=2))


class VegaForwardKinematics:
    """Forward kinematics for the fixed-base Vega-1 arm pair."""

    def __init__(self, urdf: Path):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Pinocchio is required for 58-D joint -> 62-D EEF conversion. "
                "Install the quick-start replay dependencies: "
                "pip install -e 'dataset_quickstart[replay]'"
            ) from exc
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.left_indices = self._indices([f"L_arm_j{i}" for i in range(1, 8)])
        self.right_indices = self._indices([f"R_arm_j{i}" for i in range(1, 8)])
        self.q[self._indices([f"torso_j{i}" for i in range(1, 4)])] = DEFAULT_TORSO
        self.q[self._indices([f"head_j{i}" for i in range(1, 4)])] = DEFAULT_HEAD
        self.left_frame = self.model.getFrameId("L_ee")
        self.right_frame = self.model.getFrameId("R_ee")
        if self.left_frame >= len(self.model.frames) or self.right_frame >= len(self.model.frames):
            raise ValueError("Vega URDF does not contain L_ee and R_ee frames")

    def _indices(self, names: list[str]) -> np.ndarray:
        indices = []
        for name in names:
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"Vega URDF is missing joint {name!r}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"Expected scalar joint {name!r}, got nq={joint.nq}")
            indices.append(joint.idx_q)
        return np.asarray(indices)

    def poses(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.q[self.left_indices] = left
        self.q[self.right_indices] = right
        self.pin.forwardKinematics(self.model, self.data, self.q)
        self.pin.updateFramePlacements(self.model, self.data)
        return (
            np.asarray(self.data.oMf[self.left_frame].homogeneous).copy(),
            np.asarray(self.data.oMf[self.right_frame].homogeneous).copy(),
        )


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_rgb(value: Any) -> np.ndarray:
    image = _numpy(value)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        image = image * scale
    return np.clip(image, 0, 255).astype(np.uint8)


def _processed_record(item: dict[str, Any], fk: VegaForwardKinematics) -> dict[str, Any]:
    state = _numpy(item[PUBLIC_STATE]).astype(np.float64).reshape(-1)
    target = _numpy(item[PUBLIC_ACTION]).astype(np.float64).reshape(-1)
    if state.shape != (58,) or target.shape != (58,):
        raise ValueError(f"Expected public state/action shape (58,), got {state.shape}/{target.shape}")
    state_l_pose, state_r_pose = fk.poses(state[LEFT_ARM], state[RIGHT_ARM])
    target_l_pose, target_r_pose = fk.poses(target[LEFT_ARM], target[RIGHT_ARM])
    state_62 = np.concatenate(
        [
            pose_matrix_to_9d(state_l_pose[None])[0],
            state[LEFT_HAND],
            pose_matrix_to_9d(state_r_pose[None])[0],
            state[RIGHT_HAND],
        ]
    ).astype(np.float32)
    target_62 = np.concatenate(
        [
            pose_matrix_to_9d(target_l_pose[None])[0],
            target[LEFT_HAND],
            pose_matrix_to_9d(target_r_pose[None])[0],
            target[RIGHT_HAND],
        ]
    ).astype(np.float32)
    media_keys = [PUBLIC_HEAD, PUBLIC_WRIST_L, PUBLIC_WRIST_R, PUBLIC_TACF6]
    media_keys.extend(
        _public_deform_key(side, finger)
        for side in ("left", "right")
        for finger in FINGERS
    )
    return {
        # Do not retain the 10 raw tactile-camera frames while buffering the
        # 16-step action horizon; post-training consumes deformation only.
        "item": {key: item[key] for key in media_keys},
        "state_l_pose": state_l_pose,
        "state_r_pose": state_r_pose,
        "target_l_pose": target_l_pose,
        "target_r_pose": target_r_pose,
        "target_l_hand": target[LEFT_HAND].astype(np.float32),
        "target_r_hand": target[RIGHT_HAND].astype(np.float32),
        "state_62": state_62,
        "target_62": target_62,
    }


def _action_chunk(records: list[dict[str, Any]]) -> np.ndarray:
    base_l = records[0]["state_l_pose"]
    base_r = records[0]["state_r_pose"]
    chunk = np.empty((ACTION_CHUNK, 62), dtype=np.float32)
    for step in range(ACTION_CHUNK):
        future = records[min(step, len(records) - 1)]
        chunk[step] = np.concatenate(
            [
                compute_chunk_delta_pose(base_l, future["target_l_pose"]),
                future["target_l_hand"],
                compute_chunk_delta_pose(base_r, future["target_r_pose"]),
                future["target_r_hand"],
            ]
        )
    return chunk


def _output_frame(record: dict[str, Any], chunk: np.ndarray, caption: str) -> tuple[dict, np.ndarray]:
    item = record["item"]
    tactile = _numpy(item[PUBLIC_TACF6]).astype(np.float32).reshape(10, 6)
    frame = {
        "task": caption,
        KEY_HEAD: _image_rgb(item[PUBLIC_HEAD]),
        KEY_WRIST_L: _image_rgb(item[PUBLIC_WRIST_L]),
        KEY_WRIST_R: _image_rgb(item[PUBLIC_WRIST_R]),
        KEY_STATE: record["state_62"],
        KEY_ACTION: chunk,
        KEY_ACTION_ABS: record["target_62"],
        KEY_TACF6: tactile,
    }
    out_index = 0
    for side in ("left", "right"):
        for finger in FINGERS:
            frame[DEFORM_KEYS[out_index]] = _image_rgb(item[_public_deform_key(side, finger)])
            out_index += 1
    return frame, tactile


def _caption(row: pd.Series, override: str | None = None) -> str:
    if override:
        return override
    for key in ("caption", "tasks", "task"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            return str(value[0]) if len(value) else ""
        if not pd.isna(value):
            return str(value)
    return ""


def _default_urdf() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "dataset_quickstart"
        / "third_party"
        / "dexmate-urdf"
        / "robots"
        / "humanoid"
        / "vega_1"
        / "vega_1.urdf"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="zekaiwang/trex_dataset", help="Hub repo ID or local dataset root")
    parser.add_argument("--output_root", help="Required unless --selection_only is used")
    parser.add_argument("--repo_id", default="local/trex_public_posttrain")
    parser.add_argument("--cache_dir", help="Hub download directory")
    parser.add_argument("--revision")
    parser.add_argument("--urdf", default=str(_default_urdf()))
    parser.add_argument("--episode_index", type=int, action="append", default=[])
    parser.add_argument("--episode_manifest", help="CSV or JSON episode-selection manifest")
    parser.add_argument("--motor_primitive", action="append", default=[])
    parser.add_argument("--object", action="append", default=[])
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--caption_regex")
    parser.add_argument("--instruction_override", help="Replace every selected caption")
    parser.add_argument("--num_episodes", type=int, default=0, help="0 means all matched episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_download_gb", type=float, default=100.0)
    parser.add_argument("--selection_only", action="store_true",
                        help="Report metadata and shard size, then exit before data download")
    parser.add_argument("--selection_report", help="Selection report path (.json or .csv)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.selection_only and not args.output_root:
        raise ValueError("--output_root is required unless --selection_only is used")
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else None
    if output_root is not None and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    source_repo_id, source_root, remote = _prepare_source(args.source, args.cache_dir, args.revision)
    metadata = _load_episode_metadata(source_root)
    selected = _select_episodes(metadata, args)
    print(f"Selected {len(selected)} of {len(metadata)} episodes.")
    info = json.loads((source_root / "meta" / "info.json").read_text())
    required_files = _selected_files(info, selected)
    estimated_gb = _estimate_selected_gb(
        source_repo_id, source_root, required_files, args.revision, remote=remote
    )
    print(f"Selection references {len(required_files)} unique files ({estimated_gb:.2f} GB).")
    if args.selection_report:
        _write_selection_report(
            selected, args.selection_report, args.source, args.revision,
            required_files, estimated_gb,
        )
        print(f">>> Selection report: {Path(args.selection_report).expanduser()}")
    if args.selection_only:
        return
    if remote:
        _download_selected(
            source_repo_id, source_root, selected, args.revision, args.max_download_gb
        )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot v3 is required. Install it into the T-Rex environment, e.g. "
            "pip install -e /path/to/lerobot"
        ) from exc

    fk = VegaForwardKinematics(Path(args.urdf).expanduser().resolve())
    first_episode = int(selected.iloc[0]["episode_index"])
    probe_ds = LeRobotDataset(source_repo_id, root=source_root, episodes=[first_episode])
    if len(probe_ds) == 0:
        raise RuntimeError(f"Selected episode {first_episode} contains no frames")
    probe = probe_ds[0]
    head = _image_rgb(probe[PUBLIC_HEAD])
    wrist = _image_rgb(probe[PUBLIC_WRIST_L])
    deform = _image_rgb(probe[_public_deform_key("left", "thumb")])
    features = build_trex_features(
        head_shape=(3, *head.shape[:2]),
        include_wrist=True,
        wrist_shape=(3, *wrist.shape[:2]),
        include_tactile=True,
        deform_shape=(3, *deform.shape[:2]),
    )
    del probe_ds, probe

    fps = int(info.get("fps", 30))
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)
    output = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=output_root,
        robot_type="trex_bimanual",
        use_videos=True,
    )
    stats = NormStatsAccumulator()

    for ordinal, (_, row) in enumerate(selected.iterrows(), 1):
        episode_index = int(row["episode_index"])
        caption = _caption(row, args.instruction_override)
        source_ds = LeRobotDataset(source_repo_id, root=source_root, episodes=[episode_index])
        pending: deque[dict[str, Any]] = deque()
        episode_states: list[np.ndarray] = []
        episode_targets: list[np.ndarray] = []

        def write_oldest() -> None:
            records = list(pending)
            chunk = _action_chunk(records)
            frame, tactile = _output_frame(records[0], chunk, caption)
            output.add_frame(frame)
            stats.add_frame(chunk, records[0]["state_62"], tactile)
            pending.popleft()

        for frame_index in range(len(source_ds)):
            item = source_ds[frame_index]
            record = _processed_record(item, fk)
            pending.append(record)
            episode_states.append(record["state_62"])
            episode_targets.append(record["target_62"])
            if len(pending) == ACTION_CHUNK:
                write_oldest()
        while pending:
            write_oldest()

        output.save_episode()
        stats.add_episode_tracking(np.asarray(episode_states), np.asarray(episode_targets))
        print(
            f"  [{ordinal}/{len(selected)}] episode {episode_index}: "
            f"{len(episode_states)} frames — {caption}"
        )

    output.finalize()
    stats_path = stats.write(str(output_root))
    selection_path = output_root / "meta" / "source_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "source": args.source,
                "revision": args.revision,
                "episode_indices": selected["episode_index"].astype(int).tolist(),
            },
            indent=2,
        )
    )
    print(f">>> Converted dataset: {output_root}")
    print(f">>> Normalization stats: {stats_path}")
    print(f">>> Source selection: {selection_path}")


if __name__ == "__main__":
    main()
