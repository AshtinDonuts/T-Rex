#!/usr/bin/env python3
"""Select a deterministic, object-balanced pickup subset from T-Rex metadata."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

import pandas as pd

EXCLUDE = re.compile(
    r"cloth|fabric|paper|tape|rope|cable|bag|sponge|garment|peel|wipe|wrap|fold",
    re.IGNORECASE,
)
RIGHT_HAND = re.compile(r"right[- ]hand|right arm|with (?:the )?right", re.IGNORECASE)


def normalize_primitive(value: object) -> str:
    return re.sub(r"[_-]+", " ", str(value)).strip().casefold()


def load_metadata(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No metadata parquet files under {root / 'meta/episodes'}")
    rows = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    return rows.sort_values("episode_index").drop_duplicates("episode_index")


def metadata_root(source: str, cache_dir: Path, revision: str | None) -> tuple[Path, bool]:
    local = Path(source).expanduser()
    if (local / "meta" / "info.json").exists():
        return local.resolve(), False
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=source,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["meta/**"],
        local_dir=str(cache_dir),
    )
    return cache_dir.resolve(), True


def select_pickup_rows(rows: pd.DataFrame, count: int, seed: int, max_per_object: int) -> pd.DataFrame:
    required = {"episode_index", "motor_primitive", "object", "caption"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Episode metadata is missing columns: {sorted(missing)}")
    candidates = rows[
        rows["motor_primitive"].map(normalize_primitive).str.startswith("grasp and lift")
    ].copy()
    searchable = candidates["object"].fillna("").astype(str) + " " + candidates["caption"].fillna("").astype(str)
    candidates = candidates[~searchable.map(lambda text: bool(EXCLUDE.search(text)))].copy()
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} rigid grasp-and-lift candidates; need {count}")

    candidates["selection_object"] = candidates["object"].fillna("unknown").astype(str)
    candidates["right_hand_preferred"] = candidates["caption"].fillna("").astype(str).map(
        lambda text: bool(RIGHT_HAND.search(text))
    )
    rng = random.Random(seed)
    candidates["random_rank"] = [rng.random() for _ in range(len(candidates))]
    candidates = candidates.sort_values(
        ["right_hand_preferred", "random_rank"], ascending=[False, True]
    )

    chosen: list[int] = []
    object_counts: dict[str, int] = {}
    for index, row in candidates.iterrows():
        obj = row["selection_object"].casefold()
        if object_counts.get(obj, 0) >= max_per_object:
            continue
        chosen.append(index)
        object_counts[obj] = object_counts.get(obj, 0) + 1
        if len(chosen) == count:
            break
    if len(chosen) < count:
        for index in candidates.index:
            if index not in chosen:
                chosen.append(index)
                if len(chosen) == count:
                    break
    return candidates.loc[chosen].sort_values("episode_index")


def referenced_files(info: dict, rows: pd.DataFrame) -> list[str]:
    files: set[str] = set()
    video_keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    for _, row in rows.iterrows():
        files.add(info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        ))
        for key in video_keys:
            files.add(info["video_path"].format(
                video_key=key,
                chunk_index=int(row[f"videos/{key}/chunk_index"]),
                file_index=int(row[f"videos/{key}/file_index"]),
            ))
    return sorted(files)


def estimate_gb(source: str, root: Path, files: list[str], remote: bool, revision: str | None) -> float:
    if remote:
        from huggingface_hub import HfApi

        repo = HfApi().repo_info(
            repo_id=source, repo_type="dataset", revision=revision, files_metadata=True
        )
        sizes = {entry.rfilename: (entry.size or 0) for entry in repo.siblings}
        return sum(sizes.get(path, 0) for path in files) / 1e9
    return sum((root / path).stat().st_size for path in files if (root / path).exists()) / 1e9


def write_manifests(rows: pd.DataFrame, output_dir: Path, source: str, revision: str | None,
                    files: list[str], size_gb: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["episode_index", "motor_primitive", "object", "target", "caption", "right_hand_preferred"]
    columns = [column for column in columns if column in rows]
    rows[columns].to_csv(output_dir / "selection.csv", index=False)
    payload = {
        "source": source,
        "revision": revision,
        "episode_indices": rows["episode_index"].astype(int).tolist(),
        "num_episodes": int(len(rows)),
        "num_unique_files": len(files),
        "estimated_download_gb": size_gb,
    }
    (output_dir / "selection.json").write_text(json.dumps(payload, indent=2))
    smoke = {**payload, "episode_indices": payload["episode_indices"][:4], "num_episodes": 4}
    (output_dir / "smoke_selection.json").write_text(json.dumps(smoke, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="zekaiwang/trex_dataset")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_object", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, remote = metadata_root(args.source, args.cache_dir, args.revision)
    rows = select_pickup_rows(load_metadata(root), args.count, args.seed, args.max_per_object)
    info = json.loads((root / "meta" / "info.json").read_text())
    files = referenced_files(info, rows)
    size_gb = estimate_gb(args.source, root, files, remote, args.revision)
    write_manifests(rows, args.output_dir, args.source, args.revision, files, size_gb)
    print(f"Selected {len(rows)} episodes; {len(files)} unique files; estimated {size_gb:.2f} GB")
    print(f"Review {args.output_dir / 'selection.csv'} and {args.output_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
