#!/usr/bin/env python3
"""Build task-only train/validation metadata; reuse payloads and training statistics."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "groot/vla/data/dataset"))
from abc_v3_storage import ABCV3Storage, sha256, verify_normalizer
from abc_yam_stats import atomic_json


def select_task(storage, task):
    matches = [index for index, text in storage.tasks.items() if text == task]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact task match for {task!r}; found {len(matches)}")
    task_id = matches[0]
    windows = storage.windows[storage.windows.task_index == task_id].copy()
    if windows.empty:
        raise ValueError(f"No eligible windows for task {task!r}")
    windows = windows.sort_values(["episode_index", "start"]).reset_index(drop=True)
    ids = sorted(int(ep) for ep in windows.episode_index.unique())
    episodes = storage.episodes.loc[ids].reset_index()
    if (windows.stop <= windows.start).any():
        raise ValueError("Invalid window ranges")
    lengths = storage.episodes.length.reindex(windows.episode_index).to_numpy()
    if (windows.start < 0).any() or (windows.stop.to_numpy() + 96 > lengths).any():
        raise ValueError("Window range exceeds the episode's four-block bounds")
    return task_id, episodes, windows


def write_subset(parent, output, task, split):
    """Caller owns the fresh output directory. Original episode IDs stay unchanged."""
    parent, output = Path(parent).resolve(), Path(output)
    storage = ABCV3Storage(parent)
    normalizer = verify_normalizer(parent)
    if storage.manifest["split"] != split:
        raise ValueError(f"Expected {split} metadata at {parent}")
    if split == "train" and Path(normalizer["dataset_root"]).resolve() != storage.source_root.resolve():
        raise ValueError("Training metadata must use statistics from its training source")
    if split == "validation" and Path(normalizer["dataset_root"]).resolve() == storage.source_root.resolve():
        raise ValueError("Validation must use a separate source and training statistics")
    task_id, episodes, windows = select_task(storage, task)
    output.mkdir()
    shutil.copytree(parent / "meta", output / "meta")
    # Keep source-wide task IDs, metadata, row-group offsets and normalizer provenance.
    # The sampler population is determined by episodes.parquet and windows.parquet.
    for name in ("row_groups.parquet", "norm_stats_manifest.json"):
        shutil.copyfile(parent / name, output / name)
    pq.write_table(pa.Table.from_pandas(episodes, preserve_index=False), output / "episodes.parquet")
    pq.write_table(pa.Table.from_pandas(windows, preserve_index=False), output / "windows.parquet")
    count = int((windows.stop - windows.start).sum())
    rows = int(episodes.length.sum())
    manifest = dict(storage.manifest)
    manifest.update(
        episode_count=len(episodes), selected_episode_ids=episodes.episode_index.tolist(),
        valid_windows=count, selected_rows=rows, excluded_window_starts=rows - count,
        decoded_video_checks=0,
        task_selection=dict(
            task=task, task_index=task_id, match="exact",
            parent_metadata_root=str(parent), parent_manifest_sha256=sha256(parent / "manifest.json"),
            normalization_policy="reuse-source-training-statistics",
            normalization_manifest_sha256=sha256(parent / "norm_stats_manifest.json"),
        ),
    )
    manifest["files_sha256"] = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    atomic_json(output / "manifest.json", manifest)
    # Exercise exactly the metadata/hash/normalizer checks used by the training loader.
    checked = ABCV3Storage(output)
    verify_normalizer(output)
    assert set(checked.windows.task_index) == {task_id}
    return dict(split=split, task_index=task_id, episodes=len(episodes), valid_windows=count,
                source_frames_in_selected_episodes=rows)


def prepare(train_root, validation_root, output_root, task):
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    for parent in (train_root, validation_root):
        if Path(parent).resolve() == output or Path(parent).resolve() in output.parents:
            raise ValueError("Write task metadata outside the parent metadata directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish both splits together only after both pass validation.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        staged = Path(tmp) / "subset"
        staged.mkdir()
        train = write_subset(train_root, staged / "train", task, "train")
        validation = write_subset(validation_root, staged / "validation", task, "validation")
        if sha256(staged / "train/norm_stats_manifest.json") != sha256(staged / "validation/norm_stats_manifest.json"):
            raise ValueError("Train and validation must reuse the same training normalizer")
        summary = dict(task=task, normalization_policy="reuse-source-training-statistics",
                       train=train, validation=validation)
        atomic_json(staged / "summary.json", summary)
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Exact task text; IDs are resolved independently per split")
    parser.add_argument("--train-metadata-root", type=Path, default=ROOT / "data/abc_yam_v3_train")
    parser.add_argument("--validation-metadata-root", type=Path, default=ROOT / "data/abc_yam_v3_val")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.train_metadata_root, args.validation_metadata_root, args.output_root, args.task)
    print(json.dumps(result, indent=2))
    print(f"Prepared task metadata: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
