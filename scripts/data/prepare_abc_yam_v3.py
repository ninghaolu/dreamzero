#!/usr/bin/env python3
"""Prepare compact ABC metadata in place; never copy data/video payloads."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "groot/vla/data/dataset"))
from abc_v3_storage import CAMERAS, SLICES, sha256, verify_normalizer
from abc_yam_stats import atomic_json, file_inventory, source_identity


def prepare(source, output, normalizer, pilot_episodes=0, seed=42, split="train", check_files=True):
    source, output, normalizer = map(Path, (source, output, normalizer))
    if output.exists():
        raise FileExistsError(f"Use a fresh metadata directory: {output}")
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("Write prepared metadata outside the source dataset")
    norm = verify_normalizer(normalizer)
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError("Expected LeRobot v3.0 at 30 FPS")
    files = file_inventory(source)
    if split == "train":
        identity = source_identity(source, files, norm["num_bins"], norm["num_shards"])
        if identity["source_fingerprint"] != norm["source_fingerprint"]:
            raise ValueError("Training source differs from normalizer source")
    elif source.resolve() == Path(norm["dataset_root"]).resolve():
        raise ValueError("Validation must use the separate validation source")
    columns = ["episode_index", "length", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"]
    columns += [f"videos/{key}/{field}" for key in CAMERAS.values()
                for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    tables = [pq.read_table(p, columns=columns) for p in sorted((source / "meta/episodes").glob("**/*.parquet"))]
    episodes = pa.concat_tables(tables).to_pandas().sort_values("episode_index")
    if episodes.episode_index.duplicated().any() or len(episodes) != info["total_episodes"]:
        raise ValueError("Invalid episode metadata count/IDs")
    if np.any(episodes.length != episodes.dataset_to_index - episodes.dataset_from_index):
        raise ValueError("Episode bounds/length mismatch")
    if not np.array_equal(episodes.dataset_from_index.to_numpy()[1:], episodes.dataset_to_index.to_numpy()[:-1]):
        raise ValueError("Episode bounds not contiguous")
    if episodes.dataset_from_index.iloc[0] != 0 or episodes.dataset_to_index.iloc[-1] != info["total_frames"]:
        raise ValueError("Episode bounds do not cover source")
    if pilot_episodes:
        ids = np.random.default_rng(seed).choice(episodes.episode_index, min(pilot_episodes, len(episodes)), replace=False)
        episodes = episodes[episodes.episode_index.isin(ids)].copy()
    ids = set(int(v) for v in episodes.episode_index)
    lengths = episodes.set_index("episode_index")["length"].to_dict()
    starts_by_episode = episodes.set_index("episode_index")["dataset_from_index"].to_dict()
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pandas().reset_index()
    tasks = tasks[["task_index", "task"]]
    if tasks.task_index.duplicated().any() or tasks.task.duplicated().any() or tasks.isnull().any().any():
        raise ValueError("Task ID/text map is not one-to-one")
    known_tasks = set(tasks.task_index)
    checked_paths = set()
    for _, row in episodes.iterrows():
        paths = [source / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"]
        for key in CAMERAS.values():
            prefix = f"videos/{key}/"
            start, end = row[prefix+"from_timestamp"], row[prefix+"to_timestamp"]
            if not np.isfinite([start, end]).all() or start < 0 or abs((end-start)-row.length/30) > 1e-3:
                raise ValueError("Invalid camera interval")
            paths.append(source / f"videos/{key}/chunk-{int(row[prefix+'chunk_index']):03d}/file-{int(row[prefix+'file_index']):03d}.mp4")
        if check_files:
            for path in paths:
                if path not in checked_paths and not path.is_file():
                    raise FileNotFoundError(path)
                checked_paths.add(path)
    # Scan scalar columns once, building row-group bounds and contiguous task runs.
    groups, segments = [], []
    previous_index = -1
    observed = {key: 0 for key in ids}
    for file_number, path in enumerate(files):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["index", "episode_index", "frame_index", "task_index"], use_threads=False)
            if any(table[k].null_count for k in table.column_names):
                raise ValueError("Null row index/task metadata")
            index, ep, frame, task = [table[k].to_numpy() for k in table.column_names]
            if len(index) == 0 or index[0] != previous_index + 1 or np.any(np.diff(index) != 1):
                raise ValueError("Data file/row-group global index discontinuity")
            previous_index = int(index[-1])
            groups.append(dict(path=str(path.relative_to(source)), row_group=rg, start=int(index[0]), end=int(index[-1])+1))
            boundaries = np.concatenate([[0], np.flatnonzero((np.diff(ep) != 0) | (np.diff(task) != 0))+1, [len(index)]])
            for a, b in zip(boundaries[:-1], boundaries[1:]):
                episode_id, task_id = int(ep[a]), int(task[a])
                if episode_id not in ids:
                    continue
                if task_id not in known_tasks or not np.array_equal(frame[a:b], index[a:b] - starts_by_episode[episode_id]):
                    raise ValueError("Invalid frame/task identity")
                start, stop = int(frame[a]), int(frame[b-1])+1
                if start < 0 or stop > lengths[episode_id]:
                    raise ValueError("Numeric row outside episode metadata bounds")
                observed[episode_id] += b-a
                if segments and segments[-1]["episode_index"] == episode_id and segments[-1]["task_index"] == task_id and segments[-1]["stop"] == start:
                    segments[-1]["stop"] = stop
                else:
                    segments.append(dict(episode_index=episode_id, task_index=task_id, start=start, stop=stop))
        if file_number % 50 == 0:
            print(f"indexed file {file_number+1}/{len(files)}", flush=True)
    if previous_index+1 != info["total_frames"] or any(observed[k] != lengths[k] for k in ids):
        raise ValueError("Source row coverage mismatch")
    windows = [dict(segment, stop=segment["stop"]-96) for segment in segments if segment["stop"]-segment["start"] > 96]
    count = sum(s["stop"]-s["start"] for s in windows)
    if not count:
        raise ValueError("No valid four-block windows")
    if not pilot_episodes and split == "train" and count != norm["valid_windows"]:
        raise ValueError("Loader window population does not match full statistics")
    output.mkdir(parents=True)
    meta = output / "meta"
    meta.mkdir()
    pq.write_table(pa.Table.from_pandas(episodes, preserve_index=False), output / "episodes.parquet")
    pq.write_table(pa.Table.from_pylist(groups), output / "row_groups.parquet")
    pq.write_table(pa.Table.from_pylist(windows), output / "windows.parquet")
    atomic_json(meta / "info.json", info)
    modality = {kind: {key: dict(original_key="observation.state" if kind == "state" else "action", start=a, end=b, dtype="float32", absolute=True)
                       for key, (a,b) in SLICES.items()} for kind in ("state", "action")}
    modality["video"] = {key: dict(original_key=value) for key, value in CAMERAS.items()}
    modality["annotation"] = {"task": dict(original_key="task_index")}
    atomic_json(meta / "modality.json", modality)
    atomic_json(meta / "embodiment.json", {"embodiment_tag": "yam"})
    (meta / "tasks.jsonl").write_text("".join(json.dumps(row)+"\n" for row in tasks.to_dict("records")))
    for name in ("stats.json", "relative_stats_dreamzero.json"):
        shutil.copyfile(normalizer / "meta" / name, meta / name)
    shutil.copyfile(normalizer / "norm_stats_manifest.json", output / "norm_stats_manifest.json")
    names = ["episodes.parquet", "row_groups.parquet", "windows.parquet", "norm_stats_manifest.json"]
    names += [str(p.relative_to(output)) for p in meta.iterdir()]
    manifest = dict(schema_version=1, source_root=str(source.resolve()), split=split, seed=seed,
                    episode_count=len(episodes), selected_episode_ids=sorted(ids),
                    valid_windows=count, selected_rows=int(episodes.length.sum()),
                    excluded_window_starts=int(episodes.length.sum())-count,
                    camera_file_existence_checked=check_files, decoded_video_checks=0,
                    normalizer_root=str(normalizer.resolve()),
                    population="all-valid-four-block-window-starts-uniform",
                    files_sha256={name: sha256(output / name) for name in names})
    atomic_json(output / "manifest.json", manifest)
    print(f"Prepared {len(episodes)} episodes, {count} valid windows at {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--normalizer-root", type=Path, required=True)
    parser.add_argument("--pilot-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    args = parser.parse_args()
    if args.pilot_episodes < 0:
        parser.error("pilot episode count must be nonnegative")
    prepare(args.source_root, args.output_root, args.normalizer_root, args.pilot_episodes, args.seed, args.split)


if __name__ == "__main__":
    main()
