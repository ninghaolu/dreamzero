"""CPU-only, streaming ABC v3 statistics for the four-block DreamZero recipe.

Population: each complete, task-consistent 97-row window once, four blocks of
24 relative actions per window. No boundary padding; all 14 dimensions relative.
Numeric files stay in place. State statistics come from source meta/stats.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BLOCK = 24
BLOCKS = 4
ENDPOINT = BLOCK * BLOCKS
DIM = 14
SLICES = {"left_joint_pos": (0, 6), "left_gripper_pos": (6, 7),
          "right_joint_pos": (7, 13), "right_gripper_pos": (13, 14)}
FIELDS = ("min", "max", "mean", "std", "q01", "q99")
COLUMNS = ["observation.state", "action", "episode_index", "frame_index", "index", "task_index"]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def save_part(path, identity, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        np.savez_compressed(f, identity=np.array(json.dumps(identity, sort_keys=True)), **arrays)
        tmp = f.name
    os.replace(tmp, path)


def load_part(path, identity):
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["identity"])) != identity:
            raise ValueError(f"Incompatible intermediate: {path}; use a fresh work directory")
        return {k: f[k].copy() for k in f.files if k != "identity"}


def file_inventory(root):
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise ValueError(f"No v3 data files under {root}")
    return files


def source_identity(root, files, bins, shards):
    # Metadata contents + file names/sizes/mtimes; deliberately not a payload hash.
    digest = hashlib.sha256()
    for name in ("meta/info.json", "meta/stats.json"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    for path in files:
        stat = path.stat()
        digest.update(json.dumps([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns]).encode())
    return dict(schema_version=1, dataset_root=str(root.resolve()),
                source_fingerprint=digest.hexdigest(), total_files=len(files),
                fingerprint_method="metadata-content-and-data-path-size-mtime",
                block_length=BLOCK, num_blocks=BLOCKS, action_dim=DIM,
                relative_grippers=True, population="all-valid-four-block-window-starts-uniform",
                require_video_endpoint_row=True, boundary_policy="exclude-episode-or-task-crossing",
                num_bins=bins, num_shards=shards)


def arrays_from_batch(batch):
    result = {}
    for name in COLUMNS:
        col = batch.column(batch.schema.get_field_index(name))
        if col.null_count:
            raise ValueError(f"Nulls in {name}")
        if name in ("observation.state", "action"):
            if pa.types.is_fixed_size_list(col.type):
                if col.type.list_size != DIM:
                    raise ValueError(f"Invalid width for {name}")
            elif pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                if not np.all(np.diff(col.offsets.to_numpy()) == DIM):
                    raise ValueError(f"Invalid width for {name}")
            else:
                raise ValueError(f"Expected vector column {name}")
            values = col.flatten()
            if values.null_count:
                raise ValueError(f"Null vector elements in {name}")
            arr = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, DIM)
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values in {name}")
        else:
            arr = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.int64)
        result[name] = arr
    return result


def validate_rows(rows):
    if np.any(np.diff(rows["index"]) != 1):
        raise ValueError("Non-contiguous global row indices")
    same = np.diff(rows["episode_index"]) == 0
    if np.any(np.diff(rows["episode_index"]) < 0):
        raise ValueError("Episode IDs out of order")
    if np.any(np.diff(rows["frame_index"])[same] != 1):
        raise ValueError("Non-contiguous episode frames")
    if np.any(rows["frame_index"][1:][~same] != 0):
        raise ValueError("New episode does not start at frame zero")


def owned_batches(files, file_index, batch_rows):
    """Bounded batches with 96-row lookahead across row groups and files.

    Every anchor belongs to exactly its starting data file, even across shards.
    Only the current file is consumed fully; following files supply lookahead.
    """
    remaining = pq.ParquetFile(files[file_index]).metadata.num_rows
    if remaining == 0:
        raise ValueError(f"Empty file: {files[file_index]}")
    buffer = None
    for path in files[file_index:]:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=COLUMNS, use_threads=False):
            rows = arrays_from_batch(batch)
            buffer = rows if buffer is None else {k: np.concatenate([buffer[k], rows[k]]) for k in COLUMNS}
            validate_rows(buffer)
            ready = min(remaining, len(buffer["index"]) - ENDPOINT)
            while ready > 0:
                count = min(ready, batch_rows)
                yield {k: v[:count + ENDPOINT] for k, v in buffer.items()}, count
                buffer = {k: v[count:] for k, v in buffer.items()}
                remaining -= count
                if remaining == 0:
                    return
                ready = min(remaining, len(buffer["index"]) - ENDPOINT)
    if remaining:
        # At dataset end no complete window can begin in the last 96 rows.
        yield buffer, remaining


def valid_starts(rows, owned_count):
    n = min(owned_count, max(0, len(rows["index"]) - ENDPOINT))
    starts = np.arange(n)
    # Prefix counts detect any interior task change, including A -> B -> A.
    changes = ((np.diff(rows["episode_index"]) != 0) |
               (np.diff(rows["task_index"]) != 0))
    prefix = np.concatenate([[0], np.cumsum(changes)])
    return starts[prefix[starts + ENDPOINT] == prefix[starts]]


def action_batches(rows, starts):
    for block_start in range(0, ENDPOINT, BLOCK):
        anchors = starts + block_start
        reference = rows["observation.state"][anchors]
        for offset in range(BLOCK):
            yield rows["action"][anchors + offset] - reference


def empty_moments():
    return dict(count=np.array(0, dtype=np.int64), mean=np.zeros(DIM), m2=np.zeros(DIM),
                minimum=np.full(DIM, np.inf), maximum=np.full(DIM, -np.inf))


def merge_moments(a, b):
    na, nb = int(a["count"]), int(b["count"])
    if not nb:
        return
    delta = b["mean"] - a["mean"]
    a["mean"] += delta * (nb / (na + nb))
    a["m2"] += b["m2"] + delta ** 2 * (na * nb / (na + nb))
    a["count"] = np.array(na + nb, dtype=np.int64)
    a["minimum"] = np.minimum(a["minimum"], b["minimum"])
    a["maximum"] = np.maximum(a["maximum"], b["maximum"])


def update_moments(acc, values):
    if len(values) == 0:
        return
    if not np.isfinite(values).all():
        raise ValueError("Non-finite relative actions")
    values = values.astype(np.float64)
    mean = values.mean(axis=0)
    merge_moments(acc, dict(count=len(values), mean=mean,
                           m2=np.sum((values - mean) ** 2, axis=0),
                           minimum=values.min(axis=0), maximum=values.max(axis=0)))


def update_histogram(hist, values, bounds):
    if len(values) == 0:
        return
    low, high = bounds["minimum"], bounds["maximum"]
    if not np.isfinite(values).all() or np.any(values < low) or np.any(values > high):
        raise ValueError("Values outside pass1 bounds: source changed between passes")
    span = high - low
    scaled = (values.astype(np.float64) - low) / np.where(span > 0, span, 1)
    indices = np.clip(np.floor(scaled * hist.shape[1]).astype(np.int64), 0, hist.shape[1] - 1)
    indices += np.arange(DIM) * hist.shape[1]
    hist += np.bincount(indices.ravel(), minlength=hist.size).reshape(hist.shape)


def histogram_quantile(hist, bounds, quantile):
    """Approximate NumPy's linear quantile; error <= one histogram bin width."""
    count = int(bounds["count"])
    rank = (count - 1) * quantile
    lo, hi = int(np.floor(rank)), int(np.ceil(rank))
    result = []
    for dim in range(DIM):
        cdf = np.cumsum(hist[dim])
        bins = np.searchsorted(cdf, [lo, hi], side="right")
        width = (bounds["maximum"][dim] - bounds["minimum"][dim]) / hist.shape[1]
        estimates = bounds["minimum"][dim] + (bins + 0.5) * width
        result.append(estimates[0] * (1 - (rank - lo)) + estimates[1] * (rank - lo))
    return np.asarray(result)


def part_path(work, stage, shard):
    return work / stage / f"part-{shard:05d}.npz"


def run_map(args, files, identity):
    shard = args.shard_index
    if not 0 <= shard < args.num_shards:
        raise ValueError("Invalid shard index")
    path = part_path(args.work_dir, args.stage, shard)
    part_identity = dict(identity, shard_index=shard, stage=args.stage)
    if path.exists():
        load_part(path, part_identity)
        print(f"Already complete: {path}", flush=True)
        return
    bounds = load_part(args.work_dir / "bounds.npz", dict(identity, stage="bounds")) if args.stage == "pass2" else None
    acc = empty_moments()
    hist = np.zeros((DIM, args.num_bins), dtype=np.int64)
    owned = windows = 0
    for file_index in range(shard, len(files), args.num_shards):
        for rows, count in owned_batches(files, file_index, args.batch_rows):
            starts = valid_starts(rows, count)
            owned += count
            windows += len(starts)
            for values in action_batches(rows, starts):
                if bounds is None:
                    update_moments(acc, values)
                else:
                    update_histogram(hist, values, bounds)
        print(f"{args.stage} shard={shard} file={file_index} owned_rows={owned} valid_windows={windows}", flush=True)
    arrays = acc if bounds is None else dict(histogram=hist)
    save_part(path, part_identity, owned_rows=np.array(owned), windows=np.array(windows), **arrays)


def run_reduce(args, files, identity):
    stage = "pass1" if args.stage == "reduce-pass1" else "pass2"
    acc = empty_moments()
    hist = np.zeros((DIM, args.num_bins), dtype=np.int64)
    owned = windows = 0
    for shard in range(args.num_shards):
        data = load_part(part_path(args.work_dir, stage, shard), dict(identity, shard_index=shard, stage=stage))
        owned += int(data["owned_rows"])
        windows += int(data["windows"])
        if stage == "pass1":
            if int(data["count"]) != int(data["windows"]) * ENDPOINT:
                raise ValueError("Partial action count mismatch")
            merge_moments(acc, data)
        else:
            if not np.all(data["histogram"].sum(axis=1) == int(data["windows"]) * ENDPOINT):
                raise ValueError("Partial histogram count mismatch")
            hist += data["histogram"]
    actual_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    if owned != actual_rows or (not args.max_files and owned != info["total_frames"]):
        raise ValueError("Processed row count differs from file/source metadata")
    if windows == 0:
        raise ValueError("No eligible four-block windows")
    if stage == "pass1":
        save_part(args.work_dir / "bounds.npz", dict(identity, stage="bounds"),
                  owned_rows=np.array(owned), windows=np.array(windows), **acc)
        return
    bounds = load_part(args.work_dir / "bounds.npz", dict(identity, stage="bounds"))
    if windows != int(bounds["windows"]) or owned != int(bounds["owned_rows"]):
        raise ValueError("Pass1/pass2 population mismatch")
    stats = dict(min=bounds["minimum"], max=bounds["maximum"], mean=bounds["mean"],
                 std=np.sqrt(bounds["m2"] / int(bounds["count"])),
                 q01=histogram_quantile(hist, bounds, .01), q99=histogram_quantile(hist, bounds, .99))
    relative = {key: {name: val[start:end].tolist() for name, val in stats.items()}
                for key, (start, end) in SLICES.items()}
    source_stats = json.loads((args.dataset_root / "meta/stats.json").read_text())
    state = source_stats["observation.state"]
    for key in FIELDS:
        value = np.asarray(state[key])
        if value.shape != (DIM,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid source state statistic: {key}")
    if np.any(np.asarray(state["std"]) < 0) or np.any(np.asarray(state["q01"]) > np.asarray(state["q99"])):
        raise ValueError("Invalid state statistics ordering")
    meta = args.output_dir / "meta"
    atomic_json(meta / "stats.json", source_stats)
    atomic_json(meta / "relative_stats_dreamzero.json", relative)
    outputs = {name: hashlib.sha256((meta / name).read_bytes()).hexdigest()
               for name in ("stats.json", "relative_stats_dreamzero.json")}
    atomic_json(args.output_dir / "norm_stats_manifest.json", dict(identity,
                created_utc=datetime.now(timezone.utc).isoformat(),
                processed_rows=owned, valid_windows=windows, excluded_window_starts=owned-windows,
                action_count=int(bounds["count"]), all_source_files_used=not bool(args.max_files),
                state_stats_source=str(args.dataset_root / "meta/stats.json"),
                state_stats_scope="unchanged-source-training-statistics",
                method="two-pass-streaming-moments-fixed-histogram",
                quantiles_are_approximate=True,
                quantile_absolute_error_bound=((bounds["maximum"]-bounds["minimum"])/args.num_bins).tolist(),
                output_sha256=outputs))
    print(f"Wrote statistics and manifest to {args.output_dir}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["count-files", "pass1", "reduce-pass1", "pass2", "finalize"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--num-bins", type=int, default=5000)
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--max-files", type=int, default=0, help="Debug only: truncate file inventory; final tail windows excluded")
    args = parser.parse_args(argv)
    if args.batch_rows <= 0 or args.num_bins < 2 or args.max_files < 0:
        parser.error("Invalid batch size, bin count or file limit")
    files = file_inventory(args.dataset_root)
    if args.max_files:
        files = files[:args.max_files]
    if args.stage == "count-files":
        print(len(files))
        return
    if args.work_dir is None or args.output_dir is None:
        parser.error("--work-dir and --output-dir are required")
    if not 1 <= args.num_shards <= len(files):
        parser.error("Shard count must be between 1 and the file count")
    if args.dataset_root.resolve() in [args.work_dir.resolve(), args.output_dir.resolve()] or any(
        args.dataset_root.resolve() in p.resolve().parents for p in [args.work_dir, args.output_dir]
    ):
        parser.error("Keep generated files outside the read-only source dataset")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 30:
        raise ValueError("Expected ABC LeRobot v3.0 at 30 FPS")
    identity = source_identity(args.dataset_root, files, args.num_bins, args.num_shards)
    identity["max_files"] = args.max_files
    output_manifest = args.output_dir / "norm_stats_manifest.json"
    if output_manifest.exists():
        previous = json.loads(output_manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Output directory identity mismatch; use a fresh output directory")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.work_dir / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Work directory identity mismatch; use a fresh work directory")
    atomic_json(manifest, identity)
    if args.stage in ("pass1", "pass2"):
        run_map(args, files, identity)
    else:
        run_reduce(args, files, identity)


if __name__ == "__main__":
    main()
