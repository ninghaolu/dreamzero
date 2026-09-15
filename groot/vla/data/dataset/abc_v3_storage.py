"""Reusable numeric/metadata ABC storage; no DreamZero or GPU imports."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CAMERAS = {"top_camera-images-rgb": "observation.images.top",
           "left_camera-images-rgb": "observation.images.left_wrist",
           "right_camera-images-rgb": "observation.images.right_wrist"}
SLICES = {"left_joint_pos": (0, 6), "left_gripper_pos": (6, 7),
          "right_joint_pos": (7, 13), "right_gripper_pos": (13, 14)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_normalizer(root):
    root = Path(root)
    manifest = json.loads((root / "norm_stats_manifest.json").read_text())
    expected = dict(block_length=24, num_blocks=4, action_dim=14, relative_grippers=True,
                    all_source_files_used=True, population="all-valid-four-block-window-starts-uniform")
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Incompatible or incomplete DreamZero normalizer")
    for name, digest in manifest["output_sha256"].items():
        if sha256(root / "meta" / name) != digest:
            raise ValueError(f"Normalizer hash mismatch: {name}")
    relative = json.loads((root / "meta/relative_stats_dreamzero.json").read_text())
    for key, (start, end) in SLICES.items():
        for stat in ("mean", "std", "min", "max", "q01", "q99"):
            val = np.asarray(relative[key][stat])
            if val.shape != (end-start,) or not np.isfinite(val).all():
                raise ValueError(f"Invalid normalization statistic {key}/{stat}")
    return manifest


class ABCV3Storage:
    def __init__(self, metadata_root, numeric_cache_bytes=128 * 1024**2, source_root=None):
        self.metadata_root = Path(metadata_root)
        self.manifest = json.loads((self.metadata_root / "manifest.json").read_text())
        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported ABC metadata version")
        for name, digest in self.manifest["files_sha256"].items():
            if sha256(self.metadata_root / name) != digest:
                raise ValueError(f"Prepared metadata hash mismatch: {name}")
        self.source_root = Path(source_root or self.manifest["source_root"])
        self.episodes = pq.read_table(self.metadata_root / "episodes.parquet").to_pandas().set_index("episode_index")
        self.row_groups = pq.read_table(self.metadata_root / "row_groups.parquet").to_pylist()
        self.group_ends = np.array([r["end"] for r in self.row_groups])
        self.windows = pq.read_table(self.metadata_root / "windows.parquet").to_pandas()
        self.tasks = {r["task_index"]: r["task"] for r in
                      map(json.loads, (self.metadata_root / "meta/tasks.jsonl").read_text().splitlines())}
        if numeric_cache_bytes < 1:
            raise ValueError("Numeric cache limit must be positive")
        self.numeric_cache_bytes = numeric_cache_bytes
        self.reset_cache()

    def reset_cache(self):
        self.cache = OrderedDict()
        self.cached_bytes = 0
        self.pid = os.getpid()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(cache=OrderedDict(), cached_bytes=0, pid=None)
        return state

    def read_episode(self, episode_id):
        if self.pid != os.getpid():
            self.reset_cache()
        episode_id = int(episode_id)
        if episode_id in self.cache:
            self.cache.move_to_end(episode_id)
            return self.cache[episode_id]
        episode = self.episodes.loc[episode_id]
        start, end = int(episode.dataset_from_index), int(episode.dataset_to_index)
        if (end-start) * 152 > self.numeric_cache_bytes:
            raise ValueError(f"Episode {episode_id} exceeds the numeric cache limit before reading")
        columns = ["observation.state", "action", "timestamp", "index", "frame_index", "episode_index", "task_index"]
        parts = {k: [] for k in columns}
        pos = int(np.searchsorted(self.group_ends, start, side="right"))
        for group in self.row_groups[pos:]:
            if group["start"] >= end:
                break
            pf = pq.ParquetFile(self.source_root / group["path"])
            for batch in pf.iter_batches(batch_size=8192, row_groups=[group["row_group"]], columns=columns, use_threads=False):
                indices = batch.column("index").to_numpy()
                selected = (indices >= start) & (indices < end)
                if not selected.any():
                    continue
                table = batch.filter(selected)
                for name in columns:
                    column = table.column(name)
                    if column.null_count:
                        raise ValueError(f"Null numeric column {name}")
                    if name in ("observation.state", "action"):
                        if column.flatten().null_count:
                            raise ValueError("Null vector value")
                        array = np.array(column.to_pylist(), dtype=np.float32)
                    else:
                        array = column.to_numpy()
                    parts[name].append(array)
        rows = {k: np.concatenate(v) for k, v in parts.items() if v}
        if set(rows) != set(columns) or len(rows["index"]) != end-start:
            raise ValueError(f"Episode length mismatch: {episode_id}")
        if (not np.array_equal(rows["index"], np.arange(start, end)) or
            not np.array_equal(rows["frame_index"], np.arange(end-start)) or
            not np.all(rows["episode_index"] == episode_id)):
            raise ValueError(f"Episode row identity mismatch: {episode_id}")
        for key in ("observation.state", "action"):
            if rows[key].shape != (end-start, 14) or not np.isfinite(rows[key]).all():
                raise ValueError(f"Invalid 14-D vectors: {key}")
        tolerance = max(1e-4, (end-start)/30 * np.finfo(np.float32).eps * 2)
        if not np.allclose(rows["timestamp"], np.arange(end-start)/30, atol=tolerance, rtol=0):
            raise ValueError(f"Unexpected episode-local timestamps: {episode_id}")
        size = sum(a.nbytes for a in rows.values())
        if size > self.numeric_cache_bytes:
            raise ValueError(f"Episode {episode_id} needs {size} bytes, above numeric cache limit")
        while self.cache and self.cached_bytes + size > self.numeric_cache_bytes:
            _, old = self.cache.popitem(last=False)
            self.cached_bytes -= sum(a.nbytes for a in old.values())
        self.cache[episode_id] = rows
        self.cached_bytes += size
        return rows

    def camera_request(self, episode_id, camera, local_timestamps):
        source_key = CAMERAS[camera]
        row = self.episodes.loc[int(episode_id)]
        prefix = f"videos/{source_key}/"
        start, end = float(row[prefix+"from_timestamp"]), float(row[prefix+"to_timestamp"])
        path = self.source_root / f"videos/{source_key}/chunk-{int(row[prefix+'chunk_index']):03d}/file-{int(row[prefix+'file_index']):03d}.mp4"
        return path, np.asarray(local_timestamps, dtype=np.float64) + start, (start, end)
