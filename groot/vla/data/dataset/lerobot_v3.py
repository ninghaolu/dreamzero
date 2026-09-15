"""ABC v3 adapter for DreamZero's YAM transforms and sharded trainer."""
from collections.abc import Mapping, Sequence
import json
import numpy as np
import pandas as pd
from torch.utils.data import get_worker_info

from .abc_v3_storage import ABCV3Storage, CAMERAS, SLICES, verify_normalizer
from .yam_temporal import YAMTemporalPlan, WindowRanges
from .lerobot_v3_video import V3VideoReader
from .lerobot import LeRobotSingleDataset
from .lerobot_sharded import ShardedLeRobotMixtureDataset
from ..schema.lerobot import DatasetStatisticalValues


class EpisodeWindows(Mapping):
    def __init__(self, windows, ids):
        self.ids = ids
        values = windows[["episode_index", "start", "stop"]].to_numpy(dtype=np.int64)
        boundaries = np.concatenate([[0], np.flatnonzero(np.diff(values[:, 0]))+1, [len(values)]])
        self.ranges = {int(values[a, 0]): WindowRanges(values[a:b, 1:])
                       for a, b in zip(boundaries[:-1], boundaries[1:])}
        self.empty = WindowRanges([])

    def __getitem__(self, ep):
        return self.ranges.get(int(ep), self.empty)

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)


class AllWindows(Sequence):
    def __init__(self, ids, windows):
        self.ids, self.windows = np.asarray(ids), windows
        self.ends = np.cumsum([len(windows[ep]) for ep in ids])

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        pos = int(np.searchsorted(self.ends, index, side="right"))
        ep = int(self.ids[pos])
        offset = int(self.ends[pos-1]) if pos else 0
        return ep, self.windows[ep][index-offset]


class ShardedLeRobotV3YAMDataset(LeRobotSingleDataset):
    def __init__(self, dataset_path, *args, numeric_cache_bytes=128*1024**2,
                 video_cache_bytes=32*1024**2, max_video_readers=3,
                 num_steps_per_shard=10000, source_root=None, **kwargs):
        print(f"Loading ABC v3 index: {dataset_path}", flush=True)
        self.storage = ABCV3Storage(dataset_path, numeric_cache_bytes, source_root)
        verify_normalizer(dataset_path)
        if kwargs.get("use_global_metadata", False) or kwargs.get("relative_action_per_horizon", False):
            raise ValueError("ABC requires local pooled relative normalization")
        if kwargs.get("max_chunk_size", 4) != 4:
            raise ValueError("ABC normalizer requires exactly four blocks")
        kwargs.update(use_global_metadata=False, relative_action=True,
                      relative_action_keys=list(SLICES), relative_action_per_horizon=False,
                      max_chunk_size=4, discard_bad_trajectories=False)
        self.id_to_position = {int(ep): i for i, ep in enumerate(self.storage.episodes.index)}
        self.video_reader = V3VideoReader(max_video_readers, video_cache_bytes)
        self.num_steps_per_shard = int(num_steps_per_shard)
        if self.num_steps_per_shard < 1:
            raise ValueError("Shard size must be positive")
        super().__init__(dataset_path, *args, **kwargs)
        print(f"ABC v3: {len(self.trajectory_ids)} episodes, {len(self)} valid windows", flush=True)
        self.sharded_trajectories, self.shard_lengths = [], []
        current, count = [], 0
        for ep in self.trajectory_ids:
            length = len(self.step_filter[ep])
            if not length:
                continue
            current.append(int(ep))
            count += length
            if count >= self.num_steps_per_shard:
                self.sharded_trajectories.append(current)
                self.shard_lengths.append(count)
                current, count = [], 0
        if current:
            self.sharded_trajectories.append(current)
            self.shard_lengths.append(count)
        self.shard_lengths = np.asarray(self.shard_lengths, dtype=np.int64)

    def _get_trajectories(self):
        return self.storage.episodes.index.to_numpy(), self.storage.episodes.length.to_numpy()

    def get_trajectory_index(self, trajectory_id):
        return self.id_to_position[int(trajectory_id)]

    def _get_step_filter(self):
        return EpisodeWindows(self.storage.windows, self.trajectory_ids)

    def _get_all_steps(self):
        return AllWindows(self.trajectory_ids, self.step_filter)

    def _get_chunk_size(self):
        return int(self.lerobot_info_meta.get("chunks_size", 1000))

    def _get_lerobot_stats_meta(self):
        values = json.loads((self.dataset_path / "meta/stats.json").read_text())
        return {key: DatasetStatisticalValues.model_validate(values[key]) for key in ("observation.state", "action")}

    def _get_lerobot_relative_stats_meta(self):
        values = json.loads((self.dataset_path / "meta/relative_stats_dreamzero.json").read_text())
        return {key: DatasetStatisticalValues.model_validate(values[key]) for key in SLICES}

    def _get_max_delta_index(self):
        return 96

    def get_trajectory_data(self, trajectory_id):
        rows = self.storage.read_episode(trajectory_id)
        return pd.DataFrame({k: list(v) if v.ndim == 2 else v for k,v in rows.items()})

    def raw_sample(self, trajectory_id, start, decode_video=True):
        rows = self.storage.read_episode(trajectory_id)
        plan = YAMTemporalPlan(int(start)).validate(len(rows["index"]), rows["task_index"])
        relative = plan.relative_actions(rows["observation.state"], rows["action"])
        data = {}
        for key, (a,b) in SLICES.items():
            data[f"state.{key}"] = rows["observation.state"][plan.state_indices, a:b]
            data[f"action.{key}"] = relative[:, a:b]
        data["annotation.task"] = [self.storage.tasks[int(rows["task_index"][start])]]
        if decode_video:
            for camera in CAMERAS:
                path, timestamps, interval = self.storage.camera_request(trajectory_id, camera, rows["timestamp"][plan.video_indices])
                frames, _ = self.video_reader.read(path, timestamps, interval)
                data[f"video.{camera}"] = frames
        return data

    def get_step_data(self, trajectory_id, indices):
        return self.raw_sample(trajectory_id, int(next(iter(indices.values()))[0]))


class ABCV3MixtureDataset(ShardedLeRobotMixtureDataset):
    """Uniform eligible windows with bounded indexing and a finite shard schedule.

    Subclassing preserves BaseTrainer's special iterable DataLoader integration.
    Numeric/video caches belong to each worker; there is no next-shard prefetch.
    """
    def schedule_positions(self, rank, world_size, worker_id=0, num_workers=1):
        return range(rank*num_workers+worker_id, self.num_shards_to_sample, world_size*num_workers)

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        schedule = self.generate_shards_sample_schedule()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.rank, wid]))
        for pos in self.schedule_positions(self.rank, self.world_size, wid, workers):
            dataset_id, shard = schedule[pos]
            dataset = self.datasets[dataset_id]
            windows = AllWindows(dataset.sharded_trajectories[shard], dataset.step_filter)
            count = int(dataset.num_steps_per_shard * self.shard_sampling_rate)
            if count < 1:
                raise ValueError("Shard sampling rate yields no samples")
            for index in rng.integers(len(windows), size=count):
                ep, start = windows[int(index)]
                yield dataset.transforms(dataset.raw_sample(ep, start))

    def __len__(self):
        # Length per rank before DataLoader worker partitioning.
        return sum(int(self.datasets[self.shards_sample_schedule[pos][0]].num_steps_per_shard * self.shard_sampling_rate)
                   for pos in self.schedule_positions(self.rank, self.world_size))
