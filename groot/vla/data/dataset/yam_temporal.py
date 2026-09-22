"""ABC temporal contracts shared by preparation, statistics, loading and evaluation."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class TemporalConfig:
    block_length: int = 24
    video_stride: int = 3
    num_blocks: int = 4
    action_rate: int = 30
    num_frame_per_block: int = 2
    num_state_per_block: int = 1
    num_frames: int = 33

    def __post_init__(self):
        if (self.block_length, self.video_stride) not in ((24, 3), (48, 6)):
            raise ValueError("ABC supports H24/stride3 or H48/stride6")
        if (self.num_blocks, self.action_rate, self.num_frame_per_block,
                self.num_state_per_block, self.num_frames) != (4, 30, 2, 1, 33):
            raise ValueError("ABC requires 30 Hz actions, K2/M4, one state per chunk and 33 frames")

    @classmethod
    def from_name(cls, name):
        if name not in ("h24", "h48"):
            raise ValueError(f"Unknown ABC temporal preset: {name}")
        return cls(block_length=int(name[1:]), video_stride=3 if name == "h24" else 6)

    @classmethod
    def from_manifest(cls, manifest):
        # Original version-1 artifacts predate temporal metadata and are H24 only.
        return cls(**manifest["temporal"]) if "temporal" in manifest else cls()

    @property
    def endpoint(self):
        return self.block_length * self.num_blocks

    @property
    def name(self):
        return f"h{self.block_length}"

    @property
    def video_fps(self):
        return self.action_rate / self.video_stride

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class YAMTemporalPlan:
    start: int
    temporal: TemporalConfig = TemporalConfig()

    @property
    def state_indices(self):
        return self.start + np.arange(0, self.temporal.endpoint, self.temporal.block_length)

    @property
    def action_indices(self):
        return self.start + np.arange(self.temporal.endpoint)

    @property
    def video_indices(self):
        return self.start + np.arange(0, self.temporal.endpoint + 1, self.temporal.video_stride)

    def validate(self, episode_length, task_indices=None):
        endpoint = self.start + self.temporal.endpoint
        if self.start < 0 or endpoint >= episode_length:
            raise ValueError("Incomplete four-block window")
        if task_indices is not None:
            tasks = task_indices[self.start:endpoint + 1]
            if len(tasks) != self.temporal.endpoint + 1 or np.any(tasks != tasks[0]):
                raise ValueError("Window crosses a task boundary")
        return self

    def relative_actions(self, state, action):
        t = self.temporal
        targets = action[self.action_indices].reshape(t.num_blocks, t.block_length, -1)
        return (targets - state[self.state_indices, None, :]).reshape(t.endpoint, -1)


class WindowRanges:
    """Indexable union of half-open integer ranges, without per-frame arrays."""
    def __init__(self, ranges):
        self.ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
        if np.any(self.ranges[:, 1] <= self.ranges[:, 0]):
            raise ValueError("Empty/reversed window range")
        self.ends = np.cumsum(self.ranges[:, 1] - self.ranges[:, 0])

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        indices = np.asarray(index, dtype=np.int64)
        indices = np.where(indices < 0, indices + len(self), indices)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError(index)
        pos = np.searchsorted(self.ends, indices, side="right")
        before = np.concatenate([[0], self.ends])[pos]
        result = self.ranges[pos, 0] + indices - before
        return int(result) if result.ndim == 0 else result
