"""The fixed four-block ABC/YAM temporal contract, without model dependencies."""
from dataclasses import dataclass
import numpy as np

BLOCK_LENGTH = 24
NUM_BLOCKS = 4
ENDPOINT = BLOCK_LENGTH * NUM_BLOCKS


@dataclass(frozen=True)
class YAMTemporalPlan:
    start: int

    @property
    def state_indices(self):
        return self.start + np.arange(0, ENDPOINT, BLOCK_LENGTH)

    @property
    def action_indices(self):
        return self.start + np.arange(ENDPOINT)

    @property
    def video_indices(self):
        return self.start + np.arange(0, ENDPOINT + 1, 3)

    def validate(self, episode_length, task_indices=None):
        if self.start < 0 or self.start + ENDPOINT >= episode_length:
            raise ValueError("Incomplete four-block window")
        if task_indices is not None:
            tasks = task_indices[self.start:self.start + ENDPOINT + 1]
            if np.any(tasks != tasks[0]):
                raise ValueError("Window crosses a task boundary")
        return self

    def relative_actions(self, state, action):
        targets = action[self.action_indices].reshape(NUM_BLOCKS, BLOCK_LENGTH, -1)
        return (targets - state[self.state_indices, None, :]).reshape(ENDPOINT, -1)


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
