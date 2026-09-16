#!/usr/bin/env python3
"""Verify a task subset, optionally checking real numeric windows and video decoding."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare import ABCV3Storage, select_task, sha256, verify_normalizer
from abc_v3_storage import CAMERAS
from yam_temporal import YAMTemporalPlan


def validate(root, samples=0, decode_video=False, expected_split=None):
    root = Path(root)
    storage = ABCV3Storage(root)
    verify_normalizer(root)
    manifest = storage.manifest
    if expected_split is not None and manifest["split"] != expected_split:
        raise ValueError(f"Expected {expected_split} metadata")
    selection = manifest.get("task_selection")
    if not selection:
        raise ValueError("Expected task-specific metadata; run task_specific/prepare.py first")
    task_id, episodes, windows = select_task(storage, selection["task"])
    count = int((windows.stop - windows.start).sum())
    if (task_id != selection["task_index"] or len(windows) != len(storage.windows)
            or len(episodes) != len(storage.episodes) or count != manifest["valid_windows"]
            or len(episodes) != manifest["episode_count"]):
        raise ValueError("Task subset population/count mismatch")
    if sha256(root / "norm_stats_manifest.json") != selection["normalization_manifest_sha256"]:
        raise ValueError("Task subset normalizer provenance mismatch")
    if samples < 0 or (decode_video and samples == 0):
        raise ValueError("Use positive --samples with --decode-video")
    decoder = None
    if decode_video:
        from lerobot_v3_video import V3VideoReader
        decoder = V3VideoReader()
    checks = []
    ends = (windows.stop - windows.start).cumsum().to_numpy()
    rng = np.random.default_rng(42)
    for draw in rng.integers(count, size=samples):
        pos = int(np.searchsorted(ends, draw, side="right"))
        row = windows.iloc[pos]
        start = int(row.start + draw - (ends[pos - 1] if pos else 0))
        episode = int(row.episode_index)
        data = storage.read_episode(episode)
        plan = YAMTemporalPlan(start).validate(len(data["index"]), data["task_index"])
        if int(data["task_index"][start]) != task_id:
            raise ValueError("Numeric source task does not match filtered metadata")
        relative = plan.relative_actions(data["observation.state"], data["action"])
        if relative.shape != (96, 14) or not np.isfinite(relative).all():
            raise ValueError("Invalid relative actions")
        if decoder:
            for camera in CAMERAS:
                path, times, interval = storage.camera_request(episode, camera, data["timestamp"][plan.video_indices])
                frames, _ = decoder.read(path, times, interval)
                if len(frames) != 33:
                    raise ValueError("Expected 33 frames per camera")
        checks.append(dict(episode=episode, start=start))
    return dict(task=selection["task"], task_index=task_id, split=manifest["split"],
                episodes=len(episodes), valid_windows=count, numeric_checks=checks,
                decoded_windows=len(checks) if decoder else 0,
                normalization_policy=selection["normalization_policy"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument("--expected-split", choices=["train", "validation"])
    args = parser.parse_args()
    print(json.dumps(validate(args.metadata_root, args.samples, args.decode_video, args.expected_split), indent=2))


if __name__ == "__main__":
    main()
