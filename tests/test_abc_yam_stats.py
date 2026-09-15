"""Independent brute-force targets verify boundaries, weighting and reductions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MODULE = Path(__file__).resolve().parents[1] / "groot/vla/data/dataset/abc_yam_stats.py"
spec = importlib.util.spec_from_file_location("abc_yam_stats", MODULE)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


class StatisticsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        (self.source / "meta").mkdir(parents=True)
        # Non-contiguous episode IDs, interior A -> B -> A task transitions,
        # and a short episode. Splits put valid windows across multiple files.
        lengths = [215, 40, 170]
        self.episodes = np.repeat([3, 8, 21], lengths)
        self.frames = np.concatenate([np.arange(n) for n in lengths])
        self.tasks = np.zeros(sum(lengths), dtype=np.int64)
        self.tasks[100:108] = 1
        n = len(self.frames)
        rng = np.random.default_rng(12)
        self.state = rng.normal(size=(n, 14)).astype(np.float32)
        self.action = (rng.normal(size=(n, 14)) + 3).astype(np.float32)
        # Exercise constant dimension quantile handling.
        self.state[:, 0] = 1
        self.action[:, 0] = 4
        table = pa.table({"observation.state": self.state.tolist(), "action": self.action.tolist(),
                          "episode_index": self.episodes, "frame_index": self.frames,
                          "index": np.arange(n), "task_index": self.tasks})
        boundaries = [0, 75, 113, 287, n]
        directory = self.source / "data/chunk-000"
        directory.mkdir(parents=True)
        for i, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            pq.write_table(table.slice(start, stop-start), directory / f"file-{i:03d}.parquet", row_group_size=19)
        self.source_stats = {"observation.state": {k: ([1.] * 14) for k in stats.FIELDS}}
        (self.source / "meta/stats.json").write_text(json.dumps(self.source_stats))
        (self.source / "meta/info.json").write_text(json.dumps(dict(total_frames=n, fps=30, codebase_version="v3.0")))
        targets, starts = [], []
        for start in range(n - 96):
            if len(set(self.episodes[start:start+97])) != 1 or len(set(self.tasks[start:start+97])) != 1:
                continue
            starts.append(start)
            for block in range(4):
                anchor = start + 24 * block
                for offset in range(24):
                    targets.append(self.action[anchor+offset] - self.state[anchor])
        self.expected = np.asarray(targets)
        self.starts = starts

    def test_streaming_matches_independent_reference(self):
        files = stats.file_inventory(self.source)
        for batch_rows in (7, 64, 1024):
            targets, starts = [], []
            owned = 0
            for i in range(len(files)):
                for rows, count in stats.owned_batches(files, i, batch_rows):
                    valid = stats.valid_starts(rows, count)
                    owned += count
                    starts.extend(rows["index"][valid].tolist())
                    targets.extend(v for v in stats.action_batches(rows, valid) if len(v))
            self.assertEqual(owned, len(self.frames))
            self.assertEqual(starts, self.starts)
            actual = np.concatenate(targets)
            # Order differs (offset-major streaming); compare every dimension's population.
            np.testing.assert_array_equal(np.sort(actual, axis=0), np.sort(self.expected, axis=0))

    def command(self, stage, shard=0, bins=5000):
        stats.main(["--stage", stage, "--dataset-root", str(self.source),
                    "--output-dir", str(self.root / "out"), "--work-dir", str(self.root / "work"),
                    "--num-shards", "2", "--shard-index", str(shard),
                    "--batch-rows", "31", "--num-bins", str(bins)])

    def test_two_pass_pipeline_and_resume(self):
        for stage in ("pass1", "reduce-pass1", "pass2", "finalize"):
            for shard in (range(2) if stage in ("pass1", "pass2") else [0]):
                self.command(stage, shard)
        self.command("pass1", 0)  # compatible completed part is reused
        out = self.root / "out"
        manifest = json.loads((out / "norm_stats_manifest.json").read_text())
        self.assertEqual(manifest["action_count"], len(self.expected))
        self.assertEqual(manifest["valid_windows"], len(self.starts))
        self.assertEqual(json.loads((out / "meta/stats.json").read_text()), self.source_stats)
        relative = json.loads((out / "meta/relative_stats_dreamzero.json").read_text())
        merged = {key: np.concatenate([relative[name][key] for name in stats.SLICES]) for key in stats.FIELDS}
        expected = self.expected.astype(np.float64)
        for key, reference in [("mean", expected.mean(0)), ("std", expected.std(0)),
                               ("min", expected.min(0)), ("max", expected.max(0))]:
            np.testing.assert_allclose(merged[key], reference, atol=1e-12)
        for key, q in [("q01", .01), ("q99", .99)]:
            error = np.abs(merged[key] - np.quantile(expected, q, axis=0))
            self.assertTrue(np.all(error <= np.asarray(manifest["quantile_absolute_error_bound"]) + 1e-12))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.command("pass1", bins=4000)

    def test_missing_parts_fail(self):
        self.command("pass1", 0)
        with self.assertRaises(FileNotFoundError):
            self.command("reduce-pass1")

    def test_noncontiguous_rows_rejected(self):
        files = stats.file_inventory(self.source)
        table = pq.read_table(files[1])
        indices = table["index"].to_numpy().copy()
        indices[3] += 1
        table = table.set_column(table.schema.get_field_index("index"), "index", pa.array(indices))
        pq.write_table(table, files[1])
        with self.assertRaisesRegex(ValueError, "global row"):
            list(stats.owned_batches(files, 0, 32))


if __name__ == "__main__":
    unittest.main()
