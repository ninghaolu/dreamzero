import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from test_abc_yam_v3 import make_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/train/task_specific"))
spec = importlib.util.spec_from_file_location("task_prepare", ROOT / "scripts/train/task_specific/prepare.py")
task_prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_prepare)
from validate import validate


def refresh_hashes(root):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files_sha256"] = {name: task_prepare.sha256(root / name) for name in manifest["files_sha256"]}
    (root / "manifest.json").write_text(json.dumps(manifest))


class TaskSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source, cls.train, _, _ = make_fixture(cls.root)
        cls.val = cls.root / "prepared_validation"
        val_source = cls.root / "validation_source"
        shutil.copytree(cls.train, cls.val)
        shutil.copytree(cls.source, val_source)
        # Deliberately different IDs in validation, as in the real ABC dataset.
        (cls.val / "meta/tasks.jsonl").write_text(
            '{"task_index": 8, "task": "task zero"}\n{"task_index": 7, "task": "task one"}\n')
        windows = pq.read_table(cls.val / "windows.parquet").to_pandas()
        windows.task_index = windows.task_index.map({0: 8, 1: 7})
        pq.write_table(pa.Table.from_pandas(windows, preserve_index=False), cls.val / "windows.parquet")
        for path in val_source.glob("data/**/*.parquet"):
            table = pq.read_table(path)
            column = pa.array([8 if v == 0 else 7 for v in table["task_index"].to_pylist()])
            # Retain original row-group boundaries used by row_groups.parquet.
            pq.write_table(table.set_column(table.schema.get_field_index("task_index"), "task_index", column), path, row_group_size=23)
        manifest = json.loads((cls.val / "manifest.json").read_text())
        manifest.update(split="validation", source_root=str(val_source))
        (cls.val / "manifest.json").write_text(json.dumps(manifest))
        refresh_hashes(cls.val)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_selection_different_split_ids_and_real_samples(self):
        output = self.root / "task_one"
        original_hash = task_prepare.sha256(self.train / "manifest.json")
        summary = task_prepare.prepare(self.train, self.val, output, "task one")
        self.assertEqual(summary["train"]["task_index"], 1)
        self.assertEqual(summary["validation"]["task_index"], 7)
        for split in ("train", "validation"):
            result = validate(output / split, samples=3, decode_video=True, expected_split=split)
            self.assertEqual(result["episodes"], 1)
            self.assertEqual(result["valid_windows"], 34)
            self.assertEqual({c["episode"] for c in result["numeric_checks"]}, {19})
            self.assertEqual(task_prepare.sha256(output / split / "meta/relative_stats_dreamzero.json"),
                             task_prepare.sha256(self.train / "meta/relative_stats_dreamzero.json"))
            self.assertFalse((output / split / "data").exists())
            self.assertFalse((output / split / "videos").exists())
        self.assertEqual(task_prepare.sha256(self.train / "manifest.json"), original_hash)
        with self.assertRaises(FileExistsError):
            task_prepare.prepare(self.train, self.val, output, "task one")

    def test_unknown_task_leaves_no_output(self):
        output = self.root / "unknown"
        with self.assertRaisesRegex(ValueError, "exact task match"):
            task_prepare.prepare(self.train, self.val, output, "task")
        self.assertFalse(output.exists())

    def test_bad_validation_split_does_not_publish_train(self):
        output = self.root / "bad_split"
        with self.assertRaisesRegex(ValueError, "Expected validation"):
            task_prepare.prepare(self.train, self.train, output, "task one")
        self.assertFalse(output.exists())

    def test_mixed_task_episode_filters_ranges_not_just_episode_ids(self):
        storage = task_prepare.ABCV3Storage(self.train)
        # A metadata-only mixed-task fixture: keep the matching range, not the
        # other task's range in the same episode.
        storage.windows = pa.Table.from_pylist([
            dict(episode_index=3, task_index=1, start=0, stop=5),
            dict(episode_index=3, task_index=0, start=5, stop=34),
            dict(episode_index=19, task_index=1, start=0, stop=34),
        ]).to_pandas()
        task_id, episodes, windows = task_prepare.select_task(storage, "task one")
        self.assertEqual(task_id, 1)
        self.assertEqual(set(episodes.episode_index), {3, 19})
        self.assertEqual(int((windows.stop - windows.start).sum()), 39)
        self.assertEqual(set(windows.task_index), {1})

    def test_full_dataset_is_not_a_task_subset(self):
        with self.assertRaisesRegex(ValueError, "task-specific metadata"):
            validate(self.train)


if __name__ == "__main__":
    unittest.main()
