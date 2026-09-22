import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "groot/vla/data/dataset"))
from abc_v3_storage import ABCV3Storage, CAMERAS, SLICES, sha256
from abc_yam_stats import source_identity, file_inventory
from yam_temporal import TemporalConfig, YAMTemporalPlan, WindowRanges
from lerobot_v3_video import V3VideoReader
spec = importlib.util.spec_from_file_location("prepare_abc", ROOT / "scripts/data/prepare_abc_yam_v3.py")
prepare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)


def make_fixture(root, horizon=24):
    temporal = TemporalConfig.from_name(f"h{horizon}")
    import av
    source, norm = root / "source", root / "norm"
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    (norm / "meta").mkdir(parents=True)
    length = 130 if horizon == 24 else 230
    n = 2*length
    state = np.arange(n*14, dtype=np.float32).reshape(n,14)/100
    action = state + 0.2
    task = np.repeat([0, 1], length)
    table = pa.table({"observation.state": state.tolist(), "action": action.tolist(),
                      "index": np.arange(n), "frame_index": np.tile(np.arange(length),2),
                      "episode_index": np.repeat([3, 19], length), "task_index": task,
                      "timestamp": np.tile(np.arange(length)/30,2)})
    for i,(start,stop) in enumerate([(0,100),(100,170),(170,n)]):
        pq.write_table(table.slice(start,stop-start),source/f"data/chunk-000/file-{i:03d}.parquet", row_group_size=23)
    episodes=[]
    for ep_index, ep in enumerate([3,19]):
        row=dict(episode_index=ep, length=length, dataset_from_index=ep_index*length, dataset_to_index=(ep_index+1)*length)
        row.update({"data/chunk_index":0,"data/file_index":ep_index})
        for camera_index,key in enumerate(CAMERAS.values()):
            prefix=f"videos/{key}/"
            offset=camera_index*3+ep_index*11
            file_id=ep_index+camera_index*2
            row.update({prefix+"chunk_index":0,prefix+"file_index":file_id,
                        prefix+"from_timestamp":offset/30,prefix+"to_timestamp":(offset+length)/30})
            path=source/f"videos/{key}/chunk-000/file-{file_id:03d}.mp4"
            path.parent.mkdir(parents=True,exist_ok=True)
            with av.open(str(path),"w") as container:
                stream=container.add_stream("libx264", rate=30)
                stream.width=64; stream.height=64; stream.pix_fmt="yuv420p"
                stream.options={"crf":"0","preset":"ultrafast"}
                for i in range(offset+length):
                    image=np.empty((64,64,3),np.uint8)
                    image[:]=[i % 256,camera_index*50,ep_index*90]
                    frame=av.VideoFrame.from_ndarray(image,format="rgb24")
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
        episodes.append(row)
    pq.write_table(pa.Table.from_pylist(episodes),source/"meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"task_index":[0,1]},index=pd.Index(["task zero","task one"],name="task"))),source/"meta/tasks.parquet")
    features={key:dict(shape=[64,64,3],names=["height","width","channel"],video_info={"video.fps":30}) for key in CAMERAS.values()}
    info=dict(codebase_version="v3.0",fps=30,total_frames=n,total_episodes=2,features=features,
              data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
              video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",chunks_size=1000)
    absolute={key:{stat:[1.]*14 for stat in ["min","max","mean","std","q01","q99"]} for key in ["observation.state","action"]}
    (source/"meta/info.json").write_text(json.dumps(info))
    (source/"meta/stats.json").write_text(json.dumps(absolute))
    (norm/"meta/stats.json").write_text(json.dumps(absolute))
    relative={key:{stat:[1.]*(b-a) for stat in ["min","max","mean","std","q01","q99"]} for key,(a,b) in SLICES.items()}
    (norm/"meta/relative_stats_dreamzero.json").write_text(json.dumps(relative))
    identity=source_identity(source,file_inventory(source),5000,1,temporal)
    identity.update(all_source_files_used=True,valid_windows=2*(length-temporal.endpoint),
                    output_sha256={name:sha256(norm/"meta"/name) for name in ["stats.json","relative_stats_dreamzero.json"]})
    (norm/"norm_stats_manifest.json").write_text(json.dumps(identity))
    output=root/"prepared"
    prepare_module.prepare(source,output,norm,temporal=temporal)
    return source,output,state,action


class StorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.source,cls.output,cls.state,cls.action=make_fixture(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_exact_rows_across_files_and_noncontiguous_ids(self):
        storage=ABCV3Storage(self.output)
        np.testing.assert_array_equal(storage.read_episode(19)["observation.state"],self.state[130:])
        np.testing.assert_array_equal(storage.read_episode(3)["action"],self.action[:130])
        self.assertEqual(storage.tasks[1],"task one")
        self.assertEqual(storage.manifest["valid_windows"],68)

    def test_per_camera_offsets_frame_identity_and_cache(self):
        import av
        storage=ABCV3Storage(self.output)
        decoder=V3VideoReader(max_readers=1,frame_cache_bytes=64*64*3*2,decode_batch=2)
        for ep_index,ep in enumerate([3,19]):
            for camera_index,camera in enumerate(CAMERAS):
                local=np.array([0,96,3,96])/30
                path,times,interval=storage.camera_request(ep,camera,local)
                frames,pts=decoder.read(path,times,interval)
                offset=camera_index*3+ep_index*11
                with av.open(str(path)) as container:
                    reference=np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
                np.testing.assert_array_equal(frames,reference[[offset,offset+96,offset+3,offset+96]])
                np.testing.assert_allclose(pts,times,atol=1e-6)
                self.assertLessEqual(decoder.cached_bytes,decoder.frame_cache_bytes)
                self.assertLessEqual(len(decoder.readers),1)
                with self.assertRaises(ValueError): decoder.read(path,np.array([interval[1]+1]),interval)

    def test_temporal_and_relative_roundtrip(self):
        plan=YAMTemporalPlan(10).validate(130)
        np.testing.assert_array_equal(plan.state_indices,[10,34,58,82])
        self.assertEqual(len(plan.video_indices),33)
        relative=plan.relative_actions(self.state[:130],self.action[:130])
        recovered=relative.reshape(4,24,14)+self.state[plan.state_indices,None,:]
        np.testing.assert_allclose(recovered.reshape(96,14),self.action[10:106],atol=2e-6)
        with self.assertRaises(ValueError): YAMTemporalPlan(34).validate(130)
        tasks=np.zeros(130);tasks[50]=1
        with self.assertRaises(ValueError): plan.validate(130,tasks)
        ranges=WindowRanges([[2,5],[10,12]])
        self.assertEqual(len(ranges),5)
        np.testing.assert_array_equal(ranges[np.arange(5)],[2,3,4,10,11])

    def test_numeric_cache_hard_limit(self):
        with self.assertRaisesRegex(ValueError,"cache limit"):
            ABCV3Storage(self.output,numeric_cache_bytes=10).read_episode(3)


class H48StorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.source, cls.output, cls.state, cls.action = make_fixture(Path(cls.tmp.name), 48)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sampling_timestamps_targets_and_endpoint(self):
        from abc_v3_storage import verify_normalizer
        storage = ABCV3Storage(self.output)
        t = storage.temporal
        self.assertEqual(t, TemporalConfig.from_name("h48"))
        self.assertEqual(storage.manifest["valid_windows"], 76)
        verify_normalizer(self.output, t)
        with self.assertRaisesRegex(ValueError, "temporal"):
            verify_normalizer(self.output, TemporalConfig())
        rows = storage.read_episode(19)
        plan = YAMTemporalPlan(0, t).validate(230, rows["task_index"])
        np.testing.assert_array_equal(plan.video_indices, np.arange(0, 193, 6))
        np.testing.assert_array_equal(plan.action_indices, np.arange(192))
        np.testing.assert_array_equal(plan.state_indices, [0, 48, 96, 144])
        times = rows["timestamp"][plan.video_indices]
        np.testing.assert_allclose(np.diff(times), .2)
        self.assertAlmostEqual(times[-1]-times[0], 6.4)
        relative = plan.relative_actions(rows["observation.state"], rows["action"])
        self.assertEqual(relative.shape, (192, 14))
        for block, anchor in enumerate([0, 48, 96, 144]):
            np.testing.assert_array_equal(relative[block*48:(block+1)*48],
                rows["action"][anchor:anchor+48]-rows["observation.state"][anchor])
        # Last eligible start includes the endpoint; one later must fail.
        YAMTemporalPlan(37, t).validate(230)
        with self.assertRaises(ValueError):
            YAMTemporalPlan(38, t).validate(230)
        for changed_row in (48, 191, 192):
            tasks = np.zeros(230); tasks[changed_row] = 1
            with self.assertRaisesRegex(ValueError, "task boundary"):
                plan.validate(230, tasks)
        import av
        decoder = V3VideoReader()
        for camera in CAMERAS:
            path, requested, interval = storage.camera_request(19, camera, times)
            frames, actual = decoder.read(path, requested, interval)
            self.assertEqual(frames.shape, (33, 64, 64, 3))
            np.testing.assert_allclose(actual, requested, atol=1e-6)
            with av.open(str(path)) as container:
                reference = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
            np.testing.assert_array_equal(frames, reference[np.rint(requested*30).astype(int)])
        decoder.close()

    def test_preparation_rejects_other_horizon_statistics(self):
        with self.assertRaisesRegex(ValueError, "temporal"):
            prepare_module.prepare(self.source, Path(self.tmp.name)/"bad", self.output,
                                   temporal=TemporalConfig())


if __name__ == "__main__": unittest.main()
