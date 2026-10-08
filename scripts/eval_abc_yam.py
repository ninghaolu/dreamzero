#!/usr/bin/env python3
"""Fixed held-out ABC anchors: causal configured-horizon action errors and hold-state baseline."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"groot/vla/data/dataset"))
from abc_v3_storage import ABCV3Storage, CAMERAS, SLICES, sha256, verify_normalizer
from yam_temporal import TemporalConfig, YAMTemporalPlan
from lerobot_v3_video import V3VideoReader


def metrics(predicted,target):
    error=np.asarray(predicted,dtype=np.float64)-np.asarray(target,dtype=np.float64)
    return dict(per_dimension_mae=np.abs(error).mean(axis=(0,1)).tolist(),
                per_dimension_rmse=np.sqrt(np.square(error).mean(axis=(0,1))).tolist(),
                left_gripper_mae=float(np.abs(error[:,:,6]).mean()),
                right_gripper_mae=float(np.abs(error[:,:,13]).mean()))


def validation_loss(policy, args, anchors, temporal):
    """Evaluate the native four-block denoising objective, with fixed noise draws."""
    import torch
    from hydra.utils import instantiate
    from groot.vla.data.dataset.lerobot_v3 import ShardedLeRobotV3YAMDataset
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import DreamTransform

    dataset=ShardedLeRobotV3YAMDataset(args.metadata_root,
        modality_configs=instantiate(policy.train_cfg.modality_configs.yam),
        embodiment_tag=EmbodimentTag.YAM,
        transforms=instantiate(policy.train_cfg.transforms.yam),
        action_horizon=temporal.block_length,video_stride=temporal.video_stride)
    dataset.transforms.eval()
    # Deterministic evaluation image transforms, but retain action targets in
    # the terminal packing transform (its eval mode omits them).
    for transform in dataset.transforms.transforms:
        if isinstance(transform,DreamTransform):
            transform.train()
            transform.language_dropout_prob=0.0
    collator=instantiate(policy.train_cfg.data_collator)
    model=policy.trained_model
    model.eval()
    head=model.action_head
    for attr in ["kv_cache1","kv_cache_neg","crossattn_cache","crossattn_cache_neg","clip_feas","ys","language"]:
        setattr(head,attr,None)
    head.scheduler.set_timesteps(1000,training=True)
    torch.cuda.empty_cache()
    records=[]
    with (args.output_dir/"validation-loss-samples.jsonl").open("w") as log:
        for index,anchor in enumerate(anchors):
            raw=dataset.raw_sample(anchor["episode"],anchor["start"])
            batch=collator([dataset.transforms(raw)])
            if tuple(batch["action"].shape)!=(1,temporal.endpoint,32):raise ValueError("Incorrect validation action horizon")
            draws=[]
            for draw in range(args.loss_noise_draws):
                seed=anchor["seed"]+1000000+draw*100000
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    result=model(batch)
                values={key:float(result[key].item()) for key in ("loss","action_loss","dynamics_loss")}
                if not all(np.isfinite(v) for v in values.values()):raise ValueError("Non-finite validation loss")
                draws.append(dict(seed=seed,**values))
            record=dict(anchor,draws=draws)
            records.append(record);log.write(json.dumps(record)+"\n");log.flush()
            print(f"Validation loss {index+1}/{len(anchors)}: {draws}",flush=True)
    # Average noise draws within each window before computing uncertainty.
    per_window={key:np.array([np.mean([d[key] for d in r["draws"]]) for r in records])
                for key in ("loss","action_loss","dynamics_loss")}
    summary={key:dict(mean=float(values.mean()),standard_error=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None)
             for key,values in per_window.items()}
    report=dict(checkpoint=args.checkpoint,num_windows=len(anchors),noise_draws=args.loss_noise_draws,
                temporal=temporal.to_dict(),objective="native scheduler-weighted flow-matching MSE: action + dynamics",
                conditioning="teacher-forced four-block validation windows; eval image transforms; no language dropout",
                summary=summary)
    (args.output_dir/"validation-loss.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", choices=["h24", "h48"], help="Default: infer from prepared metadata")
    parser.add_argument("--metadata-root",required=True)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--anchors",type=Path)
    parser.add_argument("--num-samples",type=int,default=32)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--checkpoint",help="Omit to evaluate only the hold-current-state baseline")
    parser.add_argument("--save-videos",action="store_true",help="Decode and save each predicted video chunk as MP4")
    parser.add_argument("--num-chunks",type=int,default=1,help="Autoregressive chunks; later states use the final predicted action")
    parser.add_argument("--save-chunk-prefixes",type=int,nargs="*",default=[],help="Also save videos ending at these chunk counts")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--validation-loss",action="store_true",help="Also measure native action/video denoising loss on the same anchors")
    parser.add_argument("--loss-noise-draws",type=int,default=2)
    args=parser.parse_args()
    if args.num_samples<1:parser.error("num-samples must be positive")
    if args.validation_loss and not args.checkpoint:parser.error("--validation-loss requires --checkpoint")
    if args.loss_noise_draws<1:parser.error("loss-noise-draws must be positive")
    if args.save_videos and not args.checkpoint:parser.error("--save-videos requires --checkpoint")
    if args.num_chunks<1:parser.error("num-chunks must be positive")
    if args.num_chunks>1 and not args.save_videos:parser.error("Multi-chunk rollout requires --save-videos")
    if any(n<1 or n>=args.num_chunks for n in args.save_chunk_prefixes):parser.error("Prefixes must be positive and smaller than num-chunks")
    storage=ABCV3Storage(args.metadata_root)
    temporal = storage.temporal
    if args.temporal and temporal != TemporalConfig.from_name(args.temporal):
        raise ValueError("Evaluation temporal configuration differs from metadata")
    verify_normalizer(args.metadata_root, temporal)
    horizon = temporal.block_length
    if storage.manifest["split"]!="validation":raise ValueError("Use separate validation-source metadata")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    identity=sha256(Path(args.metadata_root)/"manifest.json")
    anchors_path=args.anchors or args.output_dir/"anchors.json"
    if anchors_path.exists():
        anchor_doc=json.loads(anchors_path.read_text())
        if anchor_doc["metadata_sha256"]!=identity:raise ValueError("Anchor dataset identity mismatch")
        anchors=anchor_doc["anchors"][:args.num_samples]
        if not anchors:raise ValueError("Anchor file is empty")
    else:
        windows=storage.windows
        lengths=(windows.stop-windows.start).to_numpy()
        ends=np.cumsum(lengths)
        rng=np.random.default_rng(args.seed)
        anchors=[]
        for draw in rng.integers(int(ends[-1]),size=args.num_samples):
            pos=int(np.searchsorted(ends,draw,side="right"));row=windows.iloc[pos]
            start=int(row.start+draw-(ends[pos-1] if pos else 0))
            anchors.append(dict(episode=int(row.episode_index),start=start,seed=args.seed+len(anchors)))
        anchors_path.parent.mkdir(parents=True,exist_ok=True)
        anchors_path.write_text(json.dumps(dict(metadata_sha256=identity,anchors=anchors),indent=2)+"\n")
    policy=None
    if args.checkpoint:
        saved_config = json.loads((Path(args.checkpoint)/"config.json").read_text())
        head_config = saved_config["action_head_cfg"]["config"]
        if (saved_config["action_horizon"] != horizon or
            head_config["diffusion_model_cfg"]["num_action_per_block"] != horizon or
            head_config["diffusion_model_cfg"]["num_frame_per_block"] != temporal.num_frame_per_block or
            head_config["num_frames"] != temporal.num_frames):
            raise ValueError("Checkpoint temporal configuration differs from evaluation data")
        import torch
        import torch.distributed as dist
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from groot.vla.data.schema import EmbodimentTag
        from tianshou.data import Batch
        if not dist.is_initialized():
            # File rendezvous avoids a fixed TCP port collision for independent evaluations.
            rendezvous=args.output_dir.resolve()/f"rendezvous-{os.getpid()}"
            dist.init_process_group("gloo",init_method=f"file://{rendezvous}",rank=0,world_size=1)
        policy=GrootSimPolicy(embodiment_tag=EmbodimentTag.YAM,model_path=args.checkpoint,device=args.device)
    decoder=V3VideoReader()
    targets=[];baseline=[];predictions=[];tasks=[];latencies=[];records=[];rollouts=[]
    begin=time.perf_counter()
    for anchor in anchors:
        ep,start=anchor["episode"],anchor["start"]
        rows=storage.read_episode(ep)
        YAMTemporalPlan(start, temporal).validate(len(rows["index"]), rows["task_index"])
        state=rows["observation.state"][start]
        target=rows["action"][start:start+horizon]
        hold=np.repeat(state[None,:],horizon,axis=0)
        task=storage.tasks[int(rows["task_index"][start])]
        record=dict(anchor,task=task)
        if policy is not None:
            obs={f"state.{key}":state[a:b][None,:] for key,(a,b) in SLICES.items()}
            frames=[]
            for camera in CAMERAS:
                path,timestamps,interval=storage.camera_request(ep,camera,[rows["timestamp"][start]])
                frame,_=decoder.read(path,timestamps,interval)
                obs[f"video.{camera}"]=frame
                frames.append(frame[0])
            obs["annotation.task"]=task
            head=policy.trained_model.action_head
            for attr in ["kv_cache1","kv_cache_neg","crossattn_cache","crossattn_cache_neg","clip_feas","ys","language"]:
                setattr(head,attr,None)
            head.current_start_frame=0;head.seed=anchor["seed"]
            random.seed(anchor["seed"]);np.random.seed(anchor["seed"]);torch.manual_seed(anchor["seed"])
            latent_video=None;feedback_video=None;segment_latents=[];saved_frames=[];chunk_actions=[]
            record.update(chunk_latency_seconds=[],context_reset_chunks=[],chunk_seeds=[],context_end_frames=[])
            for chunk in range(args.num_chunks):
                # Keep the checkpoint's native cache limit. At a reset, the head
                # re-encodes the LAST generated image supplied in feedback_video.
                reset=chunk==0 or head.current_start_frame>=head.model.local_attn_size
                expected_end=(1 if reset else head.current_start_frame)+head.num_frame_per_block
                if reset:record["context_reset_chunks"].append(chunk+1)
                head.seed=anchor["seed"]+1000*chunk
                record["chunk_seeds"].append(head.seed)
                torch.cuda.synchronize();t0=time.perf_counter()
                result,video_pred=policy.lazy_joint_forward_causal(Batch(obs=obs),video=feedback_video,latent_video=latent_video)
                if head.current_start_frame!=expected_end:raise ValueError("Unexpected autoregressive context reset")
                record["context_end_frames"].append(head.current_start_frame)
                torch.cuda.synchronize();elapsed=time.perf_counter()-t0
                record["chunk_latency_seconds"].append(elapsed)
                if chunk==0:latencies.append(elapsed)
                values=[]
                for key,(a,b) in SLICES.items():
                    value=result.act[f"action.{key}"]
                    if torch.is_tensor(value):value=value.detach().cpu().numpy()
                    values.append(np.asarray(value).reshape(-1,b-a)[:horizon])
                prediction=np.concatenate(values,axis=1)
                if prediction.shape!=(horizon,14) or not np.isfinite(prediction).all():raise ValueError("Invalid predicted action chunk")
                chunk_actions.append(prediction)
                if args.save_videos:
                    import imageio.v2 as imageio
                    if reset:segment_latents=[]
                    segment_latents.append(video_pred)
                    with torch.inference_mode():
                        decoded=head.vae.decode(torch.cat(segment_latents,dim=2),tiled=head.tiled,
                            tile_size=(head.tile_size_height,head.tile_size_width),
                            tile_stride=(head.tile_stride_height,head.tile_stride_width))
                    if not torch.isfinite(decoded).all():raise ValueError("Non-finite generated video")
                    frames_per_chunk=horizon//temporal.video_stride
                    # First chunk includes t=0. Every later chunk adds only its
                    # eight future frames, avoiding duplicate boundary frames.
                    new_frames=decoded if chunk==0 else decoded[:,:,-frames_per_chunk:]
                    saved_frames.append(((new_frames[0].permute(1,2,3,0).float()+1)*127.5).clamp(0,255).byte().cpu().numpy())
                    feedback_video=decoded[:,:,-(frames_per_chunk+1):].permute(0,2,3,4,1).contiguous()
                    latent_video=video_pred[:,:,-head.num_frame_per_block:]
                    if chunk+1 in [args.num_chunks,*args.save_chunk_prefixes]:
                        video_frames=np.concatenate(saved_frames)
                        suffix=f"-chunks-{chunk+1}" if args.num_chunks>1 else ""
                        video_name=f"sample-{len(predictions):03d}-episode-{ep}-frame-{start}{suffix}.mp4"
                        imageio.mimsave(args.output_dir/video_name,video_frames,fps=temporal.video_fps,codec="libx264",macro_block_size=1)
                        record.setdefault("videos",{})[str(chunk+1)]=video_name
                        if chunk+1==args.num_chunks:record.update(video=video_name,video_frames=len(video_frames),video_fps=temporal.video_fps)
                        print(f"Saved {video_name}: {len(video_frames)} frames at {temporal.video_fps} fps",flush=True)
                        del video_frames
                    del decoded,new_frames
                # Absolute joint-position actions approximate the next state;
                # no future dataset observations or states enter this rollout.
                for key,(a,b) in SLICES.items():obs[f"state.{key}"]=prediction[-1:,a:b].copy()
                print(f"Sample {len(predictions)} chunk {chunk+1}/{args.num_chunks}: {elapsed:.2f}s, reset={reset}",flush=True)
            rollouts.append(np.concatenate(chunk_actions))
            prediction=chunk_actions[0]
            # Native policy.unapply already inverse-normalizes and adds the anchor state once.
            predictions.append(prediction)
            if len(predictions)<=8:
                from PIL import Image
                mosaic=np.concatenate([np.concatenate([frames[0],frames[2]],axis=1),
                                       np.concatenate([frames[1],np.zeros_like(frames[1])],axis=1)],axis=0)
                Image.fromarray(mosaic).save(args.output_dir/f"sample-{len(predictions)-1}.png")
        targets.append(target);baseline.append(hold);tasks.append(task);records.append(record)
    target=np.asarray(targets);hold=np.asarray(baseline)
    report=dict(metadata_sha256=identity,checkpoint=args.checkpoint,num_anchors=len(anchors),
                horizon=horizon,temporal=temporal.to_dict(),baseline=metrics(hold,target),per_task={},seconds=time.perf_counter()-begin,
                inference_latency_seconds=latencies,anchors=records,num_chunks=args.num_chunks,
                conditioning="initial observation only; generated video and final predicted action as next state",
                metrics_scope="first chunk only")
    for task in sorted(set(tasks)):
        selected=np.asarray(tasks)==task
        report["per_task"][task]=dict(anchors=int(selected.sum()),baseline=metrics(hold[selected],target[selected]))
    arrays=dict(target=target,hold_current_state=hold)
    if predictions:
        prediction=np.asarray(predictions);arrays["prediction"]=prediction
        arrays["rollout_prediction"]=np.asarray(rollouts)
        report["model"]=metrics(prediction,target)
        for task in report["per_task"]:
            selected=np.asarray(tasks)==task
            report["per_task"][task]["model"]=metrics(prediction[selected],target[selected])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(7,2,figsize=(12,16))
        for dim,axis in enumerate(axes.flat):
            axis.plot(target[0,:,dim],label="target");axis.plot(prediction[0,:,dim],label="prediction")
            axis.plot(hold[0,:,dim],label="hold state",linestyle="--");axis.set_title(f"Dimension {dim}")
        axes.flat[0].legend();fig.tight_layout();fig.savefig(args.output_dir/"first-anchor-actions.png");plt.close(fig)
    np.savez_compressed(args.output_dir/"actions.npz",**arrays)
    (args.output_dir/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Evaluated {len(anchors)} validation anchors; results: {args.output_dir}")
    if args.validation_loss:validation_loss(policy,args,anchors,temporal)
    if policy is not None:dist.destroy_process_group()


if __name__=="__main__":main()
