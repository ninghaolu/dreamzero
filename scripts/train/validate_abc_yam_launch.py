"""Compose and preflight without constructing a model."""
import json
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"groot/vla/data/dataset"))
from abc_v3_storage import ABCV3Storage, verify_normalizer
from yam_temporal import TemporalConfig, YAMTemporalPlan


def validate_config(cfg):
    temporal = TemporalConfig.from_name(cfg.abc_temporal)
    expected = dict(action_horizon=temporal.block_length, num_action_per_block=temporal.block_length,
                    abc_video_stride=temporal.video_stride, max_chunk_size=temporal.num_blocks,
                    num_frame_per_block=2, num_state_per_block=1, num_frames=33)
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f"{key}={cfg[key]} conflicts with {temporal.name} ({value})")
    plan = YAMTemporalPlan(0, temporal)
    for name in ("video", "state", "action"):
        if list(cfg.modality_config_yam[name].delta_indices) != getattr(plan, f"{name}_indices").tolist():
            raise ValueError(f"{name} modality indices conflict with {temporal.name}")
    dataset = cfg.train_dataset.dataset_kwargs
    if dataset.action_horizon != temporal.block_length or dataset.video_stride != temporal.video_stride:
        raise ValueError("Dataset sampling differs from model temporal configuration")
    head = cfg.action_head_cfg.config
    for key in ("num_action_per_block", "num_frame_per_block", "num_state_per_block"):
        if head.diffusion_model_cfg[key] != expected[key]:
            raise ValueError(f"Action head {key} conflicts with temporal configuration")
    if head.action_horizon != temporal.block_length or head.num_frames != temporal.num_frames:
        raise ValueError("Action head horizon/frame count conflicts with temporal configuration")
    if head.diffusion_model_cfg.max_chunk_size != temporal.num_blocks:
        raise ValueError("Diffusion chunk count conflicts with temporal configuration")
    if cfg.train_architecture != "full" or head.train_architecture != "full" or cfg.save_lora_only:
        raise ValueError("ABC launcher requires full finetuning and full checkpoint saves")
    if not head.tune_projector or not head.tune_diffusion_model:
        raise ValueError("Full finetuning requires trainable projector and diffusion model")
    return temporal


def main():
    args=sys.argv[1:]
    config_only=args and args[0]=="--config-only"
    if config_only:args=args[1:]
    with initialize_config_dir(version_base="1.3",config_dir=str(ROOT/"groot/vla/configs")):
        cfg=compose(config_name="conf",overrides=args)
    temporal = validate_config(cfg)
    if config_only:
        print(OmegaConf.to_yaml(cfg,resolve=True));return
    storage=ABCV3Storage(cfg.abc_metadata_root)
    verify_normalizer(cfg.abc_metadata_root, temporal)
    if storage.temporal != temporal:
        raise ValueError("Selected temporal configuration differs from prepared metadata")
    if storage.manifest["split"]!="train":raise ValueError("Training requires training-source metadata")
    if cfg.max_state_dim!=64 or cfg.max_action_dim!=32 or cfg.num_views!=3:
        raise ValueError("Overrides violate YAM dimensions/views")
    if cfg.save_total_limit<5:raise ValueError("DreamZero requires save_total_limit >= 5")
    for name in ["text_encoder_pretrained_path","image_encoder_pretrained_path","vae_pretrained_path"]:
        if not Path(cfg[name]).is_file():raise FileNotFoundError(cfg[name])
    for directory,filename in [(cfg.pretrained_model_path,"model.safetensors.index.json"),(cfg.tokenizer_path,"tokenizer_config.json")]:
        if not (Path(directory)/filename).is_file():raise FileNotFoundError(Path(directory)/filename)
    index=json.loads((Path(cfg.pretrained_model_path)/"model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (Path(cfg.pretrained_model_path)/shard).is_file():raise FileNotFoundError(shard)
    output=Path(cfg.output_dir);output.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg,output/"abc_resolved_config.yaml",resolve=True)
    (output/"abc_launch_overrides.json").write_text(json.dumps(args,indent=2)+"\n")
    print(f"ABC preflight passed: {storage.manifest['valid_windows']} eligible windows",flush=True)


if __name__=="__main__":main()
