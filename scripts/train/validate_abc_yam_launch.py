"""Compose and preflight without constructing a model."""
import json
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"groot/vla/data/dataset"))
from abc_v3_storage import ABCV3Storage, verify_normalizer


def main():
    args=sys.argv[1:]
    config_only=args and args[0]=="--config-only"
    if config_only:args=args[1:]
    check_only=args and args[0]=="--check-only"
    if check_only:args=args[1:]
    with initialize_config_dir(version_base="1.3",config_dir=str(ROOT/"groot/vla/configs")):
        cfg=compose(config_name="conf",overrides=args)
    if config_only:
        print(OmegaConf.to_yaml(cfg,resolve=True));return
    storage=ABCV3Storage(cfg.abc_metadata_root)
    verify_normalizer(cfg.abc_metadata_root)
    if storage.manifest["split"]!="train":raise ValueError("Training requires training-source metadata")
    if cfg.max_chunk_size!=4 or cfg.num_frames!=33 or cfg.action_horizon!=24:
        raise ValueError("Overrides violate the fixed ABC temporal contract")
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
    if not check_only:
        output=Path(cfg.output_dir);output.mkdir(parents=True,exist_ok=True)
        OmegaConf.save(cfg,output/"abc_resolved_config.yaml",resolve=True)
        (output/"abc_launch_overrides.json").write_text(json.dumps(args,indent=2)+"\n")
    print(f"ABC preflight passed: {storage.manifest['valid_windows']} eligible windows",flush=True)


if __name__=="__main__":main()
