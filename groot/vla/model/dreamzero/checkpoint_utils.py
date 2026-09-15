"""Strict base/adapter reconstruction shared by training and inference."""
import gc
import hashlib
import json
from pathlib import Path
import struct

from safetensors import safe_open
from safetensors.torch import load_file


def checkpoint_files(root):
    root = Path(root)
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        files = [root / name for name in sorted(set(mapping.values()))]
    else:
        mapping = None
        files = [root / "model.safetensors"]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError(f"Incomplete safetensors checkpoint: {root}")
    return files, mapping


def checkpoint_identity(root):
    files, _ = checkpoint_files(root)
    headers = {}
    for path in files:
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = f.read(size)
        headers[path.name] = dict(bytes=path.stat().st_size, header_sha256=hashlib.sha256(header).hexdigest())
    result = {"files": headers}
    downloads = Path(root).parent / "download_manifest.json"
    if downloads.exists():
        for entry in json.loads(downloads.read_text()):
            if Path(entry["directory"]).name == Path(root).name:
                result.update(repo=entry["repo"], revision=entry["revision"])
                break
    return result


def load_weights(model, root, trainable_only=False):
    """Validate aggregate keys and shapes before applying shards; no key rewriting."""
    files, mapping = checkpoint_files(root)
    expected = model.state_dict()
    required = ({key for key, value in model.named_parameters() if value.requires_grad}
                if trainable_only else set(expected))
    if not required:
        raise ValueError("No expected parameters to load")
    loaded = set()
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in loaded or key not in required:
                    raise ValueError(f"Duplicate/unexpected checkpoint key: {key}")
                if mapping is not None and mapping.get(key) != path.name:
                    raise ValueError(f"Checkpoint index/shard mismatch: {key}")
                if tuple(f.get_slice(key).get_shape()) != tuple(expected[key].shape):
                    raise ValueError(f"Checkpoint shape mismatch: {key}")
                loaded.add(key)
    if loaded != required or (mapping is not None and set(mapping) != loaded):
        raise ValueError(f"Checkpoint key coverage mismatch: missing={sorted(required-loaded)[:20]}")
    for path in files:
        print(f"Loading verified shard: {path}", flush=True)
        state = load_file(str(path), device="cpu")
        model.load_state_dict(state, strict=False)
        del state
        gc.collect()
    return sorted(loaded)


def load_base_and_inject(model, base_path):
    load_weights(model, base_path)
    model._base_checkpoint_path = str(Path(base_path).resolve())
    model._base_checkpoint_identity = checkpoint_identity(base_path)
    head = getattr(model, "action_head", None)
    if head is not None and getattr(head.config, "defer_lora_injection", False):
        head.inject_lora_after_loading()
    return model


def write_checkpoint_manifest(model, output_dir, dataset_metadata_root=None):
    output = Path(output_dir)
    base_path = getattr(model, "_base_checkpoint_path", None)
    if base_path is None:
        raise ValueError("Cannot save an adapter without its base checkpoint identity")
    trainable = {key: list(value.shape) for key, value in model.named_parameters() if value.requires_grad}
    manifest = dict(schema_version=1, base_path=base_path,
                    base_identity=model._base_checkpoint_identity, trainable_parameters=trainable,
                    config_sha256=hashlib.sha256((output/"config.json").read_bytes()).hexdigest())
    if dataset_metadata_root:
        root = Path(dataset_metadata_root)
        manifest["dataset_metadata_root"] = str(root.resolve())
        manifest["data_provenance"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                        for name in ("manifest.json", "norm_stats_manifest.json")}
    (output / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")


def reconstruct_adapter(cls, config, adapter_path, base_path=None):
    root = Path(adapter_path)
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Adapter checkpoint lacks checkpoint_manifest.json; explicit migration is required")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint manifest version")
    if hashlib.sha256((root/"config.json").read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Saved configuration hash mismatch")
    base_path = base_path or manifest["base_path"]
    actual_identity = checkpoint_identity(base_path)
    # Relocation need not bring the download manifest, but tensor headers/sizes must agree.
    if actual_identity["files"] != manifest["base_identity"]["files"]:
        raise ValueError("Base checkpoint identity mismatch")
    inner = config.action_head_cfg.get("config", config.action_head_cfg)
    inner.update(defer_lora_injection=True, skip_component_loading=True)
    model = load_base_and_inject(cls(config), base_path)
    trainable = {key: list(value.shape) for key, value in model.named_parameters() if value.requires_grad}
    if trainable != manifest["trainable_parameters"]:
        raise ValueError("Reconstructed adapter parameter inventory differs from saved inventory")
    load_weights(model, root, trainable_only=True)
    return model.eval()
