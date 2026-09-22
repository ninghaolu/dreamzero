"""Check completed full-finetuning smoke artifacts without loading tensor payloads."""
import argparse
import json
import math
from pathlib import Path
import struct


def validate_run(root, steps=3, ranks=8):
    root = Path(root)
    cfg = json.loads((root / 'config.json').read_text())
    if cfg['action_head_cfg']['config']['train_architecture'] != 'full':
        raise ValueError('Smoke run did not use full finetuning')
    inventory = json.loads((root / 'parameter_inventory.json').read_text())
    frozen_prefixes = ('action_head.text_encoder.', 'action_head.image_encoder.', 'action_head.vae.')
    if not inventory:
        raise ValueError('Empty parameter inventory')
    for name, parameter in inventory.items():
        expected = not name.startswith(frozen_prefixes)
        if 'lora_' in name or parameter['trainable'] != expected:
            raise ValueError(f'Incorrect full-finetuning parameter status: {name}')
    for prefix in frozen_prefixes:
        if not any(name.startswith(prefix) for name in inventory):
            raise ValueError(f'Missing frozen component: {prefix}')
    resources = []
    for rank in range(ranks):
        record = json.loads((root / f'resources-rank-{rank}.json').read_text())
        if record['global_step'] < steps or record['peak_gpu_allocated_bytes'] <= 0:
            raise ValueError(f'Rank {rank} did not finish {steps} GPU optimizer steps')
        resources.append(record)
    state = json.loads((root / 'trainer_state.json').read_text())
    if state['global_step'] < steps:
        raise ValueError('Incomplete trainer state')
    losses = [json.loads(line) for line in (root / 'loss_log.jsonl').read_text().splitlines()]
    values = [value for record in losses for key, value in record.items() if 'loss' in key]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Missing or nonfinite training losses')
    # Full checkpoint must include frozen encoders and VAE as well as tuned weights.
    index_path = root / 'model.safetensors.index.json'
    if index_path.exists():
        mapping = json.loads(index_path.read_text())['weight_map']
        files = sorted(set(mapping.values()))
    else:
        mapping = None
        files = ['model.safetensors']
    saved = {}
    for name in files:
        with (root / name).open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(length))
        for key, value in header.items():
            if key == '__metadata__':
                continue
            if key in saved or (mapping is not None and mapping.get(key) != name):
                raise ValueError(f'Invalid checkpoint index entry: {key}')
            saved[key] = value['shape']
    if mapping is not None and set(saved) != set(mapping):
        raise ValueError('Checkpoint index does not cover saved tensors')
    for name, parameter in inventory.items():
        if saved.get(name) != parameter['shape']:
            raise ValueError(f'Full checkpoint missing parameter or wrong shape: {name}')
    return dict(passed=True, steps=state['global_step'], ranks=ranks,
                parameter_count=len(inventory), checkpoint_shards=len(files), resources=resources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--ranks', type=int, default=8)
    args = parser.parse_args()
    report = validate_run(args.output_dir, args.steps, args.ranks)
    (args.output_dir / 'smoke_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
