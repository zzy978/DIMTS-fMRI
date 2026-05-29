import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from Baselines.data import build_training_loader, cycle
from Baselines.registry import SUPPORTED_BASELINES, build_adapter
from Utils.io_utils import seed_everything


def default_config_path(model_name):
    return Path('Baselines') / 'configs' / f'{model_name}_fmri.yaml'


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train fMRI conditional generation baselines.')
    parser.add_argument('--model', type=str, required=True, choices=SUPPORTED_BASELINES)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_subjects', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config_file) if args.config_file else default_config_path(args.model)
    config = load_config(config_path)
    config['model'] = args.model
    if args.max_subjects is not None:
        config['data']['max_subjects'] = int(args.max_subjects)
    if args.max_steps is not None:
        config['training']['max_steps'] = int(args.max_steps)

    checkpoint_dir = Path(args.checkpoint_dir or config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    loader, dataset = build_training_loader(config, output_dir=checkpoint_dir / 'dataset_cache')
    adapter = build_adapter(args.model, config, device)
    start_step = 0
    if args.resume:
        state = adapter.load(args.resume, map_location=device)
        start_step = int(state.get('step', 0))

    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(checkpoint_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    max_steps = int(config['training'].get('max_steps', 10000))
    save_every = int(config['training'].get('save_every', 1000))
    log_every = int(config['training'].get('log_every', 100))
    iterator = cycle(loader)
    last_losses = {}

    for step in range(start_step + 1, max_steps + 1):
        batch = next(iterator)
        last_losses = adapter.train_step(batch)
        if step % log_every == 0 or step == 1:
            loss_text = ', '.join(f'{key}={value:.6f}' for key, value in sorted(last_losses.items()))
            print(f'{args.model}: step {step}/{max_steps} | {loss_text}', flush=True)
        if step % save_every == 0:
            adapter.save(checkpoint_dir / f'checkpoint-{step}.pt', step=step, extra={'losses': last_losses})
            adapter.save(checkpoint_dir / 'latest.pt', step=step, extra={'losses': last_losses})

    adapter.save(checkpoint_dir / 'latest.pt', step=max_steps, extra={'losses': last_losses})
    with open(checkpoint_dir / 'training_summary.json', 'w') as f:
        json.dump({
            'model': args.model,
            'config_file': str(config_path),
            'checkpoint_dir': str(checkpoint_dir),
            'train_steps': max_steps,
            'train_dataset_size': int(len(dataset)),
            'last_losses': last_losses,
        }, f, indent=2)


if __name__ == '__main__':
    main()

