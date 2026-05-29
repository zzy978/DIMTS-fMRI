import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from Baselines.data import iter_batches, list_subject_files, list_subject_split_files, load_normalized_subject
from Baselines.metrics import compute_future_metrics_z, summarize_by_length
from Baselines.registry import SUPPORTED_BASELINES, build_adapter
from Utils.io_utils import seed_everything


def default_config_path(model_name):
    return Path('Baselines') / 'configs' / f'{model_name}_fmri.yaml'


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate fMRI conditional generation baselines on held-out future segments.')
    parser.add_argument('--model', type=str, required=True, choices=SUPPORTED_BASELINES)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='OUTPUT/baseline_test_set')
    parser.add_argument('--eval_source', type=str, default='test_split', choices=['test_split', 'rest1'])
    parser.add_argument('--source_dir', type=str, default='/data/home/zyzeng/data1/datasets/rest1_to_csv_500')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--pred_len', type=int, default=128)
    parser.add_argument('--extend_lengths', type=int, nargs='+', default=[128, 256, 384, 512])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_subjects', type=int, default=0)
    return parser.parse_args()


def extend_batch(adapter, normalized_histories, total_extend_len, pred_len):
    current_sequences = [history.astype(np.float32).copy() for history in normalized_histories]
    generated_chunks = [[] for _ in current_sequences]
    remaining = int(total_extend_len)

    while remaining > 0:
        current_chunk_len = min(int(pred_len), remaining)
        contexts = []
        for sequence in current_sequences:
            if sequence.shape[0] < pred_len:
                raise ValueError(f'History length {sequence.shape[0]} is shorter than context length {pred_len}.')
            # 所有 baseline 使用同一条件协议：只看当前序列最后 128 点来生成下一段未来。
            contexts.append(sequence[-pred_len:, :])
        generated = adapter.predict_future(np.stack(contexts, axis=0), pred_len=pred_len)
        generated = generated[:, :current_chunk_len, :]

        for idx, chunk in enumerate(generated):
            current_sequences[idx] = np.concatenate([current_sequences[idx], chunk], axis=0)
            generated_chunks[idx].append(chunk)
        remaining -= current_chunk_len

    return [np.concatenate(chunks, axis=0).astype(np.float32) for chunks in generated_chunks]


def main():
    args = parse_args()
    config_path = Path(args.config_file) if args.config_file else default_config_path(args.model)
    config = load_config(config_path)
    config['model'] = args.model
    config['baseline']['pred_len'] = int(args.pred_len)
    config['baseline']['context_len'] = int(args.pred_len)

    seed_everything(args.seed)
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    checkpoint_root = Path(args.checkpoint or config['training']['checkpoint_dir'])
    checkpoint = checkpoint_root if checkpoint_root.is_file() else checkpoint_root / 'latest.pt'
    adapter = build_adapter(args.model, config, device)
    adapter.load(checkpoint, map_location=device)

    output_dir = Path(args.output_root) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_source == 'test_split':
        # 默认评估训练协议中的 HCP 测试被试，不再把 Rest1-500 当作 baseline 主表测试集。
        subject_files = list_subject_split_files(config, split='test', max_subjects=args.max_subjects)
    else:
        subject_files = list_subject_files(args.source_dir, max_subjects=args.max_subjects)

    rows = []
    for extend_len in args.extend_lengths:
        for batch_files in tqdm(list(iter_batches(subject_files, args.batch_size)), desc=f'{args.model} extend_len={extend_len}'):
            records = [load_normalized_subject(path) for path in batch_files]
            histories = []
            true_futures = []
            for record in records:
                if record.normalized_sequence.shape[0] <= extend_len:
                    raise ValueError(f'{record.subject_id} length is not longer than extend_len={extend_len}.')
                histories.append(record.normalized_sequence[:-extend_len, :])
                true_futures.append(record.normalized_sequence[-extend_len:, :])

            generated_futures = extend_batch(
                adapter=adapter,
                normalized_histories=histories,
                total_extend_len=extend_len,
                pred_len=args.pred_len,
            )

            for record, generated_future, true_future in zip(records, generated_futures, true_futures):
                metrics = compute_future_metrics_z(generated_future, true_future)
                rows.append({
                    'model': args.model,
                    'subject_id': record.subject_id,
                    'extend_len': int(extend_len),
                    'autoregressive': bool(extend_len > args.pred_len),
                    'history_len': int(record.normalized_sequence.shape[0] - extend_len),
                    'future_len': int(extend_len),
                    **metrics,
                })

    tail_df = pd.DataFrame(rows)
    tail_df.to_csv(output_dir / 'tail_holdout_summary.csv', index=False)
    metrics_by_length = summarize_by_length(rows, model_name=args.model)
    metrics_by_length.to_csv(output_dir / 'metrics_by_length.csv', index=False)
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump({
            'model': args.model,
            'config_file': str(config_path),
            'checkpoint': str(checkpoint),
            'eval_source': args.eval_source,
            'source_dir': args.source_dir if args.eval_source == 'rest1' else config['data']['train_data_root'],
            'num_subjects': len(subject_files),
            'subjects': [path.stem for path in subject_files],
            'pred_len': args.pred_len,
            'extend_lengths': args.extend_lengths,
            'metrics': ['corr_z', 'mae_z', 'rmse_z', 'fc_upper_corr'],
        }, f, indent=2)


if __name__ == '__main__':
    main()
