import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from engine.logger import Logger
from engine.solver import Trainer
from Utils.extension_utils import (
    compute_extension_metrics,
    fit_sequence_scaler,
    inverse_normalize_sequence,
    load_raw_sequence,
    normalize_sequence,
)
from Utils.io_utils import load_yaml_config, merge_opts_to_config, instantiate_from_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description='Batch-extend rest1 sequences matched to test_info.csv.')
    parser.add_argument('--name', type=str, required=True, help='Output experiment name under OUTPUT/.')
    parser.add_argument('--config_file', type=str, required=True, help='Training config file.')
    parser.add_argument('--checkpoint_name', type=str, required=True, help='Checkpoint subdirectory name used during training.')
    parser.add_argument('--milestone', type=int, default=10, help='Checkpoint milestone if not using best.pt.')
    parser.add_argument('--use_best_checkpoint', action='store_true', default=False, help='Load best.pt instead of checkpoint-<milestone>.pt.')
    parser.add_argument('--output', type=str, default='OUTPUT', help='Base output directory.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
    parser.add_argument('--norm_method', type=str, default='zscore', choices=['minmax', 'zscore'], help='Normalization method for extension.')
    parser.add_argument('--pred_len', type=int, default=128, help='Single-step extension horizon.')
    parser.add_argument('--extend_lengths', type=int, nargs='+', default=[128, 256, 384, 512], help='Target extension lengths.')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of subjects to extend in parallel.')
    parser.add_argument('--info_csv', type=str, required=True, help='Path to test_info.csv.')
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing rest1 CSV files.')
    parser.add_argument('--tensorboard', action='store_true', help='Unused, kept for logger compatibility.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Optional config overrides.')
    return parser.parse_args()


def read_subject_ids(info_csv):
    subject_ids = []
    with open(info_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            subject_ids.append(row[0].strip())
    return subject_ids


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def extend_batch_to_max(trainer, normalized_sequences, max_extend_len, pred_len, coef, stepsize, sampling_steps):
    if len(normalized_sequences) == 0:
        return []

    seq_length = trainer.model.seq_length
    feature_dim = normalized_sequences[0].shape[1]
    current_sequences = [sequence.astype(np.float32).copy() for sequence in normalized_sequences]
    generated_chunks = [[] for _ in current_sequences]
    remaining = max_extend_len

    while remaining > 0:
        current_chunk_len = min(pred_len, remaining)
        context_len = seq_length - current_chunk_len
        target_windows = np.zeros((len(current_sequences), seq_length, feature_dim), dtype=np.float32)
        partial_masks = np.zeros((len(current_sequences), seq_length, feature_dim), dtype=bool)

        for idx, current_sequence in enumerate(current_sequences):
            if current_sequence.shape[0] < context_len:
                raise ValueError(
                    f'Input sequence length {current_sequence.shape[0]} is shorter than required context length {context_len}.'
                )
            target_windows[idx, :context_len, :] = current_sequence[-context_len:, :]
            partial_masks[idx, :context_len, :] = True

        restored_windows = trainer.restore_window_batch(
            target_windows=target_windows,
            partial_masks=partial_masks,
            coef=coef,
            stepsize=stepsize,
            sampling_steps=sampling_steps,
        )
        generated_batch = restored_windows[:, -current_chunk_len:, :]

        for idx in range(len(current_sequences)):
            generated_chunk = generated_batch[idx]
            current_sequences[idx] = np.concatenate([current_sequences[idx], generated_chunk], axis=0)
            generated_chunks[idx].append(generated_chunk)

        remaining -= current_chunk_len

    return [np.concatenate(chunks, axis=0) for chunks in generated_chunks]


def main():
    args = parse_args()
    args.save_dir = os.path.join(args.output, args.name)

    seed_everything(args.seed)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    config['solver']['results_folder'] = os.path.join(config['solver']['results_folder'], args.checkpoint_name)
    config['dataloader']['train_dataset']['params']['normalization'] = args.norm_method
    config['dataloader']['test_dataset']['params']['normalization'] = args.norm_method

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    trainer = Trainer(
        config=config,
        args=args,
        model=model,
        dataloader=None,
        logger=logger,
        val_dataloader=None,
    )

    checkpoint_ref = 'best' if args.use_best_checkpoint else args.milestone
    trainer.load(checkpoint_ref, verbose=True)

    dataset_name = config['dataloader']['train_dataset']['params'].get('name', '')
    neg_one_to_one = config['dataloader']['train_dataset']['params'].get('neg_one_to_one', True)
    coef = config['dataloader']['test_dataset']['coefficient']
    stepsize = config['dataloader']['test_dataset']['step_size']
    sampling_steps = config['dataloader']['test_dataset']['sampling_steps']

    subject_ids = read_subject_ids(args.info_csv)
    matched = []
    missing = []
    for subject_id in subject_ids:
        filepath = os.path.join(args.source_dir, f'{subject_id}.csv')
        if os.path.exists(filepath):
            matched.append((subject_id, filepath))
        else:
            missing.append(subject_id)

    logger.log_info(f'matched {len(matched)} files, missing {len(missing)} files')

    summary_rows = []
    extension_root = Path(args.save_dir)
    extension_root.mkdir(parents=True, exist_ok=True)

    max_extend_len = max(args.extend_lengths)
    for batch_idx, batch_items in enumerate(iter_batches(matched, args.batch_size)):
        logger.log_info(f'extending batch {batch_idx + 1} with {len(batch_items)} subjects')
        batch_records = []
        normalized_sequences = []

        for subject_id, filepath in batch_items:
            raw_sequence = load_raw_sequence(filepath, dataset_name)
            scaler = fit_sequence_scaler(raw_sequence, normalization=args.norm_method)
            normalized_sequence, auto_norm = normalize_sequence(
                raw_sequence,
                scaler,
                normalization=args.norm_method,
                neg_one_to_one=neg_one_to_one,
            )
            batch_records.append({
                'subject_id': subject_id,
                'raw_sequence': raw_sequence,
                'scaler': scaler,
                'auto_norm': auto_norm,
                'normalized_sequence': normalized_sequence,
            })
            normalized_sequences.append(normalized_sequence)

        max_extensions_norm = extend_batch_to_max(
            trainer=trainer,
            normalized_sequences=normalized_sequences,
            max_extend_len=max_extend_len,
            pred_len=args.pred_len,
            coef=coef,
            stepsize=stepsize,
            sampling_steps=sampling_steps,
        )

        for record, max_extension_norm in zip(batch_records, max_extensions_norm):
            subject_id = record['subject_id']
            raw_sequence = record['raw_sequence']
            scaler = record['scaler']
            auto_norm = record['auto_norm']
            normalized_sequence = record['normalized_sequence']

            for extend_len in args.extend_lengths:
                generated_extension_norm = max_extension_norm[:extend_len]
                extended_sequence_norm = np.concatenate([normalized_sequence, generated_extension_norm], axis=0)
                generated_extension_raw = inverse_normalize_sequence(generated_extension_norm, scaler, auto_norm=auto_norm)
                extended_sequence_raw = inverse_normalize_sequence(extended_sequence_norm, scaler, auto_norm=auto_norm)

                metrics = compute_extension_metrics(
                    history_raw=raw_sequence,
                    extension_raw=generated_extension_raw,
                    compare_len=extend_len,
                )

                length_dir = extension_root / f'extend_{extend_len}'
                length_dir.mkdir(parents=True, exist_ok=True)

                pd.DataFrame(extended_sequence_raw).to_csv(length_dir / f'{subject_id}_extended_full.csv', index=False)
                pd.DataFrame(generated_extension_raw).to_csv(length_dir / f'{subject_id}_extension_only.csv', index=False)
                np.save(length_dir / f'{subject_id}_extended_full.npy', extended_sequence_raw)
                np.save(length_dir / f'{subject_id}_extension_only.npy', generated_extension_raw)
                with open(length_dir / f'{subject_id}_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)

                summary_rows.append({
                    'subject_id': subject_id,
                    'extend_len': extend_len,
                    'autoregressive': extend_len > args.pred_len,
                    'history_len': raw_sequence.shape[0],
                    'full_len': extended_sequence_raw.shape[0],
                    **metrics,
                })

    pd.DataFrame(summary_rows).to_csv(extension_root / 'extension_summary.csv', index=False)
    with open(extension_root / 'extension_manifest.json', 'w') as f:
        json.dump({
            'checkpoint': trainer.loaded_checkpoint_meta,
            'info_csv': args.info_csv,
            'source_dir': args.source_dir,
            'matched_subjects': [subject_id for subject_id, _ in matched],
            'missing_subjects': missing,
            'extend_lengths': args.extend_lengths,
            'batch_size': args.batch_size,
            'max_extend_len': max_extend_len,
            'pred_len': args.pred_len,
            'normalization': args.norm_method,
        }, f, indent=2)

    logger.log_info(
        f'batch extension done | matched={len(matched)} | missing={len(missing)} | '
        f'lengths={args.extend_lengths}'
    )


if __name__ == '__main__':
    main()
