import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from engine.logger import Logger
from engine.solver import Trainer
from Utils.extension_utils import (
    compute_dynamic_fc_similarity,
    compute_groundtruth_extension_metrics,
    fit_sequence_scaler,
    inverse_normalize_sequence,
    load_raw_sequence,
    normalize_sequence,
)
from Utils.io_utils import load_yaml_config, merge_opts_to_config, instantiate_from_config, seed_everything


Z_SCORE_NORMALIZATION = 'zscore'


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Rest1-500 ground-truth BOLD extension and dynamic FC trajectories.')
    parser.add_argument('--name', type=str, default='rest1_500_groundtruth_DFCGraph_lambda4')
    parser.add_argument('--output', type=str, default='OUTPUT')
    parser.add_argument('--config_file', type=str, default='./Config/fmri_seq256_dfc.yaml')
    parser.add_argument('--checkpoint_name', type=str, default='fmri_seq256_zscore_DFCGraph_stride64_subj1000_lambda4_256')
    parser.add_argument('--checkpoint_ref', type=str, default='best', help='Use "best" or a numeric milestone such as "10".')
    parser.add_argument('--source_dir', type=str, default='/data/home/zyzeng/data1/datasets/rest1_to_csv_500')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--pred_len', type=int, default=128)
    parser.add_argument('--extend_lengths', type=int, nargs='+', default=[128, 256, 384, 512])
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fc_window', type=int, default=128)
    parser.add_argument('--fc_stride', type=int, default=32)
    parser.add_argument('--max_subjects', type=int, default=0, help='Use only the first N sorted subjects for smoke tests. <=0 uses all.')
    parser.add_argument('--num_plot_subjects', type=int, default=3, help='Number of sorted subjects to use for example figures.')
    parser.add_argument('--skip_dynamic_fc', action='store_true', help='Skip dynamic FC trajectory CSV and summary metrics.')
    parser.add_argument('--skip_sliding_windows', action='store_true', help='Skip Rest1 sliding-window prediction outputs.')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Optional config overrides.')
    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, args.name)
    return args


def checkpoint_name_for_trainer(checkpoint_name, seq_length):
    suffix = f'_{seq_length}'
    # Trainer 会自动在 results_folder 后追加 "_seq_length"，因此默认传入的真实目录名需要去掉一次后缀。
    if checkpoint_name.endswith(suffix):
        return checkpoint_name[:-len(suffix)]
    return checkpoint_name


def checkpoint_ref_value(checkpoint_ref):
    if checkpoint_ref == 'best':
        return 'best'
    if checkpoint_ref.isdigit():
        return int(checkpoint_ref)
    raise ValueError(f'checkpoint_ref must be "best" or an integer string, got {checkpoint_ref}.')


def list_subject_files(source_dir, max_subjects):
    files = sorted(Path(source_dir).glob('*.csv'))
    if len(files) == 0:
        raise ValueError(f'No CSV files found in {source_dir}.')
    if max_subjects > 0:
        files = files[:max_subjects]
    return files


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_trainer(args):
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    seq_length = int(config['model']['params']['seq_length'])
    checkpoint_base_name = checkpoint_name_for_trainer(args.checkpoint_name, seq_length)
    config['solver']['results_folder'] = os.path.join(config['solver']['results_folder'], checkpoint_base_name)
    config['dataloader']['train_dataset']['params']['normalization'] = Z_SCORE_NORMALIZATION
    config['dataloader']['test_dataset']['params']['normalization'] = Z_SCORE_NORMALIZATION

    logger = Logger(args)
    logger.save_config(config)

    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
        logger.log_info('CUDA is not available or --gpu is None; running on CPU will be slow.')

    model = instantiate_from_config(config['model']).to(device)
    trainer = Trainer(
        config=config,
        args=args,
        model=model,
        dataloader=None,
        logger=logger,
        val_dataloader=None,
    )
    trainer.load(checkpoint_ref_value(args.checkpoint_ref), verbose=True)
    return config, trainer, logger


def inverse_window_batch(normalized_windows, scaler, auto_norm):
    original_shape = normalized_windows.shape
    feature_dim = normalized_windows.shape[-1]
    restored = inverse_normalize_sequence(
        normalized_windows.reshape(-1, feature_dim),
        scaler,
        auto_norm=auto_norm,
    )
    return restored.reshape(original_shape)


def compute_zscore_error_metrics(generated_future_norm, true_future_norm):
    if generated_future_norm.shape != true_future_norm.shape:
        raise ValueError(
            f'generated_future_norm and true_future_norm must have the same shape, '
            f'got {generated_future_norm.shape} and {true_future_norm.shape}.'
        )
    # z-score 指标保留在模型实际生成空间中，便于跨 subject 比较误差大小。
    err = generated_future_norm - true_future_norm
    return {
        'mae_z': float(np.mean(np.abs(err))),
        'mse_z': float(np.mean(err ** 2)),
        'rmse_z': float(np.sqrt(np.mean(err ** 2))),
    }


def extend_batch_to_length(trainer, normalized_histories, total_extend_len, pred_len, coef, stepsize, sampling_steps):
    if len(normalized_histories) == 0:
        return []

    seq_length = int(trainer.model.seq_length)
    feature_dim = normalized_histories[0].shape[1]
    current_sequences = [sequence.astype(np.float32).copy() for sequence in normalized_histories]
    generated_chunks = [[] for _ in current_sequences]
    remaining = int(total_extend_len)

    while remaining > 0:
        current_chunk_len = min(pred_len, remaining)
        context_len = seq_length - current_chunk_len
        target_windows = np.zeros((len(current_sequences), seq_length, feature_dim), dtype=np.float32)
        partial_masks = np.zeros((len(current_sequences), seq_length, feature_dim), dtype=bool)

        for idx, current_sequence in enumerate(current_sequences):
            if current_sequence.shape[0] < context_len:
                raise ValueError(
                    f'Input history length {current_sequence.shape[0]} is shorter than required context length {context_len}.'
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

        for idx, generated_chunk in enumerate(generated_batch):
            current_sequences[idx] = np.concatenate([current_sequences[idx], generated_chunk], axis=0)
            generated_chunks[idx].append(generated_chunk)
        remaining -= current_chunk_len

    return [np.concatenate(chunks, axis=0) for chunks in generated_chunks]


def restore_sliding_future_batch(trainer, context_windows, pred_len, coef, stepsize, sampling_steps):
    if len(context_windows) == 0:
        return np.empty((0, pred_len, 0), dtype=np.float32)

    seq_length = int(trainer.model.seq_length)
    feature_dim = context_windows[0].shape[1]
    context_len = seq_length - pred_len
    target_windows = np.zeros((len(context_windows), seq_length, feature_dim), dtype=np.float32)
    partial_masks = np.zeros((len(context_windows), seq_length, feature_dim), dtype=bool)

    for idx, context_window in enumerate(context_windows):
        if context_window.shape[0] != context_len:
            raise ValueError(f'Expected context length {context_len}, got {context_window.shape[0]}.')
        target_windows[idx, :context_len, :] = context_window
        partial_masks[idx, :context_len, :] = True

    restored_windows = trainer.restore_window_batch(
        target_windows=target_windows,
        partial_masks=partial_masks,
        coef=coef,
        stepsize=stepsize,
        sampling_steps=sampling_steps,
    )
    return restored_windows[:, -pred_len:, :]


def sliding_starts(length, window, stride):
    if length < window:
        return []
    return list(range(0, length - window + 1, stride))


def corr_matrix(sequence):
    corr = np.corrcoef(sequence, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_tail_example(history_raw, generated_future_raw, true_future_raw, output_path):
    plt = try_import_matplotlib()
    if plt is None:
        return

    roi_indices = [0, 1, 2]
    roi_indices = [idx for idx in roi_indices if idx < history_raw.shape[1]]
    history_steps = np.arange(history_raw.shape[0])
    future_steps = np.arange(history_raw.shape[0], history_raw.shape[0] + true_future_raw.shape[0])

    fig, axes = plt.subplots(len(roi_indices), 1, figsize=(10, 3 * len(roi_indices)), sharex=True)
    if len(roi_indices) == 1:
        axes = [axes]
    for ax, roi_idx in zip(axes, roi_indices):
        ax.plot(history_steps, history_raw[:, roi_idx], label='History', linewidth=1.0)
        ax.plot(future_steps, true_future_raw[:, roi_idx], label='True future', linewidth=1.0)
        ax.plot(future_steps, generated_future_raw[:, roi_idx], label='Generated future', linewidth=1.0)
        ax.axvline(history_raw.shape[0] - 1, color='r', linestyle='--', linewidth=1.0)
        ax.set_title(f'ROI {roi_idx}')
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fc_heatmap(generated_future_raw, true_future_raw, output_path):
    plt = try_import_matplotlib()
    if plt is None:
        return

    gen_fc = corr_matrix(generated_future_raw)
    true_fc = corr_matrix(true_future_raw)
    diff_fc = gen_fc - true_fc
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ('Generated FC', gen_fc, -1.0, 1.0),
        ('True future FC', true_fc, -1.0, 1.0),
        ('Generated - True', diff_fc, -1.0, 1.0),
    ]
    for ax, (title, matrix, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(matrix, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_dynamic_fc_lines(dynamic_records, output_path):
    plt = try_import_matplotlib()
    if plt is None or len(dynamic_records) == 0:
        return

    df = pd.DataFrame(dynamic_records)
    future_ids = sorted(df['future_window_idx'].unique())
    fig, axes = plt.subplots(len(future_ids), 1, figsize=(10, 3 * len(future_ids)), sharex=True)
    if len(future_ids) == 1:
        axes = [axes]

    for ax, future_idx in zip(axes, future_ids):
        sub_df = df[df['future_window_idx'] == future_idx].sort_values('history_window_idx')
        ax.plot(sub_df['history_center'], sub_df['gen_hist_fc_corr'], label='Generated vs history FC', linewidth=1.4)
        ax.plot(sub_df['history_center'], sub_df['true_hist_fc_corr'], label='True future vs history FC', linewidth=1.4)
        ax.set_ylabel('FC correlation')
        ax.set_title(f'Future window {future_idx}')
        ax.legend(loc='best')
    axes[-1].set_xlabel('History window center')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_dynamic_fc_heatmap(dynamic_records, output_path):
    plt = try_import_matplotlib()
    if plt is None or len(dynamic_records) == 0:
        return

    df = pd.DataFrame(dynamic_records)
    if df['future_window_idx'].nunique() <= 1:
        return
    gen_grid = df.pivot(index='future_window_idx', columns='history_window_idx', values='gen_hist_fc_corr')
    true_grid = df.pivot(index='future_window_idx', columns='history_window_idx', values='true_hist_fc_corr')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (title, grid) in zip(axes, [('Generated trajectory', gen_grid), ('True future trajectory', true_grid)]):
        image = ax.imshow(grid.values, aspect='auto', cmap='viridis', vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel('History window idx')
        ax.set_ylabel('Future window idx')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metrics_by_length(summary_df, output_path):
    plt = try_import_matplotlib()
    if plt is None or summary_df.empty:
        return

    metrics = ['rmse', 'fc_upper_corr', 'psd_l1']
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        mean_col = f'{metric}_mean'
        if mean_col not in summary_df.columns:
            continue
        ax.plot(summary_df['extend_len'], summary_df[mean_col], marker='o')
        ax.set_xlabel('Extend length')
        ax.set_ylabel(metric)
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize_by_length(tail_df):
    metric_cols = [
        'mae', 'mse', 'rmse', 'future_mean_shift_mae', 'future_std_shift_mae',
        'mae_z', 'mse_z', 'rmse_z',
        'fc_upper_corr', 'fc_abs_diff', 'psd_l1', 'seam_mae', 'seam_rmse',
        'gen_max_hist_fc_corr', 'gen_last_hist_fc_corr',
        'true_max_hist_fc_corr', 'true_last_hist_fc_corr',
        'gen_true_trajectory_corr', 'gen_true_best_window_match',
    ]
    rows = []
    for extend_len, group in tail_df.groupby('extend_len'):
        row = {'extend_len': int(extend_len), 'num_subjects': int(group['subject_id'].nunique())}
        for metric in metric_cols:
            if metric not in group:
                continue
            values = group[metric].dropna()
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_std'] = float(values.std(ddof=0))
            row[f'{metric}_median'] = float(values.median())
            row[f'{metric}_q25'] = float(values.quantile(0.25))
            row[f'{metric}_q75'] = float(values.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values('extend_len')


def load_normalized_subject(filepath, dataset_name):
    raw_sequence = load_raw_sequence(str(filepath), dataset_name)
    scaler = fit_sequence_scaler(raw_sequence, normalization=Z_SCORE_NORMALIZATION)
    # 本实验为对齐当前 prediction 代码，用完整 subject 序列拟合 z-score；这不是严格的 history-only normalization。
    normalized_sequence, auto_norm = normalize_sequence(
        raw_sequence,
        scaler,
        normalization=Z_SCORE_NORMALIZATION,
        neg_one_to_one=False,
    )
    return raw_sequence, normalized_sequence, scaler, auto_norm


def process_tail_holdout(args, trainer, config, subject_files, output_dir, plots_dir):
    dataset_name = config['dataloader']['train_dataset']['params'].get('name', '')
    coef = config['dataloader']['test_dataset']['coefficient']
    stepsize = config['dataloader']['test_dataset']['step_size']
    sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
    example_subject_ids = {path.stem for path in subject_files[:max(args.num_plot_subjects, 0)]}

    tail_rows = []
    dynamic_rows = []
    for extend_len in args.extend_lengths:
        batches = list(iter_batches(subject_files, args.batch_size))
        for batch_files in tqdm(batches, desc=f'tail holdout extend_len={extend_len}'):
            batch_records = []
            normalized_histories = []

            for filepath in batch_files:
                raw_sequence, normalized_sequence, scaler, auto_norm = load_normalized_subject(filepath, dataset_name)
                if raw_sequence.shape[0] <= extend_len:
                    raise ValueError(f'{filepath} length {raw_sequence.shape[0]} is not longer than extend_len={extend_len}.')
                history_raw = raw_sequence[:-extend_len, :]
                true_future_raw = raw_sequence[-extend_len:, :]
                normalized_history = normalized_sequence[:-extend_len, :]
                true_future_norm = normalized_sequence[-extend_len:, :]
                batch_records.append({
                    'subject_id': filepath.stem,
                    'history_raw': history_raw,
                    'true_future_raw': true_future_raw,
                    'true_future_norm': true_future_norm,
                    'scaler': scaler,
                    'auto_norm': auto_norm,
                })
                normalized_histories.append(normalized_history)

            generated_norms = extend_batch_to_length(
                trainer=trainer,
                normalized_histories=normalized_histories,
                total_extend_len=extend_len,
                pred_len=args.pred_len,
                coef=coef,
                stepsize=stepsize,
                sampling_steps=sampling_steps,
            )

            for record, generated_norm in zip(batch_records, generated_norms):
                generated_future_raw = inverse_normalize_sequence(
                    generated_norm,
                    record['scaler'],
                    auto_norm=record['auto_norm'],
                )
                metrics = compute_groundtruth_extension_metrics(
                    history_raw=record['history_raw'],
                    generated_future_raw=generated_future_raw,
                    true_future_raw=record['true_future_raw'],
                )
                metrics.update(compute_zscore_error_metrics(
                    generated_future_norm=generated_norm,
                    true_future_norm=record['true_future_norm'],
                ))
                dynamic_records = []
                dynamic_summary = {}
                if not args.skip_dynamic_fc:
                    dynamic_records, dynamic_summary = compute_dynamic_fc_similarity(
                        history_raw=record['history_raw'],
                        generated_future_raw=generated_future_raw,
                        true_future_raw=record['true_future_raw'],
                        window=args.fc_window,
                        stride=args.fc_stride,
                        future_stride=args.fc_window,
                    )

                row = {
                    'subject_id': record['subject_id'],
                    'extend_len': int(extend_len),
                    'autoregressive': bool(extend_len > args.pred_len),
                    'history_len': int(record['history_raw'].shape[0]),
                    'future_len': int(record['true_future_raw'].shape[0]),
                    **metrics,
                    **dynamic_summary,
                }
                tail_rows.append(row)

                for dynamic_record in dynamic_records:
                    dynamic_rows.append({
                        'subject_id': record['subject_id'],
                        'extend_len': int(extend_len),
                        **dynamic_record,
                    })

                if record['subject_id'] in example_subject_ids:
                    prefix = f"{record['subject_id']}_extend{extend_len}"
                    plot_tail_example(
                        record['history_raw'],
                        generated_future_raw,
                        record['true_future_raw'],
                        plots_dir / f'{prefix}_timeseries.png',
                    )
                    plot_fc_heatmap(
                        generated_future_raw,
                        record['true_future_raw'],
                        plots_dir / f'{prefix}_fc_heatmap.png',
                    )
                    plot_dynamic_fc_lines(
                        dynamic_records,
                        plots_dir / f'{prefix}_dynamic_fc_lines.png',
                    )
                    plot_dynamic_fc_heatmap(
                        dynamic_records,
                        plots_dir / f'{prefix}_dynamic_fc_heatmap.png',
                    )

    tail_df = pd.DataFrame(tail_rows)
    dynamic_df = pd.DataFrame(dynamic_rows)
    tail_df.to_csv(output_dir / 'tail_holdout_summary.csv', index=False)
    if not args.skip_dynamic_fc:
        dynamic_df.to_csv(output_dir / 'dynamic_fc_similarity.csv', index=False)

    metrics_by_length = summarize_by_length(tail_df)
    metrics_by_length.to_csv(output_dir / 'metrics_by_length.csv', index=False)
    plot_metrics_by_length(metrics_by_length, plots_dir / 'metrics_by_length.png')
    return tail_df, dynamic_df, metrics_by_length


def process_sliding_windows(args, trainer, config, subject_files, output_dir):
    dataset_name = config['dataloader']['train_dataset']['params'].get('name', '')
    coef = config['dataloader']['test_dataset']['coefficient']
    stepsize = config['dataloader']['test_dataset']['step_size']
    sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
    seq_length = int(trainer.model.seq_length)
    context_len = seq_length - args.pred_len
    if context_len <= 0:
        raise ValueError('pred_len must be smaller than seq_length for sliding-window prediction.')

    rows = []
    for filepath in tqdm(subject_files, desc='sliding-window prediction'):
        raw_sequence, normalized_sequence, scaler, auto_norm = load_normalized_subject(filepath, dataset_name)
        starts = sliding_starts(raw_sequence.shape[0], seq_length, args.stride)
        for start_batch in iter_batches(starts, args.batch_size):
            context_windows = [
                normalized_sequence[start:start + context_len, :]
                for start in start_batch
            ]
            generated_norm = restore_sliding_future_batch(
                trainer=trainer,
                context_windows=context_windows,
                pred_len=args.pred_len,
                coef=coef,
                stepsize=stepsize,
                sampling_steps=sampling_steps,
            )
            generated_raw = inverse_window_batch(generated_norm, scaler, auto_norm)

            for batch_idx, start in enumerate(start_batch):
                future_start = start + context_len
                true_future_raw = raw_sequence[future_start:start + seq_length, :]
                true_future_norm = normalized_sequence[future_start:start + seq_length, :]
                history_raw = raw_sequence[start:future_start, :]
                metrics = compute_groundtruth_extension_metrics(
                    history_raw=history_raw,
                    generated_future_raw=generated_raw[batch_idx],
                    true_future_raw=true_future_raw,
                )
                metrics.update(compute_zscore_error_metrics(
                    generated_future_norm=generated_norm[batch_idx],
                    true_future_norm=true_future_norm,
                ))
                rows.append({
                    'subject_id': filepath.stem,
                    'window_idx': int(start // args.stride),
                    'window_start': int(start),
                    'window_end': int(start + seq_length),
                    'pred_len': int(args.pred_len),
                    **metrics,
                })

    sliding_df = pd.DataFrame(rows)
    sliding_df.to_csv(output_dir / 'sliding_window_summary.csv', index=False)

    metric_cols = [
        'mae', 'mse', 'rmse', 'future_mean_shift_mae', 'future_std_shift_mae',
        'mae_z', 'mse_z', 'rmse_z',
        'fc_upper_corr', 'fc_abs_diff', 'psd_l1', 'seam_mae', 'seam_rmse',
    ]
    subject_summary = sliding_df.groupby('subject_id')[metric_cols].mean().reset_index()
    subject_summary.insert(1, 'num_windows', sliding_df.groupby('subject_id').size().values)
    subject_summary.to_csv(output_dir / 'sliding_window_subject_summary.csv', index=False)
    return sliding_df, subject_summary


def save_manifest(args, trainer, subject_files, output_dir):
    manifest = {
        'source_dir': args.source_dir,
        'num_subjects': len(subject_files),
        'subjects': [path.stem for path in subject_files],
        'checkpoint': trainer.loaded_checkpoint_meta,
        'config_file': args.config_file,
        'normalization': Z_SCORE_NORMALIZATION,
        'normalization_fit': 'full_subject_sequence',
        'zscore_error_metrics': ['mae_z', 'mse_z', 'rmse_z'],
        'skip_dynamic_fc': args.skip_dynamic_fc,
        'skip_sliding_windows': args.skip_sliding_windows,
        'pred_len': args.pred_len,
        'extend_lengths': args.extend_lengths,
        'stride': args.stride,
        'fc_window': args.fc_window,
        'fc_stride': args.fc_stride,
        'batch_size': args.batch_size,
        'max_subjects': args.max_subjects,
    }
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.save_dir)
    plots_dir = output_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    subject_files = list_subject_files(args.source_dir, args.max_subjects)
    config, trainer, logger = build_trainer(args)
    logger.log_info(f'Loaded {len(subject_files)} Rest1 subject CSV files from {args.source_dir}')

    tail_df, dynamic_df, metrics_by_length = process_tail_holdout(
        args=args,
        trainer=trainer,
        config=config,
        subject_files=subject_files,
        output_dir=output_dir,
        plots_dir=plots_dir,
    )
    if args.skip_sliding_windows:
        sliding_df = pd.DataFrame()
        sliding_subject_df = pd.DataFrame()
    else:
        sliding_df, sliding_subject_df = process_sliding_windows(
            args=args,
            trainer=trainer,
            config=config,
            subject_files=subject_files,
            output_dir=output_dir,
        )
    save_manifest(args, trainer, subject_files, output_dir)

    logger.log_info(
        f'rest1 ground-truth extension done | subjects={len(subject_files)} | '
        f'tail_rows={len(tail_df)} | dynamic_rows={len(dynamic_df)} | '
        f'sliding_rows={len(sliding_df)} | output={output_dir}'
    )


if __name__ == '__main__':
    main()
