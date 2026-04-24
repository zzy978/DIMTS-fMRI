import json
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)


def load_raw_sequence(filepath, name=''):
    if filepath.endswith('.npy'):
        data = np.load(filepath)
    else:
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
    if data.ndim != 2:
        raise ValueError(f'Expected a 2D sequence, got shape {data.shape} from {filepath}.')
    return data.astype(np.float32)


def fit_sequence_scaler(raw_sequence, normalization='minmax'):
    if normalization == 'minmax':
        scaler = MinMaxScaler()
    elif normalization == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError(f'Unsupported normalization method: {normalization}')
    scaler.fit(raw_sequence)
    return scaler


def normalize_sequence(raw_sequence, scaler, normalization='minmax', neg_one_to_one=True):
    normalized = scaler.transform(raw_sequence)
    auto_norm = normalization == 'minmax' and neg_one_to_one
    if auto_norm:
        normalized = normalize_to_neg_one_to_one(normalized)
    return normalized.astype(np.float32), auto_norm


def inverse_normalize_sequence(normalized_sequence, scaler, auto_norm=False):
    restored = normalized_sequence
    if auto_norm:
        restored = unnormalize_to_zero_to_one(restored)
    restored = scaler.inverse_transform(restored)
    return restored.astype(np.float32)


def _safe_corrcoef(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        return 0.0
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _lag1_autocorr(sequence):
    if sequence.shape[0] < 2:
        return np.zeros(sequence.shape[1], dtype=np.float32)
    x0 = sequence[:-1, :]
    x1 = sequence[1:, :]
    x0_centered = x0 - x0.mean(axis=0, keepdims=True)
    x1_centered = x1 - x1.mean(axis=0, keepdims=True)
    numerator = np.sum(x0_centered * x1_centered, axis=0)
    denominator = np.sqrt(
        np.sum(x0_centered ** 2, axis=0) * np.sum(x1_centered ** 2, axis=0)
    ) + 1e-8
    return numerator / denominator


def _connectivity_upper(sequence):
    corr = np.corrcoef(sequence, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    return upper


def _normalized_psd(sequence):
    spectrum = np.abs(np.fft.rfft(sequence, axis=0))
    denom = spectrum.sum(axis=0, keepdims=True) + 1e-8
    return spectrum / denom


def compute_extension_metrics(history_raw, extension_raw, compare_len=None):
    if history_raw.ndim != 2 or extension_raw.ndim != 2:
        raise ValueError('history_raw and extension_raw must be 2D arrays shaped [time, feature_dim].')

    if compare_len is None:
        compare_len = min(history_raw.shape[0], extension_raw.shape[0])
    compare_len = int(min(compare_len, history_raw.shape[0], extension_raw.shape[0]))
    if compare_len <= 0:
        raise ValueError('compare_len must be positive for extension metrics.')

    history_context = history_raw[-compare_len:, :]
    extension_compare = extension_raw[:compare_len, :]

    seam_diff = extension_raw[0, :] - history_raw[-1, :]
    mean_diff = extension_compare.mean(axis=0) - history_context.mean(axis=0)
    std_diff = extension_compare.std(axis=0) - history_context.std(axis=0)
    lag1_diff = _lag1_autocorr(extension_compare) - _lag1_autocorr(history_context)

    history_fc = _connectivity_upper(history_context)
    extension_fc = _connectivity_upper(extension_compare)
    history_psd = _normalized_psd(history_context)
    extension_psd = _normalized_psd(extension_compare)

    metrics = {
        'compare_len': compare_len,
        'seam_mae': float(np.mean(np.abs(seam_diff))),
        'seam_rmse': float(np.sqrt(np.mean(seam_diff ** 2))),
        'mean_shift_mae': float(np.mean(np.abs(mean_diff))),
        'std_shift_mae': float(np.mean(np.abs(std_diff))),
        'lag1_autocorr_mae': float(np.mean(np.abs(lag1_diff))),
        'fc_upper_corr': _safe_corrcoef(history_fc, extension_fc),
        'psd_l1': float(np.mean(np.abs(history_psd - extension_psd))),
    }
    return metrics


def compute_prediction_metrics(pred, gt, pred_len, eps=1e-8):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f'pred and gt must have the same shape, got {pred.shape} and {gt.shape}.')
    if pred.ndim != 3:
        raise ValueError(f'Expected pred and gt shaped [batch, seq_len, feature_dim], got {pred.shape}.')
    if pred_len <= 0 or pred_len > pred.shape[1]:
        raise ValueError(f'pred_len must be in [1, {pred.shape[1]}], got {pred_len}.')

    pred_future = pred[:, -pred_len:, :]
    gt_future = gt[:, -pred_len:, :]
    err = pred_future - gt_future

    fc_corrs = []
    psd_l1s = []
    for pred_sample, gt_sample in zip(pred_future, gt_future):
        fc_corrs.append(_safe_corrcoef(_connectivity_upper(gt_sample), _connectivity_upper(pred_sample)))
        psd_l1s.append(float(np.mean(np.abs(_normalized_psd(gt_sample) - _normalized_psd(pred_sample)))))

    metrics = {
        'num_samples': int(pred.shape[0]),
        'seq_len': int(pred.shape[1]),
        'pred_len': int(pred_len),
        'mae': float(np.mean(np.abs(err))),
        'mse': float(np.mean(err ** 2)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mape': float(np.mean(np.abs(err) / (np.abs(gt_future) + eps)) * 100.0),
        'future_mean_shift_mae': float(np.mean(np.abs(pred_future.mean(axis=1) - gt_future.mean(axis=1)))),
        'future_std_shift_mae': float(np.mean(np.abs(pred_future.std(axis=1) - gt_future.std(axis=1)))),
        'fc_upper_corr': float(np.mean(fc_corrs)) if len(fc_corrs) > 0 else 0.0,
        'psd_l1': float(np.mean(psd_l1s)) if len(psd_l1s) > 0 else 0.0,
    }
    return metrics


def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_extension_summary(history_raw, extension_raw, output_path, roi_indices=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if roi_indices is None:
        roi_indices = [0, min(1, history_raw.shape[1] - 1), min(2, history_raw.shape[1] - 1)]
    roi_indices = [idx for idx in roi_indices if 0 <= idx < history_raw.shape[1]]
    if len(roi_indices) == 0:
        return None

    history_steps = np.arange(history_raw.shape[0])
    extension_steps = np.arange(history_raw.shape[0], history_raw.shape[0] + extension_raw.shape[0])

    fig, axes = plt.subplots(len(roi_indices), 1, figsize=(10, 3 * len(roi_indices)), sharex=True)
    if len(roi_indices) == 1:
        axes = [axes]

    for ax, roi_idx in zip(axes, roi_indices):
        ax.plot(history_steps, history_raw[:, roi_idx], label='History')
        ax.plot(extension_steps, extension_raw[:, roi_idx], label='Extension')
        ax.axvline(history_raw.shape[0] - 1, color='r', linestyle='--', linewidth=1)
        ax.set_title(f'ROI {roi_idx}')
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
