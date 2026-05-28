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


def compute_groundtruth_extension_metrics(history_raw, generated_future_raw, true_future_raw):
    if history_raw.ndim != 2 or generated_future_raw.ndim != 2 or true_future_raw.ndim != 2:
        raise ValueError('history_raw, generated_future_raw and true_future_raw must be 2D arrays shaped [time, feature_dim].')
    if generated_future_raw.shape != true_future_raw.shape:
        raise ValueError(
            f'generated_future_raw and true_future_raw must have the same shape, '
            f'got {generated_future_raw.shape} and {true_future_raw.shape}.'
        )
    if history_raw.shape[1] != generated_future_raw.shape[1]:
        raise ValueError('history and future arrays must have the same feature dimension.')

    err = generated_future_raw - true_future_raw
    seam_diff = generated_future_raw[0, :] - history_raw[-1, :]

    # 这里是真实未来段可用的评估：直接比较生成未来和隐藏的 Rest1 未来真值。
    true_fc = _connectivity_upper(true_future_raw)
    generated_fc = _connectivity_upper(generated_future_raw)

    metrics = {
        'future_len': int(generated_future_raw.shape[0]),
        'mae': float(np.mean(np.abs(err))),
        'mse': float(np.mean(err ** 2)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'future_mean_shift_mae': float(np.mean(np.abs(
            generated_future_raw.mean(axis=0) - true_future_raw.mean(axis=0)
        ))),
        'future_std_shift_mae': float(np.mean(np.abs(
            generated_future_raw.std(axis=0) - true_future_raw.std(axis=0)
        ))),
        'fc_upper_corr': _safe_corrcoef(true_fc, generated_fc),
        # FC 平均绝对差用于直接衡量生成 FC 和真实未来 FC 在 ROI-pair 连接强度上的偏差。
        'fc_abs_diff': float(np.mean(np.abs(generated_fc - true_fc))),
        'psd_l1': float(np.mean(np.abs(
            _normalized_psd(true_future_raw) - _normalized_psd(generated_future_raw)
        ))),
        'seam_mae': float(np.mean(np.abs(seam_diff))),
        'seam_rmse': float(np.sqrt(np.mean(seam_diff ** 2))),
    }
    return metrics


def fc_upper_vector(sequence):
    if sequence.ndim != 2:
        raise ValueError('sequence must be a 2D array shaped [time, feature_dim].')
    # 动态 FC 比较只使用相关矩阵上三角，避免对角线恒为 1 影响相似性。
    return _connectivity_upper(sequence)


def _window_starts(length, window, stride):
    if window <= 0:
        raise ValueError('window must be positive.')
    if stride <= 0:
        raise ValueError('stride must be positive.')
    if length < window:
        return []
    return list(range(0, length - window + 1, stride))


def compute_dynamic_fc_similarity(
    history_raw,
    generated_future_raw,
    true_future_raw,
    window=128,
    stride=32,
    future_stride=None,
):
    if history_raw.ndim != 2 or generated_future_raw.ndim != 2 or true_future_raw.ndim != 2:
        raise ValueError('history_raw, generated_future_raw and true_future_raw must be 2D arrays shaped [time, feature_dim].')
    if generated_future_raw.shape != true_future_raw.shape:
        raise ValueError(
            f'generated_future_raw and true_future_raw must have the same shape, '
            f'got {generated_future_raw.shape} and {true_future_raw.shape}.'
        )
    if future_stride is None:
        future_stride = window

    history_starts = _window_starts(history_raw.shape[0], window, stride)
    future_starts = _window_starts(generated_future_raw.shape[0], window, future_stride)
    if len(history_starts) == 0:
        raise ValueError('history_raw is shorter than the dynamic FC window.')
    if len(future_starts) == 0:
        raise ValueError('future arrays are shorter than the dynamic FC window.')

    # 历史段用滑窗形成 subject-specific dynamic FC repertoire。
    history_vectors = [
        fc_upper_vector(history_raw[start:start + window, :])
        for start in history_starts
    ]
    records = []
    gen_trajectories = []
    true_trajectories = []

    for future_idx, future_start in enumerate(future_starts):
        gen_vec = fc_upper_vector(generated_future_raw[future_start:future_start + window, :])
        true_vec = fc_upper_vector(true_future_raw[future_start:future_start + window, :])
        gen_corrs = []
        true_corrs = []

        for hist_idx, (hist_start, hist_vec) in enumerate(zip(history_starts, history_vectors)):
            gen_corr = _safe_corrcoef(gen_vec, hist_vec)
            true_corr = _safe_corrcoef(true_vec, hist_vec)
            gen_corrs.append(gen_corr)
            true_corrs.append(true_corr)
            records.append({
                'future_window_idx': int(future_idx),
                'future_start': int(future_start),
                'history_window_idx': int(hist_idx),
                'history_start': int(hist_start),
                'history_center': float(hist_start + window / 2.0),
                'gen_hist_fc_corr': gen_corr,
                'true_hist_fc_corr': true_corr,
            })

        gen_trajectories.append(gen_corrs)
        true_trajectories.append(true_corrs)

    gen_trajectories = np.asarray(gen_trajectories, dtype=np.float32)
    true_trajectories = np.asarray(true_trajectories, dtype=np.float32)
    gen_argmax = np.argmax(gen_trajectories, axis=1)
    true_argmax = np.argmax(true_trajectories, axis=1)
    trajectory_corrs = [
        _safe_corrcoef(gen_row, true_row)
        for gen_row, true_row in zip(gen_trajectories, true_trajectories)
    ]

    summary = {
        'dynamic_num_history_windows': int(len(history_starts)),
        'dynamic_num_future_windows': int(len(future_starts)),
        'gen_max_hist_fc_corr': float(np.mean(np.max(gen_trajectories, axis=1))),
        'gen_last_hist_fc_corr': float(np.mean(gen_trajectories[:, -1])),
        'true_max_hist_fc_corr': float(np.mean(np.max(true_trajectories, axis=1))),
        'true_last_hist_fc_corr': float(np.mean(true_trajectories[:, -1])),
        'gen_true_trajectory_corr': float(np.mean(trajectory_corrs)),
        # 该值表示生成未来和真实未来是否最像同一个历史 FC 窗口，用于描述动态 FC 状态匹配。
        'gen_true_best_window_match': float(np.mean(gen_argmax == true_argmax)),
    }
    return records, summary


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
