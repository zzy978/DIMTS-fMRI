import numpy as np
import pandas as pd

from Utils.extension_utils import _connectivity_upper, _safe_corrcoef


MAIN_METRICS = ('corr_z', 'mae_z', 'rmse_z', 'fc_upper_corr')


def compute_future_metrics_z(generated_future_norm, true_future_norm):
    generated_future_norm = np.asarray(generated_future_norm, dtype=np.float32)
    true_future_norm = np.asarray(true_future_norm, dtype=np.float32)
    if generated_future_norm.shape != true_future_norm.shape:
        raise ValueError(
            f'generated_future_norm and true_future_norm must have the same shape, '
            f'got {generated_future_norm.shape} and {true_future_norm.shape}.'
        )

    err = generated_future_norm - true_future_norm
    true_fc = _connectivity_upper(true_future_norm)
    generated_fc = _connectivity_upper(generated_future_norm)

    # corr_z 直接把 [time, ROI] 展平，衡量隐藏未来段整体波形是否一致。
    return {
        'corr_z': _safe_corrcoef(true_future_norm, generated_future_norm),
        'mae_z': float(np.mean(np.abs(err))),
        'mse_z': float(np.mean(err ** 2)),
        'rmse_z': float(np.sqrt(np.mean(err ** 2))),
        'fc_upper_corr': _safe_corrcoef(true_fc, generated_fc),
    }


def summarize_by_length(rows, model_name=None):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    group_cols = ['extend_len']
    if model_name is not None:
        df['model'] = model_name
        group_cols = ['model', 'extend_len']

    summary_rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row['num_subjects'] = int(group['subject_id'].nunique()) if 'subject_id' in group else int(len(group))
        for metric in MAIN_METRICS:
            values = group[metric].dropna()
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_std'] = float(values.std(ddof=0))
            row[f'{metric}_median'] = float(values.median())
            row[f'{metric}_q25'] = float(values.quantile(0.25))
            row[f'{metric}_q75'] = float(values.quantile(0.75))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(group_cols)

