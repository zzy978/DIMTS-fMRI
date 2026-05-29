import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Baselines.metrics import compute_future_metrics_z, summarize_by_length
from Baselines.registry import SUPPORTED_BASELINES, build_adapter
from Utils.io_utils import seed_everything


def tiny_config(model_name):
    return {
        'model': model_name,
        'baseline': {
            'seq_length': 256,
            'context_len': 128,
            'pred_len': 128,
            'feature_size': 219,
            'hidden_dim': 8,
            'latent_dim': 4,
            'noise_dim': 4,
            'timesteps': 4,
            'sampling_steps': 4,
            'fft_loss_weight': 0.01 if model_name in {'fourierdiff', 'pad_ts'} else 0.0,
            'continuity_weight': 0.01 if model_name == 'pad_ts' else 0.0,
            'corr_loss_weight': 0.01 if model_name == 'pad_ts' else 0.0,
            'reconstruction_weight': 2.0,
        },
        'training': {
            'lr': 1.0e-3,
            'weight_decay': 0.0,
            'grad_clip': 1.0,
        },
    }


def autoregressive_extend(adapter, history, extend_len, pred_len):
    current = history.copy()
    chunks = []
    remaining = extend_len
    while remaining > 0:
        chunk_len = min(pred_len, remaining)
        # smoke test 也固定只取最后 128 点作为条件，验证真实评估路径的长度协议。
        context = current[-pred_len:, :][None, :, :]
        chunk = adapter.predict_future(context, pred_len=pred_len)[0, :chunk_len, :]
        chunks.append(chunk)
        current = np.concatenate([current, chunk], axis=0)
        remaining -= chunk_len
    return np.concatenate(chunks, axis=0)


def main():
    seed_everything(123)
    device = torch.device('cpu')
    batch = torch.randn(2, 256, 219)
    rows = []

    with tempfile.TemporaryDirectory(prefix='dimts_baseline_smoke_') as tmpdir:
        tmpdir = Path(tmpdir)
        for model_name in SUPPORTED_BASELINES:
            adapter = build_adapter(model_name, tiny_config(model_name), device)
            for _ in range(2):
                adapter.train_step(batch)
            context = batch[:, :128, :].numpy()
            pred = adapter.predict_future(context, pred_len=128)
            assert pred.shape == (2, 128, 219), f'{model_name} produced {pred.shape}'

            history = np.random.randn(256, 219).astype(np.float32)
            true_future = np.random.randn(512, 219).astype(np.float32)
            for extend_len in [128, 256, 384, 512]:
                generated = autoregressive_extend(adapter, history, extend_len, 128)
                assert generated.shape == (extend_len, 219), f'{model_name} extend {extend_len} failed'
                metrics = compute_future_metrics_z(generated, true_future[:extend_len])
                rows.append({
                    'model': model_name,
                    'subject_id': 'smoke',
                    'extend_len': extend_len,
                    **metrics,
                })

        summary = []
        for model_name in SUPPORTED_BASELINES:
            model_rows = [row for row in rows if row['model'] == model_name]
            summary.extend(summarize_by_length(model_rows, model_name=model_name).to_dict('records'))
        summary_df = pd.DataFrame(summary)
        output_path = tmpdir / 'metrics_by_model_length.csv'
        summary_df.to_csv(output_path, index=False)
        assert summary_df['model'].nunique() == len(SUPPORTED_BASELINES)
        assert set(summary_df['extend_len'].astype(int)) == {128, 256, 384, 512}
        for metric in ['corr_z_mean', 'mae_z_mean', 'rmse_z_mean', 'fc_upper_corr_mean']:
            assert metric in summary_df.columns
        print(f'smoke test passed: {output_path}')


if __name__ == '__main__':
    main()

