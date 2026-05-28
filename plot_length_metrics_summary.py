import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Plot extension metrics by generated length.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metrics_by_length.csv.')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['rmse_z', 'fc_upper_corr', 'psd_l1'],
        help='Metric base names. The script reads <metric>_mean columns.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    if 'extend_len' not in df.columns:
        raise ValueError('metrics_by_length.csv must contain extend_len.')

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('matplotlib is required to draw metric curves.') from exc

    fig, axes = plt.subplots(1, len(args.metrics), figsize=(6 * len(args.metrics), 4))
    if len(args.metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, args.metrics):
        mean_col = f'{metric}_mean'
        if mean_col not in df.columns:
            raise ValueError(f'Missing required column: {mean_col}')
        ax.plot(df['extend_len'], df[mean_col], marker='o', linewidth=1.8)
        ax.set_xlabel('Extend length')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.2)

    # 这里默认使用 rmse_z，避免把 raw BOLD 空间的 RMSE 和 z-score 空间指标混在一起展示。
    fig.tight_layout()
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.csv_path), 'plots', 'metrics_by_length_zscore.png')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'saved plot to {output_path}')


if __name__ == '__main__':
    main()
