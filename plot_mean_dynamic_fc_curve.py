import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Plot cohort-mean dynamic FC similarity curves.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to dynamic_fc_similarity.csv.')
    parser.add_argument('--extend_len', type=int, default=128)
    parser.add_argument('--future_window_idx', type=int, default=0)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--show_band', action='store_true', help='Draw mean +/- standard error bands.')
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    subset = df[
        (df['extend_len'] == args.extend_len)
        & (df['future_window_idx'] == args.future_window_idx)
    ].copy()
    if subset.empty:
        raise ValueError(
            f'No rows found for extend_len={args.extend_len}, '
            f'future_window_idx={args.future_window_idx}.'
        )

    # 对同一历史 FC 窗口位置聚合所有 subject，得到群体平均动态 FC 相似性轨迹。
    grouped = subset.groupby(['history_window_idx', 'history_center'])
    summary = grouped.agg(
        gen_mean=('gen_hist_fc_corr', 'mean'),
        gen_sem=('gen_hist_fc_corr', 'sem'),
        true_mean=('true_hist_fc_corr', 'mean'),
        true_sem=('true_hist_fc_corr', 'sem'),
        num_subjects=('subject_id', 'nunique'),
    ).reset_index().sort_values('history_window_idx')

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('matplotlib is required to draw the mean dynamic FC curve.') from exc

    fig, ax = plt.subplots(figsize=(12, 4))
    x = summary['history_center']
    ax.plot(x, summary['gen_mean'], label='Generated vs history FC', linewidth=1.8)
    ax.plot(x, summary['true_mean'], label='True future vs history FC', linewidth=1.8)

    if args.show_band:
        ax.fill_between(
            x,
            summary['gen_mean'] - summary['gen_sem'],
            summary['gen_mean'] + summary['gen_sem'],
            alpha=0.18,
        )
        ax.fill_between(
            x,
            summary['true_mean'] - summary['true_sem'],
            summary['true_mean'] + summary['true_sem'],
            alpha=0.18,
        )

    num_subjects = int(summary['num_subjects'].min())
    ax.set_title(
        f'Mean Dynamic FC Similarity | extend_len={args.extend_len}, '
        f'future_window={args.future_window_idx}, n={num_subjects}'
    )
    ax.set_xlabel('History window center')
    ax.set_ylabel('FC correlation')
    ax.legend(loc='best')
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(args.csv_path),
            'plots',
            f'mean_extend{args.extend_len}_future{args.future_window_idx}_dynamic_fc_lines.png',
        )
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    summary_path = os.path.splitext(output_path)[0] + '.csv'
    summary.to_csv(summary_path, index=False)
    print(f'saved plot to {output_path}')
    print(f'saved summary to {summary_path}')


if __name__ == '__main__':
    main()
