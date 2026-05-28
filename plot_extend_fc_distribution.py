import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Plot FC absolute-difference and Pearson-correlation distributions.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to tail_holdout_summary.csv with fc_abs_diff and fc_upper_corr columns.')
    parser.add_argument('--extend_len', type=int, default=128)
    parser.add_argument('--all_lengths', action='store_true', help='Plot every extend_len found in the CSV.')
    parser.add_argument('--output_path', type=str, default=None)
    return parser.parse_args()


def add_summary_lines(ax, x_center, values, color):
    stats = {
        'min': float(values.min()),
        'mean': float(values.mean()),
        'max': float(values.max()),
    }
    # 三条水平线分别标出该分布的最小值、均值和最大值。
    for value in stats.values():
        ax.hlines(value, x_center - 0.16, x_center + 0.16, color=color, linewidth=1.0)
    return stats


def style_violin(violin):
    for body in violin['bodies']:
        body.set_facecolor('#9ecae1')
        body.set_edgecolor('#9ecae1')
        body.set_alpha(0.65)


def plot_single_length(ax_left, ax_right, subset, extend_len):
    mad_values = subset['fc_abs_diff'].dropna()
    corr_values = subset['fc_upper_corr'].dropna()

    left_violin = ax_left.violinplot([mad_values], positions=[0], widths=0.55, showextrema=False)
    right_violin = ax_right.violinplot([corr_values], positions=[1], widths=0.55, showextrema=False)
    style_violin(left_violin)
    style_violin(right_violin)

    left_stats = add_summary_lines(ax_left, 0, mad_values, '#2c7fb8')
    right_stats = add_summary_lines(ax_right, 1, corr_values, '#2c7fb8')

    ax_left.set_xlim(-0.6, 1.6)
    ax_left.set_xticks([0, 1])
    ax_left.set_xticklabels(['Mean Absolute Difference', "Pearson's Correlation"])
    ax_left.set_ylabel('Mean Absolute Difference')
    ax_right.set_ylabel("Pearson's Correlation")
    ax_left.set_title(f'extend_len={extend_len}')

    # # 用文本记录每个分布的 min/mean/max，避免只看水平线时无法读出具体数值。
    # text = (
    #     f"MAD min/mean/max: {left_stats['min']:.4f} / {left_stats['mean']:.4f} / {left_stats['max']:.4f}\n"
    #     f"Corr min/mean/max: {right_stats['min']:.4f} / {right_stats['mean']:.4f} / {right_stats['max']:.4f}"
    # )
    # ax_left.text(0.02, 0.98, text, transform=ax_left.transAxes, va='top', fontsize=8)


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    required_cols = ['fc_abs_diff', 'fc_upper_corr']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}. Re-run evaluation with the updated script first.')

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('matplotlib is required to draw the FC distribution plot.') from exc

    if args.all_lengths:
        extend_lengths = sorted(df['extend_len'].dropna().unique().astype(int))
    else:
        extend_lengths = [args.extend_len]

    fig, left_axes = plt.subplots(len(extend_lengths), 1, figsize=(7, 4 * len(extend_lengths)))
    if len(extend_lengths) == 1:
        left_axes = [left_axes]
    for ax_left, extend_len in zip(left_axes, extend_lengths):
        subset = df[df['extend_len'] == extend_len].copy()
        if subset.empty:
            raise ValueError(f'No rows found for extend_len={extend_len}.')
        ax_right = ax_left.twinx()
        plot_single_length(ax_left, ax_right, subset, extend_len)

    title = 'Generated FC vs True Future FC Distribution'
    if args.all_lengths:
        title += ' (all lengths)'
    else:
        title += f' (extend_len={args.extend_len})'
    fig.suptitle(title, y=1.0)

    fig.tight_layout()
    output_path = args.output_path
    if output_path is None:
        filename = 'all_lengths_fc_distribution.png' if args.all_lengths else f'extend{args.extend_len}_fc_distribution.png'
        output_path = os.path.join(os.path.dirname(args.csv_path), filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'saved plot to {output_path}')


if __name__ == '__main__':
    main()
