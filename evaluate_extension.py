import argparse
import json
import os

import numpy as np

from Utils.extension_utils import compute_extension_metrics, save_extension_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate extended BOLD sequences without future ground truth.')
    parser.add_argument('--history', type=str, required=True, help='Path to the original raw history sequence (.npy).')
    parser.add_argument('--extension', type=str, required=True, help='Path to the generated raw extension (.npy).')
    parser.add_argument('--compare_len', type=int, default=0, help='Length used for distribution comparison. <=0 uses min(history_len, extension_len).')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save metrics and plot.')
    parser.add_argument('--tag', type=str, default='extension_eval', help='Filename prefix for saved artifacts.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    history = np.load(args.history)
    extension = np.load(args.extension)
    compare_len = None if args.compare_len <= 0 else args.compare_len

    metrics = compute_extension_metrics(history_raw=history, extension_raw=extension, compare_len=compare_len)
    metrics_path = os.path.join(args.output_dir, f'{args.tag}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_path = os.path.join(args.output_dir, f'{args.tag}_summary.png')
    save_extension_summary(history_raw=history, extension_raw=extension, output_path=plot_path)

    print(json.dumps(metrics, indent=2))
    print(f'saved metrics to {metrics_path}')
    if os.path.exists(plot_path):
        print(f'saved summary plot to {plot_path}')


if __name__ == '__main__':
    main()
