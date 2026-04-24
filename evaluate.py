import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from Utils.extension_utils import compute_prediction_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate predict-mode outputs.')
    parser.add_argument('--pred', type=str, required=True, help='Path to prediction array (.npy).')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground-truth array (.npy).')
    parser.add_argument('--pred_len', type=int, required=True, help='Prediction horizon length.')
    parser.add_argument('--output_json', type=str, default=None, help='Optional path to save metrics JSON.')
    parser.add_argument('--plot_path', type=str, default=None, help='Optional path to save one prediction-vs-ground-truth plot.')
    parser.add_argument('--sample_id', type=int, default=0, help='Sample index for the optional plot.')
    parser.add_argument('--feature_id', type=int, default=0, help='Feature index for the optional plot.')
    return parser.parse_args()


def main():
    args = parse_args()
    pred = np.load(args.pred)
    gt = np.load(args.gt)

    metrics = compute_prediction_metrics(pred, gt, args.pred_len)
    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'saved metrics to {args.output_json}')

    if args.plot_path is not None:
        os.makedirs(os.path.dirname(args.plot_path) or '.', exist_ok=True)
        pred_seq = pred[args.sample_id, :, args.feature_id]
        gt_seq = gt[args.sample_id, :, args.feature_id]
        plt.plot(gt_seq, label='Ground Truth')
        plt.plot(pred_seq, label='Prediction')
        plt.axvline(gt.shape[1] - args.pred_len - 1, color='r', linestyle='--', label='Prediction Start')
        plt.legend()
        plt.savefig(args.plot_path)
        plt.close()
        print(f'saved plot to {args.plot_path}')


if __name__ == '__main__':
    main()
