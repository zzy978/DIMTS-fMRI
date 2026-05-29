import argparse
from pathlib import Path

import pandas as pd

from Baselines.metrics import MAIN_METRICS, summarize_by_length


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize fMRI baseline comparison metrics.')
    parser.add_argument('--results_root', type=str, default='OUTPUT/baseline_test_set')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--include_dimts_csv', type=str, default=None)
    return parser.parse_args()


def load_tail_csv(path, model_name):
    df = pd.read_csv(path)
    if 'model' not in df:
        df['model'] = model_name
    missing = [metric for metric in MAIN_METRICS if metric not in df.columns]
    if missing:
        raise ValueError(f'{path} is missing metrics: {missing}')
    return df


def main():
    args = parse_args()
    root = Path(args.results_root)
    frames = []
    for tail_path in sorted(root.glob('*/tail_holdout_summary.csv')):
        frames.append(load_tail_csv(tail_path, tail_path.parent.name))

    if args.include_dimts_csv:
        dimts_path = Path(args.include_dimts_csv)
        if dimts_path.exists():
            frames.append(load_tail_csv(dimts_path, 'dim_ts'))

    if len(frames) == 0:
        raise ValueError(f'No tail_holdout_summary.csv files found under {root}.')

    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for model_name, group in combined.groupby('model'):
        rows.extend(summarize_by_length(group.to_dict('records'), model_name=model_name).to_dict('records'))
    summary = pd.DataFrame(rows).sort_values(['model', 'extend_len'])

    output_csv = Path(args.output_csv) if args.output_csv else root / 'metrics_by_model_length.csv'
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f'saved {output_csv}')


if __name__ == '__main__':
    main()
