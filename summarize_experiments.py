import argparse
import csv
import glob
import json
import os


def flatten_dict(d, prefix=''):
    flat = {}
    for key, value in d.items():
        full_key = f'{prefix}{key}' if prefix == '' else f'{prefix}.{key}'
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def collect_summaries(output_root):
    rows = {}

    for path in glob.glob(os.path.join(output_root, '*', 'training_summary.json')):
        with open(path, 'r') as f:
            data = json.load(f)
        exp_name = data.get('experiment_name', os.path.basename(os.path.dirname(path)))
        rows.setdefault(exp_name, {}).update(flatten_dict(data))

    for path in glob.glob(os.path.join(output_root, '*', 'predict_summary.json')):
        with open(path, 'r') as f:
            data = json.load(f)
        exp_name = data.get('experiment_name', os.path.basename(os.path.dirname(path)))
        rows.setdefault(exp_name, {}).update(flatten_dict(data))

    return rows


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment JSON outputs into one CSV.')
    parser.add_argument('--output_root', type=str, default='OUTPUT', help='Directory containing experiment folders.')
    parser.add_argument('--csv_path', type=str, default='OUTPUT/experiment_summary.csv', help='Output CSV path.')
    args = parser.parse_args()

    rows = collect_summaries(args.output_root)
    if len(rows) == 0:
        raise ValueError(f'No training_summary.json or predict_summary.json files found under {args.output_root}.')

    fieldnames = sorted({key for row in rows.values() for key in row.keys()})
    os.makedirs(os.path.dirname(args.csv_path) or '.', exist_ok=True)
    with open(args.csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp_name in sorted(rows.keys()):
            writer.writerow(rows[exp_name])

    print(f'saved summary to {args.csv_path}')


if __name__ == '__main__':
    main()
