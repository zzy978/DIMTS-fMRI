from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from Utils.Data_utils.real_datasets import SubjectSplitCSVDataset
from Utils.extension_utils import fit_sequence_scaler, load_raw_sequence, normalize_sequence


def build_training_loader(config, output_dir):
    data_cfg = config['data']
    baseline_cfg = config['baseline']
    training_cfg = config['training']
    dataset = SubjectSplitCSVDataset(
        name='fmri',
        data_root=data_cfg['train_data_root'],
        window=int(baseline_cfg['seq_length']),
        stride=int(data_cfg.get('stride', 64)),
        proportion=0.9,
        save2npy=False,
        neg_one_to_one=False,
        normalization='zscore',
        seed=int(data_cfg.get('seed', 123)),
        period='train',
        output_dir=str(output_dir),
        subject_train_ratio=float(data_cfg.get('subject_train_ratio', 0.8)),
        subject_val_ratio=float(data_cfg.get('subject_val_ratio', 0.1)),
        subject_test_ratio=float(data_cfg.get('subject_test_ratio', 0.1)),
        subject_shuffle=bool(data_cfg.get('subject_shuffle', True)),
        max_subjects=data_cfg.get('max_subjects'),
        drop_nan_subjects=bool(data_cfg.get('drop_nan_subjects', True)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(training_cfg.get('batch_size', 16)),
        shuffle=True,
        num_workers=int(training_cfg.get('num_workers', 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, dataset


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def list_subject_files(source_dir, max_subjects=0):
    files = sorted(Path(source_dir).glob('*.csv'))
    if len(files) == 0:
        raise ValueError(f'No CSV files found in {source_dir}.')
    if max_subjects and int(max_subjects) > 0:
        files = files[:int(max_subjects)]
    return files


def list_subject_split_files(config, split='test', max_subjects=0):
    data_cfg = config['data']
    subject_files = SubjectSplitCSVDataset._list_subject_files(
        data_cfg['train_data_root'],
        data_cfg.get('file_pattern', '.csv'),
    )
    subject_files, _ = SubjectSplitCSVDataset._filter_nan_subjects(
        subject_files,
        bool(data_cfg.get('drop_nan_subjects', True)),
    )
    subject_files = SubjectSplitCSVDataset._limit_subjects(
        subject_files,
        seed=int(data_cfg.get('seed', 123)),
        max_subjects=data_cfg.get('max_subjects'),
        shuffle=bool(data_cfg.get('subject_shuffle', True)),
    )
    split_subjects = SubjectSplitCSVDataset._split_subjects(
        subject_files,
        seed=int(data_cfg.get('seed', 123)),
        train_ratio=float(data_cfg.get('subject_train_ratio', 0.8)),
        val_ratio=float(data_cfg.get('subject_val_ratio', 0.1)),
        shuffle=bool(data_cfg.get('subject_shuffle', True)),
    )
    files = [Path(path) for path in split_subjects[split]]
    if max_subjects and int(max_subjects) > 0:
        files = files[:int(max_subjects)]
    return files


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def load_normalized_subject(filepath):
    raw_sequence = load_raw_sequence(str(filepath), 'fmri')
    scaler = fit_sequence_scaler(raw_sequence, normalization='zscore')
    # 与训练集 subject-level z-score 保持一致：每个被试用自身整条序列拟合 scaler。
    normalized_sequence, auto_norm = normalize_sequence(
        raw_sequence,
        scaler,
        normalization='zscore',
        neg_one_to_one=False,
    )
    return SimpleNamespace(
        subject_id=Path(filepath).stem,
        raw_sequence=raw_sequence,
        normalized_sequence=normalized_sequence.astype(np.float32),
        scaler=scaler,
        auto_norm=auto_norm,
    )
