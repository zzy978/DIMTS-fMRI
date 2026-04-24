import os
import json
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        stride=1,
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        normalization='minmax',
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        assert normalization in ['minmax', 'zscore'], 'normalization must be minmax or zscore.'
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.normalization = normalization
        self.rawdata, self.scaler = self.read_data(data_root, self.name, normalization)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.stride = stride
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = self._num_windows(self.len, self.window, self.stride)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one and self.normalization == 'minmax'

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i, start in enumerate(self._window_starts(self.len, self.window, self.stride)):
            x[i, :, :] = data[start:start + self.window, :]

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def _window_starts(length, window, stride):
        if stride <= 0:
            raise ValueError('stride must be a positive integer.')
        if length < window:
            return []
        return list(range(0, length - window + 1, stride))

    @classmethod
    def _num_windows(cls, length, window, stride):
        return len(cls._window_starts(length, window, stride))

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name='', normalization='minmax'):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler() if normalization == 'minmax' else StandardScaler()
        scaler = scaler.fit(data)
        return data, scaler
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    

class SubjectSplitCSVDataset(Dataset):
    def __init__(
        self,
        name,
        data_root,
        window=64,
        stride=1,
        proportion=0.8,
        save2npy=True,
        neg_one_to_one=True,
        normalization='minmax',
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate',
        distribution='geometric',
        mean_mask_length=3,
        subject_train_ratio=0.8,
        subject_val_ratio=0.1,
        subject_test_ratio=0.1,
        subject_shuffle=True,
        file_pattern='.csv',
        max_subjects=None,
        drop_nan_subjects=False
    ):
        super(SubjectSplitCSVDataset, self).__init__()
        assert period in ['train', 'val', 'test'], 'period must be train, val or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        assert normalization in ['minmax', 'zscore'], 'normalization must be minmax or zscore.'
        ratios = np.array([subject_train_ratio, subject_val_ratio, subject_test_ratio], dtype=np.float64)
        assert np.isclose(ratios.sum(), 1.0), 'subject split ratios must sum to 1.0.'

        self.name = name
        self.pred_len = predict_length
        self.missing_ratio = missing_ratio
        self.style = style
        self.distribution = distribution
        self.mean_mask_length = mean_mask_length
        self.normalization = normalization
        self.window = window
        self.stride = stride
        self.period = period
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one and self.normalization == 'minmax'
        self.max_subjects = max_subjects
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        subject_files = self._list_subject_files(data_root, file_pattern)
        subject_files, excluded_nan_files = self._filter_nan_subjects(subject_files, drop_nan_subjects)
        subject_files = self._limit_subjects(
            subject_files,
            seed=seed,
            max_subjects=max_subjects,
            shuffle=subject_shuffle,
        )
        self.selected_subject_pool = subject_files
        split_subjects = self._split_subjects(
            subject_files,
            seed=seed,
            train_ratio=subject_train_ratio,
            val_ratio=subject_val_ratio,
            shuffle=subject_shuffle,
        )
        self.subject_files = split_subjects[period]
        self._save_split_manifest(split_subjects, excluded_nan_files)

        self.raw_windows, self.samples, self.sample_subjects = self._build_windows(self.subject_files)
        self.sample_num = self.samples.shape[0]
        self.var_num = self.samples.shape[-1] if self.sample_num > 0 else 0

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_{self.period}.npy"), self.raw_windows)
            if self.auto_norm:
                np.save(
                    os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_{self.period}.npy"),
                    unnormalize_to_zero_to_one(self.samples)
                )
            else:
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_{self.period}.npy"), self.samples)

        if period in ['val', 'test'] and missing_ratio is not None:
            self.masking = self.mask_data(seed)
        elif period in ['val', 'test'] and predict_length is not None:
            masks = np.ones(self.samples.shape)
            masks[:, -predict_length:, :] = 0
            self.masking = masks.astype(bool)

    @staticmethod
    def _list_subject_files(data_root, file_pattern):
        if not os.path.isdir(data_root):
            raise ValueError(f"Expected subject directory for subject split mode, got: {data_root}")
        files = [
            os.path.join(data_root, f)
            for f in sorted(os.listdir(data_root))
            if f.endswith(file_pattern)
        ]
        if len(files) == 0:
            raise ValueError(f"No subject files matching '*{file_pattern}' found in {data_root}")
        return files

    @classmethod
    def _filter_nan_subjects(cls, subject_files, drop_nan_subjects):
        if not drop_nan_subjects:
            return subject_files, []

        valid_files = []
        excluded_files = []
        for filepath in subject_files:
            data = cls._read_subject_csv(filepath)
            if np.isnan(data).any():
                excluded_files.append(filepath)
            else:
                valid_files.append(filepath)

        if len(valid_files) == 0:
            raise ValueError('All subject files were excluded because they contain NaN values.')
        return valid_files, excluded_files

    @staticmethod
    def _limit_subjects(subject_files, seed, max_subjects=None, shuffle=True):
        if max_subjects is None:
            return subject_files
        max_subjects = int(max_subjects)
        if max_subjects <= 0 or len(subject_files) <= max_subjects:
            return subject_files

        indices = np.arange(len(subject_files))
        st0 = np.random.get_state()
        np.random.seed(seed)
        if shuffle:
            indices = np.random.permutation(indices)
        np.random.set_state(st0)
        selected = np.sort(indices[:max_subjects])
        return [subject_files[idx] for idx in selected]

    @staticmethod
    def _split_subjects(subject_files, seed, train_ratio, val_ratio, shuffle=True):
        num_subjects = len(subject_files)
        indices = np.arange(num_subjects)
        st0 = np.random.get_state()
        np.random.seed(seed)
        if shuffle:
            indices = np.random.permutation(indices)
        np.random.set_state(st0)

        train_end = int(np.floor(num_subjects * train_ratio))
        val_end = train_end + int(np.floor(num_subjects * val_ratio))
        if num_subjects >= 3:
            train_end = max(train_end, 1)
            val_end = max(val_end, train_end + 1)
            val_end = min(val_end, num_subjects - 1)

        split_indices = {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:],
        }
        return {
            split: [subject_files[idx] for idx in split_ids]
            for split, split_ids in split_indices.items()
        }

    def _save_split_manifest(self, split_subjects, excluded_nan_files=None):
        if not self.save2npy:
            return
        manifest = {
            split: [os.path.basename(path) for path in paths]
            for split, paths in split_subjects.items()
        }
        manifest['num_subjects_per_split'] = {
            split: len(paths) for split, paths in split_subjects.items()
        }
        manifest['selected_subjects'] = [os.path.basename(path) for path in self.selected_subject_pool]
        manifest['excluded_nan_subjects'] = [] if excluded_nan_files is None else [
            os.path.basename(path) for path in excluded_nan_files
        ]
        manifest['window'] = self.window
        manifest['stride'] = self.stride
        manifest['max_subjects'] = self.max_subjects
        manifest['period'] = self.period
        manifest_path = os.path.join(self.dir, f"{self.name}_subject_split_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        for split, paths in split_subjects.items():
            filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
            info_path = os.path.join(self.dir, f"{split}_info.csv")
            pd.DataFrame(filenames).to_csv(info_path, index=False, header=False)

    @staticmethod
    def _read_subject_csv(filepath, name=''):
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        return df.values

    def _build_windows(self, subject_files):
        raw_windows = []
        norm_windows = []
        sample_subjects = []

        for filepath in subject_files:
            rawdata = self._read_subject_csv(filepath, self.name)
            scaler = MinMaxScaler() if self.normalization == 'minmax' else StandardScaler()
            scaler = scaler.fit(rawdata)
            data = scaler.transform(rawdata)
            if self.auto_norm:
                data = normalize_to_neg_one_to_one(data)

            starts = CustomDataset._window_starts(rawdata.shape[0], self.window, self.stride)
            for start in starts:
                end = start + self.window
                raw_windows.append(rawdata[start:end, :])
                norm_windows.append(data[start:end, :])
                sample_subjects.append(os.path.basename(filepath))

        if len(norm_windows) == 0:
            raise ValueError(
                f"No windows created for split '{self.period}'. "
                f"Check subject counts and window length={self.window}."
            )

        return np.asarray(raw_windows), np.asarray(norm_windows), sample_subjects

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style, self.distribution)
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}_{self.period}.npy"), masks)

        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if hasattr(self, 'masking'):
            x = self.samples[ind, :, :]
            m = self.masking[ind, :, :]
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num


class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name='', normalization='minmax'):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler() if normalization == 'minmax' else StandardScaler()
        scaler = scaler.fit(data)
        return data, scaler
