import os
import copy
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond, build_dataloader_val
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config
from Utils.extension_utils import (
    compute_extension_metrics,
    fit_sequence_scaler,
    inverse_normalize_sequence,
    load_raw_sequence,
    normalize_sequence,
    save_extension_summary,
    save_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None) # 给这次实验命名

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')  # 加上这个参数后会启用 TensorBoard 日志

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')  # 用于要求 CuDNN 走确定性模式，目的是增强复现性，但通常会变慢
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.') # 加上这个参数后会进入训练模式，否则默认是测试模式（即采样模式）
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')  # 0表示无条件采样，1表示有条件采样（即infill或predict）
    parser.add_argument('--mode', type=str, default='infill',
                        help='infill or predict') # 这个参数只有在 sample=1 时才有意义，表示是进行插值（infill）还是预测（predict）。插值是指在已知数据的基础上填补缺失部分，而预测则是根据已知数据预测未来的值。
    parser.add_argument('--milestone', type=int, default=10) # 要加载的 checkpoint 编号

    parser.add_argument('--missing_ratio', type=float, default=0.1, help='Ratio of Missing Values.') # 只在 mode=infill 时有意义。表示要人为遮掉多少比例的数据点，然后让模型补全
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.') # 只在 mode=predict 时有意义。表示要预测未来多少个时间步。
    parser.add_argument('--norm_method', type=str, default='minmax',
                        choices=['minmax', 'zscore'],
                        help='Normalization method for dataset preprocessing.')
    parser.add_argument('--data_input_mode', type=str, default='single_csv',
                        choices=['single_csv', 'subject_split'],
                        help='Dataset input mode: single CSV file or subject-level directory split.')
    parser.add_argument('--subject_train_ratio', type=float, default=0.8,
                        help='Train subject ratio for subject_split mode.')
    parser.add_argument('--subject_val_ratio', type=float, default=0.1,
                        help='Validation subject ratio for subject_split mode.')
    parser.add_argument('--subject_test_ratio', type=float, default=0.1,
                        help='Test subject ratio for subject_split mode.')
    parser.add_argument('--subject_shuffle', action='store_true', default=False,
                        help='Shuffle subject list before splitting in subject_split mode.')
    parser.add_argument('--max_subjects', type=int, default=0,
                        help='Maximum number of subjects to keep in subject_split mode. <=0 means use all.')
    parser.add_argument('--drop_nan_subjects', action='store_true', default=False,
                        help='Exclude subject CSV files that contain NaN values in subject_split mode.')
    parser.add_argument('--extend_sequence', action='store_true', default=False,
                        help='Extend a single raw sequence instead of using dataset-based predict/infill.')
    parser.add_argument('--extend_input', type=str, default=None,
                        help='Path to a raw sequence file (.csv or .npy) for extension.')
    parser.add_argument('--extend_len', type=int, default=0,
                        help='Total number of future timepoints to generate for sequence extension.')
    parser.add_argument('--autoregressive', action='store_true', default=False,
                        help='Use iterative autoregressive continuation for sequence extension.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)   # 这是一个“附加覆盖参数”列表，用来临时修改 YAML 配置里的字段，而不用直接改配置文件。

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    config['dataloader']['train_dataset']['params']['normalization'] = args.norm_method
    config['dataloader']['test_dataset']['params']['normalization'] = args.norm_method
    if args.data_input_mode == 'subject_split':
        split_target = 'Utils.Data_utils.real_datasets.SubjectSplitCSVDataset'
        subject_root = config['dataloader']['train_dataset']['params']['data_root']

        if 'val_dataset' not in config['dataloader']:
            config['dataloader']['val_dataset'] = copy.deepcopy(config['dataloader']['test_dataset'])

        split_specs = {
            'train_dataset': 'train',
            'val_dataset': 'val',
            'test_dataset': 'test',
        }
        for dataset_key, period in split_specs.items():
            config['dataloader'][dataset_key]['target'] = split_target
            params = config['dataloader'][dataset_key]['params']
            params['data_root'] = subject_root
            params['period'] = period
            params['normalization'] = args.norm_method
            params['subject_train_ratio'] = args.subject_train_ratio
            params['subject_val_ratio'] = args.subject_val_ratio
            params['subject_test_ratio'] = args.subject_test_ratio
            params['subject_shuffle'] = args.subject_shuffle
            params['max_subjects'] = None if args.max_subjects <= 0 else args.max_subjects
            params['drop_nan_subjects'] = args.drop_nan_subjects

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    trainer = Trainer(
        config=config,
        args=args,
        model=model,
        dataloader=None,
        logger=logger,
        val_dataloader=None,
    )

    if args.extend_sequence:
        if args.extend_input is None:
            raise ValueError('Please provide --extend_input when using --extend_sequence.')
        if args.extend_len <= 0:
            raise ValueError('Please provide a positive --extend_len when using --extend_sequence.')
        if args.pred_len <= 0:
            raise ValueError('Please provide a positive --pred_len for sequence extension.')

        trainer.load(args.milestone)
        dataset_name = config['dataloader']['train_dataset']['params'].get('name', '')
        neg_one_to_one = config['dataloader']['train_dataset']['params'].get('neg_one_to_one', True)
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']

        raw_sequence = load_raw_sequence(args.extend_input, dataset_name)
        scaler = fit_sequence_scaler(raw_sequence, normalization=args.norm_method)
        normalized_sequence, auto_norm = normalize_sequence(
            raw_sequence,
            scaler,
            normalization=args.norm_method,
            neg_one_to_one=neg_one_to_one,
        )

        extension_outputs = trainer.extend_sequence(
            normalized_sequence=normalized_sequence,
            total_extend_len=args.extend_len,
            pred_len=args.pred_len,
            coef=coef,
            stepsize=stepsize,
            sampling_steps=sampling_steps,
            autoregressive=args.autoregressive,
        )

        generated_extension_norm = extension_outputs['generated_extension']
        extended_sequence_norm = extension_outputs['extended_sequence']
        generated_extension_raw = inverse_normalize_sequence(generated_extension_norm, scaler, auto_norm=auto_norm)
        extended_sequence_raw = inverse_normalize_sequence(extended_sequence_norm, scaler, auto_norm=auto_norm)
        metrics = compute_extension_metrics(
            history_raw=raw_sequence,
            extension_raw=generated_extension_raw,
            compare_len=min(args.pred_len, generated_extension_raw.shape[0]),
        )

        np.save(os.path.join(args.save_dir, f'extension_input_raw_{args.name}.npy'), raw_sequence)
        np.save(os.path.join(args.save_dir, f'extension_input_norm_{args.name}.npy'), normalized_sequence)
        np.save(os.path.join(args.save_dir, f'extension_norm_{args.name}.npy'), generated_extension_norm)
        np.save(os.path.join(args.save_dir, f'extension_raw_{args.name}.npy'), generated_extension_raw)
        np.save(os.path.join(args.save_dir, f'extended_full_norm_{args.name}.npy'), extended_sequence_norm)
        np.save(os.path.join(args.save_dir, f'extended_full_raw_{args.name}.npy'), extended_sequence_raw)
        np.save(os.path.join(args.save_dir, f'extension_windows_norm_{args.name}.npy'), extension_outputs['restored_windows'])
        save_metrics(metrics, os.path.join(args.save_dir, f'extension_metrics_{args.name}.json'))
        save_extension_summary(
            history_raw=raw_sequence,
            extension_raw=generated_extension_raw,
            output_path=os.path.join(args.save_dir, f'extension_summary_{args.name}.png'),
        )
        logger.log_info(
            f"extension done | input_len={raw_sequence.shape[0]} | extend_len={generated_extension_raw.shape[0]} | "
            f"full_len={extended_sequence_raw.shape[0]} | seam_mae={metrics['seam_mae']:.6f} | "
            f"fc_upper_corr={metrics['fc_upper_corr']:.6f}"
        )
        return

    val_dataloader_info = None
    if 'val_dataset' in config['dataloader']:
        val_dataloader_info = build_dataloader_val(config, args)
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    if logger is not None and val_dataloader_info is not None:
        logger.log_info(
            f"dataset sizes | train: {len(dataloader_info['dataset'])}, "
            f"val: {len(val_dataloader_info['dataset'])}, "
            f"test: {len(test_dataloader_info['dataset']) if args.sample == 1 and args.mode in ['infill', 'predict'] else 'not built'}"
        )
    trainer = Trainer(
        config=config,
        args=args,
        model=model,
        dataloader=dataloader_info,
        logger=logger,
        val_dataloader=val_dataloader_info,
    )

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size'] 
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps'] ## 采样步数：例如训练时的t设置为500，采样步数设为250，那就是隔一个采样一次
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples) # 先把数据从[-1,1]范围还原到[0,1]范围，因为在训练时我们把数据归一化到了[-1,1]，所以采样出来的也是这个范围的。要得到原始数据的数值，我们需要先把它们还原到[0,1]范围，然后再根据原始数据的分布进行反归一化，反归一化的代码是下一行，可以得到原始数据。
            samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape) # 反变换到原始数据
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}.npy'), samples)
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
