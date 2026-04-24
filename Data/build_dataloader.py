import torch
from Utils.io_utils import instantiate_from_config


def _build_named_dataloader(dataset_config, batch_size, shuffle, args=None, drop_last=None):
    dataset_config['params']['output_dir'] = args.save_dir
    dataset = instantiate_from_config(dataset_config)
    if drop_last is None:
        drop_last = shuffle
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=drop_last)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }
    return dataload_info


def build_dataloader(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    return _build_named_dataloader(config['dataloader']['train_dataset'], batch_size, jud, args=args, drop_last=jud)


def build_dataloader_val(config, args=None):
    if 'val_dataset' not in config['dataloader']:
        return None
    batch_size = config['dataloader'].get('sample_size', config['dataloader']['batch_size'])
    return _build_named_dataloader(config['dataloader']['val_dataset'], batch_size, False, args=args, drop_last=False)

def build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    if args.mode == 'infill':
        config['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio
    elif args.mode == 'predict':
        config['dataloader']['test_dataset']['params']['predict_length'] = args.pred_len
    return _build_named_dataloader(config['dataloader']['test_dataset'], batch_size, False, args=args, drop_last=False)


if __name__ == '__main__':
    pass
