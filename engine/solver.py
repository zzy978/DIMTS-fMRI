import os
import sys
import time
import json
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader=None, logger=None, val_dataloader=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = None if dataloader is None else cycle(dataloader['dataloader'])
        self.dataloader = None if dataloader is None else dataloader['dataloader']
        self.val_dataloader = None if val_dataloader is None else val_dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger
        self.best_val_loss = None
        self.best_val_step = None
        self.loaded_checkpoint_meta = None

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def evaluate(self):
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)
                loss = self.model(batch, target=batch)
                total_loss += loss.item()
                total_batches += 1
        self.model.train()

        if total_batches == 0:
            return None
        return total_loss / total_batches

    def _checkpoint_state(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_step': self.best_val_step,
        }
        return data

    def save(self, milestone, verbose=False, filename=None):
        checkpoint_path = self.results_folder / (filename if filename is not None else f'checkpoint-{milestone}.pt')
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(checkpoint_path)))
        data = self._checkpoint_state()
        torch.save(data, str(checkpoint_path))

    def save_best(self, verbose=False):
        self.save(milestone='best', verbose=verbose, filename='best.pt')

    def save_training_summary(self):
        summary = {
            'experiment_name': self.args.name,
            'seq_len': int(self.model.seq_length),
            'stride': int(self.config['dataloader']['train_dataset']['params'].get('stride', 1)),
            'normalization': self.config['dataloader']['train_dataset']['params'].get('normalization'),
            'max_subjects': self.config['dataloader']['train_dataset']['params'].get('max_subjects'),
            'l_loss': self.config['model']['params'].get('l_loss'),
            'mmd_alpha': self.config['model']['params'].get('mmd_alpha'),
            'train_steps': int(self.step),
            'best_val_loss': self.best_val_loss,
            'best_val_step': self.best_val_step,
            'results_folder': str(self.results_folder),
        }
        if self.dataloader is not None:
            summary['train_dataset_size'] = int(len(self.dataloader.dataset))
        if self.val_dataloader is not None:
            summary['val_dataset_size'] = int(len(self.val_dataloader.dataset))
        with open(os.path.join(self.args.save_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    def load(self, milestone, verbose=False):
        if milestone == 'best':
            checkpoint_path = self.results_folder / 'best.pt'
        else:
            checkpoint_path = self.results_folder / f'checkpoint-{milestone}.pt'
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(checkpoint_path)))
        device = self.device
        data = torch.load(str(checkpoint_path), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.best_val_loss = data.get('best_val_loss')
        self.best_val_step = data.get('best_val_step')
        self.milestone = milestone if isinstance(milestone, int) else self.milestone
        self.loaded_checkpoint_meta = {
            'checkpoint_ref': milestone,
            'step': data.get('step'),
            'best_val_loss': data.get('best_val_loss'),
            'best_val_step': data.get('best_val_step'),
            'path': str(checkpoint_path),
        }
        return data
    
    def save_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current classifer to {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        data = {
            'step': self.step_classifier,
            'classifier': self.classifier.state_dict()
        }
        torch.save(data, str(self.results_folder / f'ckpt_classfier-{milestone}.pt'))

    def load_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'ckpt_classfier-{milestone}.pt'), map_location=device)
        self.classifier.load_state_dict(data['classifier'])
        self.step_classifier = data['step']
        self.milestone_classifier = milestone

    def train(self):
        if self.dl is None:
            raise ValueError('Training requested without a training dataloader.')
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                # self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)
                        val_loss = self.evaluate()
                        if val_loss is not None:
                            if self.best_val_loss is None or val_loss < self.best_val_loss:
                                self.best_val_loss = float(val_loss)
                                self.best_val_step = int(self.step)
                                self.save_best(verbose=True)
                            self.logger.add_scalar(tag='val/loss', scalar_value=val_loss, global_step=self.step)
                            self.logger.log_info(
                                f'{self.args.name}: step {self.step}/{self.train_num_steps} | '
                                f'train_loss={total_loss:.6f} | val_loss={val_loss:.6f}'
                            )
                        else:
                            self.logger.log_info(
                                f'{self.args.name}: step {self.step}/{self.train_num_steps} | '
                                f'train_loss={total_loss:.6f}'
                            )

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))
        self.save_training_summary()

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def restore_sequence(self, target_window, partial_mask, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if target_window.ndim != 2 or partial_mask.ndim != 2:
            raise ValueError('target_window and partial_mask must both be 2D arrays shaped [seq_len, feature_dim].')
        if target_window.shape != partial_mask.shape:
            raise ValueError('target_window and partial_mask must have the same shape.')

        model_kwargs = {
            'coef': coef,
            'learning_rate': stepsize,
        }
        x = torch.from_numpy(target_window).float().unsqueeze(0).to(self.device)
        t_m = torch.from_numpy(partial_mask.astype(bool)).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                )
            else:
                sample = self.ema.ema_model.fast_sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                    sampling_timesteps=sampling_steps,
                )
        return sample[0].detach().cpu().numpy()

    def restore_window_batch(self, target_windows, partial_masks, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if target_windows.ndim != 3 or partial_masks.ndim != 3:
            raise ValueError('target_windows and partial_masks must be shaped [batch, seq_len, feature_dim].')
        if target_windows.shape != partial_masks.shape:
            raise ValueError('target_windows and partial_masks must have the same shape.')

        model_kwargs = {
            'coef': coef,
            'learning_rate': stepsize,
        }
        x = torch.from_numpy(target_windows).float().to(self.device)
        t_m = torch.from_numpy(partial_masks.astype(bool)).to(self.device)

        with torch.no_grad():
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                )
            else:
                sample = self.ema.ema_model.fast_sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                    sampling_timesteps=sampling_steps,
                )
        return sample.detach().cpu().numpy()

    def extend_sequence(
        self,
        normalized_sequence,
        total_extend_len,
        pred_len,
        coef=1e-1,
        stepsize=1e-1,
        sampling_steps=50,
        autoregressive=False,
    ):
        if normalized_sequence.ndim != 2:
            raise ValueError('normalized_sequence must be shaped [time, feature_dim].')
        seq_length = self.model.seq_length
        feature_dim = normalized_sequence.shape[1]

        if pred_len <= 0:
            raise ValueError('pred_len must be positive for sequence extension.')
        if pred_len >= seq_length:
            raise ValueError('pred_len must be smaller than the model seq_length for extension.')
        if total_extend_len <= 0:
            raise ValueError('total_extend_len must be positive.')
        if not autoregressive and total_extend_len > pred_len:
            raise ValueError('Without autoregressive mode, total_extend_len cannot exceed pred_len.')

        current_sequence = normalized_sequence.astype(np.float32).copy()
        generated_chunks = []
        window_records = []
        remaining = total_extend_len

        while remaining > 0:
            current_chunk_len = min(pred_len, remaining) if autoregressive else total_extend_len
            context_len = seq_length - current_chunk_len
            if current_sequence.shape[0] < context_len:
                raise ValueError(
                    f'Input sequence length {current_sequence.shape[0]} is shorter than required context length {context_len}.'
                )

            target_window = np.zeros((seq_length, feature_dim), dtype=np.float32)
            partial_mask = np.zeros((seq_length, feature_dim), dtype=bool)
            context_window = current_sequence[-context_len:, :]
            target_window[:context_len, :] = context_window
            partial_mask[:context_len, :] = True

            restored_window = self.restore_sequence(
                target_window=target_window,
                partial_mask=partial_mask,
                coef=coef,
                stepsize=stepsize,
                sampling_steps=sampling_steps,
            )
            generated_chunk = restored_window[-current_chunk_len:, :]
            current_sequence = np.concatenate([current_sequence, generated_chunk], axis=0)
            generated_chunks.append(generated_chunk)
            window_records.append(restored_window)
            remaining -= current_chunk_len

            if not autoregressive:
                break

        return {
            'extended_sequence': current_sequence,
            'generated_extension': np.concatenate(generated_chunks, axis=0),
            'restored_windows': np.stack(window_records, axis=0),
        }

    def forward_sample(self, x_start):
       b, c, h = x_start.shape
       noise = torch.randn_like(x_start, device=self.device)
       t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
       x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
       return x_t, t

    def train_classfier(self, classifier):
        device = self.device
        step = 0
        self.milestone_classifier = 0
        self.step_classifier = 0
        dataloader = self.dataloader
        dataloader.dataset.shift_period('test')
        dataloader = cycle(dataloader)

        self.classifier = classifier
        self.opt_classifier = Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=5.0e-4)
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training classifier...'.format(self.args.name), check_primary=False)
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)
                    x_t, t = self.forward_sample(x)
                    logits = classifier(x_t, t)
                    loss = F.cross_entropy(logits, y)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                self.opt_classifier.step()
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

        # return classifier
