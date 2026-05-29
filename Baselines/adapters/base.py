from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class TorchForecastAdapter:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        training = config.get('training', {})
        self.context_len = int(config['baseline']['context_len'])
        self.pred_len = int(config['baseline']['pred_len'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(training.get('lr', 1e-4)),
            weight_decay=float(training.get('weight_decay', 0.0)),
        )

    def split_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(self.device).float()
        context = batch[:, :self.context_len, :]
        future = batch[:, self.context_len:self.context_len + self.pred_len, :]
        if future.shape[1] != self.pred_len:
            raise ValueError(f'Expected future length {self.pred_len}, got {future.shape[1]}.')
        return context, future

    def train_step(self, batch):
        self.model.train()
        context, future = self.split_batch(batch)
        self.optimizer.zero_grad()
        loss, loss_dict = self.model.loss(context, future)
        loss.backward()
        grad_clip = float(self.config.get('training', {}).get('grad_clip', 1.0))
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()
        return {key: float(value.detach().cpu()) for key, value in loss_dict.items()}

    @torch.no_grad()
    def predict_future(self, context_batch, pred_len=None):
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f'This adapter predicts fixed pred_len={self.pred_len}, got {pred_len}.')
        self.model.eval()
        context = torch.as_tensor(context_batch, dtype=torch.float32, device=self.device)
        pred = self.model.predict(context)
        return pred.detach().cpu().numpy().astype('float32')

    def save(self, path, step, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step': int(step),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'extra': extra or {},
        }, path)

    def load(self, path, map_location=None):
        data = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(data['model'])
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
        return data


class DiffusionAdapter(TorchForecastAdapter):
    @torch.no_grad()
    def predict_future(self, context_batch, pred_len=None):
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f'This adapter predicts fixed pred_len={self.pred_len}, got {pred_len}.')
        self.model.eval()
        context = torch.as_tensor(context_batch, dtype=torch.float32, device=self.device)
        sampling_steps = self.config.get('baseline', {}).get('sampling_steps')
        pred = self.model.predict(context, sampling_steps=sampling_steps)
        return pred.detach().cpu().numpy().astype('float32')


class TimeGANAdapter(TorchForecastAdapter):
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        training = config.get('training', {})
        self.context_len = int(config['baseline']['context_len'])
        self.pred_len = int(config['baseline']['pred_len'])
        lr = float(training.get('lr', 1e-4))
        self.generator_optimizer = torch.optim.Adam(
            list(self.model.context_encoder.parameters()) + list(self.model.generator.parameters()),
            lr=lr,
            weight_decay=float(training.get('weight_decay', 0.0)),
        )
        self.discriminator_optimizer = torch.optim.Adam(
            list(self.model.discriminator.parameters()) + list(self.model.disc_head.parameters()),
            lr=lr,
            weight_decay=float(training.get('weight_decay', 0.0)),
        )

    def train_step(self, batch):
        self.model.train()
        context, future = self.split_batch(batch)
        real_full = torch.cat([context, future], dim=1)

        self.discriminator_optimizer.zero_grad()
        with torch.no_grad():
            fake_future = self.model.generate(context)
        fake_full = torch.cat([context, fake_future], dim=1)
        real_logits = self.model.discriminate(real_full)
        fake_logits = self.model.discriminate(fake_full)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()
        fake_future = self.model.generate(context)
        fake_full = torch.cat([context, fake_future], dim=1)
        fake_logits = self.model.discriminate(fake_full)
        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        recon_loss = F.mse_loss(fake_future, future)
        recon_weight = float(self.config.get('baseline', {}).get('reconstruction_weight', 10.0))
        # TimeGAN 本身不是前缀条件预测模型，这里用重建项把生成未来段绑定到隐藏真值。
        g_loss = adv_loss + recon_weight * recon_loss
        g_loss.backward()
        grad_clip = float(self.config.get('training', {}).get('grad_clip', 1.0))
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.generator_optimizer.step()

        return {
            'loss': float((g_loss + d_loss).detach().cpu()),
            'g_loss': float(g_loss.detach().cpu()),
            'd_loss': float(d_loss.detach().cpu()),
            'recon': float(recon_loss.detach().cpu()),
            'adv': float(adv_loss.detach().cpu()),
        }

    @torch.no_grad()
    def predict_future(self, context_batch, pred_len=None):
        if pred_len is not None and int(pred_len) != self.pred_len:
            raise ValueError(f'This adapter predicts fixed pred_len={self.pred_len}, got {pred_len}.')
        self.model.eval()
        context = torch.as_tensor(context_batch, dtype=torch.float32, device=self.device)
        return self.model.generate(context).detach().cpu().numpy().astype('float32')

    def save(self, path, step, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step': int(step),
            'model': self.model.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'config': self.config,
            'extra': extra or {},
        }, path)

    def load(self, path, map_location=None):
        data = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(data['model'])
        if 'generator_optimizer' in data:
            self.generator_optimizer.load_state_dict(data['generator_optimizer'])
        if 'discriminator_optimizer' in data:
            self.discriminator_optimizer.load_state_dict(data['discriminator_optimizer'])
        return data

