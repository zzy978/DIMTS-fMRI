import math

import torch
from torch import nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self, feature_size, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(feature_size, hidden_dim, batch_first=True)

    def forward(self, context):
        _, hidden = self.gru(context)
        return hidden[-1]


class FutureDecoder(nn.Module):
    def __init__(self, feature_size, pred_len, hidden_dim, latent_dim=0, noise_dim=0):
        super().__init__()
        self.pred_len = pred_len
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        input_dim = hidden_dim + latent_dim + noise_dim
        self.state_proj = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(pred_len, hidden_dim) * 0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, feature_size),
        )

    def forward(self, context_state, latent=None, noise=None):
        parts = [context_state]
        if self.latent_dim > 0:
            if latent is None:
                latent = torch.zeros(context_state.shape[0], self.latent_dim, device=context_state.device)
            parts.append(latent)
        if self.noise_dim > 0:
            if noise is None:
                noise = torch.zeros(context_state.shape[0], self.noise_dim, device=context_state.device)
            parts.append(noise)

        state = torch.tanh(self.state_proj(torch.cat(parts, dim=-1)))
        tokens = state[:, None, :] + self.position[None, :, :]
        return self.net(tokens)


class ConditionalTimeVAE(nn.Module):
    def __init__(self, feature_size, context_len, pred_len, hidden_dim=128, latent_dim=64, kl_weight=1e-3):
        super().__init__()
        self.feature_size = feature_size
        self.context_len = context_len
        self.pred_len = pred_len
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.context_encoder = ContextEncoder(feature_size, hidden_dim)
        self.full_encoder = ContextEncoder(feature_size, hidden_dim)
        self.mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.decoder = FutureDecoder(feature_size, pred_len, hidden_dim, latent_dim=latent_dim)

    def encode(self, context, future):
        context_state = self.context_encoder(context)
        full_state = self.full_encoder(torch.cat([context, future], dim=1))
        stats = torch.cat([context_state, full_state], dim=-1)
        return context_state, self.mu(stats), self.logvar(stats)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, context, future=None):
        if future is None:
            context_state = self.context_encoder(context)
            latent = torch.randn(context.shape[0], self.latent_dim, device=context.device)
            return self.decoder(context_state, latent=latent), None, None
        context_state, mu, logvar = self.encode(context, future)
        latent = self.reparameterize(mu, logvar)
        return self.decoder(context_state, latent=latent), mu, logvar

    def loss(self, context, future):
        pred, mu, logvar = self(context, future)
        recon = F.mse_loss(pred, future)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.kl_weight * kl, {'loss': recon + self.kl_weight * kl, 'recon': recon, 'kl': kl}

    @torch.no_grad()
    def predict(self, context):
        pred, _, _ = self(context, future=None)
        return pred


class ConditionalTimeGAN(nn.Module):
    def __init__(self, feature_size, context_len, pred_len, hidden_dim=128, noise_dim=64):
        super().__init__()
        self.feature_size = feature_size
        self.context_len = context_len
        self.pred_len = pred_len
        self.noise_dim = noise_dim
        self.context_encoder = ContextEncoder(feature_size, hidden_dim)
        self.generator = FutureDecoder(feature_size, pred_len, hidden_dim, noise_dim=noise_dim)
        self.discriminator = nn.GRU(feature_size, hidden_dim, batch_first=True)
        self.disc_head = nn.Linear(hidden_dim, 1)

    def sample_noise(self, batch_size, device):
        return torch.randn(batch_size, self.noise_dim, device=device)

    def generate(self, context, noise=None):
        state = self.context_encoder(context)
        if noise is None:
            noise = self.sample_noise(context.shape[0], context.device)
        return self.generator(state, noise=noise)

    def discriminate(self, full_sequence):
        _, hidden = self.discriminator(full_sequence)
        return self.disc_head(hidden[-1]).squeeze(-1)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class DenoiseBackbone(nn.Module):
    def __init__(self, feature_size, context_len, pred_len, hidden_dim=128, time_dim=64):
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.context_encoder = ContextEncoder(feature_size, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.initial = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_proj = nn.Linear(feature_size + hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, feature_size),
        )

    def forward(self, context, noisy_future, t):
        context_state = self.context_encoder(context)
        time_state = self.time_embed(t)
        h0 = torch.tanh(self.initial(torch.cat([context_state, time_state], dim=-1))).unsqueeze(0)
        time_tokens = time_state[:, None, :].expand(-1, noisy_future.shape[1], -1)
        tokens = self.input_proj(torch.cat([noisy_future, time_tokens], dim=-1))
        hidden, _ = self.gru(tokens, h0)
        return self.output(hidden)


class ConditionalDiffusionForecaster(nn.Module):
    def __init__(
        self,
        feature_size,
        context_len,
        pred_len,
        hidden_dim=128,
        timesteps=100,
        fft_loss_weight=0.0,
        continuity_weight=0.0,
        corr_loss_weight=0.0,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.context_len = context_len
        self.pred_len = pred_len
        self.timesteps = timesteps
        self.fft_loss_weight = fft_loss_weight
        self.continuity_weight = continuity_weight
        self.corr_loss_weight = corr_loss_weight
        self.denoiser = DenoiseBackbone(feature_size, context_len, pred_len, hidden_dim=hidden_dim)

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1.0 - alpha_cumprod))

    def q_sample(self, future, t, noise):
        shape = (future.shape[0],) + (1,) * (future.ndim - 1)
        return (
            self.sqrt_alpha_cumprod[t].reshape(shape) * future
            + self.sqrt_one_minus_alpha_cumprod[t].reshape(shape) * noise
        )

    def _fft_loss(self, pred, future):
        pred_fft = torch.fft.rfft(pred, dim=1)
        future_fft = torch.fft.rfft(future, dim=1)
        return torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(future_fft)))

    def _corr_loss(self, pred, future):
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        future_centered = future - future.mean(dim=1, keepdim=True)
        numerator = torch.sum(pred_centered * future_centered, dim=1)
        denominator = torch.sqrt(
            torch.sum(pred_centered ** 2, dim=1) * torch.sum(future_centered ** 2, dim=1) + 1e-8
        )
        return torch.mean(1.0 - numerator / denominator)

    def loss(self, context, future):
        batch = future.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=future.device)
        noise = torch.randn_like(future)
        noisy_future = self.q_sample(future, t, noise)
        pred = self.denoiser(context, noisy_future, t)
        recon = F.mse_loss(pred, future)
        total = recon
        losses = {'loss': total, 'recon': recon}

        if self.fft_loss_weight > 0:
            # 频域损失用于约束生成未来段的 BOLD 低频能量结构，避免只优化点误差。
            fft = self._fft_loss(pred, future)
            total = total + self.fft_loss_weight * fft
            losses['fft'] = fft
        if self.continuity_weight > 0:
            # 首个未来点与历史最后一点的连续性约束，降低自回归扩展时的拼接突变。
            continuity = F.mse_loss(pred[:, 0, :], context[:, -1, :])
            total = total + self.continuity_weight * continuity
            losses['continuity'] = continuity
        if self.corr_loss_weight > 0:
            # ROI 维度相关趋势约束用于 PaD-TS 条件适配，保持未来段整体协同变化。
            corr = self._corr_loss(pred, future)
            total = total + self.corr_loss_weight * corr
            losses['corr'] = corr

        losses['loss'] = total
        return total, losses

    @torch.no_grad()
    def predict(self, context, sampling_steps=None):
        steps = self.timesteps if sampling_steps is None else min(int(sampling_steps), self.timesteps)
        indices = torch.linspace(self.timesteps - 1, 0, steps, device=context.device).long()
        future = torch.randn(context.shape[0], self.pred_len, self.feature_size, device=context.device)

        for idx, t_value in enumerate(indices):
            t = torch.full((context.shape[0],), int(t_value.item()), device=context.device, dtype=torch.long)
            pred_x0 = self.denoiser(context, future, t)
            if idx == len(indices) - 1:
                future = pred_x0
            else:
                next_t = int(indices[idx + 1].item())
                noise = torch.randn_like(future)
                future = (
                    self.sqrt_alpha_cumprod[next_t] * pred_x0
                    + self.sqrt_one_minus_alpha_cumprod[next_t] * noise
                )
        return future

