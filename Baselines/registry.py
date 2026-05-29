from Baselines.adapters import DiffusionAdapter, TimeGANAdapter, TorchForecastAdapter
from Baselines.models import (
    ConditionalDiffusionForecaster,
    ConditionalTimeGAN,
    ConditionalTimeVAE,
)


SUPPORTED_BASELINES = ('timegan', 'timevae', 'diffusion_ts', 'fourierdiff', 'pad_ts')


def build_adapter(model_name, config, device):
    model_name = model_name.lower()
    params = config['baseline']
    common = {
        'feature_size': int(params['feature_size']),
        'context_len': int(params['context_len']),
        'pred_len': int(params['pred_len']),
        'hidden_dim': int(params.get('hidden_dim', 128)),
    }

    if model_name == 'timegan':
        model = ConditionalTimeGAN(
            **common,
            noise_dim=int(params.get('noise_dim', 64)),
        )
        return TimeGANAdapter(model=model, config=config, device=device)
    if model_name == 'timevae':
        model = ConditionalTimeVAE(
            **common,
            latent_dim=int(params.get('latent_dim', 64)),
            kl_weight=float(params.get('kl_weight', 1e-3)),
        )
        return TorchForecastAdapter(model=model, config=config, device=device)
    if model_name == 'diffusion_ts':
        model = ConditionalDiffusionForecaster(
            **common,
            timesteps=int(params.get('timesteps', 100)),
            fft_loss_weight=float(params.get('fft_loss_weight', 0.0)),
            continuity_weight=float(params.get('continuity_weight', 0.0)),
            corr_loss_weight=float(params.get('corr_loss_weight', 0.0)),
        )
        return DiffusionAdapter(model=model, config=config, device=device)
    if model_name == 'fourierdiff':
        model = ConditionalDiffusionForecaster(
            **common,
            timesteps=int(params.get('timesteps', 100)),
            fft_loss_weight=float(params.get('fft_loss_weight', 0.1)),
            continuity_weight=float(params.get('continuity_weight', 0.0)),
            corr_loss_weight=float(params.get('corr_loss_weight', 0.0)),
        )
        return DiffusionAdapter(model=model, config=config, device=device)
    if model_name == 'pad_ts':
        model = ConditionalDiffusionForecaster(
            **common,
            timesteps=int(params.get('timesteps', 100)),
            fft_loss_weight=float(params.get('fft_loss_weight', 0.05)),
            continuity_weight=float(params.get('continuity_weight', 0.05)),
            corr_loss_weight=float(params.get('corr_loss_weight', 0.02)),
        )
        return DiffusionAdapter(model=model, config=config, device=device)

    raise ValueError(f'Unsupported baseline model: {model_name}. Choices: {SUPPORTED_BASELINES}')

