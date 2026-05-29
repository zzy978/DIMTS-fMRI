# fMRI Baseline Adapters

This directory contains lightweight conditional adapters for the baseline comparison protocol used in this repo.

The implementations share a strict fMRI interface: each model receives the latest 128 z-score normalized BOLD time points and predicts the next 128 time points. Longer horizons are generated autoregressively in 128-point chunks.

Default evaluation uses the HCP subject-level `test` split reconstructed from each baseline config. Rest1-500 evaluation is still available with `evaluate_fmri_baseline.py --eval_source rest1`, but it is not the default baseline comparison target.

Notes:

- `TimeGAN` and `TimeVAE` are adapted with an explicit context input, because their standard forms are not future-prefix conditional predictors.
- `Diffusion-TS`, `FourierDiff`, and `PaD-TS` use a common conditional diffusion forecaster with model-specific loss weights.
- These adapters are meant to provide a reproducible in-repo comparison protocol. If exact official implementations are required, replace the model internals while preserving the adapter interface.
