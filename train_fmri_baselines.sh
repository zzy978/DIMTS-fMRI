#!/bin/bash
#SBATCH --partition=partition_1
#SBATCH --gres=gpu:1
#SBATCH --job-name=fmri_baseline_train
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd /data/home/zyzeng/project/DiMTS
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-/data/home/zyzeng/.conda/envs/dimts/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

MODELS=(timegan timevae diffusion_ts fourierdiff pad_ts)
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_NAME="${MODEL_NAME:-${MODELS[$ARRAY_ID]}}"
GPU_ID="${GPU_ID:-0}"
MAX_STEPS="${MAX_STEPS:-10000}"
MAX_SUBJECTS="${MAX_SUBJECTS:-1000}"
CONFIG_FILE="${CONFIG_FILE:-Baselines/configs/${MODEL_NAME}_fmri.yaml}"

"${PYTHON_BIN}" train_fmri_baseline.py \
  --model "${MODEL_NAME}" \
  --config_file "${CONFIG_FILE}" \
  --gpu "${GPU_ID}" \
  --max_steps "${MAX_STEPS}" \
  --max_subjects "${MAX_SUBJECTS}"

