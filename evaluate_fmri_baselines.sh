#!/bin/bash
#SBATCH --partition=partition_1
#SBATCH --gres=gpu:1
#SBATCH --job-name=fmri_baseline_eval
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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
CONFIG_FILE="${CONFIG_FILE:-Baselines/configs/${MODEL_NAME}_fmri.yaml}"
EVAL_SOURCE="${EVAL_SOURCE:-test_split}"
SOURCE_DIR="${SOURCE_DIR:-/data/home/zyzeng/data1/datasets/rest1_to_csv_500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-OUTPUT/baseline_test_set}"
PRED_LEN="${PRED_LEN:-128}"
EXTEND_LENGTHS="${EXTEND_LENGTHS:-128 256 384 512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SUBJECTS="${MAX_SUBJECTS:-0}"

read -r -a EXTEND_LENGTH_ARRAY <<< "${EXTEND_LENGTHS}"

"${PYTHON_BIN}" evaluate_fmri_baseline.py \
  --model "${MODEL_NAME}" \
  --config_file "${CONFIG_FILE}" \
  --eval_source "${EVAL_SOURCE}" \
  --source_dir "${SOURCE_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --gpu "${GPU_ID}" \
  --pred_len "${PRED_LEN}" \
  --extend_lengths "${EXTEND_LENGTH_ARRAY[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --max_subjects "${MAX_SUBJECTS}"
