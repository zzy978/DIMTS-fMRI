#!/bin/bash
#SBATCH --partition=partition_1
#SBATCH --gres=gpu:1
#SBATCH --job-name=rest1_gt_ext
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /data/home/zyzeng/project/DiMTS
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-/data/home/zyzeng/.conda/envs/dimts/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

CONFIG_FILE="${CONFIG_FILE:-./Config/fmri_seq256_dfc.yaml}"
SOURCE_DIR="${SOURCE_DIR:-/data/home/zyzeng/data1/datasets/rest1_to_csv_500}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-fmri_seq256_zscore_DFCGraph_stride64_subj1000_lambda4_256}"
CHECKPOINT_REF="${CHECKPOINT_REF:-best}"
RUN_NAME="${RUN_NAME:-rest1_500_groundtruth_DFCGraph_lambda4}"
OUTPUT_DIR="${OUTPUT_DIR:-OUTPUT}"
GPU_ID="${GPU_ID:-0}"
PRED_LEN="${PRED_LEN:-128}"
STRIDE="${STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FC_WINDOW="${FC_WINDOW:-128}"
FC_STRIDE="${FC_STRIDE:-32}"
MAX_SUBJECTS="${MAX_SUBJECTS:-0}"
NUM_PLOT_SUBJECTS="${NUM_PLOT_SUBJECTS:-3}"

# Rest1-500 内部留出未来段做真值评估；z-score 拟合策略由 Python 脚本固定为完整 subject 序列。
"${PYTHON_BIN}" evaluate_rest1_groundtruth_extension.py \
  --name "${RUN_NAME}" \
  --output "${OUTPUT_DIR}" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_name "${CHECKPOINT_NAME}" \
  --checkpoint_ref "${CHECKPOINT_REF}" \
  --source_dir "${SOURCE_DIR}" \
  --gpu "${GPU_ID}" \
  --pred_len "${PRED_LEN}" \
  --extend_lengths 128 256 384 512 \
  --stride "${STRIDE}" \
  --batch_size "${BATCH_SIZE}" \
  --fc_window "${FC_WINDOW}" \
  --fc_stride "${FC_STRIDE}" \
  --max_subjects "${MAX_SUBJECTS}" \
  --num_plot_subjects "${NUM_PLOT_SUBJECTS}"
