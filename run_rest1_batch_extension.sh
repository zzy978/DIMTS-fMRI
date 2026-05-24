#!/bin/bash

set -euo pipefail

cd /data/home/zyzeng/project/DiMTS

PYTHON_BIN="${PYTHON_BIN:-/data/home/zyzeng/.conda/envs/dimts/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

CHECKPOINT_NAME="${CHECKPOINT_NAME:-fmri_seq256_zscore_stride64_subj1000_lambda3}"
RUN_NAME="${RUN_NAME:-rest1_extend_last500_${CHECKPOINT_NAME}}"
CONFIG_FILE="${CONFIG_FILE:-./Config/fmri_seq256.yaml}"
INFO_CSV="${INFO_CSV:-OUTPUT/${CHECKPOINT_NAME}/samples/last500.csv}"
SOURCE_DIR="${SOURCE_DIR:-/data/home/zyzeng/data1/datasets/rest1_to_csv}"
GPU_ID="${GPU_ID:-0}"
PRED_LEN="${PRED_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"

"${PYTHON_BIN}" batch_extend_rest1.py \
  --name "${RUN_NAME}" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_name "${CHECKPOINT_NAME}" \
  --use_best_checkpoint \
  --gpu "${GPU_ID}" \
  --pred_len "${PRED_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --extend_lengths 128 256 384 512 \
  --info_csv "${INFO_CSV}" \
  --source_dir "${SOURCE_DIR}" \
  --output '/data/home/zyzeng/data1/datasets/rest1_extensions'
