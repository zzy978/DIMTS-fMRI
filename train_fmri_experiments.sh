#!/bin/bash
#SBATCH --partition=partition_1
#SBATCH --gres=gpu:1
#SBATCH --job-name=ROIGraph_IdentityBias_seq256_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd /data/home/zyzeng/project/DiMTS

PYTHON_BIN="${PYTHON_BIN:-/data/home/zyzeng/.conda/envs/dimts/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

CONFIG_FILE="./Config/fmri_seq256.yaml"
GPU_ID="${GPU_ID:-0}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-phase_a}"
BEST_SUBJECTS="${BEST_SUBJECTS:-400}"
PRED_LEN="${PRED_LEN:-128}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
STRIDE="${STRIDE:-64}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"

COMMON_ARGS=(
  --config_file "${CONFIG_FILE}"
  --gpu "${GPU_ID}"
  --train
  --data_input_mode subject_split
  --subject_train_ratio 0.8
  --subject_val_ratio 0.1
  --subject_test_ratio 0.1
  --subject_shuffle
  --drop_nan_subjects
  --stride "${STRIDE}"
)

if [ "${EXPERIMENT_GROUP}" = "phase_a" ]; then
  SUBJECT_COUNTS=(100 200 400 600 800)
  LAMBDA1=0.1
  LAMBDA2=0.01

  MAX_SUBJECTS="${SUBJECT_COUNTS[$ARRAY_ID]}"
  EXP_NAME="fmri_seq256_zscore_stride${STRIDE}_subj${MAX_SUBJECTS}_base"
  CHECKPOINT_NAME="${EXP_NAME}"

  "${PYTHON_BIN}" main.py \
    --name "${EXP_NAME}" \
    --checkpoint_name "${CHECKPOINT_NAME}" \
    --lambda1 "${LAMBDA1}" \
    --lambda2 "${LAMBDA2}" \
    --max_subjects "${MAX_SUBJECTS}" \
    "${COMMON_ARGS[@]}" \
    solver.max_epochs "${TRAIN_STEPS}"
elif [ "${EXPERIMENT_GROUP}" = "phase_b" ]; then
  # Phase B：尝试兼顾 FC / 结构相似性的损失权重组合。
  LAMBDA1S=(0.3 1.0 0.3 0.1 0.1)
  LAMBDA2S=(0.001 0.001 0.003 0.003 0.01)

  LAMBDA1="${LAMBDA1S[$ARRAY_ID]}"
  LAMBDA2="${LAMBDA2S[$ARRAY_ID]}"
  EXP_IDX=$((ARRAY_ID + 1))
  # 改实验名，实验名是跟模型保存名相关的，改了实验名就相当于改了模型名，注意还要跟predict_fmri_experiments里对应的实验名一致
  EXP_NAME="fmri_seq256_zscore_ROIGraph_IdentityBias_stride${STRIDE}_subj${BEST_SUBJECTS}_lambda${EXP_IDX}"
  CHECKPOINT_NAME="${EXP_NAME}"

  "${PYTHON_BIN}" main.py \
    --name "${EXP_NAME}" \
    --checkpoint_name "${CHECKPOINT_NAME}" \
    --lambda1 "${LAMBDA1}" \
    --lambda2 "${LAMBDA2}" \
    --max_subjects "${BEST_SUBJECTS}" \
    "${COMMON_ARGS[@]}" \
    solver.max_epochs "${TRAIN_STEPS}"
else
  echo "Unsupported EXPERIMENT_GROUP=${EXPERIMENT_GROUP}. Use phase_a or phase_b."
  exit 1
fi
