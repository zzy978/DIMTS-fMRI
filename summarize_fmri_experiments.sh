#!/bin/bash
#SBATCH --partition=partition_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=fmri_summary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /data/home/zyzeng/project/DiMTS

PYTHON_BIN="${PYTHON_BIN:-/data/home/zyzeng/.conda/envs/dimts/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-OUTPUT}"
CSV_PATH="${CSV_PATH:-OUTPUT/experiment_summary.csv}"

"${PYTHON_BIN}" summarize_experiments.py --output_root "${OUTPUT_ROOT}" --csv_path "${CSV_PATH}"
