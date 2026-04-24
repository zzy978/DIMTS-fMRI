#!/bin/bash

set -euo pipefail

cd /data/home/zyzeng/project/DiMTS

echo "Submit Phase A predict/evaluate sweep using best checkpoints"
PREDICT_JOB_ID=$(sbatch --parsable --array=0-4 --export=EXPERIMENT_GROUP=phase_a predict_fmri_experiments.sh)
echo "Submitted predict array job: ${PREDICT_JOB_ID}"

SUMMARY_JOB_ID=$(sbatch --parsable --dependency=afterok:${PREDICT_JOB_ID} summarize_fmri_experiments.sh)
echo "Submitted summary job: ${SUMMARY_JOB_ID}"
