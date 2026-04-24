#!/bin/bash

set -euo pipefail

cd /data/home/zyzeng/project/DiMTS

BEST_SUBJECTS="${BEST_SUBJECTS:-1000}"

echo "Submit Phase B lambda sweep with BEST_SUBJECTS=${BEST_SUBJECTS}"
sbatch --array=0-4 --export=EXPERIMENT_GROUP=phase_b,BEST_SUBJECTS="${BEST_SUBJECTS}" train_fmri_experiments.sh
