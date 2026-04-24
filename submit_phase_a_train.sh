#!/bin/bash

set -euo pipefail

cd /data/home/zyzeng/project/DiMTS

echo "Submit Phase A subject sweep (100/200/400/600/800)"
sbatch --array=0-4 train_fmri_experiments.sh

echo
echo "Phase B should be submitted after you decide BEST_SUBJECTS."
echo "Example:"
echo "sbatch --array=0-4 --export=EXPERIMENT_GROUP=phase_b,BEST_SUBJECTS=400 train_fmri_experiments.sh"
