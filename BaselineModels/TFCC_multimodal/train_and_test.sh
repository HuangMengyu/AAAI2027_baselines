#!/bin/bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC_multimodal
#SBATCH --output=SlurmRecords_train_and_test/tfcc_multimodal_30iters_lr1e-3_%A.out
#SBATCH --chdir=/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC_multimodal

set -e
cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC_multimodal

TRAIN_SCRIPT="${1:?Usage: sbatch train_and_test.sh TRAIN_SCRIPT}"

# 1. Submit the Array Job (Folds)
# --parsable makes sbatch return ONLY the Job ID (e.g., 12345)
JOB_ID=$(sbatch --parsable "$TRAIN_SCRIPT")

echo "Submitted Array Job $JOB_ID for 5 folds."

# 2. Submit the Summary Job with a dependency
# This job will NOT run until Job $JOB_ID (all 5 tasks) is COMPLETED (afterok)
JOB_ID_test=$(sbatch --parsable --dependency=afterok:$JOB_ID TFCC_test_5fold.sh)
echo "Submitted Array Job $JOB_ID_test for testing."