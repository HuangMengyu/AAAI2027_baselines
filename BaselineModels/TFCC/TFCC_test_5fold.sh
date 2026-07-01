#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1224 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=TFCC-test
#SBATCH --output=SlurmRecords_results_balanced_accu/TFCC_SleepEDFx_%A.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

echo ${CUDA_VISIBLE_DEVICES}

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM:$PYTHONPATH"

# cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM

python BaselineModels/TFCC/test.py \
    --dataset_name SleepEDFx \
    --checkpoint_path checkpoints/TFCC_SleepEDFx \
    --frozen_backbone False