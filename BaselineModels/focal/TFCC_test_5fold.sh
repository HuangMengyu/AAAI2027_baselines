#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=FOCAL-test
#SBATCH --output=SlurmRecords_results_stft_balanced_accu/FOCAL_UCI-HAR_stft_%A.out
#SBATCH --chdir=/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load IPython/8.14.0-GCCcore-12.3.0

# python -m venv torch_env
source /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_FNattract/experiment_venv/bin/activate

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PYTHONPATH="/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines:$PYTHONPATH"

# cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/focal
cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines

python BaselineModels/focal/src/test.py \
    --dataset_name UCI-HAR \
    --checkpoint_path checkpoints/FOCAL_UCI-HAR_stft \
    --final_out_channels 128 \
    --output_dims 127