#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=DCL_UCI-HAR_total
#SBATCH --array=1-5
#SBATCH --output=logs/UCI-HAR_total/DCL_UCI-HAR_total_%A_%a.out
#SBATCH --chdir=/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/DebiasedContrastiveLearning

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
source /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_FNattract/experiment_venv/bin/activate

# python -m venv torch_env
# source torch_env/bin/activate
# pip install torchtune

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/DebiasedContrastiveLearning:/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines:$PYTHONPATH"

cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/DebiasedContrastiveLearning

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python main_5fold.py \
    --seq_len 3000 --in_dim 3 \
    --dataset_name UCI-HAR_total \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --warm_epochs 0 \
    --model_save_path checkpoints/DCL_UCI-HAR_total \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \

    