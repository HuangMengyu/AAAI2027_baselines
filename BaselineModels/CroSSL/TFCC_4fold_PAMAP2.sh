#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC_multimodal
#SBATCH --array=1-4
#SBATCH --output=logs/CroSSL_PAMAP2_%A_%a.out
#SBATCH --chdir=/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/CroSSL

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

export PYTHONPATH="/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/CroSSL:/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines:$PYTHONPATH"

cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/CroSSL

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python src/main_5fold.py \
    --seq_len 3000 --in_dim 3 \
    --dataset_name PAMAP2 \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --option 'None_None' \
    --model_save_path checkpoints/CroSSL_PAMAP2 \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \
    