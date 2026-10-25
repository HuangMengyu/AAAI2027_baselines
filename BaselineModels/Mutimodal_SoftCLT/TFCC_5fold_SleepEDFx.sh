#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -t 20:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
# #SBATCH -C NOGPU
#SBATCH --job-name=SoftCLT_SleepEDFx
#SBATCH --array=1-5
#SBATCH --output=logs/SoftCLT_SleepEDFx_%A_%a.out
#SBATCH --chdir=/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/Mutimodal_SoftCLT

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
source /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_FNattract/experiment_venv/bin/activate
# python -m venv torch_env
# source torch_env/bin/activate
# pip install torchtune

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/Mutimodal_SoftCLT:/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines:$PYTHONPATH"

cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/Mutimodal_SoftCLT


# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python -u main_5fold.py \
    --seq_len 3000 --in_dim 3 \
    --dataset_name SleepEDFx \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --option 'None_None' \
    --model_save_path checkpoints/SoftCLT_SleepEDFx_newDTW \
    --batch_size 128 \
    --num_iter 600 \
    --lr 1e-3 \
    --ssl True \
