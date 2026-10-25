#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -t 05:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC_fnAB_attract_SleepEDFx
#SBATCH --array=1-5
#SBATCH --output=logs/SleepEDFx/warmup4_fnABC_attract_SleepEDFx_0.035_%A_%a.out
#SBATCH --chdir=/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC_multimodal

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
source /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_FNattract/experiment_venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines:$PYTHONPATH"

# cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC_multimodal
cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC_multimodal/


# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python main_5fold.py \
    --seq_len 3000 \
    --dataset_name SleepEDFx \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --epochs_ft 10 \
    --warm_epochs 4 \
    --option 'fn_ABC' \
    --fn_mode 'attract' \
    --model_save_path checkpoints/warmup4_fnABC_attract_SleepEDFx_0.035 \
    --batch_size 128 \
    --topk 0.035 \
    --use_iteration False \
    --num_iter 300 \
    --lr 1e-3 \
    --ssl True \
    --supervised_baseline False
    # --topk 0.08 \
    