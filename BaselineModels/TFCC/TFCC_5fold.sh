#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -t 01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC
#SBATCH --array=3-3
#SBATCH --output=SlurmRecords_finetune/tfcc_5fold_pretrain_37iter_bs128_finetune_5epoch_%A_%a.out
#SBATCH --chdir=/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
source /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_FNattract/experiment_venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}

export PYTHONPATH="/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC:/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines:$PYTHONPATH"

cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC
# For Testing:
python main_5fold.py \
    --seq_len 3000 --in_dim 3 --n_classes 5 \
    --model_save_path '/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/TFCC/tfcc_5fold_pretrain_37iter_bs128_finetune_10epoch' \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 120 \
    --epochs_ft 10 \
    --batch_size 128 \
    --train_with_iter True \
    --num_iter 37 \
    --ssl True \
    --frozen_backbone False


    