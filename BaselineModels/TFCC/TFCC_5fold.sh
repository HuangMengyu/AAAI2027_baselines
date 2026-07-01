#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-52 -p alvis
#SBATCH -t 01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC
#SBATCH --array=3-3
#SBATCH --output=SlurmRecords_finetune/tfcc_5fold_pretrain_37iter_bs128_finetune_5epoch_%A_%a.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC
# For Testing:
python main_5fold.py \
    --seq_len 3000 --in_dim 3 --n_classes 5 \
    --dataset_path '/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette' \
    --model_save_path '/mimer/NOBACKUP/groups/naiss2025-22-1224/Codes/TFCC/tfcc_5fold_pretrain_37iter_bs128_finetune_10epoch' \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 120 \
    --epochs_ft 10 \
    --batch_size 128 \
    --train_with_iter True \
    --num_iter 37 \
    --ssl True \
    --frozen_backbone False


    