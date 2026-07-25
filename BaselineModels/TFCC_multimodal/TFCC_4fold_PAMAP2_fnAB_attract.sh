#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-351 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC_fnAB_attract_PAMAP2
#SBATCH --array=1-4
#SBATCH --output=logs/PAMAP2/fnABC_attract_PAMAP2_0.035_%A_%a.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC_multimodal/

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python main_5fold.py \
    --seq_len 3000 \
    --dataset_name PAMAP2 \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --epochs_ft 10 \
    --warm_epochs 1 \
    --option 'fn_ABC' \
    --fn_mode 'attract' \
    --model_save_path checkpoints/warmup1_fnABC_attract_PAMAP2_0.035 \
    --batch_size 128 \
    --topk 0.035 \
    --use_iteration False \
    --num_iter 300 \
    --lr 1e-3 \
    --ssl True \
    --supervised_baseline False
    # --topk 0.08 \
    