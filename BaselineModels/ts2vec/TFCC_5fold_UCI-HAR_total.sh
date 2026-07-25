#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-117 -p alvis
#SBATCH -t 01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TS2Vec_UCI-HAR_total
#SBATCH --array=1-5
#SBATCH --output=logs/TS2Vec_UCI-HAR_total_%A_%a.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0
source my_venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/ts2vec:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/ts2vec

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python main_5fold.py \
    --seq_len 128 \
    --dataset_name UCI-HAR_total \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --model_save_path checkpoints/TS2Vec_UCI-HAR_total \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \
    