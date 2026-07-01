#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-117 -p alvis
#SBATCH -t 08:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=SoftCLT_PAMAP2
#SBATCH --array=1-4
#SBATCH --output=SlurmRecords_newDTW_20260207/SoftCLT_PAMAP2_%A_%a.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
# module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

# python -m venv torch_env
source torch_env/bin/activate
# pip install soft-dtw-pytorch
# pip install fastdtw
# pip install tslearn

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM:$PYTHONPATH"

# cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/Mutimodal_SoftCLT
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python -u BaselineModels/Mutimodal_SoftCLT/main_5fold.py \
    --seq_len 3000 --in_dim 3 \
    --dataset_name PAMAP2 \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --option 'None_None' \
    --model_save_path checkpoints/SoftCLT_PAMAP2_newDTW_20260207 \
    --batch_size 128 \
    --num_iter 200 \
    --lr 1e-3 \
    --ssl True \