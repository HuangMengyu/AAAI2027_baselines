#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-117 -p alvis
#SBATCH -t 03:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=FOCAL
#SBATCH --array=1-5
#SBATCH --output=SlurmRecords_stft/FOCAL_UCI-HAR_%A_%a.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0

python -m venv torch_env
source torch_env/bin/activate
# pip install tsai==0.3.9 --no-deps
# pip install fastai py-cpuinfo
# 1. Uninstall the problematic newer versions
# pip uninstall -y fastcore fastai tsai

# # 2. Install the specific compatible versions
# pip install fastcore==1.7.29 fastai==2.7.18

# # 3. Re-install tsai 0.3.9 (without letting it update the others)
# pip install tsai==0.3.9 --no-deps

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/focal:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/focal

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python src/main_5fold.py \
    --seq_len 3000 --in_dim 3 \
    --dataset_name UCI-HAR \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --model_save_path checkpoints/FOCAL_UCI-HAR_stft \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \
    