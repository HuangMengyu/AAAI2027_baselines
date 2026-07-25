#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-351 -p alvis
#SBATCH -t 04:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
# #SBATCH -C NOGPU
#SBATCH --job-name=StatioCL-ADF_PAMAP2
#SBATCH --array=1-4
#SBATCH --output=logs_noautolag/PAMAP2/PAMAP2_lag2_%A_%a.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

python -m venv statsmodels_env
source statsmodels_env/bin/activate
python -m pip install statsmodels

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/StatioCL-ADF:/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/StatioCL-ADF

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python main_5fold.py \
    --seq_len 3000 --in_dim 3 \
    --dataset_name PAMAP2 \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --warm_epochs 0 \
    --model_save_path checkpoints/PAMAP2_noautolag_lag2 \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \
    --final_out_channels 128 \
    --output_dims 127 \
    --adf_cache_only False \
    --adf_sample_autolag none \
    --adf_sample_maxlag 2


    
