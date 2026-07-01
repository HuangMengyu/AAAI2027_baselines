#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-52 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TS2Vec-test
#SBATCH --output=SlurmRecords_results_same_encoder_balanced_accu/TS2Vec_UCI-HAR_%A.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM:$PYTHONPATH"

# cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/ts2vec
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM

python BaselineModels/ts2vec/test.py \
    --dataset_name UCI-HAR \
    --checkpoint_path checkpoints/TS2Vec_UCI-HAR \
    --final_out_channels 128 \
    --output_dims 127