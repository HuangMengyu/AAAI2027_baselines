#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-52 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC-test
#SBATCH --output=SlurmRecords_results_rerun_paper_settings/TFCC_multimodal_fnAB_cancel_UCI-HAR_0.0025_%A.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM:$PYTHONPATH"

# cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC_multimodal
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM

python BaselineModels/TFCC_multimodal/test.py \
    --dataset_name UCI-HAR \
    --checkpoint_path checkpoints/TFCC_multimodal_fnAB_cancel_UCI-HAR_topk0.0025 \
    --final_out_channels 128 \
    --output_dims 127