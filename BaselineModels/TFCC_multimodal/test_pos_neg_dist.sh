#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1224 -p alvis
#SBATCH -t 00:40:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC_test_pos_neg_dist
#SBATCH --output=SlurmRecords_test_pos_neg_dist/tfcc_test_pos_neg_dist_10epoch_projector_statistics_%A.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load plotly.py/5.16.0-GCCcore-12.3.0
module load einops/0.7.0-GCCcore-12.3.0

python -m venv t-SNE_env
source t-SNE_env/bin/activate
# pip install kaleido==0.2.1

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC_multimodal:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC_multimodal
# For Testing:
python test_pos_neg_dist.py \
    --data_path '/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette' \
    --checkpoint_path checkpoints/TFCC_10epochs_bs128_lr1e-3/0 \
    --img_save_dir '10_epoch' \
    --experiment_log 'fold_1_TFCC_10epochs' \
    --fold '1' \
    --intra_modal False \
    --inter_modal True \
    --real_label False

    