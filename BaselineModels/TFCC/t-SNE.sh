#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-52 -p alvis
#SBATCH -t 00:40:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=TFCC_t-SNE
#SBATCH --output=SlurmRecords_t-SNE/tfcc_t-SNE_%A.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load plotly.py/5.16.0-GCCcore-12.3.0
module load einops/0.7.0-GCCcore-12.3.0
# module load RAPIDS/24.4-foss-2023a-CUDA-12.1.1

python -m venv t-SNE_env
source t-SNE_env/bin/activate
# pip install kaleido==0.2.1

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/TFCC
# For Testing:
python t-SNE.py \
    --raw_input False \
    --data_path '/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette' \
    --checkpoint_path '/mimer/NOBACKUP/groups/naiss2025-22-1224/Codes/TFCC/ExperimentLog/tfcc_5fold_pretrain_37iter_bs128/0' \
    --img_save_dir 't-SNE_imgs' \
    --experiment_log '37_iters_pretrain_batchsize_128' \

    