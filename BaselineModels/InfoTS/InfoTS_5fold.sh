#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-117 -p alvis
#SBATCH -t 1-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=InfoTS
#SBATCH --array=1-5
#SBATCH --output=SlurmRecords_MetaTraining_AllSampleLoss/Infots_5fold_%A_%a.out

module purge
# module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
# module load scikit-learn/1.4.2-gfbf-2023a
# module load einops/0.7.0-GCCcore-12.3.0
# virtualenv --system-site-packages my_venv
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0
source my_venv/bin/activate

# pip install tsaug

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/InfoTS:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027_baselines/BaselineModels/InfoTS
# For Testing:
python main_5fold.py \
    --seq_len 3000 --in_dim 3 --n_classes 5 \
    --dataset_path '/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette' \
    --model_save_path '/mimer/NOBACKUP/groups/naiss2025-22-1224/Codes/InfoTS/ExperimentLog/5fold_withMetaTraining_withAllSampleLoss' \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 100 \
    --batch_size 128 

    