#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-12 -p vera
#SBATCH -t 01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TS2Vec_PAMAP2
#SBATCH --array=1-4
#SBATCH --output=logs/TS2Vec_PAMAP2_%A_%a.out
#SBATCH --chdir=/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/ts2vec

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
# virtualenv --system-site-packages my_venv
source /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_FNattract/experiment_venv/bin/activate
# pip install tsaug

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export PYTHONPATH="/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/ts2vec:/cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines:$PYTHONPATH"

cd /cephyr/NOBACKUP/groups/multimodal_learning_cse/AAAI2027_baselines/BaselineModels/ts2vec

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python main_5fold.py \
    --seq_len 256 \
    --dataset_name PAMAP2 \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --model_save_path checkpoints/TS2Vec_PAMAP2 \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \
    