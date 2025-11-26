#!/bin/bash
#SBATCH --job-name=cnn_ctcf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cnn_%j.out
#SBATCH --error=logs/cnn_%j.err

mkdir -p logs

module load python/3.11
module load scipy-stack
module load cuda/12.2

source /home/ekourb/tf/tfbinding_env/bin/activate

python train_cnn_simple.py \
    --data /home/ekourb/tf/datasets_chr1_1000bp/ctcf_chr1_dataset_struct.npz \
    --prefix cnn_ctcf

echo "Done!"