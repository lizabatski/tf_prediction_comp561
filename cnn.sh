#!/bin/bash
#SBATCH --job-name=cnn_debug_ctcf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/debug_cnn_%j.out
#SBATCH --error=logs/debug_cnn_%j.err

# ---------------------------------------
# Setup
# ---------------------------------------

mkdir -p logs

module load python/3.11
module load scipy-stack
module load cuda/12.2

source /home/ekourb/tf/tfbinding_env/bin/activate

echo "========================================="
echo " Debug CNN + TF-MoDISco job started"
echo " Date: $(date)"
echo " Node: $(hostname)"
echo " Job ID: $SLURM_JOB_ID"
echo "========================================="



# ---------------------------------------
# Dataset Check
# ---------------------------------------

DATA_PATH="/home/ekourb/tf/datasets_chr1_1000bp/ctcf_chr1_dataset_struct.npz"

echo -e "\nDataset:"
echo "  Path: $DATA_PATH"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: Dataset not found!"
    ls -la /home/ekourb/tf/datasets_chr1_1000bp/
    exit 1
fi

echo "  Size: $(du -h $DATA_PATH | cut -f1)"

# ---------------------------------------
# GPU Info
# ---------------------------------------

if command -v nvidia-smi > /dev/null; then
    echo -e "\nGPU info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
fi

echo "========================================="
echo "Launching debug CNN training..."
echo "========================================="

export CUDA_LAUNCH_BLOCKING=0

# ---------------------------------------
# Run Your New Script
# ---------------------------------------

python train_cnn_simple.py \
    --data "$DATA_PATH" \
    --prefix "cnn_debug_ctcf"

EXIT_CODE=$?

echo -e "\n========================================="
echo "Job finished: $(date)"
echo "Exit code: $EXIT_CODE"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ Job completed successfully!"
else
    echo "✗ Job failed. Check logs."
fi

echo "========================================="
