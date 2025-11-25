#!/bin/bash
#SBATCH --job-name=cnn_ctcf_1000bp
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cnn_1000bp_%j.out
#SBATCH --error=logs/cnn_1000bp_%j.err

# Create logs directory
mkdir -p logs

# Load required modules (adjust based on your cluster)
module load python/3.11
module load scipy-stack
module load cuda/12.2

# Activate your environment
source /home/ekourb/tf/tfbinding_env/bin/activate

echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "========================================="

# Environment verification
echo -e "\nEnvironment check:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check key packages
python -c "
import sys
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU device: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
except ImportError as e:
    print(f'PyTorch import error: {e}')
    sys.exit(1)

try:
    import sklearn
    print(f'scikit-learn: {sklearn.__version__}')
except ImportError as e:
    print(f'sklearn import error: {e}')
    sys.exit(1)

try:
    import modiscolite
    print(f'modiscolite: Available')
except ImportError as e:
    print(f'modiscolite import error: {e}')
    print('TF-MoDISco will be disabled')

try:
    import h5py
    print(f'h5py: {h5py.__version__}')
except ImportError as e:
    print(f'h5py import error: {e}')
    sys.exit(1)
"

# Check if data file exists
DATA_PATH="/home/ekourb/tf/datasets_chr1_1000bp/ctcf_chr1_dataset_struct.npz"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    echo "Available files in directory:"
    ls -la /home/ekourb/tf/datasets_chr1_1000bp/
    exit 1
fi

echo -e "\nData file check:"
echo "Data path: $DATA_PATH"
echo "File size: $(du -h $DATA_PATH | cut -f1)"

# GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo -e "\nGPU information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

echo "========================================="
echo "Starting CNN training on 1000bp CTCF sequences..."
echo "Dataset: $(basename $DATA_PATH)"
echo "========================================="

# Set CUDA environment variables for better performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Run the training script
python train_cnn.py \
    --data-path "$DATA_PATH" \
    --output-prefix "cnn_ctcf_1000bp" \
    --use-modisco true

# Capture exit status
EXIT_STATUS=$?

echo -e "\n========================================="
echo "Job finished at: $(date)"
echo "Exit status: $EXIT_STATUS"

if [[ $EXIT_STATUS -eq 0 ]]; then
    echo -e "\n✓ Training completed successfully!"
    echo -e "\nGenerated files:"
    ls -la cnn_ctcf_1000bp_*
else
    echo -e "\n✗ Training failed with exit status $EXIT_STATUS"
    echo "Check the error log for details."
fi

echo "========================================="