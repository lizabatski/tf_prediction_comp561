#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --account=def-majewski     
#SBATCH --time=12:00:00            
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=prep-%j.out
#SBATCH --error=prep-%j.err

echo "Starting preprocessing..."
module load python/3.10
source ~/envs/tf/bin/activate   # <-- your environment

python prepare_data.py
