#!/bin/bash
#SBATCH --job-name=street
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

module purge
module load anaconda3/2023.3
conda activate hypll

# Array of temperatures


# Run the main script with temperature argument
kernprof -l train_streets.py --config_file configs/streets.yaml