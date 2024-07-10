#!/bin/bash
#SBATCH --job-name=island_euc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --array=0-1  # Adjust this range based on the number of temperatures you want to test

module purge
module load anaconda3/2023.3
conda activate hypll

# Array of temperatures
temperatures=(0.1 1.0)

# Run the main script with temperature argument
kernprof -l train_streets.py --config_file configs/config.yaml --temperature ${temperatures[$SLURM_ARRAY_TASK_ID]}