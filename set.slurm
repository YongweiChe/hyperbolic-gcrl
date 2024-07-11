#!/bin/bash
#SBATCH --job-name=island_euc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

module purge
module load anaconda3/2023.3
conda activate hypll

# Run the main script with temperature argument
kernprof -l train_set.py --config_file configs/set_hyperbolic.yaml --temperature 0.1