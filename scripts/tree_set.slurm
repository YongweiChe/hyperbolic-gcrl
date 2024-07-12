#!/bin/bash
#SBATCH --job-name=set
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

module purge
module load anaconda3/2023.3
conda activate hypll

# Run the main script with temperature argument
python train_tree_set.py --config_file configs/treeset.yaml