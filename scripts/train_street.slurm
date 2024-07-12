#!/bin/bash
#SBATCH --job-name=street
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Check if a config file was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch script.sh <path_to_yaml_config>"
    exit 1
fi

CONFIG_FILE=$1

module purge
module load anaconda3/2023.3
conda activate hypll

# Run the main script with the provided config file
kernprof -l train_streets.py --config_file $CONFIG_FILE