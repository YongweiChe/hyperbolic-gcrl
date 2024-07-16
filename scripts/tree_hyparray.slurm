#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --array=1-6  # Adjust based on total number of combinations and desired parallelism

module purge
module load anaconda3/2023.3
conda activate hypll

# Base config file
CONFIG_FILE="configs/tree_hyp.yaml"

# Read parameters for this job
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/hyp_combinations.txt)
read -r HYPERBOLIC DEPTH EMBEDDING_DIM <<< "$PARAMS"

# Run the Python script with parameters
python train_tree.py --config_file $CONFIG_FILE \
                     --hyperbolic $HYPERBOLIC \
                     --depth $DEPTH \
                     --embedding_dim $EMBEDDING_DIM

# Optionally, you can save results to unique files
# RESULTS_DIR="results"
# mkdir -p $RESULTS_DIR
# RESULTS_FILE="${RESULTS_DIR}/results_${SLURM_ARRAY_TASK_ID}.txt"
# python train_tree.py --config_file $CONFIG_FILE \
#                      --hyperbolic $HYPERBOLIC \
#                      --depth $DEPTH \
#                      --embedding_dim $EMBEDDING_DIM \
#                      > $RESULTS_FILE 2>&1