#!/bin/bash
#SBATCH --job-name=nested_pyramid          # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yongweic@princeton.edu
#SBATCH --array=0-9              # This line creates a job array with indices 0 to 9

module purge
module load anaconda3/2023.3
conda activate hypll

# Define arrays for embedding_dim and gamma
embedding_dims=(8 16 32 64 128)
gammas=(0.01 0.001)

# Calculate indices for embedding_dim and gamma
embedding_dim_index=$((SLURM_ARRAY_TASK_ID / 2))
gamma_index=$((SLURM_ARRAY_TASK_ID % 2))

# Get the actual values
embedding_dim=${embedding_dims[$embedding_dim_index]}
gamma=${gammas[$gamma_index]}

python train.py --project "nested_pyramid" --num_trajectories 10000 --batch_size 128 --embedding_dim $embedding_dim --num_epochs=512 --maze_type "nested_pyramid" --gamma $gamma --num_workers 8