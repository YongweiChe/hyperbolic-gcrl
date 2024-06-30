#!/bin/bash
#SBATCH --job-name=embed_dim_exp  # create a short name for your job
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=2         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G          # memory per cpu-core (4G is default)
#SBATCH --time=02:05:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin         # send email when job begins
#SBATCH --mail-type=end           # send email when job ends
#SBATCH --mail-user=yongweic@princeton.edu
#SBATCH --array=0-6               # job array with index values 0 to 6

module purge
module load anaconda3/2023.3
conda activate hypll

# Define an array of embedding dimensions
embedding_dims=(2 4 8 16 32 64 128)

# Use the SLURM_ARRAY_TASK_ID to select the embedding dimension
embedding_dim=${embedding_dims[$SLURM_ARRAY_TASK_ID]}

python train.py --project "geometric" --embedding_dim $embedding_dim --num_epochs=256 --maze_type "slit" --gamma 0.1