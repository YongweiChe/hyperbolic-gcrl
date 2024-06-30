#!/bin/bash
#SBATCH --job-name=gamma_exp     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --time=04:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yongweic@princeton.edu
#SBATCH --array=0-3              # job array with index values 0, 1, 2, 3

module purge
module load anaconda3/2023.3
conda activate hypll

# Define an array of gamma values
gamma_values=(0.01 0.1 0.25 0.5)

# Use the SLURM_ARRAY_TASK_ID to select the gamma value
gamma=${gamma_values[$SLURM_ARRAY_TASK_ID]}

python train.py --project "gamma_sweep" --embedding_dim 16 --num_epochs=256 --maze_type "pyramid" --gamma $gamma