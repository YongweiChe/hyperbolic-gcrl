#!/bin/bash
#SBATCH --job-name=long          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --time=04:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yongweic@princeton.edu
#SBATCH --array=1-4              # job array with index values 1-4

module purge
module load anaconda3/2023.3
conda activate hypll

python train.py --project "blocker" --custom $SLURM_ARRAY_TASK_ID --embedding_dim 16 --num_epochs=256 --maze_type "blocker" --gamma 0.1