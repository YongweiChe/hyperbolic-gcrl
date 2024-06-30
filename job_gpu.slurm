#!/bin/bash
#SBATCH --job-name=gpu          # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yongweic@princeton.edu
#SBATCH --array=1-4              # This line creates a job array with indices 1 to 4

module purge
module load anaconda3/2023.3
conda activate hypll

python train.py --project "exp_vs_hyp" --custom "$SLURM_ARRAY_TASK_ID" --hyperbolic True --embedding_dim 16 --num_epochs=256 --maze_type "pyramid" --gamma 0.01 --num_workers 8