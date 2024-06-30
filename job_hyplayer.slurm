#!/bin/bash
#SBATCH --job-name=gpu_array    # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --mem-per-cpu=8G        # memory per cpu-core (4G is default)
#SBATCH --time=08:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=yongweic@princeton.edu
#SBATCH --array=1-3             # array job with index from 1 to 3

module purge
module load anaconda3/2023.3
conda activate hypll

# Use the SLURM_ARRAY_TASK_ID as the number of hyperbolic layers
hyp_layers=${SLURM_ARRAY_TASK_ID}

python train.py --project "hyplayers" --hyperbolic True --hyp_layers $hyp_layers --num_trajectories 10000 --batch_size 128 --embedding_dim 8 --num_epochs=128 --maze_type "pyramid" --gamma 0.01 --num_workers 8 --max_steps 100