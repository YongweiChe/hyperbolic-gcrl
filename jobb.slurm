#!/bin/bash
#SBATCH --job-name=gpu          # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yongweic@princeton.edu

module purge
module load anaconda3/2023.3
conda activate hypll

python train.py --project "nested_pyramid2" --hyperbolic True --num_trajectories 100000 --batch_size 128 --embedding_dim 8 --num_epochs=1024 --maze_type "nested_pyramid" --gamma 0.01 --num_workers 8 --max_steps 500