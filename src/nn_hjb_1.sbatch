#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=nn_hjb_1
#SBATCH --output=nn_hjb_1.out
#SBATCH --error=nn_hjb_1.err
#SBATCH --time=0-1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 fd_BFGS_1.py
srun python3 fd_Plots_1.py