#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=nn_hjb
#SBATCH --output=nn_hjb.out
#SBATCH --error=nn_hjb.err
#SBATCH --time=0-1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

container=/apps/containers/tensorflow-gpu/tensorflow-2.4.3-gpu-jupyter.sif

srun singularity run --nv ${container} python3 BFGS.py