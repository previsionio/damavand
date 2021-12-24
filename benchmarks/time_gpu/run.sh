#!/bin/bash

#SBATCH --job-name=two_nodes
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=benchmarks.listing
#SBATCH --time=10:00

module purge

module load openmpi/4.1.1
module load cuda/10.2

srun python3 benchmarks.py
