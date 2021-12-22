#!/bin/bash

#SBATCH --job-name=single_node_multiple_gpus
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --output=single_node_multiple_gpus.listing
#SBATCH --time=10:00

module purge

module load openmpi/4.1.1
module load cuda/10.2

srun cuda-memcheck python3 single_node_multiple_gpus.py
