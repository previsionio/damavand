#!/bin/bash

#SBATCH --job-name=single_node_single_gpu
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=single_node_single_gpu.listing
#SBATCH --time=00:05:00

export LD_LIBRARY_PATH="/gpfswork/rech/rmq/ubt65ti/damavand_genci/damavand-gpu/build/"

export RUST_BACKTRACE=1
module purge

module load openmpi/3.1.4
module load cuda/10.2

srun python3 single_node_single_gpu.py
