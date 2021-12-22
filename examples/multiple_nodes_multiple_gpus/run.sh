#!/bin/bash

#SBATCH --job-name=single_node_multiple_gpus
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --output=multiple_nodes_multiple_gpus.listing
#SBATCH --time=10:00

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/gpfswork/rech/rmq/ubt65ti/damavand_genci/damavand-gpu/build"
export RUST_BACKTRACE=1

module purge

module load openmpi/4.1.1
module load cuda/10.2

srun python3 multiple_nodes_multiple_gpus.py
