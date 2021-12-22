#!/bin/bash

#SBATCH --job-name=single_node
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=single_node.listing
#SBATCH --time=5:00

export LD_LIBRARY_PATH="/gpfswork/rech/rmq/ubt65ti/damavand_genci/damavand-gpu/build/"

module purge

module load openmpi/3.1.4
module load cuda/10.2

srun python3 single_node.py
