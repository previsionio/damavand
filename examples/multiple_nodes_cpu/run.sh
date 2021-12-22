#!/bin/bash

#SBATCH --job-name=two_nodes
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=two_nodes.listing
#SBATCH --time=5:00

export LD_LIBRARY_PATH="/gpfswork/rech/rmq/ubt65ti/damavand_genci/damavand-gpu/build/:$LD_LIBRARY_PATH"

module purge

module load openmpi/3.1.4
module load cuda/10.2

srun python3 two_nodes.py
