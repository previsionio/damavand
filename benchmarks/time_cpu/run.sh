#!/bin/bash

#SBATCH --job-name=two_nodes
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=benchmarks.listing
#SBATCH --time=10:00

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/gpfswork/rech/rmq/ubt65ti/damavand_genci/damavand-gpu/build"

module purge

# module load hpctoolkit/2020.08.03
module load openmpi/4.1.1
module load cuda/10.2
# module load gcc/8.3.1
# module load intel-vtune/2020.3

srun python3 benchmarks.py
# srun hpcrun python3 benchmarks.py
