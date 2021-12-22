#!/bin/bash

#SBATCH --job-name=sample_benchmark
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --output=sample_benchmark.listing
#SBATCH --time=00:45:00

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/gpfswork/rech/rmq/ubt65ti/damavand_genci/damavand-gpu/build"

module purge

module load openmpi/4.1.1
module load cuda/10.2

export RAYON_NUM_THREADS=40
srun python3 sample_benchmarks.py
