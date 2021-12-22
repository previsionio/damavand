---
sidebarDepth: 2
---

# High Performance Computing

<p align="center">
  <img src="/hpc_logo.png" width="300em" />
</p>

**Damavand** is written to support computations on conventional laptops or HPC architectures.
Here, we describe the distributed implementation and provide with some scripts to start with.


## Distributed CPU

First, let us pass through the distributed CPU implementation. Consider that you have access to a supercomputer, or an
architecture that supports multiple nodes. One advantage of this is that quantum circuit simulations are **memory
bound**. This means that the computation bottleneck does not reside on the operations - that are pretty simple - but
rather on how to store the excessive amount of memory required to simulate a quantum state.

As shown in the illustration below, condier that the state vector can be splitted into a fixed number N. It is then
straightforward to assign each chunk of the state vector to a different node.

![Distributed GPU](/damavand_cpu_distributed.png)

Each node is built with a certain number of processors, thus allowing to run multithreaded experiments. However,
referring to the multithreading implementation presented in Guide, we can infer that some communications will be
required between distant nodes to share a part of the state vector. Indeed, the stride between two amplitudes is
necessarily of length 2^k, where k is the target qubit on which one wants to perform some operation.

Damavand implements an MPI scheme in order to share this information across multiple nodes.

## Distributed GPU

Another possibility is to leverage architectures - mainly built to support Artificial Intelligence applications -
composed of multiple nodes, each containing multiple GPUs. As seen in the Guide section, the single GPU implementation
is almost straightforward, and although some optimizations could be further thought of, it is reasonable to consider the
multi GPU implementation. We focus on [CUDA](https://developer.nvidia.com/cuda-zone) capable devices, thus restraining
the developments to **Nvidia** hardware.

In the case of a supercomputer, the GPUs on the same node will often allow direct inter device communications through
high thoughput NVLinks.

![Distributed GPU](/damavand_gpu_distributed.png)

The illustration above shows that memory can be loaded on from the CPUs to the GPUs. We choose to treat inter-node
communications with MPI ([OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/)).

## Slurm scripts

```bash
#!/bin/bash

$SBATCH --job-name=multi_gpu_mpi_cuda-aware     # nom du job
$SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
$SBATCH --ntasks=8                   # nombre total de tache MPI
$SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
$SBATCH --gres=gpu:4                 # nombre de GPU par noeud (max 8 avec gpu_p2)
$SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
$SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
$SBATCH --hint=nomultithread         # hyperthreading desactive
$SBATCH --time=00:10:00              # temps d'execution maximum demande (HH:MM:SS)
$SBATCH --output=multi_gpu_mpi%j.out # nom du fichier de sortie
$SBATCH --error=multi_gpu_mpi%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load ...
 
# echo des commandes lancees
set -x
 
# execution du code
srun python distributed_cpu.py
```
## Experimentations on Jean-Zay

<p align="center">
  <img src="/jean-zay.jpg" width="600em" />
</p>
