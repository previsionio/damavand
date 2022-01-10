
use crate::qubit_backend::circuit::Circuit;
use crate::utils::ApplyMethod;

use num::complex::Complex;
use ndarray::prelude::*;

use mpi::topology::SystemCommunicator;
use mpi::traits::*;

impl Circuit {
    pub fn get_fidelity(
        &mut self,
        state_1: Array1<Complex<f64>>,
        state_2: Array1<Complex<f64>>
    ) -> f64 {

        let fidelity = match self.apply_method {
            ApplyMethod::BruteForce |
            ApplyMethod::Shuffle |
            ApplyMethod::Multithreading |
            ApplyMethod::GPU => {
                state_2.dot(&state_1.mapv(|a| a.conj())).norm_sqr()
            },
            ApplyMethod::DistributedCPU |
            ApplyMethod::DistributedGPU => {
                self.distributed_dot(&state_1, &state_2)
            }
        };
        self.reset();

        fidelity
    }

    pub fn distributed_dot(
        &self,
        state_1: &Array1<Complex<f64>>,
        state_2: &Array1<Complex<f64>>
    ) -> f64 {

        let world = SystemCommunicator::world();
        let num_nodes = world.size() as usize;

        let root_node_rank = 0;
        let root_node = world.process_at_rank(root_node_rank);

        let current_node_rank = world.rank();

        let local_fidelity = state_2.dot(&state_1.mapv(|a| a.conj()))

        let mut local_fidelities = vec![0_f64; num_nodes];

        let mut fidelity: f64 = 0.;

        if current_node_rank == root_node_rank {
            root_node.gather_into_root(&local_fidelity, &mut local_fidelities[..]);
            fidelity = local_fidelities.iter().sum();
        } else {
            root_node.gather_into(&local_fidelity);
        }

        root_node.broadcast_into(&mut fidelity);

        fidelity.norm_sqr()
    }
}
