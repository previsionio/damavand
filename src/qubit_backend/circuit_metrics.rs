
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
                state_1.mapv(|a| a.conj()).dot(&state_2).norm_sqr()
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

        let local_fidelity = state_1.mapv(|a| a.conj()).dot(&state_2);
        let local_fidelity_real = local_fidelity.re;
        let local_fidelity_imaginary = local_fidelity.im;

        let mut local_fidelities_real = vec![0_f64; num_nodes];
        let mut local_fidelities_imaginary = vec![0_f64; num_nodes];

        if current_node_rank == root_node_rank {
            root_node.gather_into_root(
                &local_fidelity_real,
                &mut local_fidelities_real[..]);
        } else {
            root_node.gather_into(&local_fidelity_real);
        }

        if current_node_rank == root_node_rank {
            root_node.gather_into_root(
                &local_fidelity_imaginary,
                &mut local_fidelities_imaginary[..]);
        } else {
            root_node.gather_into(&local_fidelity_imaginary);
        }


        let fidelity_helper = if current_node_rank == root_node_rank {
            let mut fidelity_helper = Complex::<f64>{re: 0., im: 0.};
            for i in 0..num_nodes {
                fidelity_helper += Complex::<f64>{
                    re: local_fidelities_real[i],
                    im: local_fidelities_imaginary[i]
                };
            }
            fidelity_helper
        } else {
            Complex::<f64>{re: 0., im: 0.}
        };

        let mut fidelity = fidelity_helper.norm_sqr();

        root_node.broadcast_into(&mut fidelity);

        fidelity

    }
}
