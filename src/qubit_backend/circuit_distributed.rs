use crate::qubit_backend::circuit::Circuit;
use crate::utils;
use num::complex::Complex;

use mpi::topology::SystemCommunicator;
use mpi::traits::*;

#[cfg(feature = "gpu")]
#[link(name = "damavand-gpu", kind = "static")]
extern "C" {

    fn exchange_amplitudes_between_gpus(
        current_gpu_rank: libc::c_int,
        partner_gpu_rank: libc::c_int,
    );
}

impl Circuit {
    /// Exchanges amplitudes between current and partner node
    ///
    /// # Attributes
    /// * `partner_node_rank` rank of node treated by partner process
    pub fn exchange_amplitudes_between_nodes(&mut self, partner_node_rank: usize) {
        let world = SystemCommunicator::world();

        // retrieve partner node
        let partner_node = world.process_at_rank(partner_node_rank as i32);

        // split state into real and imaginary amplitudes
        let local_amplitudes_real: Vec<f64> = self.local_amplitudes.iter().map(|a| a.re).collect();

        let local_amplitudes_imaginary: Vec<f64> =
            self.local_amplitudes.iter().map(|a| a.im).collect();

        // send real part of local amplitudes
        partner_node.send(&local_amplitudes_real[..]);
        let (partner_amplitudes_real, _) = partner_node.receive_vec::<f64>();

        // send imaginary part of local amplitudes
        partner_node.send(&local_amplitudes_imaginary[..]);
        let (partner_amplitudes_imaginary, _) = partner_node.receive_vec::<f64>();

        self.partner_amplitudes = partner_amplitudes_real
            .iter()
            .zip(&partner_amplitudes_imaginary)
            .map(|(re, im)| Complex::<f64> { re: *re, im: *im })
            .collect();
    }

    /// Exchanges amplitudes between two partner gpus on the same node
    ///
    /// # Attributes
    /// * `current_node_rank` rank of node treated by current process
    /// * `partner_node_rank` rank of node treated by partner process
    #[cfg(feature = "gpu")]
    pub fn exchange_amplitudes_between_gpus(
        &mut self,
        current_gpu_rank: usize,
        partner_gpu_rank: usize,
    ) {
        unsafe {
            exchange_amplitudes_between_gpus(current_gpu_rank as i32, partner_gpu_rank as i32);
        }
    }

    /// 
    pub fn sample_distributed(
        &mut self,
        num_samples: usize,
        node_probabilities: Vec<f64>,
    ) -> Vec<usize> {
        let num_observables = self.observables.len();
        let mut samples = vec![0_usize; num_samples];

        let world = SystemCommunicator::world();

        world.barrier();

        // first, compute cumulated distribution locally on all nodes
        let mut cumulative_distribution = vec![0.];
        for probability in &node_probabilities {
            let value = cumulative_distribution.last().unwrap() + probability;
            cumulative_distribution.push(value);
        }

        let num_nodes = world.size() as usize;
        let root_node_rank = 0;
        let current_node_rank = world.rank();
        let mut sampled_node_ranks = vec![0_usize; num_samples];
        let num_amplitudes_per_node = self.local_amplitudes.len();

        let root_node = world.process_at_rank(root_node_rank);

        // gather cumulative distributions per node at root node
        let mut node_cumulated_probabilities = vec![0_f64; num_nodes];

        world.barrier();

        if current_node_rank == root_node_rank {
            root_node.gather_into_root(
                cumulative_distribution.last().unwrap(),
                &mut node_cumulated_probabilities[..],
            );
        } else {
            root_node.gather_into(cumulative_distribution.last().unwrap());
        }

        world.barrier();
        if current_node_rank == root_node_rank {
            for sample_index in 0..num_samples {
                let sampled_node_rank =
                    utils::sample_from_discrete_distribution(&node_cumulated_probabilities);
                if sampled_node_rank.is_some() {
                    sampled_node_ranks[sample_index] = sampled_node_rank.unwrap();
                }
            }
        }
        world.barrier();

        root_node.broadcast_into(&mut sampled_node_ranks[..]);

        // gather sampled indexes
        let mut sample_index = 0_usize;

        for sampled_node_rank in sampled_node_ranks {
            // sampled locally, no need for communication
            if current_node_rank == root_node_rank && current_node_rank == sampled_node_rank as i32
            {
                let amplitude_index =
                    utils::sample_from_discrete_distribution(&node_probabilities).unwrap();
                samples[sample_index] = amplitude_index;
            }

            // sampled on other nodes than root_node: need for communication
            if current_node_rank != root_node_rank && current_node_rank == sampled_node_rank as i32
            {
                let amplitude_index =
                    utils::sample_from_discrete_distribution(&node_probabilities).unwrap();
                root_node.send(
                    &(amplitude_index + num_amplitudes_per_node * current_node_rank as usize),
                );
            } else if current_node_rank == root_node_rank
                && current_node_rank != sampled_node_rank as i32
            {
                let (msg, _) = world.any_process().receive::<usize>();
                samples[sample_index] = msg;
            }
            sample_index += 1;
        }

        root_node.broadcast_into(&mut samples[..]);
        world.barrier();

        samples
    }
}
