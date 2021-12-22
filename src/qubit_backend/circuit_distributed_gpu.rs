use crate::qubit_backend::circuit::Circuit;

use mpi::topology::SystemCommunicator;
use mpi::traits::*;

#[link(name = "damavand-gpu", kind = "static")]
extern "C" {
    // fn load_amplitudes_local_on_device(
    //     local_amplitudes_real: *mut libc::c_double,
    //     local_amplitudes_imaginary: *mut libc::c_double,
    // ) -> libc::c_void;

    // fn load_amplitudes_distributed_on_device(
    //     local_amplitudes_real: *mut libc::c_double,
    //     local_amplitudes_imaginary: *mut libc::c_double,
    //     partner_amplitudes_real: *mut libc::c_double,
    //     partner_amplitudes_imaginary: *mut libc::c_double,
    // ) -> libc::c_void;

    fn split_amplitudes_between_gpus(
        num_amlpitudes_per_gpu: libc::c_int,
        local_amplitudes_real: *mut libc::c_double,
        local_amplitudes_imaginary: *mut libc::c_double,
        partner_amplitudes_real: *mut libc::c_double,
        partner_amplitudes_imaginary: *mut libc::c_double,
    );

    fn exchange_amplitudes_between_gpus(
        current_gpu_rank: libc::c_int,
        partner_gpu_rank: libc::c_int,
    );
}

impl Circuit {
    /// Applies distributed kernel across all nodes
    ///
    /// # Attributes
    /// * `gate_index` index of the gate that needs to be applied.
    ///
    /// # Examples
    /// ```
    /// use damavand::qubit_backend::circuit::Circuit;
    ///
    /// let num_qubits = 3;
    /// let circuit = Circuit::new(num_qubits);
    ///
    /// circuit.apply_distributed(0);
    /// ```
    pub fn apply_distributed_gpu(&mut self, gate_index: usize) {
        let world = SystemCommunicator::world();

        let gate = self.gates[gate_index].lock().unwrap();

        let target_qubit = gate.get_target_qubit() as usize;
        drop(gate);

        let num_nodes = world.size() as usize;
        let num_amplitudes = 1_usize << self.num_qubits;
        let amplitude_gap = 1_usize << target_qubit;
        let num_amplitudes_per_node = num_amplitudes / num_nodes;
        let num_amplitudes_per_gpu = num_amplitudes_per_node / self.num_gpus_per_node;

        // amplitudes fit in single gpu
        if amplitude_gap < num_amplitudes_per_gpu {
            self.apply_gpu_local(gate_index);
        // amplitudes fit in single node, communication between gpus required
        } else if num_amplitudes_per_gpu <= amplitude_gap && amplitude_gap < num_amplitudes_per_node
        {
            for local_gpu_rank in 0..self.num_gpus_per_node {
                let partner_gpu_rank = Circuit::compute_partner_rank(
                    local_gpu_rank,
                    num_amplitudes_per_gpu,
                    amplitude_gap,
                );

                if local_gpu_rank < partner_gpu_rank {
                    unsafe {
                        exchange_amplitudes_between_gpus(
                            local_gpu_rank as i32,
                            partner_gpu_rank as i32,
                        );
                    }
                }
            }

            self.apply_gpu_distributed(gate_index);

        // amplitudes are positioned in distant nodes, communication between nodes required
        } else {
            // iterate on all nodes and assign task to each of them
            let current_node_rank = world.rank() as usize;

            // loop on all gpus per node
            for local_gpu_rank in 0..self.num_gpus_per_node {
                // compute global gpu rank
                let global_gpu_rank = current_node_rank * self.num_gpus_per_node + local_gpu_rank;

                // compute global gpu partner rank
                let global_gpu_partner_rank = Circuit::compute_partner_rank(
                    global_gpu_rank,
                    num_amplitudes_per_gpu,
                    amplitude_gap,
                );

                // retrieve partner node rank
                let partner_gpu_rank = global_gpu_partner_rank / self.num_gpus_per_node;

                self.retrieve_amplitudes_on_host();

                self.exchange_amplitudes_between_nodes(partner_gpu_rank);

                // exchange amplitudes between nodes
                unsafe {
                    split_amplitudes_between_gpus(
                        self.num_amplitudes_per_gpu as i32,
                        self.local_amplitudes
                            .iter()
                            .map(|a| a.re)
                            .collect::<Vec<f64>>()
                            .as_mut_ptr(),
                        self.local_amplitudes
                            .iter()
                            .map(|a| a.im)
                            .collect::<Vec<f64>>()
                            .as_mut_ptr(),
                        self.partner_amplitudes
                            .iter()
                            .map(|a| a.re)
                            .collect::<Vec<f64>>()
                            .as_mut_ptr(),
                        self.partner_amplitudes
                            .iter()
                            .map(|a| a.im)
                            .collect::<Vec<f64>>()
                            .as_mut_ptr(),
                    );
                }

                self.apply_gpu_distributed(gate_index);
            }
        }
    }
}
