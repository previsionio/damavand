use crate::qubit_backend::circuit::Circuit;
use mpi::topology::SystemCommunicator;
use mpi::traits::*;

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
    /// circuit.apply_distributed_cpu(0);
    /// ```
    pub fn apply_distributed_cpu(&mut self, gate_index: usize) {
        let world = SystemCommunicator::world();

        let gate = self.gates[gate_index].lock().unwrap();

        let target_qubit = gate.get_target_qubit() as usize;
        drop(gate);

        let num_nodes = world.size() as usize;
        let num_amplitudes = 1_usize << self.num_qubits;
        let amplitude_gap = 1_usize << target_qubit;
        let num_amplitudes_per_node = num_amplitudes / num_nodes;

        // amplitudes fit in single node
        if amplitude_gap < num_amplitudes_per_node {
            self.apply_multithreading_local(gate_index);
        // amplitudes are positioned in distant nodes, communication required
        } else {
            let current_node_rank = world.rank() as usize;

            let partner_node_rank = Circuit::compute_partner_rank(
                current_node_rank,
                num_amplitudes_per_node,
                amplitude_gap,
            );

            self.exchange_amplitudes_between_nodes(partner_node_rank);

            self.apply_multithreading_distributed(gate_index, current_node_rank, partner_node_rank);
        }
    }
}
