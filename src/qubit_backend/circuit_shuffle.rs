use crate::qubit_backend::circuit::Circuit;
use crate::utils;

impl Circuit {
    /// Apply Shuffle method
    ///
    /// #Arguments
    /// `gate_index` the index of the gate to be applied on the quantum state.
    pub fn apply_shuffle(&mut self, gate_index: usize) {
        let gate = self.gates[gate_index].lock().unwrap();
        if gate.get_control_qubit().is_some() {
            let (active_matrix_list, inactive_matrix_list) = utils::get_cnot_list_of_matrices(
                self.num_qubits as usize,
                gate.get_control_qubit().unwrap() as usize,
                gate.get_target_qubit() as usize,
            );
            let active_state = utils::kron_shuffle(&active_matrix_list, &self.local_amplitudes);
            let inactive_state = utils::kron_shuffle(&inactive_matrix_list, &self.local_amplitudes);
            self.local_amplitudes = active_state + inactive_state;
        } else {
            let matrix_list = utils::get_list_of_matrices(
                self.num_qubits as usize,
                gate.get_target_qubit() as usize,
                &gate.get_matrix(),
            );
            self.local_amplitudes = utils::kron_shuffle(&matrix_list, &self.local_amplitudes);
        }
    }
}
