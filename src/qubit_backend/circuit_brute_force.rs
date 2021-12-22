use crate::qubit_backend::circuit::Circuit;
use crate::utils;

impl Circuit {
    pub fn apply_brute_force(&mut self, gate_index: usize) {
        let operation = utils::convert_gate_to_operation(self.num_qubits, &self.gates[gate_index]);
        self.local_amplitudes = operation.dot(&self.local_amplitudes);
    }
}
