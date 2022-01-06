use crate::qubit_backend::circuit::Circuit;

#[link(name = "damavand-gpu", kind = "static")]
extern "C" {

    fn apply_one_qubit_gate_gpu_local(
        gate_matrix_real: *mut libc::c_double,
        gate_matrix_imaginary: *mut libc::c_double,
        num_qubits: libc::c_int,
        num_amplitudes_per_gpu: libc::c_int,
        control_qubit: libc::c_int,
        target_qubit: libc::c_int,
    ) -> libc::c_void;

    fn apply_one_qubit_gate_gpu_distributed(
        gate_matrix_real: *mut libc::c_double,
        gate_matrix_imaginary: *mut libc::c_double,
        num_qubits: libc::c_int,
        num_amplitudes_per_gpu: libc::c_int,
        control_qubit: libc::c_int,
        target_qubit: libc::c_int,
    ) -> libc::c_void;

}

impl Circuit {
    /// Applies local kernel on GPU
    ///
    /// # Attributes
    /// * `gate_index` index of the gate that needs to be applied.
    pub fn apply_gpu_local(&mut self, gate_index: usize) {
        let gate = self.gates[gate_index].lock().unwrap();
        let target_qubit = gate.get_target_qubit();

        let control_qubit = gate.get_control_qubit();
        let control = if control_qubit.is_some() {
            control_qubit.unwrap() as i32
        } else {
            -1_i32
        };

        unsafe {
            apply_one_qubit_gate_gpu_local(
                gate.get_matrix()
                    .iter_mut()
                    .map(|a| a.re)
                    .collect::<Vec<f64>>()
                    .as_mut_ptr(),
                gate.get_matrix()
                    .iter_mut()
                    .map(|a| a.im)
                    .collect::<Vec<f64>>()
                    .as_mut_ptr(),
                self.num_qubits as i32,
                self.num_amplitudes_per_gpu as i32,
                control,
                target_qubit as i32,
            )
        };
    }
    /// Applies local kernel on GPU for multiple GPUs runs.
    ///
    /// # Attributes
    /// * `gate_index` index of the gate that needs to be applied.
   pub fn apply_gpu_distributed(&mut self, gate_index: usize) {
        let gate = self.gates[gate_index].lock().unwrap();
        let target_qubit = gate.get_target_qubit();

        let control_qubit = gate.get_control_qubit();
        let control = if control_qubit.is_some() {
            control_qubit.unwrap() as i32
        } else {
            -1_i32
        };

        unsafe {
            apply_one_qubit_gate_gpu_distributed(
                gate.get_matrix()
                    .iter()
                    .map(|a| a.re)
                    .collect::<Vec<f64>>()
                    .as_mut_ptr(),
                gate.get_matrix()
                    .iter()
                    .map(|a| a.im)
                    .collect::<Vec<f64>>()
                    .as_mut_ptr(),
                self.num_qubits as i32,
                self.num_amplitudes_per_gpu as i32,
                control,
                target_qubit as i32,
            )
        };
    }
}
