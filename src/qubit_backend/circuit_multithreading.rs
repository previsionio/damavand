use crate::qubit_backend::circuit::Circuit;
use num::complex::Complex;

impl Circuit {
    /// Apply multithreading method on a run that contains a single node.
    ///
    /// #Arguments
    /// `gate_index` the index of the gate to be applied on the quantum state.
    pub fn apply_multithreading_local(&mut self, gate_index: usize) {
        let gate = self.gates[gate_index].lock().unwrap();

        let target_qubit = gate.get_target_qubit();
        let control_qubit = gate.get_control_qubit();
        let (gm_00, gm_01, gm_10, gm_11) = if gate.get_name() == "CNOT" {
            (
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 0., im: 0. },
            )
        } else {
            (
                gate.get_matrix()[[0, 0]],
                gate.get_matrix()[[0, 1]],
                gate.get_matrix()[[1, 0]],
                gate.get_matrix()[[1, 1]],
            )
        };
        drop(gate);

        let partner_amplitudes = self.local_amplitudes.clone();

        ndarray::Zip::indexed(&mut self.local_amplitudes).par_for_each(
            |amplitude_index, local_amplitude| {
                let apply_gate = if control_qubit.is_some() {
                    amplitude_index & (1 << control_qubit.unwrap())
                } else {
                    1_usize
                };
                let partner_amplitude_index =
                    Circuit::compute_partner_rank(amplitude_index, 1, 1 << target_qubit);

                if apply_gate == 1_usize {
                    let lower_amplitude = local_amplitude.clone();
                    let upper_amplitude = partner_amplitudes[partner_amplitude_index].clone();

                    if partner_amplitude_index > amplitude_index {
                        *local_amplitude = lower_amplitude * gm_00 + upper_amplitude * gm_01;
                    } else {
                        *local_amplitude = upper_amplitude * gm_10 + lower_amplitude * gm_11;
                    }
                }
            },
        );
    }
    /// Apply multithreading method on a run that contains a multiple nodes.
    ///
    /// #Arguments
    /// `gate_index`: the index of the gate to be applied on the quantum state.
    /// `current_node_rank`: the rank of the current node.
    /// `partner_node_rank`: the rank of the partner node.
    pub fn apply_multithreading_distributed(
        &mut self,
        gate_index: usize,
        current_node_rank: usize,
        partner_node_rank: usize,
    ) {
        let gate = self.gates[gate_index].lock().unwrap();

        let control_qubit = gate.get_control_qubit();
        let (gm_00, gm_01, gm_10, gm_11) = if gate.get_name() == "CNOT" {
            (
                Complex::<f64> { re: 0., im: 0. },
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 1., im: 0. },
                Complex::<f64> { re: 0., im: 0. },
            )
        } else {
            (
                gate.get_matrix()[[0, 0]],
                gate.get_matrix()[[0, 1]],
                gate.get_matrix()[[1, 0]],
                gate.get_matrix()[[1, 1]],
            )
        };
        drop(gate);

        ndarray::Zip::indexed(&mut self.local_amplitudes)
            .and(&self.partner_amplitudes)
            .par_for_each(|amplitude_index, local_amplitude, partner_amplitude| {
                let apply_gate = if control_qubit.is_some() {
                    amplitude_index & (1 << control_qubit.unwrap())
                } else {
                    1_usize
                };

                if apply_gate == 1_usize {
                    if partner_node_rank > current_node_rank {
                        *local_amplitude = *local_amplitude * gm_00 + partner_amplitude * gm_01;
                    } else {
                        *local_amplitude = partner_amplitude * gm_10 + *local_amplitude * gm_11;
                    }
                }
            });
    }
}
