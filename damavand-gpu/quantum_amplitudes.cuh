#ifndef QUANTUM_AMPLITUDES_H
#define QUANTUM_AMPLITUDES_H

class QuantumAmplitudes
{
public:
    double *real_parts;
    double *imaginary_parts;

public:
    void set_zero_state(int num_amplitudes_per_gpu, bool is_first_gpu);

    void load_on_device(
        int num_amplitudes_per_gpu,
        double *amplitudes_real,
        double *amplitudes_imaginary);

    void apply_one_qubit_gate(
        double *gate_matrix_real,
        double *gate_matrix_imaginary,
        int num_amplitudes_per_gpu,
        int control_qubit,
        int target_qubit);

    void apply_one_qubit_gate_distributed(
        QuantumAmplitudes partner_amplitudes,
        double *gate_matrix_real,
        double *gate_matrix_imaginary,
        int num_amplitudes_per_gpu,
        int control_qubit,
        int target_qubit);

    double *measure(int num_amplitudes_per_gpu);
};
#endif
