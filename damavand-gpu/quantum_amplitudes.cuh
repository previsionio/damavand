#ifndef QUANTUM_AMPLITUDES_CUH
#define QUANTUM_AMPLITUDES_CUH
#include "utils.cuh"

class QuantumAmplitudes
{
public:
    double *real_parts;
    double *imaginary_parts;
    OccupancyStrategy occupancy_strategy;

public:
    QuantumAmplitudes();
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

    void measure(
        int num_amplitudes_per_gpu,
        int first_amplitudes_index,
        double* device_probabilities,
        cudaStream_t stream);

};

#endif
