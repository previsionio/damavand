#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

#include "utils.cuh"
#include "kernels.cuh"
#include "quantum_amplitudes.cuh"

QuantumAmplitudes::QuantumAmplitudes()
{
    occupancy_strategy = Automatic;
}

void
QuantumAmplitudes::set_zero_state(int num_amplitudes_per_gpu,
                                  bool is_first_gpu)
{

    checkCudaErrors(cudaMalloc((void **) &real_parts,
                               sizeof(double) * num_amplitudes_per_gpu));

    checkCudaErrors(cudaMalloc((void **) &imaginary_parts,
                               sizeof(double) * num_amplitudes_per_gpu));

    if(is_first_gpu)
    {

      auto parameters = get_launching_parameters(
          occupancy_strategy,
          num_amplitudes_per_gpu,
          init_zero_state_on_first_gpu);

#ifdef HAS_PROFILER
        sdkStartTimer(&init_kernel_timer);
#endif
        init_zero_state_on_first_gpu <<< parameters.grid_size, parameters.block_size >>>(
            num_amplitudes_per_gpu,
            real_parts,
            imaginary_parts);

        checkCudaErrors(cudaDeviceSynchronize());

#ifdef HAS_PROFILER
        sdkStopTimer(&init_kernel_timer);
#endif
    }
    else
    {

      auto parameters = get_launching_parameters(
          occupancy_strategy,
          num_amplitudes_per_gpu,
          init_zero_state_on_other_gpu);

#ifdef HAS_PROFILER
        sdkStartTimer(&init_kernel_timer);
#endif
        init_zero_state_on_other_gpu <<< parameters.grid_size, parameters.block_size >>>(
            num_amplitudes_per_gpu,
            real_parts,
            imaginary_parts);

        checkCudaErrors(cudaDeviceSynchronize());

#ifdef HAS_PROFILER
        sdkStopTimer(&init_kernel_timer);
#endif
    }
}

void
QuantumAmplitudes::load_on_device(
    int num_amplitudes_per_gpu,
    double *amplitudes_real,
    double *amplitudes_imaginary)
{
#ifdef HAS_PROFILER
    sdkStartTimer(&copy_host_to_device_timer);
#endif
    checkCudaErrors(cudaMemcpy(real_parts,
                               amplitudes_real,
                               sizeof(double) * num_amplitudes_per_gpu,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(imaginary_parts,
                               amplitudes_real,
                               sizeof(double) * num_amplitudes_per_gpu,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef HAS_PROFILER
    sdkStopTimer(&copy_host_to_device_timer);
#endif
}

void
QuantumAmplitudes::apply_one_qubit_gate(
    double *gate_matrix_real,
    double *gate_matrix_imaginary,
    int num_amplitudes_per_gpu,
    int control_qubit, int target_qubit)
{

    cuDoubleComplex gate_matrix[4];

    gate_matrix[0] =
        make_cuDoubleComplex(gate_matrix_real[0], gate_matrix_imaginary[0]);
    gate_matrix[1] =
        make_cuDoubleComplex(gate_matrix_real[1], gate_matrix_imaginary[1]);
    gate_matrix[2] =
        make_cuDoubleComplex(gate_matrix_real[2], gate_matrix_imaginary[2]);
    gate_matrix[3] =
        make_cuDoubleComplex(gate_matrix_real[3], gate_matrix_imaginary[3]);

    auto parameters = get_launching_parameters(
        occupancy_strategy,
        num_amplitudes_per_gpu,
        apply_one_qubit_gate_kernel_local);

#ifdef HAS_PROFILER
    sdkStartTimer(&apply_kernel_timer);
#endif

    apply_one_qubit_gate_kernel_local <<< parameters.grid_size, parameters.block_size >>>(
        num_amplitudes_per_gpu,
        control_qubit,
        target_qubit,
        gate_matrix[0],
        gate_matrix[1],
        gate_matrix[2],
        gate_matrix[3],
        real_parts,
        imaginary_parts);

    checkCudaErrors(cudaDeviceSynchronize());

#ifdef HAS_PROFILER
    sdkStopTimer(&apply_kernel_timer);
#endif

}

void
QuantumAmplitudes::apply_one_qubit_gate_distributed(
    QuantumAmplitudes partner_amplitudes,
    double *gate_matrix_real,
    double
    *gate_matrix_imaginary,
    int
    num_amplitudes_per_gpu,
    int control_qubit,
    int target_qubit)
{

    cuDoubleComplex gate_matrix[4];

    gate_matrix[0] =
        make_cuDoubleComplex(gate_matrix_real[0], gate_matrix_imaginary[0]);
    gate_matrix[1] =
        make_cuDoubleComplex(gate_matrix_real[1], gate_matrix_imaginary[1]);
    gate_matrix[2] =
        make_cuDoubleComplex(gate_matrix_real[2], gate_matrix_imaginary[2]);
    gate_matrix[3] =
        make_cuDoubleComplex(gate_matrix_real[3], gate_matrix_imaginary[3]);

    auto parameters = get_launching_parameters(
        occupancy_strategy,
        num_amplitudes_per_gpu,
        apply_one_qubit_gate_kernel_distributed);

#ifdef HAS_PROFILER
    sdkStartTimer(&apply_kernel_timer);
#endif
    apply_one_qubit_gate_kernel_distributed <<< parameters.grid_size, parameters.block_size >>>(
        num_amplitudes_per_gpu,
        control_qubit, target_qubit,
        gate_matrix[0],
        gate_matrix[1],
        gate_matrix[2],
        gate_matrix[3],
        real_parts,
        imaginary_parts,
        partner_amplitudes.real_parts,
        partner_amplitudes.imaginary_parts);

    checkCudaErrors(cudaDeviceSynchronize());

#ifdef HAS_PROFILER
    sdkStopTimer(&apply_kernel_timer);
#endif

}

void
QuantumAmplitudes::measure(
    int num_amplitudes_per_gpu,
    int first_amplitudes_index,
    double* device_probabilities,
    cudaStream_t stream)
{
#ifdef USE_SHARED_MEMORY
    auto parameters = get_launching_parameters(
        occupancy_strategy,
        num_amplitudes_per_gpu,
        measure_amplitudes_on_device_shared);
#else
    auto parameters = get_launching_parameters(
        occupancy_strategy,
        num_amplitudes_per_gpu,
        measure_amplitudes_on_device_global);
#endif

#ifdef HAS_PROFILER
    sdkStartTimer(&measure_kernel_timer);
#endif

#ifdef USE_SHARED_MEMORY
    measure_amplitudes_on_device_shared <<< parameters.grid_size, parameters.block_size, 0, stream>>>(
        num_amplitudes_per_gpu,
        first_amplitudes_index,
        device_probabilities,
        real_parts,
        imaginary_parts);
#else
    measure_amplitudes_on_device_global <<< parameters.block_size, parameters.block_size, 0, stream>>>(
        num_amplitudes_per_gpu,
        first_amplitudes_index,
        device_probabilities,
        real_parts,
        imaginary_parts);
#endif

    checkCudaErrors(cudaDeviceSynchronize());

#ifdef HAS_PROFILER
    sdkStopTimer(&measure_kernel_timer);
#endif
}
