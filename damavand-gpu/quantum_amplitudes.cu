#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <iostream>
#include <vector>

#include "quantum_amplitudes.cuh"
#include "kernels.cuh"

void
QuantumAmplitudes::set_zero_state(int num_amplitudes_per_gpu,
                                  bool is_first_gpu)
{

    int block_size;
    int min_grid_size;
    int grid_size;

    checkCudaErrors(cudaMalloc((void **) &real_parts,
                               sizeof(double) * num_amplitudes_per_gpu));

    checkCudaErrors(cudaMalloc((void **) &imaginary_parts,
                               sizeof(double) * num_amplitudes_per_gpu));

    if(is_first_gpu)
    {
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            init_zero_state_on_first_gpu,
            0,
            num_amplitudes_per_gpu);

        grid_size = (num_amplitudes_per_gpu + block_size - 1) / block_size;

        init_zero_state_on_first_gpu <<< grid_size, block_size >>>(
            num_amplitudes_per_gpu,
            real_parts,
            imaginary_parts);
    }
    else
    {
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            init_zero_state_on_other_gpu,
            0,
            num_amplitudes_per_gpu);

        grid_size = (num_amplitudes_per_gpu + block_size - 1) / block_size;
        init_zero_state_on_other_gpu <<< grid_size, block_size >>>(
            num_amplitudes_per_gpu,
            real_parts,
            imaginary_parts);
    }
    checkCudaErrors(cudaDeviceSynchronize());
}

void
QuantumAmplitudes::load_on_device(
    int num_amplitudes_per_gpu,
    double *amplitudes_real,
    double *amplitudes_imaginary)
{
    checkCudaErrors(cudaMemcpy(real_parts,
                               amplitudes_real,
                               sizeof(double) * num_amplitudes_per_gpu,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(imaginary_parts,
                               amplitudes_real,
                               sizeof(double) * num_amplitudes_per_gpu,
                               cudaMemcpyHostToDevice));
}

void
QuantumAmplitudes::apply_one_qubit_gate(
    double *gate_matrix_real,
    double *gate_matrix_imaginary,
    int num_amplitudes_per_gpu,
    int control_qubit, int target_qubit)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    cuDoubleComplex gate_matrix[4];

    gate_matrix[0] =
        make_cuDoubleComplex(gate_matrix_real[0], gate_matrix_imaginary[0]);
    gate_matrix[1] =
        make_cuDoubleComplex(gate_matrix_real[1], gate_matrix_imaginary[1]);
    gate_matrix[2] =
        make_cuDoubleComplex(gate_matrix_real[2], gate_matrix_imaginary[2]);
    gate_matrix[3] =
        make_cuDoubleComplex(gate_matrix_real[3], gate_matrix_imaginary[3]);

    // get optimized block_size
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        apply_one_qubit_gate_kernel_local,
        0,
        num_amplitudes_per_gpu);

    grid_size =(num_amplitudes_per_gpu + block_size - 1) / block_size;

    apply_one_qubit_gate_kernel_local <<< grid_size, block_size >>>(
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
    int block_size;
    int min_grid_size;
    int grid_size;

    cuDoubleComplex gate_matrix[4];

    gate_matrix[0] =
        make_cuDoubleComplex(gate_matrix_real[0], gate_matrix_imaginary[0]);
    gate_matrix[1] =
        make_cuDoubleComplex(gate_matrix_real[1], gate_matrix_imaginary[1]);
    gate_matrix[2] =
        make_cuDoubleComplex(gate_matrix_real[2], gate_matrix_imaginary[2]);
    gate_matrix[3] =
        make_cuDoubleComplex(gate_matrix_real[3], gate_matrix_imaginary[3]);

    // get optimized block_size
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        apply_one_qubit_gate_kernel_distributed,
        0,
        num_amplitudes_per_gpu);

    grid_size =(num_amplitudes_per_gpu + block_size - 1) / block_size;

    apply_one_qubit_gate_kernel_distributed <<< grid_size, block_size >>>(
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
}

double *
QuantumAmplitudes::measure(int num_amplitudes_per_gpu)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        apply_one_qubit_gate_kernel_local,
        0, num_amplitudes_per_gpu);

    grid_size = (num_amplitudes_per_gpu + block_size - 1) / block_size;


    double *gpu_probabilities;
    gpu_probabilities = (double *) malloc(num_amplitudes_per_gpu * sizeof(double));

    double *prob;
    checkCudaErrors(cudaMalloc((void **) &prob,
                               sizeof(double) * num_amplitudes_per_gpu));

    measure_amplitudes_on_device <<< grid_size, block_size >>>(
        num_amplitudes_per_gpu,
        prob,
        real_parts,
        imaginary_parts);

    checkCudaErrors(cudaMemcpy(gpu_probabilities,
                               prob,
                               sizeof(double) * num_amplitudes_per_gpu,
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());

    return gpu_probabilities;
}
