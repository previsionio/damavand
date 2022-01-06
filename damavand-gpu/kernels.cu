#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <cuComplex.h>
#include "kernels.cuh"
#include <cmath>

#ifdef HAS_CUDA_PROFILING_CONTROL
#include <cuda_profiler_api.h>
#endif

__global__ void measure_amplitudes_on_device_shared(
    int num_amplitudes_per_gpu,
    int first_amplitude_index,
    double *prob,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_id >= num_amplitudes_per_gpu)
        return;

    extern __shared__ double shared_amplitudes_real[];
    extern __shared__ double shared_amplitudes_imaginary[];

    shared_amplitudes_real[thread_id] =
      device_local_amplitudes_real[first_amplitude_index + thread_id];

    shared_amplitudes_imaginary[thread_id] =
      device_local_amplitudes_imaginary[first_amplitude_index + thread_id];

    __syncthreads();

    // do not use pow(x, 2) as it is heavy-weighted (CUDA C++ best practices guide)
    prob[first_amplitude_index + thread_id] =
        shared_amplitudes_real[thread_id] * shared_amplitudes_real[thread_id] +
        shared_amplitudes_imaginary[thread_id] * shared_amplitudes_real[thread_id];
}

__global__ void measure_amplitudes_on_device_global(
    int num_amplitudes_per_gpu,
    int first_amplitude_index,
    double *prob,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_id >= num_amplitudes_per_gpu)
        return;

    // do not use pow(x, 2) as it is heavy-weighted (CUDA C++ best practices guide)
    prob[thread_id] =
        device_local_amplitudes_real[thread_id] * device_local_amplitudes_real[thread_id] +
        device_local_amplitudes_imaginary[thread_id] * device_local_amplitudes_imaginary[thread_id];
}

__global__ void init_zero_state_on_first_gpu(
    int num_amplitudes_per_gpu,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if(thread_id >= num_amplitudes_per_gpu)
        return;

    if(thread_id == 0)
    {
        device_local_amplitudes_real[thread_id] = 1.;
    }
    else
    {
        device_local_amplitudes_real[thread_id] = 0.;
    }
    device_local_amplitudes_imaginary[thread_id] = 0.;
}

__global__ void init_zero_state_on_other_gpu(
    int num_amplitudes_per_gpu,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary)
{

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if(thread_id >= num_amplitudes_per_gpu)
        return;

    device_local_amplitudes_real[thread_id] = 0.;
    device_local_amplitudes_imaginary[thread_id] = 0.;
}

__device__ inline int compute_partner_thread_id(
    int thread_id,
    int num_amplitudes_per_node,
    int amplitude_gap)
{
    int partner_thread_id;

    auto num_amplitudes_before_node = thread_id * num_amplitudes_per_node;

    if(num_amplitudes_before_node %(2 * amplitude_gap) < amplitude_gap)
    {
        // node is linked with a node of higher rank
        partner_thread_id = thread_id + amplitude_gap / num_amplitudes_per_node;
    }
    else
    {
        // node is linked with a node of lower rank
        partner_thread_id = thread_id - amplitude_gap / num_amplitudes_per_node;
    }
    return partner_thread_id;
}

__global__ void apply_one_qubit_gate_kernel_local(
    int num_qubits,
    int num_amplitudes_per_gpu,
    int control_qubit,
    int target_qubit,
    cuDoubleComplex device_gate_matrix_0,
    cuDoubleComplex device_gate_matrix_1,
    cuDoubleComplex device_gate_matrix_2,
    cuDoubleComplex device_gate_matrix_3,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_id >= num_amplitudes_per_gpu)
        return;

    // controlled gate
    bool apply_gate = true;

    if(control_qubit >= 0)
        apply_gate = (thread_id >> (num_qubits - 1 - control_qubit)) & 1;

    if(!apply_gate)
        return;

    int partner_id = compute_partner_thread_id(thread_id,
                     1,
                     1 << (num_qubits - 1 - target_qubit));

    cuDoubleComplex local_amplitude =
        make_cuDoubleComplex(device_local_amplitudes_real[thread_id],
                             device_local_amplitudes_imaginary[thread_id]);

    cuDoubleComplex partner_amplitude =
        make_cuDoubleComplex(device_local_amplitudes_real[partner_id],
                             device_local_amplitudes_imaginary[partner_id]);

    if(partner_id > thread_id)
    {
        auto output = cuCadd(cuCmul(device_gate_matrix_0, local_amplitude),
                             cuCmul(device_gate_matrix_1, partner_amplitude));
        device_local_amplitudes_real[thread_id] = output.x;
        device_local_amplitudes_imaginary[thread_id] = output.y;
    }
    else
    {
        auto output = cuCadd(cuCmul(device_gate_matrix_2, partner_amplitude),
                             cuCmul(device_gate_matrix_3, local_amplitude));
        device_local_amplitudes_real[thread_id] = output.x;
        device_local_amplitudes_imaginary[thread_id] = output.y;
    }
}

__global__ void apply_one_qubit_gate_kernel_distributed(
    int num_qubits,
    int num_amplitudes_per_gpu,
    int control_qubit,
    int target_qubit,
    cuDoubleComplex device_gate_matrix_0,
    cuDoubleComplex device_gate_matrix_1,
    cuDoubleComplex device_gate_matrix_2,
    cuDoubleComplex device_gate_matrix_3,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary,
    double *device_partner_amplitudes_real,
    double *device_partner_amplitudes_imaginary)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_id >= num_amplitudes_per_gpu)
        return;

    bool apply_gate = true;

    if(control_qubit >= 0)
        apply_gate = (thread_id >> (num_qubits - 1 - control_qubit)) & 1;

    if(!apply_gate)
        return;

    int partner_id = compute_partner_thread_id(thread_id,
                     1,
                     1 << (num_qubits - 1 - target_qubit));

    cuDoubleComplex local_amplitude;
    cuDoubleComplex partner_amplitude;

    local_amplitude =
        make_cuDoubleComplex(device_local_amplitudes_real[thread_id],
                             device_local_amplitudes_imaginary[thread_id]);

    partner_amplitude =
        make_cuDoubleComplex(device_partner_amplitudes_real[thread_id],
                             device_partner_amplitudes_imaginary[thread_id]);

    if(partner_id > thread_id)
    {
        auto output = cuCadd(cuCmul(device_gate_matrix_0, local_amplitude),
                             cuCmul(device_gate_matrix_1, partner_amplitude));
        device_local_amplitudes_real[thread_id] = output.x;
        device_local_amplitudes_imaginary[thread_id] = output.y;
    }
    else
    {
        auto output = cuCadd(cuCmul(device_gate_matrix_2, partner_amplitude),
                             cuCmul(device_gate_matrix_3, local_amplitude));
        device_local_amplitudes_real[thread_id] = output.x;
        device_local_amplitudes_imaginary[thread_id] = output.y;
    }
}
