
#ifndef kernels_cuh
#define kernels_cuh

#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <cuComplex.h>
#include <math.h>

#ifdef HAS_CUDA_PROFILING_CONTROL
#include <cuda_profiler_api.h>
#endif

__global__ void measure_amplitudes_on_device(
    int num_amplitudes_per_gpu,
    double *prob,
    double *real_parts,
    double *imaginary_aprts);

__global__ void init_zero_state_on_first_gpu(
    int num_amplitudes_per_gpu,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary);

__global__ void init_zero_state_on_other_gpu(
    int num_amplitudes_per_gpu,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary);

__global__ void apply_one_qubit_gate_kernel_local(
    int num_amplitudes_per_gpu,
    int control_qubit,
    int target_qubit,
    cuDoubleComplex device_gate_matrix_0,
    cuDoubleComplex device_gate_matrix_1,
    cuDoubleComplex device_gate_matrix_2,
    cuDoubleComplex device_gate_matrix_3,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary);

__global__ void apply_one_qubit_gate_kernel_distributed(
    int num_amplitudes_per_gpu,
    int control_qubit,
    int target_qubit,
    cuDoubleComplex
    device_gate_matrix_0,
    cuDoubleComplex
    device_gate_matrix_1,
    cuDoubleComplex
    device_gate_matrix_2,
    cuDoubleComplex
    device_gate_matrix_3,
    double *device_local_amplitudes_real,
    double *device_local_amplitudes_imaginary,
    double *device_partner_amplitudes_real,
    double *device_partner_amplitudes_imaginary);
#endif
