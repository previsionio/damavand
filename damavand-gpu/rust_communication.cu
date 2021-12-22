
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <vector>
#include <memory>
#include <complex>
#include <omp.h>

#include "quantum_amplitudes.cuh"
#include "kernels.cuh"

// contains the amplitudes for each gpu in a single node.
// we thus avoid MPI communication between GPUs in the same node.
// There is one MPI process for each node containing multiple GPUs

std::vector<QuantumAmplitudes> local_amplitudes;
std::vector<QuantumAmplitudes> partner_amplitudes;

int num_gpus_per_node_used;

extern "C" int get_number_of_available_gpus()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

extern "C" double get_memory_for_gpu(int local_gpu_rank)
{
    checkCudaErrors(cudaSetDevice(local_gpu_rank));
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, local_gpu_rank));
    return static_cast<double>(deviceProp.totalGlobalMem / 1048576.0f);
}

extern "C" int peer_access_allowed(int source_gpu_id, int target_gpu_id)
{
    int can_access_peer;
    checkCudaErrors(cudaDeviceCanAccessPeer(
                        &can_access_peer,
                        source_gpu_id,
                        target_gpu_id));
    return can_access_peer;
}

extern "C" void exchange_amplitudes_between_gpus(int current_gpu_rank, int partner_gpu_rank,
        int num_amplitudes_per_gpu)
{
    checkCudaErrors(cudaMemcpy(
                        partner_amplitudes[current_gpu_rank].real_parts,
                        local_amplitudes[partner_gpu_rank].real_parts,
                        sizeof(double) * num_amplitudes_per_gpu,
                        cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(
                        partner_amplitudes[current_gpu_rank].imaginary_parts,
                        local_amplitudes[partner_gpu_rank].imaginary_parts,
                        sizeof(double) * num_amplitudes_per_gpu,
                        cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(
                        partner_amplitudes[partner_gpu_rank].real_parts,
                        local_amplitudes[current_gpu_rank].real_parts,
                        sizeof(double) * num_amplitudes_per_gpu,
                        cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(
                        partner_amplitudes[partner_gpu_rank].imaginary_parts,
                        local_amplitudes[current_gpu_rank].imaginary_parts,
                        sizeof(double) * num_amplitudes_per_gpu,
                        cudaMemcpyDeviceToDevice));
}

extern "C" void init_quantum_state(
    int num_amplitudes_per_gpu,
    int num_gpus_per_node,
    bool is_first_node)
{
    num_gpus_per_node_used = num_gpus_per_node;

    local_amplitudes.clear();
    partner_amplitudes.clear();

    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int local_gpu_rank = 0; local_gpu_rank < num_gpus_per_node_used; local_gpu_rank++)
    {
        checkCudaErrors(cudaSetDevice(local_gpu_rank));
        for(int partner_gpu_rank = 0; partner_gpu_rank < num_gpus_per_node_used; partner_gpu_rank++)
        {
            if(partner_gpu_rank == local_gpu_rank) continue;
            int access_allowed;
            cudaDeviceCanAccessPeer(&access_allowed, local_gpu_rank, partner_gpu_rank);
            if(access_allowed)
            {
                checkCudaErrors(cudaSetDevice(local_gpu_rank));
                checkCudaErrors(cudaDeviceEnablePeerAccess(partner_gpu_rank, 0));
                printf("Peer access from GPU %d to GPU %d enabled\n", local_gpu_rank, partner_gpu_rank);
            }
            else
            {
                printf("WARNING: could not enable peeer access between GPUs\n");
            }
        }
    }

    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int local_gpu_rank = 0; local_gpu_rank < num_gpus_per_node_used; local_gpu_rank++)
    {
        checkCudaErrors(cudaSetDevice(local_gpu_rank));
        auto quantum_amplitudes = QuantumAmplitudes();
        auto partner_quantum_amplitudes = QuantumAmplitudes();

        if(local_gpu_rank == 0 && is_first_node)
        {
            quantum_amplitudes.set_zero_state(num_amplitudes_per_gpu, true);
            partner_quantum_amplitudes.set_zero_state(num_amplitudes_per_gpu, false);
        }
        else
        {
            quantum_amplitudes.set_zero_state(num_amplitudes_per_gpu, false);
            partner_quantum_amplitudes.set_zero_state(num_amplitudes_per_gpu, false);
        }
        local_amplitudes.push_back(quantum_amplitudes);
        partner_amplitudes.push_back(partner_quantum_amplitudes);
    }
}

extern "C" void measure_on_gpu(int num_amplitudes_per_gpu, double *probabilities)
{

    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));

        double * gpu_probabilities =
            local_amplitudes[gpu_id].measure(num_amplitudes_per_gpu);

        for(int amplitude_id_on_gpu = 0;
                amplitude_id_on_gpu < num_amplitudes_per_gpu;
                amplitude_id_on_gpu++)
        {
            probabilities[amplitude_id_on_gpu + gpu_id * num_amplitudes_per_gpu] =
                gpu_probabilities[amplitude_id_on_gpu];
        }
        checkCudaErrors(cudaDeviceReset());
    }
}

extern "C" void apply_one_qubit_gate_gpu_local(double *gate_matrix_real,
        double *gate_matrix_imaginary,
        int num_amplitudes_per_gpu,
        int control_qubit, int target_qubit)
{
    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));
        local_amplitudes[gpu_id].apply_one_qubit_gate(gate_matrix_real,
                gate_matrix_imaginary,
                num_amplitudes_per_gpu,
                control_qubit,
                target_qubit);
    }
}

extern "C" void apply_one_qubit_gate_gpu_distributed(double *gate_matrix_real,
        double *gate_matrix_imaginary,
        int num_amplitudes_per_gpu,
        int control_qubit, int target_qubit)
{
    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));
        local_amplitudes[gpu_id].apply_one_qubit_gate_distributed
        (partner_amplitudes[gpu_id], gate_matrix_real, gate_matrix_imaginary,
         num_amplitudes_per_gpu, control_qubit, target_qubit);
    }
}

//TODO amplitude encoding
extern "C" void load_amplitudes_local_on_device(int num_amplitudes_per_gpu,
        double *local_amplitudes_real,
        double *local_amplitudes_imaginary)
{
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));
        local_amplitudes[gpu_id].load_on_device(num_amplitudes_per_gpu,
                                                local_amplitudes_real,
                                                local_amplitudes_imaginary);
    }
}

extern "C" void split_amplitudes_between_gpus(int num_amplitudes_per_gpu,
        double *local_amplitudes_real,
        double *local_amplitudes_imaginary,
        double *partner_amplitudes_real,
        double *partner_amplitudes_imaginary)
{
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));

        int
        start_index = gpu_id * num_amplitudes_per_gpu;

        checkCudaErrors(cudaMemcpy(
                            local_amplitudes[gpu_id].real_parts,
                            &local_amplitudes_real[start_index],
                            sizeof(double) * num_amplitudes_per_gpu,
                            cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(
                            local_amplitudes[gpu_id].imaginary_parts,
                            &local_amplitudes_imaginary[start_index],
                            sizeof(double) * num_amplitudes_per_gpu,
                            cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(
                            partner_amplitudes[gpu_id].real_parts,
                            &partner_amplitudes_real[start_index],
                            sizeof(double) * num_amplitudes_per_gpu,
                            cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(
                            partner_amplitudes[gpu_id].imaginary_parts,
                            &partner_amplitudes_imaginary[start_index],
                            sizeof(double) * num_amplitudes_per_gpu,
                            cudaMemcpyHostToDevice));
    }
}

extern "C" void retrieve_amplitudes_on_host(
    int num_amplitudes_per_gpu,
    double *local_amplitudes_real,
    double *local_amplitudes_imaginary)
{
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));

        int start_index = gpu_id * num_amplitudes_per_gpu;
        checkCudaErrors(cudaMemcpy(
                            &local_amplitudes_real[start_index],
                            local_amplitudes[gpu_id].real_parts,
                            sizeof(double) * num_amplitudes_per_gpu,
                            cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaMemcpy(
                            &local_amplitudes_imaginary[start_index],
                            local_amplitudes[gpu_id].imaginary_parts,
                            sizeof(double) * num_amplitudes_per_gpu,
                            cudaMemcpyDeviceToHost));
    }
}
