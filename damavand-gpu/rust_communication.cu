
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <vector>
#include <memory>
#include <complex>
#include <omp.h>

#include "quantum_amplitudes.cuh"
#include "kernels.cuh"
#include "utils.cuh"

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

extern "C" void print_timers()
{
#ifdef HAS_PROFILER
    float init_average_time = 0.;
    float apply_average_time = 0.;
    float measure_average_time = 0.;
    float copy_device_to_host_time = 0.;

    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        auto parameters = get_launching_parameters(
            local_amplitudes[gpu_id].occupancy_strategy,
            num_gpus_per_node_used,
            init_zero_state_on_first_gpu);

        auto init_potential_occupancy = get_occupancy(
            num_gpus_per_node_used,
            init_zero_state_on_first_gpu,
            parameters.block_size);

        parameters = get_launching_parameters(
            local_amplitudes[gpu_id].occupancy_strategy,
            num_gpus_per_node_used,
            apply_one_qubit_gate_kernel_local);

        auto apply_potential_occupancy = get_occupancy(
            num_gpus_per_node_used,
            apply_one_qubit_gate_kernel_local,
            parameters.block_size);

        parameters = get_launching_parameters(
            local_amplitudes[gpu_id].occupancy_strategy,
            num_gpus_per_node_used,
            measure_amplitudes_on_device_shared);

        auto measure_potential_occupancy = get_occupancy(
            num_gpus_per_node_used,
            measure_amplitudes_on_device_shared,
            parameters.block_size);

        init_average_time += get_init_kernel_elapsed_time();
        apply_average_time += get_apply_kernel_elapsed_time();
        measure_average_time += get_measure_kernel_elapsed_time();
        copy_device_to_host_time += get_copy_device_to_host_elapsed_time();

        // printf("initoccupancy %f\n", init_potential_occupancy);
        // printf("apply occupancy %f\n", apply_potential_occupancy);
        // printf("measure occupancy %f\n", measure_potential_occupancy);
    }

    printf("init %f\n", init_average_time / num_gpus_per_node_used);
    printf("apply %f\n", apply_average_time / num_gpus_per_node_used);
    printf("measure %f\n", measure_average_time / num_gpus_per_node_used);
    printf("copy_device_to_host %f\n", copy_device_to_host_time/ num_gpus_per_node_used);
#endif
}

extern "C" void exchange_amplitudes_between_gpus(
    int current_gpu_rank,
    int partner_gpu_rank,
    int num_amplitudes_per_gpu)
{
    #ifdef HAS_PROFILER
        sdkStartTimer(&copy_device_to_device_timer);
    #endif

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
    #ifdef HAS_PROFILER
        sdkStopTimer(&copy_device_to_device_timer);
    #endif
}

extern "C" void init_quantum_state(
    int num_amplitudes_per_gpu,
    int num_gpus_per_node,
    bool is_first_node)
{
    #ifdef HAS_PROFILER
        init_timers();
    #endif
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

extern "C" void sequential_measure_on_gpu(int num_amplitudes_per_gpu, double *probabilities)
{

    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // allocate memory on device for results
        double *device_probabilities;
        checkCudaErrors(cudaMalloc((void **) &device_probabilities,
                                   sizeof(double) * num_amplitudes_per_gpu));

        // allocate memory on host for results
        double *host_probabilities;
        checkCudaErrors(cudaMallocHost((void **) &host_probabilities,
                                       sizeof(double) * num_amplitudes_per_gpu));

        #ifdef HAS_PROFILER
            sdkStartTimer(&copy_device_to_host_timer);
        #endif

        // run measure kernel
        local_amplitudes[gpu_id].measure(
            num_amplitudes_per_gpu,
            0,
            device_probabilities,
            stream);


        checkCudaErrors(cudaMemcpy(host_probabilities,
                                   device_probabilities,
                                   sizeof(double) * num_amplitudes_per_gpu,
                                   cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaDeviceSynchronize());

        #ifdef HAS_PROFILER
            sdkStopTimer(&copy_device_to_host_timer);
        #endif

        for(int amplitude_id_on_gpu = 0;
                amplitude_id_on_gpu < num_amplitudes_per_gpu;
                amplitude_id_on_gpu++)
            probabilities[amplitude_id_on_gpu + gpu_id * num_amplitudes_per_gpu] =
                host_probabilities[amplitude_id_on_gpu];


        checkCudaErrors(cudaDeviceReset());
    }
}

extern "C" void concurrent_measure_on_gpu(int num_amplitudes_per_gpu, double *probabilities)
{

    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));

        int stream_size = 64;

        // compute number of streams necessary
        int num_streams = num_amplitudes_per_gpu / stream_size;

        cudaStream_t streams[num_streams];

        // allocate memory on host for results
        double *host_probabilities;
        checkCudaErrors(cudaMallocHost((void **) &host_probabilities,
                                       sizeof(double) * num_amplitudes_per_gpu));

        // allocate memory on device for results
        double *device_probabilities;
        checkCudaErrors(cudaMalloc((void **) &device_probabilities,
                                   sizeof(double) * stream_size));

        // create all streams
        for (int stream_id = 0; stream_id < num_streams; ++stream_id) {
            checkCudaErrors(cudaStreamCreate(&streams[stream_id]));
        }

        #ifdef HAS_PROFILER
            sdkStartTimer(&copy_device_to_host_timer);
        #endif

        // loop on all streams
        for (int stream_id = 0; stream_id < num_streams; ++stream_id) {

            // run measure kernel
            local_amplitudes[gpu_id].measure(
                stream_size,
                stream_id * stream_size,
                device_probabilities,
                streams[stream_id]);
        }

        for (int stream_id = 0; stream_id < num_streams; ++stream_id) {

            checkCudaErrors(cudaMemcpyAsync(device_probabilities,
                                            host_probabilities + stream_id * stream_size,
                                            sizeof(double) * stream_size,
                                            cudaMemcpyDeviceToHost,
                                            streams[stream_id]));

        }

        for (int stream_id = 0; stream_id < num_streams; ++stream_id) {
            checkCudaErrors(cudaStreamSynchronize(streams[stream_id]));
        }

        #ifdef HAS_PROFILER
            sdkStopTimer(&copy_device_to_host_timer);
        #endif

        for (int stream_id = 0; stream_id < num_streams; ++stream_id) {
            checkCudaErrors(cudaStreamDestroy(streams[stream_id]));
        }

        for(int id = 0; id < num_amplitudes_per_gpu; id++)
        {
            probabilities[id] = host_probabilities[id];
        }

        checkCudaErrors(cudaDeviceReset());
    }
}

extern "C" void measure_on_gpu(int num_amplitudes_per_gpu, double *probabilities)
{
    #ifdef USE_CONCURRENT_COPY
        concurrent_measure_on_gpu(num_amplitudes_per_gpu, probabilities);
    #else 
        sequential_measure_on_gpu(num_amplitudes_per_gpu, probabilities);
    #endif
}

extern "C" void apply_one_qubit_gate_gpu_local(
    double *gate_matrix_real,
    double *gate_matrix_imaginary,
    int num_qubits,
    int num_amplitudes_per_gpu,
    int control_qubit,
    int target_qubit)
{
    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));
        local_amplitudes[gpu_id].apply_one_qubit_gate(
            gate_matrix_real,
            gate_matrix_imaginary,
            num_qubits,
            num_amplitudes_per_gpu,
            control_qubit,
            target_qubit);
    }
}

extern "C" void apply_one_qubit_gate_gpu_distributed(
    double *gate_matrix_real,
    double *gate_matrix_imaginary,
    int num_qubits,
    int num_amplitudes_per_gpu,
    int control_qubit,
    int target_qubit)
{
    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));
        local_amplitudes[gpu_id].apply_one_qubit_gate_distributed(
            partner_amplitudes[gpu_id],
            gate_matrix_real,
            gate_matrix_imaginary,
            num_qubits,
            num_amplitudes_per_gpu,
            control_qubit,
            target_qubit);
    }
}

//TODO amplitude encoding
extern "C" void load_amplitudes_local_on_device(
    int num_amplitudes_per_gpu,
    double *local_amplitudes_real,
    double *local_amplitudes_imaginary)
{
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));
        local_amplitudes[gpu_id].load_on_device(
            num_amplitudes_per_gpu,
            local_amplitudes_real,
            local_amplitudes_imaginary);
    }
}

extern "C" void split_amplitudes_between_gpus(
    int num_amplitudes_per_gpu,
    double *local_amplitudes_real,
    double *local_amplitudes_imaginary,
    double *partner_amplitudes_real,
    double *partner_amplitudes_imaginary)
{
    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));

        int
        start_index = gpu_id * num_amplitudes_per_gpu;

        #ifdef HAS_PROFILER
            sdkStartTimer(&copy_host_to_device_timer);
        #endif

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
        #ifdef HAS_PROFILER
            sdkStopTimer(&copy_host_to_device_timer);
        #endif
    }
}

extern "C" void retrieve_amplitudes_on_host(
    int num_amplitudes_per_gpu,
    double *local_amplitudes_real,
    double *local_amplitudes_imaginary)
{
    #pragma omp parallel for num_threads(num_gpus_per_node_used)
    for(int gpu_id = 0; gpu_id < num_gpus_per_node_used; gpu_id++)
    {
        checkCudaErrors(cudaSetDevice(gpu_id));

        int start_index = gpu_id * num_amplitudes_per_gpu;

        #ifdef HAS_PROFILER
            sdkStartTimer(&copy_device_to_host_timer);
        #endif

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

        #ifdef HAS_PROFILER
            sdkStopTimer(&copy_device_to_host_timer);
        #endif
    }
}
