#ifndef UTILS_CUH
#define UTILS_CUH

#include "helper_cuda.h"

#ifdef HAS_PROFILER
#include <helper_timer.h>
#endif

enum OccupancyStrategy{
    Linear,
    Automatic,
};

struct LaunchingParameters
{
  public:
    int grid_size, block_size;
};

#ifdef HAS_PROFILER
extern StopWatchInterface *copy_host_to_device_timer;
extern StopWatchInterface *copy_device_to_host_timer;
extern StopWatchInterface *copy_device_to_device_timer;
extern StopWatchInterface *init_kernel_timer;
extern StopWatchInterface *apply_kernel_timer;
extern StopWatchInterface *measure_kernel_timer;

void init_timers();

inline float get_copy_host_to_device_elapsed_time()
{
    return sdkGetAverageTimerValue(&copy_host_to_device_timer);
}

inline float get_copy_device_to_host_elapsed_time()
{
    return sdkGetAverageTimerValue(&copy_device_to_host_timer);
}

inline float get_copy_device_to_device_elapsed_time()
{
    return sdkGetAverageTimerValue(&copy_device_to_device_timer);
}

inline float get_init_kernel_elapsed_time()
{
    return sdkGetAverageTimerValue(&init_kernel_timer);
}

inline float get_apply_kernel_elapsed_time()
{
    return sdkGetAverageTimerValue(&apply_kernel_timer);
}

inline float get_measure_kernel_elapsed_time()
{
    return sdkGetAverageTimerValue(&measure_kernel_timer);
}
#endif

template<class T>
LaunchingParameters get_launching_parameters(
    OccupancyStrategy occupancy_strategy,
    int num_amplitudes_per_gpu,
    T kernel)
{
    int block_size;
    int min_grid_size;
    int grid_size;

    // get max num threads per block
    int device;
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));
    int max_num_threads_per_block = prop.maxThreadsPerBlock;

    if (occupancy_strategy == Automatic)
    {
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            kernel,
            0,
            num_amplitudes_per_gpu);
        
        grid_size = (num_amplitudes_per_gpu + block_size - 1) / block_size;
    }
    else if (occupancy_strategy == Linear)
    {
        auto potential_grid_size = (int) ceil(num_amplitudes_per_gpu / max_num_threads_per_block);

        if (potential_grid_size < 1) {
          grid_size = 1;
        } else {
          grid_size = potential_grid_size;
        }

        block_size = max_num_threads_per_block;
    }

    LaunchingParameters parameters = {grid_size, block_size};
    return parameters;
};

template<class T>
double get_occupancy(int num_amplitudes_per_gpu, T kernel, int block_size)
{
    double potential_occupancy;

    int device;
    cudaDeviceProp prop;
    int numBlocks;
    int activeWarps;
    int maxWarps;

    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &numBlocks,
                        kernel,
                        block_size,
                        0));

    activeWarps = numBlocks * block_size/ prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    potential_occupancy = (double)activeWarps / maxWarps;

    return potential_occupancy * 100.;
};

#endif
