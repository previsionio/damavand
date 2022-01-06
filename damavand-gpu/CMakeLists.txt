project(damavand-gpu LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.0)

option(HAS_PROFILER "Enables profiling" FALSE)
option(USE_CONCURRENT_COPY "Uses concurrent copy and kernel execution when possible" FALSE)
option(USE_SHARED_MEMORY "Uses shared memory when possbile" FALSE)
option(AUTOMATIC_OCCUPANCY_STRATEGY "Automatic occupancy strategy" TRUE)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --generate-line-info")

if (HAS_PROFILER)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DHAS_PROFILER")
endif()

if (AUTOMATIC_OCCUPANCY_STRATEGY)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DAUTOMATIC_OCCUPANCY_STRATEGY")
endif()

if (USE_CONCURRENT_COPY)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DUSE_CONCURRENT_COPY")
endif()

if (USE_SHARED_MEMORY)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DUSE_SHARED_MEMORY")
endif()

include_directories("$ENV{CUDA_SAMPLES_COMMON_INCLUDE_DIR}/")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include_directories(${PROJECT_SOURCE_DIR}/damavand-gpu)
include_directories(${PROJECT_SOURCE_DIR})

cuda_add_library(damavand-gpu SHARED 
                 utils.cu
                 kernels.cu
                 quantum_amplitudes.cu
                 rust_communication.cu)

target_link_libraries(damavand-gpu)

install(TARGETS damavand-gpu LIBRARY DESTINATION .)
