cmake_minimum_required(VERSION 3.0)
project(damavand-gpu LANGUAGES CXX CUDA)

find_package(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $ENV{CUDA_NVCC_FLAGS} --generate-line-info")
include_directories("$ENV{CUDA_SAMPLES_PATH}/common/inc/")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include_directories(${PROJECT_SOURCE_DIR}/damavand-gpu)

cuda_add_library(damavand-gpu SHARED 
                 rust_communication.cu
                 quantum_amplitudes.cu
                 kernels.cu)

include_directories(${PROJECT_SOURCE_DIR})

target_link_libraries(damavand-gpu)

install(TARGETS damavand-gpu DESTINATION .)
