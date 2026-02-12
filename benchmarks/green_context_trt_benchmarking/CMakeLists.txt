# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(green_context_trt_benchmarking CXX CUDA)

find_package(holoscan 3.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find CUDA and CUPTI
find_package(CUDAToolkit REQUIRED)

add_executable(green_context_trt_benchmarking
    green_context_trt_benchmark.cpp
    benchmark_cuda_kernel.cu
)

target_link_libraries(green_context_trt_benchmarking
    PRIVATE
    holoscan::core
    holoscan::ops::inference
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cupti
)

# Set CUDA properties
set_property(TARGET green_context_trt_benchmarking PROPERTY CUDA_STANDARD 17)
set_property(TARGET green_context_trt_benchmarking PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET green_context_trt_benchmarking PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Include CUDA headers
target_include_directories(green_context_trt_benchmarking PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Find and configure CCCL (CUDA C++ Core Libraries) for CUDA standard library headers
find_package(CCCL CONFIG QUIET)
if(TARGET CCCL::CCCL)
    target_link_libraries(green_context_trt_benchmarking PRIVATE CCCL::CCCL)
    message(STATUS "Using CCCL::CCCL target for CUDA standard library headers")
else()
    target_include_directories(green_context_trt_benchmarking PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
    message(STATUS "Using manual CCCL include path: ${CUDAToolkit_INCLUDE_DIRS}/cccl")
endif()

# Copy YAML config file to build directory
add_custom_target(green_context_trt_benchmark_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/green_context_trt_benchmark.yaml"
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "green_context_trt_benchmark.yaml"
    BYPRODUCTS "green_context_trt_benchmark.yaml"
)
add_dependencies(green_context_trt_benchmarking green_context_trt_benchmark_yaml)

# Generate ONNX model at build time with proper dependency tracking.
# Auto-generation is enabled only when Python3 can import both onnx and numpy.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import onnx, numpy"
        RESULT_VARIABLE TRT_BENCH_ONNX_IMPORT_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(TRT_BENCH_ONNX_IMPORT_RESULT EQUAL 0)
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/benchmark_model.onnx"
            COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx_model.py"
                --output "${CMAKE_CURRENT_BINARY_DIR}/benchmark_model.onnx"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx_model.py"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Generating benchmark ONNX model (requires Python onnx/numpy)"
            VERBATIM
        )

        add_custom_target(green_context_trt_benchmark_model
            DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/benchmark_model.onnx"
        )
        add_dependencies(green_context_trt_benchmarking green_context_trt_benchmark_model)
    else()
        message(WARNING
            "Python3 is available but missing required modules (onnx, numpy). "
            "Skipping auto-generation of benchmark_model.onnx. "
            "Generate manually and pass --model-path at runtime.")
    endif()
else()
    message(WARNING "Python3 not found. Generate benchmark_model.onnx manually and pass --model-path.")
endif()
