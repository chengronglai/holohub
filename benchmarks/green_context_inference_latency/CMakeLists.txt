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
project(green_context_inference_latency CXX)

find_package(holoscan 3.10 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(CUDAToolkit REQUIRED)

add_executable(green_context_inference_latency
    inference_scheduling_benchmark.cpp
)

target_link_libraries(green_context_inference_latency
    PRIVATE
    holoscan::core
    holoscan::ops::inference
    CUDA::cuda_driver
    CUDA::cudart
)

target_include_directories(green_context_inference_latency PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Copy YAML config and model generation script to build directory
add_custom_target(inference_scheduling_assets
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/inference_scheduling_benchmark.yaml"
        ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx_model.py"
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "inference_scheduling_benchmark.yaml" "generate_onnx_model.py"
    BYPRODUCTS "inference_scheduling_benchmark.yaml" "generate_onnx_model.py"
)
add_dependencies(green_context_inference_latency inference_scheduling_assets)
