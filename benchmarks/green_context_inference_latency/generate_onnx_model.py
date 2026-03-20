#!/usr/bin/env python3
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

"""Generate a simple ONNX model for the Green Context TRT benchmark.

Creates a fully-connected (dense) network with configurable layers and hidden size.
The model does MatMul + ReLU operations that create meaningful GPU work when run
through TensorRT, without requiring any training data or external dependencies
beyond the 'onnx' Python package.

Usage:
    python generate_onnx_model.py [--output PATH] [--input-size N] [--hidden-size N] [--num-layers N]
"""

import argparse
import os
import sys

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    import subprocess
    print("'onnx' package not found -- installing...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
    import onnx
    from onnx import TensorProto, helper


def generate_fc_model(
    input_size: int = 1024,
    hidden_size: int = 4096,
    num_layers: int = 6,
    output_path: str = "benchmark_model.onnx",
) -> None:
    """Generate a simple fully-connected ONNX model.

    Architecture:
        Input(1, input_size) -> [MatMul(hidden_size) -> ReLU] x num_layers -> Output(1, input_size)

    The first layer projects from input_size to hidden_size, intermediate layers
    are hidden_size -> hidden_size, and the last layer projects back to input_size.

    The default hidden_size=4096 and num_layers=6 are chosen to create meaningful
    GPU contention on modern GPUs (e.g. RTX 6000 Ada, 142 SMs). On smaller GPUs
    (e.g. Orin), reduce hidden_size to 2048 and num_layers to 3.

    Args:
        input_size: Dimension of the input and output tensors.
        hidden_size: Width of hidden layers (controls GPU workload).
        num_layers: Number of MatMul + ReLU stages.
        output_path: Where to save the ONNX model file.
    """
    print(f"Generating ONNX model: {num_layers} FC layers, "
          f"input_size={input_size}, hidden_size={hidden_size}")

    nodes = []
    initializers = []

    # Track current tensor name and dimension through the network
    current_tensor = "input"
    current_dim = input_size

    rng = np.random.default_rng(seed=42)

    for i in range(num_layers):
        # Determine output dimension for this layer
        if i == num_layers - 1:
            # Last layer projects back to input_size
            out_dim = input_size
        else:
            out_dim = hidden_size

        # Weight matrix: (current_dim, out_dim)
        weight_name = f"W{i}"
        # Use small random values (Xavier-like initialization) to avoid numerical issues
        scale = np.sqrt(2.0 / (current_dim + out_dim))
        weight_data = (rng.standard_normal((current_dim, out_dim)) * scale).astype(np.float32)

        initializers.append(
            helper.make_tensor(
                weight_name,
                TensorProto.FLOAT,
                [current_dim, out_dim],
                weight_data.flatten().tolist(),
            )
        )

        # MatMul node
        matmul_output = f"matmul_{i}"
        nodes.append(
            helper.make_node("MatMul", [current_tensor, weight_name], [matmul_output],
                             name=f"MatMul_{i}")
        )

        # ReLU activation
        relu_output = f"relu_{i}"
        nodes.append(
            helper.make_node("Relu", [matmul_output], [relu_output], name=f"Relu_{i}")
        )

        current_tensor = relu_output
        current_dim = out_dim

    # Define input and output value infos
    # Batch size is 1 (fixed) -- TensorRT will optimize for this
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_size])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, input_size])

    # Add an Identity node to rename the last relu output to "output"
    nodes.append(
        helper.make_node("Identity", [current_tensor], ["output"], name="Output_Identity")
    )

    # Build the graph and model
    graph = helper.make_graph(
        nodes,
        "benchmark_fc_network",
        [input_info],
        [output_info],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8

    # Validate and save -- use external data format for models > 2 GB
    model_size = model.ByteSize()
    if model_size > 2 * 1024 * 1024 * 1024:
        print(f"Model proto is {model_size / (1024**3):.1f} GB -- using external data format")
        # Save with external data (weights stored in a sibling .data file)
        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(output_path) + ".data",
        )
        # check_model needs the path when using external data
        onnx.checker.check_model(output_path)
    else:
        onnx.checker.check_model(model)
        onnx.save(model, output_path)

    # Report file size (include external data file if present)
    total_bytes = os.path.getsize(output_path)
    ext_data_path = output_path + ".data"
    if os.path.exists(ext_data_path):
        total_bytes += os.path.getsize(ext_data_path)
    file_size_mb = total_bytes / (1024 * 1024)
    print(f"Model saved to: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Input shape:  [1, {input_size}]")
    print(f"  Output shape: [1, {input_size}]")
    print(f"  Layers: {num_layers} x (MatMul + ReLU)")
    print(f"  Hidden size: {hidden_size}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a simple ONNX model for Green Context TRT benchmarking"
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_model.onnx",
        help="Output ONNX file path (default: benchmark_model.onnx)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1024,
        help="Input/output tensor dimension (default: 1024)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden layer width -- controls GPU workload (default: 4096). "
             "Use 2048 for smaller GPUs (e.g. Orin).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of MatMul + ReLU layers (default: 6). "
             "Use 3 for smaller GPUs (e.g. Orin).",
    )

    args = parser.parse_args()
    generate_fc_model(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()