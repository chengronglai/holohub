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

    # Estimate total weight size upfront to decide serialization strategy.
    weight_bytes = 0
    prev = input_size
    for i in range(num_layers):
        out = input_size if i == num_layers - 1 else hidden_size
        weight_bytes += prev * out * 4
        prev = out
    use_external = weight_bytes > 1.5 * 1024 * 1024 * 1024

    if use_external:
        print(f"Weights are {weight_bytes / (1024**3):.1f} GB -- using external data format")

    nodes = []
    initializers = []
    current_tensor = "input"
    current_dim = input_size
    rng = np.random.default_rng(seed=42)

    data_filename = os.path.basename(output_path) + ".data"
    data_filepath = os.path.join(os.path.dirname(os.path.abspath(output_path)), data_filename)
    ext_file = open(data_filepath, "wb") if use_external else None

    try:
        for i in range(num_layers):
            if i == num_layers - 1:
                out_dim = input_size
            else:
                out_dim = hidden_size

            weight_name = f"W{i}"
            scale = np.sqrt(2.0 / (current_dim + out_dim))
            weight_data = (rng.standard_normal((current_dim, out_dim)) * scale).astype(np.float32)

            if use_external:
                raw = weight_data.tobytes()
                offset = ext_file.tell()
                ext_file.write(raw)
                tensor = TensorProto()
                tensor.name = weight_name
                tensor.data_type = TensorProto.FLOAT
                tensor.dims.extend([current_dim, out_dim])
                tensor.data_location = TensorProto.EXTERNAL
                for k, v in [("location", data_filename),
                             ("offset", str(offset)),
                             ("length", str(len(raw)))]:
                    entry = tensor.external_data.add()
                    entry.key = k
                    entry.value = v
                initializers.append(tensor)
            else:
                initializers.append(
                    helper.make_tensor(
                        weight_name,
                        TensorProto.FLOAT,
                        [current_dim, out_dim],
                        weight_data.flatten().tolist(),
                    )
                )

            matmul_output = f"matmul_{i}"
            nodes.append(
                helper.make_node("MatMul", [current_tensor, weight_name], [matmul_output],
                                 name=f"MatMul_{i}")
            )
            relu_output = f"relu_{i}"
            nodes.append(
                helper.make_node("Relu", [matmul_output], [relu_output], name=f"Relu_{i}")
            )
            current_tensor = relu_output
            current_dim = out_dim
    finally:
        if ext_file is not None:
            ext_file.close()

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_size])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, input_size])

    nodes.append(
        helper.make_node("Identity", [current_tensor], ["output"], name="Output_Identity")
    )

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

    if use_external:
        onnx.save(model, output_path)
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