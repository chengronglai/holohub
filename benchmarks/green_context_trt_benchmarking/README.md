# Green Context TRT Inference Benchmark

This benchmark measures CUDA kernel launch-start time improvements provided by NVIDIA CUDA Green Context technology when TensorRT inference is used as the contending GPU workload. It uses NVIDIA CUPTI (CUDA Profiling Tools Interface) for precise GPU timing measurements within the Holoscan SDK framework.

## Overview

This is an extension of the [Green Context Benchmarking](../green_context_benchmarking/) benchmark. While the original uses a raw CUDA kernel (`DummyLoadOp`) as background GPU load, this variant uses **Holoscan's `InferenceOp` with TensorRT** as the contending workload. This exercises the full Holoscan inference stack to uncover potential scheduling issues specific to TensorRT inference pipelines.

### Architecture

The benchmark runs two concurrent workloads:

1. **Contending Workload Pipeline**: `TensorSourceOp -> InferenceOp (TRT) -> SinkOp`
   - `TensorSourceOp` generates random CUDA tensors
   - `InferenceOp` runs TensorRT inference on a simple FC network
   - `SinkOp` discards the output and signals that the pipeline is warmed up

2. **Measurement Operator**: `TimingBenchmarkOp`
   - Launches a lightweight CUDA kernel and measures launch-start time via CUPTI
   - Identical to the original Green Context benchmark's measurement
   - **Waits for the inference pipeline to be ready** before recording measurements, ensuring that timing samples are only collected under actual TRT inference contention
   - Also collects **per-kernel GPU execution times** for all TRT inference kernels via CUPTI, which can be used to verify whether Green Context is actually partitioning TRT inference

### A/B Testing Design

- **Baseline**: Both workloads run on separate non-default CUDA streams WITHOUT Green Context
- **Green Context**: InferenceOp runs in Partition 0, TimingBenchmarkOp in Partition 1, each with dedicated GPU SMs

### Warmup

The benchmark includes an automatic warmup mechanism:
1. **Pre-ready warmup**: `TimingBenchmarkOp` runs the timing kernel but discards measurements while the TensorRT engine is being built/loaded. Measurement only begins after the inference pipeline produces its first output.
2. **Post-ready warmup**: An additional configurable number of iterations (default: 100) are discarded after the pipeline is ready, to let GPU scheduling stabilize before recording.

This ensures both baseline and Green Context runs measure under identical conditions with real TRT inference contention.

## Prerequisites

### Generate the ONNX Model

Before running the benchmark, generate the dummy ONNX model:

```bash
pip install onnx numpy
python generate_onnx_model.py --output benchmark_model.onnx
```

This creates a simple fully-connected network (MatMul + ReLU layers) that produces meaningful GPU work through TensorRT without requiring any training data.

Options for model generation:
```bash
# Default (recommended for modern dGPUs like RTX 6000 Ada)
python generate_onnx_model.py --output benchmark_model.onnx

# Smaller model for embedded GPUs (e.g. Orin)
python generate_onnx_model.py --output benchmark_model.onnx \
    --hidden-size 2048 --num-layers 3

# Custom configuration
python generate_onnx_model.py \
    --output benchmark_model.onnx \
    --input-size 1024 \
    --hidden-size 4096 \
    --num-layers 6
```

## Usage

### Basic Usage

**Default (older GPUs and Orin systems):**
```bash
./holohub run green_context_trt_benchmarking --docker-opts="--user root"
```

**Modern GPUs and Jetson Thor:**
```bash
./holohub run green_context_trt_benchmarking --docker-opts="--user root" --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu
```

### Command Line Options

```bash
./holohub run green_context_trt_benchmarking --docker-opts="--user root" --run-args="[OPTIONS]"

Options:
  --samples N             Number of timing samples to measure (default: 1000)
  --warmup-samples N      Post-ready warmup iterations to discard (default: 100)
  --model-path PATH       Path to ONNX model file (default: ./benchmark_model.onnx)
  --input-size N          Input tensor size matching the model (default: 1024)
  --mode MODE             Run mode: 'baseline', 'green-context', or 'all' (default: all)
  --help                  Show help message

Examples:
  ./holohub run green_context_trt_benchmarking --docker-opts="--user root" --run-args="--help"
  ./holohub run green_context_trt_benchmarking --docker-opts="--user root" --run-args="--samples 500 --mode all"
  ./holohub run green_context_trt_benchmarking --docker-opts="--user root" --run-args="--model-path /path/to/custom_model.onnx --input-size 2048"
```

### Sweep Script

A sweep script is provided to run the benchmark across multiple model sizes and sample counts, then print a comparison summary table:

```bash
# Full sweep (3 model sizes x 3 sample counts = 9 runs)
./benchmarks/green_context_trt_benchmarking/run_sweep.sh \
    --base-img nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu

# Quick single-config run
./benchmarks/green_context_trt_benchmarking/run_sweep.sh \
    --samples "1000" --models "medium:4096:6:1024" \
    --base-img nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu

# Run with a pre-built binary (no docker)
./benchmarks/green_context_trt_benchmarking/run_sweep.sh \
    --binary ./build/benchmarks/green_context_trt_benchmarking/green_context_trt_benchmarking
```

Run `./benchmarks/green_context_trt_benchmarking/run_sweep.sh --help` for all options.

## Measurements

The benchmark reports the following metrics:

1. **CUDA Kernel Launch-Start Time**: The primary measurement. Time from `cudaLaunchKernel` (CPU) to GPU execution start for the `simple_benchmark_kernel`, measured via CUPTI. Shows whether Green Context reduces scheduling latency under TRT inference contention.

2. **CUDA Kernel Execution Time**: GPU execution duration of the `simple_benchmark_kernel` itself.

3. **TRT InferenceOp Compute Time**: Wall-clock time of `InferenceOp::compute()`, measured by wrapping the call in `CustomInferenceOp`. If Green Context is partitioning TRT (fewer SMs), this should increase.

4. **TRT Inference Per-Kernel GPU Execution Time (CUPTI)**: GPU execution duration of each individual TensorRT kernel (cuBLAS GEMMs, activations, etc.), measured via CUPTI during the measurement phase. This is the most precise measurement for verifying whether Green Context is actually partitioning TRT inference:
   - **Positive delta** (GC kernels take longer): Green Context IS partitioning TRT, as expected with fewer SMs.
   - **Near-zero delta**: Green Context is NOT partitioning TRT -- TensorRT may be creating its own internal CUDA streams/contexts that bypass the Holoscan-provided Green Context stream pool.

## Important Disclaimers

### CUPTI Profiling Overhead
This benchmark uses NVIDIA CUPTI for timing measurements, which introduces profiling overhead that affects absolute timing values. However, **the relative performance comparison between baseline and Green Context configurations remains valid**.

### Not Official SOL Numbers
The absolute timing values reported should not be used as reference numbers for CUDA kernel launch performance, as they include CUPTI profiling overhead and are specific to this benchmark's testing methodology.

## Technical Details

### ONNX Model

The default generated model is a fully-connected (dense) network:
```
Input(1, 1024) -> [MatMul(4096) -> ReLU] x 6 -> Output(1, 1024)
```

Each MatMul creates GPU work through TensorRT. The model size (hidden_size and num_layers) can be tuned to produce more or less GPU contention depending on the target GPU. Models exceeding 2 GB are automatically saved with ONNX external data format.

### Green Context Configuration

The benchmark dynamically allocates GPU SM partitions:
- **Partition 0**: InferenceOp (contending workload)
- **Partition 1**: TimingBenchmarkOp (measurement kernel)
- Each partition gets roughly half the GPU's SMs (minimum 4 per partition)

### Timing Kernel

The `simple_benchmark_kernel` is identical to the original Green Context benchmark -- a lightweight sin/cos computation designed to isolate launch-start latency from execution time.

### CUPTI Timestamp Handling

Unlike the original benchmark where the GPU is always busy (continuous background kernel), TRT inference has periodic idle gaps between iterations. During these gaps, the GPU can pick up the benchmark kernel almost instantly -- sometimes before the CUPTI `cuptiGetTimestamp()` callback records the launch time. The profiler uses signed arithmetic for the latency calculation and clamps slightly-negative values to zero, treating them as near-instant kernel pickup rather than rejecting them.
