# Green Context Inference Latency Benchmark

This benchmark measures **end-to-end inference pipeline latency** at high frequency (e.g., 1 kHz) under contending GPU inference workload, with and without NVIDIA CUDA Green Context. It demonstrates how Green Context SM partitioning improves latency determinism for real-time control loops sharing a GPU with heavier background workloads.

## Overview

The benchmark runs two concurrent TensorRT inference pipelines on the same GPU:

1. **Measured pipeline** -- a small, fast model triggered at a fixed rate (default 1 kHz), simulating a real-time control policy. End-to-end latency from tensor emission to inference completion is measured.
2. **Contending pipeline** -- a large, heavy model running free (or at a configurable rate), simulating a lower-priority perception or planning workload that competes for GPU SMs.

Both pipelines use **synthetic ONNX models** auto-generated at startup -- no external model files or training data are required. Model complexity is controlled entirely through command-line flags.

### What it Measures

- **End-to-end pipeline latency**: Wall-clock time from `PeriodicTxOp` emission to `TimingRxOp` receipt (includes inference + scheduling overhead)
- **Latency distribution**: Average, P50, P95, P99, min, max, standard deviation
- **Green Context improvement**: Side-by-side comparison of baseline (shared GPU) vs. Green Context (partitioned SMs)
- **Contending pipeline throughput**: Iterations completed and throughput (Hz) of the background workload

### Architecture

```text
Measured pipeline (PeriodicCondition @ frequency-hz):
  PeriodicTxOp ──► InferenceOp ──► TimingRxOp
                   [GC Partition 0]

Contending pipeline (free-running or PeriodicCondition @ contending-frequency-hz):
  PeriodicTxOp ──► InferenceOp ──► ContendingSinkOp
                   [GC Partition 1]
```

### A/B Testing Design

**Baseline:** Both pipelines run on separate CUDA streams without Green Context. The contending inference competes for all GPU SMs.

**Green Context:** Each pipeline gets a dedicated SM partition via `CudaGreenContextPool`. The measured pipeline's inference runs in an isolated partition, shielding it from contention.

## Usage

### Basic Usage

```bash
./holohub run green_context_inference_latency --docker-opts="--user root"
```

For modern GPUs (compute capability >= 7.0) or Jetson Thor, use a CUDA 13 base image:

```bash
./holohub run green_context_inference_latency \
  --docker-opts="--user root" \
  --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.10.0-cuda13
```

### Command Line Options

```text
Options:
  --samples N                Measurement samples (default: 1000)
  --warmup-samples N         Warmup iterations (default: 100)
  --backend BACKEND          'trt' or 'onnxrt' (default: trt)
  --frequency-hz N           Measured pipeline frequency in Hz (default: 1000)
  --mode MODE                'baseline', 'green-context', or 'all' (default: all)

  Measured model (high-frequency control loop):
  --measured-input-size N     Input/output dimension (default: 64)
  --measured-hidden-size N    Hidden layer width (default: 256)
  --measured-layers N         Number of FC layers (default: 3)

  Contending model (heavy lower-priority workload):
  --contending-input-size N   Input/output dimension (default: 1024)
  --contending-hidden-size N  Hidden layer width (default: 4096)
  --contending-layers N       Number of FC layers (default: 6)
  --contending-frequency-hz N Contending pipeline Hz; 0=free-running (default: 0)

  Green Context partitioning:
  --sms-per-partition N       SMs for both partitions, 0=auto (default: 0)
  --measured-sms N            SMs for measured partition only, 0=auto (default: 0)
  --contending-sms N          SMs for contending partition only, 0=auto (default: 0)

  Paths:
  --model-dir PATH           Directory for generated models (default: exe dir)
  --help                     Show this message
```

### Examples

Run with all defaults (1 kHz measured, free-running contending, TRT backend, baseline + GC):

```bash
./holohub run green_context_inference_latency \
  --docker-opts="--user root" \
  --run-args="--samples 2000"
```

Quick smoke test with ONNX Runtime (skips TRT engine compilation):

```bash
./holohub run green_context_inference_latency \
  --docker-opts="--user root" \
  --run-args="--backend onnxrt --samples 200 --warmup-samples 20"
```

Custom SM partitioning (16 SMs for measured, 120 for contending):

```bash
./holohub run green_context_inference_latency \
  --docker-opts="--user root" \
  --run-args="--measured-sms 16 --contending-sms 120 --mode green-context"
```

Heavier contending model:

```bash
./holohub run green_context_inference_latency \
  --docker-opts="--user root" \
  --run-args="--contending-hidden-size 8192 --contending-layers 10"
```

Contending pipeline at a fixed 60 Hz instead of free-running:

```bash
./holohub run green_context_inference_latency \
  --docker-opts="--user root" \
  --run-args="--contending-frequency-hz 60"
```

Show help:

```bash
./holohub run green_context_inference_latency \
  --docker-opts="--user root" \
  --run-args="--help"
```

## Synthetic Model Generation

Both models are fully-connected (dense) networks generated at runtime by `generate_onnx_model.py`. The script is called automatically if the ONNX file does not already exist. On subsequent runs with the same parameters, the cached model is reused.

**Architecture per model:**

```text
Input[1, input_size] → [MatMul(hidden_size) → ReLU] × N → Output[1, input_size]
```

Model files are placed alongside the executable with parameter-encoded names:
- `measured_i64_h256_l3.onnx`
- `contending_i1024_h4096_l6.onnx`

The `onnx` Python package is auto-installed inside the container if not already present.

### Default Model Sizes

| Parameter | Measured | Contending |
|-----------|----------|------------|
| Input/output dim | 64 | 1024 |
| Hidden dim | 256 | 4096 |
| Layers | 3 | 6 |
| Purpose | Fast control policy | Heavy background load |
| Approx. FLOPs | ~0.2M | ~151M (~768x measured) |
