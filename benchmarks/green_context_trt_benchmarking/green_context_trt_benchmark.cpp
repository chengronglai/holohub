/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// CUPTI headers
#include <cupti.h>
#include "cupti_profiler.hpp"
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <gxf/std/tensor.hpp>
#include "benchmark_cuda_kernel.cu.hpp"

using namespace holoscan;

// ============================================================================
// Statistics utilities (reused from green_context_benchmarking)
// ============================================================================

struct BenchmarkStats {
  double avg = 0.0;
  double std_dev = 0.0;
  double min_val = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max_val = 0.0;
  size_t sample_count = 0;
  std::vector<double> sorted_data;
};

double calculate_percentile(const std::vector<double>& sorted_data, double percentile) {
  if (sorted_data.empty())
    return 0.0;

  double index = (percentile / 100.0) * (sorted_data.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(index));
  size_t upper = static_cast<size_t>(std::ceil(index));

  if (lower == upper) {
    return sorted_data[lower];
  }

  double weight = index - lower;
  return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
}

double calculate_std_dev(const std::vector<double>& data, double mean) {
  if (data.size() <= 1)
    return 0.0;

  double sum_sq_diff = 0.0;
  for (double value : data) {
    double diff = value - mean;
    sum_sq_diff += diff * diff;
  }

  return std::sqrt(sum_sq_diff / (data.size() - 1));
}

BenchmarkStats calculate_benchmark_stats(
  const std::vector<double>& raw_values, bool skip_negative_values = false) {
  BenchmarkStats stats;

  for (const auto& value : raw_values) {
    if (value >= 0.0 || !skip_negative_values) {
      stats.sorted_data.push_back(value);
    }
  }

  if (stats.sorted_data.empty())
    return stats;

  std::sort(stats.sorted_data.begin(), stats.sorted_data.end());
  stats.sample_count = stats.sorted_data.size();

  stats.avg =
      std::accumulate(stats.sorted_data.begin(), stats.sorted_data.end(), 0.0) / stats.sample_count;
  stats.std_dev = calculate_std_dev(stats.sorted_data, stats.avg);

  stats.min_val = stats.sorted_data.front();
  stats.max_val = stats.sorted_data.back();
  stats.p50 = calculate_percentile(stats.sorted_data, 50.0);
  stats.p95 = calculate_percentile(stats.sorted_data, 95.0);
  stats.p99 = calculate_percentile(stats.sorted_data, 99.0);

  return stats;
}

// ============================================================================
// TensorSourceOp: Emits random CUDA tensors to feed InferenceOp
// ============================================================================

class TensorSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorSourceOp)

  TensorSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(input_size_, "input_size", "Input Size",
               "Number of float elements in the input tensor", 1024);
    spec.param(tensor_name_, "tensor_name", "Tensor Name",
               "Name of the output tensor (must match InferenceOp pre_processor_map)",
               std::string("input_tensor"));
    spec.output<holoscan::gxf::Entity>("output");
  }

  void initialize() override {
    Operator::initialize();

    int size = input_size_.get();
    size_t bytes = size * sizeof(float);

    // Allocate GPU buffer for input tensor
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&d_input_data_, bytes),
                                   "Failed to allocate GPU input buffer");

    // Fill with random data
    std::vector<float> host_data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& val : host_data) {
      val = dis(gen);
    }

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(d_input_data_, host_data.data(), bytes, cudaMemcpyHostToDevice),
        "Failed to copy input data to GPU");

    HOLOSCAN_LOG_INFO("[TensorSourceOp] Initialized: input_size={}, tensor_name={}",
                      size, tensor_name_.get());
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    int size = input_size_.get();

    // Create a GXF entity containing the tensor
    auto maybe_entity = nvidia::gxf::Entity::New(context.context());
    if (!maybe_entity) {
      throw std::runtime_error("[TensorSourceOp] Failed to create GXF entity");
    }
    auto entity = std::move(maybe_entity.value());

    // Add a tensor component to the entity
    auto maybe_tensor = entity.add<nvidia::gxf::Tensor>(tensor_name_.get().c_str());
    if (!maybe_tensor) {
      throw std::runtime_error("[TensorSourceOp] Failed to add tensor to entity");
    }
    auto tensor = maybe_tensor.value();

    // Wrap pre-allocated GPU memory into the tensor (no allocation per tick)
    const nvidia::gxf::Shape shape({1, size});
    const uint32_t element_size = sizeof(float);

    auto result = tensor->wrapMemory(
        shape,
        nvidia::gxf::PrimitiveType::kFloat32,
        element_size,
        nvidia::gxf::ComputeTrivialStrides(shape, element_size),
        nvidia::gxf::MemoryStorageType::kDevice,
        d_input_data_,
        [](void*) { return nvidia::gxf::Success; });  // no-op: memory owned by this op

    if (!result) {
      throw std::runtime_error("[TensorSourceOp] Failed to wrap memory in tensor");
    }

    // Emit the entity to the output port
    auto holoscan_entity = holoscan::gxf::Entity(std::move(entity));
    op_output.emit(holoscan_entity, "output");
  }

  ~TensorSourceOp() {
    if (d_input_data_) {
      cudaError_t err = cudaFree(d_input_data_);
      if (err != cudaSuccess) {
        HOLOSCAN_LOG_WARN("[TensorSourceOp] cudaFree failed during destruction: {}",
                          cudaGetErrorString(err));
      }
    }
  }

 private:
  Parameter<int> input_size_;
  Parameter<std::string> tensor_name_;
  float* d_input_data_ = nullptr;
};

// ============================================================================
// InferenceComputeTracker: Thread-safe collection of compute() durations
// ============================================================================

class InferenceComputeTracker {
 public:
  void record_tick(double duration_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    execution_times_us_.push_back(duration_us);
  }

  BenchmarkStats get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (execution_times_us_.empty()) {
      return BenchmarkStats{};
    }
    return calculate_benchmark_stats(execution_times_us_, false);
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    execution_times_us_.clear();
  }

 private:
  mutable std::mutex mutex_;
  std::vector<double> execution_times_us_;
};

// Global tracker for InferenceOp compute time (wall-clock around InferenceOp::compute)
static InferenceComputeTracker g_inference_compute_tracker;

// Set once inference pipeline produces its first output; gates TimingBenchmarkOp
// measurement so samples are only collected under actual inference contention.
static std::atomic<bool> g_inference_pipeline_ready{false};

// ============================================================================
// CustomInferenceOp: Wraps InferenceOp to measure compute() wall-clock time
// ============================================================================

class CustomInferenceOp : public ops::InferenceOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(CustomInferenceOp, ops::InferenceOp)
  CustomInferenceOp() = default;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto start = std::chrono::steady_clock::now();

    ops::InferenceOp::compute(op_input, op_output, context);

    auto end = std::chrono::steady_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    g_inference_compute_tracker.record_tick(static_cast<double>(duration_us));
  }
};

// ============================================================================
// InferenceTimingSinkOp: Consumes inference output and signals pipeline readiness
// ============================================================================

class InferenceTimingSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(InferenceTimingSinkOp)

  InferenceTimingSinkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::any>("in");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    // Consume input to act as a sink and signal pipeline readiness.
    (void)op_input.receive<std::any>("in");

    // Signal that inference is running; unblocks TimingBenchmarkOp measurement.
    if (!g_inference_pipeline_ready.load(std::memory_order_relaxed)) {
      g_inference_pipeline_ready.store(true, std::memory_order_release);
      HOLOSCAN_LOG_INFO("[InferenceTimingSinkOp] Inference pipeline is warmed up "
                        "(first output received). Timing measurement will begin.");
    }
  }
};

// ============================================================================
// TimingBenchmarkOp: CUPTI-based kernel launch-start time measurement
// (Reused from green_context_benchmarking)
// ============================================================================

class TimingBenchmarkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimingBenchmarkOp)

  TimingBenchmarkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(
        workload_size_, "workload_size", "Workload Size", "Size of timing workload", 1024);
    spec.param(total_samples_,
               "total_samples",
               "Total Samples",
               "Total samples to measure in the benchmark",
               100);
    spec.param(warmup_samples_,
               "warmup_samples",
               "Warmup Samples",
               "Additional samples to discard after inference pipeline is ready, "
               "to let scheduling stabilize before measurement",
               100);
  }

  void initialize() override {
    Operator::initialize();
    execution_count_ = 0;

    threads_per_block_ = 256;

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMalloc(&d_benchmark_data_, workload_size_.get() * sizeof(float)), "cudaMalloc failed");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemset(d_benchmark_data_, 0, workload_size_.get() * sizeof(float)), "cudaMemset failed");

    cupti_profiler_ = cupti_timing::CuptiSchedulingProfiler::getInstance();
    if (!cupti_profiler_->initialize()) {
      HOLOSCAN_LOG_WARN("[TimingBenchmarkOp] CUPTI initialization failed, no "
                        "CUDA kernel launch-start time measurements will be available");
      cupti_profiler_ = nullptr;
    }
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Allocate CUDA stream on first call (needed for both warmup and measurement)
    if (cached_stream_ == nullptr) {
      auto maybe_stream = context.allocate_cuda_stream("timing_stream");
      if (!maybe_stream) {
        HOLOSCAN_LOG_ERROR("[TimingBenchmarkOp] Failed to allocate non-default CUDA stream");
        throw std::runtime_error("Failed to allocate non-default CUDA stream");
      }
      cached_stream_ = maybe_stream.value();
      HOLOSCAN_LOG_INFO("[TimingBenchmarkOp] Using allocated non-default CUDA stream: {}",
        reinterpret_cast<long long>(cached_stream_));
    }

    // ---- Phase 1: Wait for inference pipeline to be ready ----
    // Don't record until inference is running, otherwise baseline measures
    // zero-contention while the GC run (with cached engine) sees real load.
    if (!g_inference_pipeline_ready.load(std::memory_order_acquire)) {
      // Run the kernel to keep the CUDA stream/context active, but discard results
      async_run_simple_benchmark_kernel(d_benchmark_data_, workload_size_.get(),
                                        threads_per_block_, cached_stream_);
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(cached_stream_),
                                     "Failed to synchronize timing stream");
      // Drain CUPTI data to prevent accumulation from warmup kernels
      if (cupti_profiler_) {
        cuptiActivityFlushAll(0);
        cupti_profiler_->getLatestSchedulingLatency();
        cupti_profiler_->getLatestExecutionDuration();
      }
      pre_ready_warmup_count_++;
      if (pre_ready_warmup_count_ == 1 || pre_ready_warmup_count_ % 500 == 0) {
        HOLOSCAN_LOG_INFO("[TimingBenchmarkOp] Waiting for inference pipeline to warm up "
                          "({} iterations so far)", pre_ready_warmup_count_);
      }
      return;
    }

    // ---- Phase 2: Post-ready stabilization warmup ----
    // After the inference pipeline starts, run a few more iterations to let
    // the GPU scheduler and TensorRT execution settle before measuring.
    if (post_ready_warmup_count_ < warmup_samples_.get()) {
      async_run_simple_benchmark_kernel(d_benchmark_data_, workload_size_.get(),
                                        threads_per_block_, cached_stream_);
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(cached_stream_),
                                     "Failed to synchronize timing stream");
      if (cupti_profiler_) {
        cuptiActivityFlushAll(0);
        cupti_profiler_->getLatestSchedulingLatency();
        cupti_profiler_->getLatestExecutionDuration();
      }
      post_ready_warmup_count_++;
      if (post_ready_warmup_count_ == warmup_samples_.get()) {
        HOLOSCAN_LOG_INFO("[TimingBenchmarkOp] Warmup complete: {} pre-ready + {} post-ready "
                          "iterations. Starting measurement...",
                          pre_ready_warmup_count_, post_ready_warmup_count_);
        // Start collecting inference kernel execution times now that warmup is done
        if (cupti_profiler_) {
          cupti_profiler_->startCollectingInferenceKernels();
        }
      }
      return;
    }

    // ---- Phase 3: Actual measurement ----
    execution_count_++;

    int log_output_interval = std::max(1, total_samples_.get() / 10);
    if (execution_count_ % log_output_interval == 0 || execution_count_ == 1) {
      HOLOSCAN_LOG_INFO("[TimingBenchmarkOp] Collecting {}/{} samples",
                        execution_count_, total_samples_.get());
    }

    // Launch kernel - CUPTI will capture launch and execution timestamps
    async_run_simple_benchmark_kernel(d_benchmark_data_, workload_size_.get(),
                                       threads_per_block_, cached_stream_);
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(cached_stream_),
                                   "Failed to synchronize timing stream");

    double cuda_kernel_launch_start_time = -1.0;
    double cuda_kernel_execution_time = -1.0;
    if (cupti_profiler_) {
      cuptiActivityFlushAll(0);

      const int max_poll_attempts = 500;
      const int initial_poll_interval_ms = 1;
      int poll_count = 0;
      bool data_ready = false;
      int backoff_factor = 1;

      while (poll_count < max_poll_attempts && !data_ready) {
        if (cupti_profiler_->hasMeasurements()) {
          data_ready = true;
          break;
        }

        if (!cupti_profiler_->hasPendingLaunches()) {
          break;
        }

        int current_interval = std::min(initial_poll_interval_ms * backoff_factor, 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(current_interval));

        if (poll_count % 5 == 0) {
          cuptiActivityFlushAll(0);
        }

        if (poll_count % 50 == 0 && backoff_factor < 5) {
          backoff_factor++;
        }

        poll_count++;
      }

      cuda_kernel_launch_start_time = cupti_profiler_->getLatestSchedulingLatency();
      cuda_kernel_execution_time = cupti_profiler_->getLatestExecutionDuration();

      if (poll_count >= max_poll_attempts) {
        HOLOSCAN_LOG_WARN("[CUPTI] Data polling timed out after {} attempts (~{}ms). "
                          "Possible buffer overflow or severe contention.", poll_count,
                          (poll_count * 2));
      }
    } else {
      HOLOSCAN_LOG_WARN("[TimingBenchmarkOp] CUPTI profiler not available!");
    }

    cuda_kernel_launch_start_times_us_.push_back(cuda_kernel_launch_start_time);
    cuda_kernel_execution_times_us_.push_back(cuda_kernel_execution_time);

    if (execution_count_ >= total_samples_.get()) {
      fragment()->stop_execution();
    }
  }

  BenchmarkStats get_cuda_kernel_launch_start_time_benchmark_stats() const {
    return calculate_benchmark_stats(cuda_kernel_launch_start_times_us_, true);
  }

  BenchmarkStats get_cuda_kernel_execution_time_benchmark_stats() const {
    return calculate_benchmark_stats(cuda_kernel_execution_times_us_, true);
  }

  BenchmarkStats get_inference_kernel_execution_time_stats() {
    if (!cupti_profiler_) return BenchmarkStats{};
    auto durations = cupti_profiler_->stopAndGetInferenceKernelDurations();
    if (durations.empty()) return BenchmarkStats{};
    return calculate_benchmark_stats(durations, false);
  }

  ~TimingBenchmarkOp() {
    if (d_benchmark_data_) {
      cudaError_t err = cudaFree(d_benchmark_data_);
      if (err != cudaSuccess) {
        HOLOSCAN_LOG_WARN("[TimingBenchmarkOp] cudaFree failed during destruction: {}",
                          cudaGetErrorString(err));
      }
    }
  }

 private:
  Parameter<int> workload_size_;
  Parameter<int> total_samples_;
  Parameter<int> warmup_samples_;
  int execution_count_ = 0;
  int pre_ready_warmup_count_ = 0;
  int post_ready_warmup_count_ = 0;
  float* d_benchmark_data_ = nullptr;
  std::vector<double> cuda_kernel_launch_start_times_us_;
  std::vector<double> cuda_kernel_execution_times_us_;
  cudaStream_t cached_stream_ = nullptr;
  cupti_timing::CuptiSchedulingProfiler* cupti_profiler_ = nullptr;
  int threads_per_block_;
};

// ============================================================================
// GreenContextTrtBenchmarkApp
// ============================================================================

class GreenContextTrtBenchmarkApp : public holoscan::Application {
 public:
  explicit GreenContextTrtBenchmarkApp(bool use_green_context, int total_samples,
                                       int warmup_samples, const std::string& model_path,
                                       int input_size, const std::string& backend = "trt",
                                       int sms_per_partition = 0)
      : use_green_context_(use_green_context),
        total_samples_(total_samples),
        warmup_samples_(warmup_samples),
        model_path_(model_path),
        input_size_(input_size),
        backend_(backend),
        sms_per_partition_(sms_per_partition) {}

  void compose() override {
    std::shared_ptr<CudaStreamPool> inference_stream_pool;
    std::shared_ptr<CudaStreamPool> timing_stream_pool;
    // GC resources need to stay in scope for add_arg to TimingBenchmarkOp
    std::shared_ptr<CudaGreenContextPool> cuda_green_context_pool;
    std::shared_ptr<CudaGreenContext> timing_green_context;

    if (use_green_context_) {
      HOLOSCAN_LOG_INFO("Initializing green context partitions for TRT benchmark");
      try {
        cudaDeviceProp prop;
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDeviceProperties(&prop, 0),
                                       "Failed to get device properties");
        HOLOSCAN_LOG_INFO("GPU: {}, SMs: {}, Compute: {}.{}", prop.name,
          prop.multiProcessorCount, prop.major, prop.minor);

        int total_sms = prop.multiProcessorCount;
        int sms_per_partition;
        if (sms_per_partition_ > 0) {
          // User-specified SM count per partition (rounded down to multiple of 4)
          sms_per_partition = sms_per_partition_ & ~3;
          sms_per_partition = std::max(4, sms_per_partition);
          HOLOSCAN_LOG_INFO("Using user-specified {} SMs per partition (requested: {})",
                            sms_per_partition, sms_per_partition_);
        } else {
          // Auto: use half the GPU SMs
          sms_per_partition = std::max(4, (total_sms / 2) & ~3);
        }

        if (sms_per_partition * 2 > total_sms) {
          throw std::runtime_error("GPU too small for requested partition size (need at least " +
                                   std::to_string(sms_per_partition * 2) + " SMs, have " +
                                   std::to_string(total_sms) + ")");
        }

        std::vector<uint32_t> partitions = {static_cast<uint32_t>(sms_per_partition),
                                            static_cast<uint32_t>(sms_per_partition)};

        HOLOSCAN_LOG_INFO("Configuring green context with {} partitions, {} SMs each "
                          "(total: {} SMs, available: {} SMs)", partitions.size(),
                          sms_per_partition, sms_per_partition * 2, total_sms);

        cuda_green_context_pool = make_resource<CudaGreenContextPool>(
            "cuda_green_context_pool",
            Arg("dev_id", 0),
            Arg("num_partitions", static_cast<uint32_t>(partitions.size())),
            Arg("sms_per_partition", partitions));

        // Partition 0 - for inference pipeline
        auto inference_green_context =
            make_resource<CudaGreenContext>("inference_green_context",
                                            Arg("cuda_green_context_pool", cuda_green_context_pool),
                                            Arg("index", static_cast<int32_t>(0)));

        inference_stream_pool = make_resource<CudaStreamPool>(
            "inference_stream_pool", 0, 0, 0, 1, 5, inference_green_context);

        // Partition 1 - for TimingBenchmarkOp
        timing_green_context =
            make_resource<CudaGreenContext>("timing_green_context",
                                            Arg("cuda_green_context_pool", cuda_green_context_pool),
                                            Arg("index", static_cast<int32_t>(1)));

        timing_stream_pool = make_resource<CudaStreamPool>(
            "timing_stream_pool", 0, 0, 0, 1, 5, timing_green_context);

        HOLOSCAN_LOG_INFO("Green context enabled with separate partitions:");
        HOLOSCAN_LOG_INFO("  - InferenceOp: Partition 0 ({} SMs)", sms_per_partition);
        HOLOSCAN_LOG_INFO("  - TimingBenchmarkOp: Partition 1 ({} SMs)", sms_per_partition);
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Failed to setup green context: {}", e.what());
        throw;
      }
    } else {
      // Baseline: Non-default streams WITHOUT Green Context partitions
      inference_stream_pool =
          make_resource<CudaStreamPool>("inference_stream_pool", 0, 0, 0, 1, 5);
      timing_stream_pool =
          make_resource<CudaStreamPool>("timing_stream_pool", 0, 0, 0, 1, 5);

      HOLOSCAN_LOG_INFO("Baseline mode: Non-default streams WITHOUT Green Context partitions");
    }

    // --- Contending workload pipeline: TensorSourceOp -> InferenceOp -> SinkOp ---

    tensor_source_op_ = make_operator<TensorSourceOp>(
        "tensor_source_op",
        Arg("input_size", input_size_),
        Arg("tensor_name", std::string("input_tensor")));

    // Configure InferenceOp model and tensor maps programmatically
    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("benchmark_model", model_path_);

    ops::InferenceOp::DataVecMap pre_processor_map;
    pre_processor_map.insert("benchmark_model", {"input_tensor"});

    ops::InferenceOp::DataVecMap inference_map;
    inference_map.insert("benchmark_model", {"output_tensor"});

    auto pool_resource = make_resource<UnboundedAllocator>("pool");

    // Must pass cuda_stream_pool as a named arg; positional does not populate
    // the Parameter so the backend falls back to a non-GC stream.
    inference_op_ = make_operator<CustomInferenceOp>(
        "inference_op",
        from_config("inference"),
        Arg("model_path_map", model_path_map),
        Arg("pre_processor_map", pre_processor_map),
        Arg("inference_map", inference_map),
        Arg("allocator") = pool_resource,
        Arg("cuda_stream_pool") = inference_stream_pool);
    if (!backend_.empty()) {
      inference_op_->add_arg(Arg("backend", backend_));
    }

    sink_op_ = make_operator<InferenceTimingSinkOp>("sink_op");

    // Wire the inference pipeline
    add_flow(tensor_source_op_, inference_op_, {{"output", "receivers"}});
    add_flow(inference_op_, sink_op_, {{"transmitter", "in"}});

    // --- Measurement operator (independent, no connections) ---

    timing_benchmark_op_ = make_operator<TimingBenchmarkOp>(
        "timing_benchmark_op",
        Arg("workload_size", 1024),
        Arg("total_samples", total_samples_),
        Arg("warmup_samples", warmup_samples_));
    add_operator(timing_benchmark_op_);

    // TimingBenchmarkOp needs the full GC resource chain attached to its entity
    // so GXF can initialize them in dependency order
    timing_benchmark_op_->add_arg(timing_stream_pool);
    if (use_green_context_) {
      timing_benchmark_op_->add_arg(timing_green_context);
      timing_benchmark_op_->add_arg(cuda_green_context_pool);
    }
  }

  BenchmarkStats get_cuda_kernel_launch_start_time_benchmark_stats() const {
    return timing_benchmark_op_->get_cuda_kernel_launch_start_time_benchmark_stats();
  }

  BenchmarkStats get_cuda_kernel_execution_time_benchmark_stats() const {
    return timing_benchmark_op_->get_cuda_kernel_execution_time_benchmark_stats();
  }

  BenchmarkStats get_inference_compute_time_benchmark_stats() const {
    return g_inference_compute_tracker.get_stats();
  }

  BenchmarkStats get_inference_kernel_execution_time_stats() {
    return timing_benchmark_op_->get_inference_kernel_execution_time_stats();
  }

 private:
  bool use_green_context_;
  int total_samples_;
  int warmup_samples_;
  std::string model_path_;
  int input_size_;
  std::string backend_;
  int sms_per_partition_;  // 0 = auto (half the GPU SMs)
  std::shared_ptr<TensorSourceOp> tensor_source_op_;
  std::shared_ptr<CustomInferenceOp> inference_op_;
  std::shared_ptr<InferenceTimingSinkOp> sink_op_;
  std::shared_ptr<TimingBenchmarkOp> timing_benchmark_op_;
};

// ============================================================================
// Output formatting (reused from green_context_benchmarking)
// ============================================================================

void print_comprehensive_timing_results(const BenchmarkStats& launch_start_stats,
                                       const BenchmarkStats& execution_stats,
                                       const std::string& context_type) {
  std::cout << "=== " << context_type << " ===" << std::endl;
  std::cout << std::fixed << std::setprecision(2) << std::dec;

  std::cout << "CUDA Kernel Launch-Start Time:" << std::endl;
  if (launch_start_stats.sample_count == 0) {
    std::cout << "  Not available" << std::endl;
    std::cout << "  (CUPTI initialization may have failed or no measurements captured)"
    << std::endl;
  } else {
    std::cout << "  Average: " << launch_start_stats.avg << " μs" << std::endl;
    std::cout << "  Std Dev: " << launch_start_stats.std_dev << " μs" << std::endl;
    std::cout << "  Min:     " << launch_start_stats.min_val << " μs" << std::endl;
    std::cout << "  P50:     " << launch_start_stats.p50 << " μs" << std::endl;
    std::cout << "  P95:     " << launch_start_stats.p95 << " μs" << std::endl;
    std::cout << "  P99:     " << launch_start_stats.p99 << " μs" << std::endl;
    std::cout << "  Max:     " << launch_start_stats.max_val << " μs" << std::endl;
    std::cout << "  Samples: " << launch_start_stats.sample_count << std::endl;
  }

  std::cout << std::endl;

  std::cout << "CUDA Kernel Execution Time:" << std::endl;
  if (execution_stats.sample_count == 0) {
    std::cout << "  Not available" << std::endl;
    std::cout << "  (CUPTI initialization may have failed or no measurements captured)"
    << std::endl;
  } else {
    std::cout << "  Average: " << execution_stats.avg << " μs" << std::endl;
    std::cout << "  Std Dev: " << execution_stats.std_dev << " μs" << std::endl;
    std::cout << "  Min:     " << execution_stats.min_val << " μs" << std::endl;
    std::cout << "  P50:     " << execution_stats.p50 << " μs" << std::endl;
    std::cout << "  P95:     " << execution_stats.p95 << " μs" << std::endl;
    std::cout << "  P99:     " << execution_stats.p99 << " μs" << std::endl;
    std::cout << "  Max:     " << execution_stats.max_val << " μs" << std::endl;
    std::cout << "  Samples: " << execution_stats.sample_count << std::endl;
  }
}

void print_title(const std::string& title) {
  std::cout << std::string(80, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(80, '=') << std::endl;
}

void print_benchmark_config(const std::string& mode, int total_samples,
                            int warmup_samples, const std::string& model_path,
                            int input_size, const std::string& backend,
                            int sms_per_partition) {
  std::cout << "  Benchmark Mode: " << mode << std::endl;
  std::cout << "  Backend: " << backend << std::endl;
  std::cout << "  Measurement Samples: " << total_samples << std::endl;
  std::cout << "  Warmup Samples: " << warmup_samples
            << " (+ engine build wait)" << std::endl;
  std::cout << "  Model Path: " << model_path << std::endl;
  std::cout << "  Input Size: " << input_size << std::endl;
  std::cout << "  SMs Per Partition: "
            << (sms_per_partition > 0 ? std::to_string(sms_per_partition) : "auto (half GPU)")
            << std::endl;
}

void print_usage(const char* program_name) {
  std::cout << "Green Context TRT Inference Benchmark\n\n";
  std::cout
      << "Measures CUDA kernel launch-start time with inference as contending "
         "workload\n\n";
  std::cout << "Usage: " << program_name << " [OPTIONS]\n";
  std::cout << "Options:\n";
  std::cout << "  --samples N          Number of timing samples to measure (default: 1000)\n";
  std::cout << "  --warmup-samples N   Post-ready warmup iterations to discard (default: 100)\n";
  std::cout << "  --model-path PATH    Path to ONNX model file (default: ./benchmark_model.onnx)\n";
  std::cout << "  --input-size N       Input tensor size matching the model (default: 1024)\n";
  std::cout << "  --backend BACKEND    Inference backend: 'trt' or 'onnxrt' (default: from YAML)\n";
  std::cout << "  --sms-per-partition N SMs per green context partition (default: auto = half GPU)\n";
  std::cout << "                        Use smaller values (8, 16) to stress-test SM partitioning\n";
  std::cout << "  --mode MODE          Run mode: 'baseline', 'green-context', "
               "or 'all' (default: all)\n";
  std::cout << "                        baseline: Run only without green context\n";
  std::cout << "                        green-context: Run only with green context\n";
  std::cout << "                        all: Run both and show comparison\n";
  std::cout << "  --help               Show this help message\n";
  std::cout << "\nBefore running, generate the ONNX model:\n";
  std::cout << "  python generate_onnx_model.py --output benchmark_model.onnx\n";
  std::cout << "\nExample:\n";
  std::cout << "  " << program_name
            << " --samples 1000 --model-path ./benchmark_model.onnx --mode all\n";
}

int main(int argc, char* argv[]) {
  // Default values
  int total_samples = 1000;
  int warmup_samples = 100;
  std::string model_path = "benchmark_model.onnx";
  int input_size = 1024;
  std::string backend;  // empty = use YAML config value
  int sms_per_partition = 0;  // 0 = auto (half the GPU)
  std::string mode = "all";

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--samples" && i + 1 < argc) {
      total_samples = std::atoi(argv[++i]);
      if (total_samples <= 0) {
        std::cerr << "Error: samples must be positive\n";
        return 1;
      }
    } else if (arg == "--warmup-samples" && i + 1 < argc) {
      warmup_samples = std::atoi(argv[++i]);
      if (warmup_samples < 0) {
        std::cerr << "Error: warmup-samples must be non-negative\n";
        return 1;
      }
    } else if (arg == "--model-path" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--input-size" && i + 1 < argc) {
      input_size = std::atoi(argv[++i]);
      if (input_size <= 0) {
        std::cerr << "Error: input-size must be positive\n";
        return 1;
      }
    } else if (arg == "--backend" && i + 1 < argc) {
      backend = argv[++i];
      if (backend != "trt" && backend != "onnxrt") {
        std::cerr << "Error: backend must be 'trt' or 'onnxrt'\n";
        return 1;
      }
    } else if (arg == "--sms-per-partition" && i + 1 < argc) {
      sms_per_partition = std::atoi(argv[++i]);
      if (sms_per_partition < 0) {
        std::cerr << "Error: sms-per-partition must be non-negative (0 = auto)\n";
        return 1;
      }
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
      if (mode != "baseline" && mode != "green-context" && mode != "all") {
        std::cerr << "Error: mode must be 'baseline', 'green-context', or 'all'\n";
        return 1;
      }
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  // Verify model file exists
  if (!std::filesystem::exists(model_path)) {
    std::cerr << "Error: Model file not found: " << model_path << "\n";
    std::cerr << "Generate it first: python generate_onnx_model.py --output " << model_path
              << "\n";
    return 1;
  }

  // Resolve to absolute path for InferenceOp
  model_path = std::filesystem::absolute(model_path).string();

  // Resolve YAML config path (same directory as the executable).
  std::filesystem::path exe_dir;
  try {
    exe_dir = std::filesystem::canonical(argv[0]).parent_path();
  } catch (const std::filesystem::filesystem_error&) {
    exe_dir = std::filesystem::absolute(argv[0]).parent_path();
  }
  auto config_path = exe_dir / "green_context_trt_benchmark.yaml";
  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Error: Benchmark config file not found: " << config_path << "\n";
    return 1;
  }

  // If --backend was not specified on CLI, read it from the YAML config
  // so that the printed configuration always shows the resolved value.
  if (backend.empty()) {
    auto yaml_config = holoscan::Config(config_path);
    auto& yaml_node = yaml_config.yaml_nodes()[0];
    if (yaml_node["inference"] && yaml_node["inference"]["backend"]) {
      backend = yaml_node["inference"]["backend"].as<std::string>();
    } else {
      backend = "trt";
    }
  }

  print_title("Green Context Inference Benchmark");
  std::cout << "Benchmark Configurations:" << std::endl;
  print_benchmark_config(mode, total_samples, warmup_samples, model_path, input_size, backend,
                         sms_per_partition);

  // Initialize CUDA
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(0), "Failed to set CUDA device");

  BenchmarkStats cuda_kernel_launch_start_time_stats_without_gc,
                 cuda_kernel_launch_start_time_stats_with_gc;
  BenchmarkStats inference_compute_stats_without_gc, inference_compute_stats_with_gc;
  BenchmarkStats cuda_kernel_execution_stats_without_gc, cuda_kernel_execution_stats_with_gc;
  BenchmarkStats inference_kernel_exec_stats_without_gc, inference_kernel_exec_stats_with_gc;

  // Run benchmark without green context (baseline)
  if (mode == "baseline" || mode == "all") {
    print_title("Running benchmark for baseline\n"
                "(non-default CUDA streams, TRT inference contention, without green context)");

    // Reset global state for this run
    g_inference_compute_tracker.reset();
    g_inference_pipeline_ready.store(false, std::memory_order_release);

    try {
      auto app_no_gc = std::make_unique<GreenContextTrtBenchmarkApp>(
          false, total_samples, warmup_samples, model_path, input_size, backend,
          sms_per_partition);
      app_no_gc->config(config_path);
      app_no_gc->scheduler(app_no_gc->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));
      app_no_gc->run();
      cuda_kernel_launch_start_time_stats_without_gc =
          app_no_gc->get_cuda_kernel_launch_start_time_benchmark_stats();
      inference_compute_stats_without_gc =
          app_no_gc->get_inference_compute_time_benchmark_stats();
      cuda_kernel_execution_stats_without_gc =
          app_no_gc->get_cuda_kernel_execution_time_benchmark_stats();
      inference_kernel_exec_stats_without_gc =
          app_no_gc->get_inference_kernel_execution_time_stats();
      std::cout << "Baseline benchmark completed" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Baseline benchmark failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Run benchmark with green context
  if (mode == "green-context" || mode == "all") {
    print_title("Running main benchmark\n"
                "(with green context, TRT inference in separate partition)");

    // Reset global state for this run
    g_inference_compute_tracker.reset();
    g_inference_pipeline_ready.store(false, std::memory_order_release);

    try {
      auto app_with_gc = std::make_unique<GreenContextTrtBenchmarkApp>(
          true, total_samples, warmup_samples, model_path, input_size, backend,
          sms_per_partition);
      app_with_gc->config(config_path);
      app_with_gc->scheduler(app_with_gc->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));
      app_with_gc->run();
      cuda_kernel_launch_start_time_stats_with_gc =
          app_with_gc->get_cuda_kernel_launch_start_time_benchmark_stats();
      inference_compute_stats_with_gc =
          app_with_gc->get_inference_compute_time_benchmark_stats();
      cuda_kernel_execution_stats_with_gc =
          app_with_gc->get_cuda_kernel_execution_time_benchmark_stats();
      inference_kernel_exec_stats_with_gc =
          app_with_gc->get_inference_kernel_execution_time_stats();
      std::cout << "Main benchmark completed" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Main benchmark failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Display benchmark configurations
  print_title("Benchmark Configurations");
  print_benchmark_config(mode, total_samples, warmup_samples, model_path, input_size, backend,
                         sms_per_partition);
  std::cout << std::endl;

  // Display comprehensive benchmark results
  print_title("Comprehensive Timing Results");

  if (mode == "baseline" || mode == "all") {
    print_comprehensive_timing_results(cuda_kernel_launch_start_time_stats_without_gc,
                                      cuda_kernel_execution_stats_without_gc,
                                      "Without Green Context (Baseline)");
    std::cout << std::endl;
  }

  if (mode == "green-context" || mode == "all") {
    print_comprehensive_timing_results(cuda_kernel_launch_start_time_stats_with_gc,
                                      cuda_kernel_execution_stats_with_gc,
                                      "With Green Context");
    std::cout << std::endl;
  }

  // Performance Comparison - only show when mode is "all"
  if (mode == "all" && cuda_kernel_launch_start_time_stats_without_gc.sample_count > 0 &&
      cuda_kernel_launch_start_time_stats_with_gc.sample_count > 0) {
    print_title("Baseline and Green Context Benchmark Comparison");

    double avg_improvement = ((cuda_kernel_launch_start_time_stats_without_gc.avg -
                               cuda_kernel_launch_start_time_stats_with_gc.avg) /
                              cuda_kernel_launch_start_time_stats_without_gc.avg) * 100.0;
    double p95_improvement = ((cuda_kernel_launch_start_time_stats_without_gc.p95 -
                               cuda_kernel_launch_start_time_stats_with_gc.p95) /
                              cuda_kernel_launch_start_time_stats_without_gc.p95) * 100.0;
    double p99_improvement = ((cuda_kernel_launch_start_time_stats_without_gc.p99 -
                               cuda_kernel_launch_start_time_stats_with_gc.p99) /
                              cuda_kernel_launch_start_time_stats_without_gc.p99) * 100.0;

    std::cout << std::fixed << std::setprecision(2) << std::dec;

    std::cout << "Launch-Start Latency:" << std::endl;
    std::cout << "  Average Latency:  " << std::setw(8)
              << cuda_kernel_launch_start_time_stats_without_gc.avg << " μs → "
              << std::setw(8) << cuda_kernel_launch_start_time_stats_with_gc.avg
              << " μs  (" << std::showpos << avg_improvement << std::noshowpos << "%)"
              << std::endl;
    std::cout << "  95th Percentile:  " << std::setw(8)
              << cuda_kernel_launch_start_time_stats_without_gc.p95 << " μs → "
              << std::setw(8) << cuda_kernel_launch_start_time_stats_with_gc.p95
              << " μs  (" << std::showpos << p95_improvement << std::noshowpos << "%)"
              << std::endl;
    std::cout << "  99th Percentile:  " << std::setw(8)
              << cuda_kernel_launch_start_time_stats_without_gc.p99 << " μs → "
              << std::setw(8) << cuda_kernel_launch_start_time_stats_with_gc.p99
              << " μs  (" << std::showpos << p99_improvement << std::noshowpos << "%)"
              << std::endl << std::endl;

    if (cuda_kernel_execution_stats_without_gc.sample_count > 0 &&
        cuda_kernel_execution_stats_with_gc.sample_count > 0) {
      double exec_avg_improvement = ((cuda_kernel_execution_stats_without_gc.avg -
                                     cuda_kernel_execution_stats_with_gc.avg) /
                                    cuda_kernel_execution_stats_without_gc.avg) * 100.0;
      double exec_p95_improvement = ((cuda_kernel_execution_stats_without_gc.p95 -
                                     cuda_kernel_execution_stats_with_gc.p95) /
                                    cuda_kernel_execution_stats_without_gc.p95) * 100.0;
      double exec_p99_improvement = ((cuda_kernel_execution_stats_without_gc.p99 -
                                     cuda_kernel_execution_stats_with_gc.p99) /
                                    cuda_kernel_execution_stats_without_gc.p99) * 100.0;

      std::cout << "Kernel Execution Time:" << std::endl;
      std::cout << "  Average Duration: " << std::setw(8)
                << cuda_kernel_execution_stats_without_gc.avg << " μs → "
                << std::setw(8) << cuda_kernel_execution_stats_with_gc.avg
                << " μs  (" << std::showpos << exec_avg_improvement << std::noshowpos << "%)"
                << std::endl;
      std::cout << "  95th Percentile:  " << std::setw(8)
                << cuda_kernel_execution_stats_without_gc.p95 << " μs → "
                << std::setw(8) << cuda_kernel_execution_stats_with_gc.p95
                << " μs  (" << std::showpos << exec_p95_improvement << std::noshowpos << "%)"
                << std::endl;
      std::cout << "  99th Percentile:  " << std::setw(8)
                << cuda_kernel_execution_stats_without_gc.p99 << " μs → "
                << std::setw(8) << cuda_kernel_execution_stats_with_gc.p99
                << " μs  (" << std::showpos << exec_p99_improvement << std::noshowpos << "%)"
                << std::endl << std::endl;
    }
  }

  // Reusable printer for any BenchmarkStats section
  auto print_stats_section = [](const BenchmarkStats& stats, const std::string& label,
                                const std::string& count_label = "Samples",
                                const std::string& empty_msg = "Not available") {
    std::cout << "=== " << label << " ===" << std::endl;
    if (stats.sample_count == 0) {
      std::cout << "  " << empty_msg << std::endl << std::endl;
      return;
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Average: " << stats.avg << " μs" << std::endl;
    std::cout << "  Std Dev: " << stats.std_dev << " μs" << std::endl;
    std::cout << "  Min:     " << stats.min_val << " μs" << std::endl;
    std::cout << "  P50:     " << stats.p50 << " μs" << std::endl;
    std::cout << "  P95:     " << stats.p95 << " μs" << std::endl;
    std::cout << "  P99:     " << stats.p99 << " μs" << std::endl;
    std::cout << "  Max:     " << stats.max_val << " μs" << std::endl;
    std::cout << "  " << count_label << ": " << stats.sample_count << std::endl << std::endl;
  };

  // Display InferenceOp compute() wall-clock time
  print_title("TRT InferenceOp Compute Time (Wall-Clock)");

  if (mode == "baseline" || mode == "all") {
    print_stats_section(inference_compute_stats_without_gc,
                        "Without Green Context (Baseline)");
  }
  if (mode == "green-context" || mode == "all") {
    print_stats_section(inference_compute_stats_with_gc,
                        "With Green Context");
  }

  if (mode == "all" &&
      inference_compute_stats_without_gc.sample_count > 0 &&
      inference_compute_stats_with_gc.sample_count > 0) {
    double avg_change = ((inference_compute_stats_with_gc.avg -
                          inference_compute_stats_without_gc.avg) /
                         inference_compute_stats_without_gc.avg) * 100.0;
    std::cout << "=== InferenceOp Compute Time Change (BL → GC) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << std::showpos;
    std::cout << "  Average: " << avg_change << "%" << std::endl;
    std::cout << std::noshowpos;
    std::cout << "  (Positive = inference takes longer with GC → GC is partitioning)"
              << std::endl << std::endl;
  }

  // Display per-kernel inference GPU execution time (CUPTI-measured)
  print_title("TRT Inference Per-Kernel GPU Execution Time (CUPTI)");

  if (mode == "baseline" || mode == "all") {
    print_stats_section(inference_kernel_exec_stats_without_gc,
                        "Without Green Context (Baseline)", "Kernels",
                        "Not available (no inference kernels captured)");
  }
  if (mode == "green-context" || mode == "all") {
    print_stats_section(inference_kernel_exec_stats_with_gc,
                        "With Green Context", "Kernels",
                        "Not available (no inference kernels captured)");
  }

  if (mode == "all" &&
      inference_kernel_exec_stats_without_gc.sample_count > 0 &&
      inference_kernel_exec_stats_with_gc.sample_count > 0) {
    double avg_change = ((inference_kernel_exec_stats_with_gc.avg -
                          inference_kernel_exec_stats_without_gc.avg) /
                         inference_kernel_exec_stats_without_gc.avg) * 100.0;
    double p50_change = ((inference_kernel_exec_stats_with_gc.p50 -
                          inference_kernel_exec_stats_without_gc.p50) /
                         inference_kernel_exec_stats_without_gc.p50) * 100.0;
    double p95_change = ((inference_kernel_exec_stats_with_gc.p95 -
                          inference_kernel_exec_stats_without_gc.p95) /
                         inference_kernel_exec_stats_without_gc.p95) * 100.0;

    std::cout << "=== Inference Kernel Execution Time Change (BL → GC) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << std::showpos;
    std::cout << "  Average: " << avg_change << "%" << std::endl;
    std::cout << "  P50:     " << p50_change << "%" << std::endl;
    std::cout << "  P95:     " << p95_change << "%" << std::endl;
    std::cout << std::noshowpos;
    std::cout << "  (Positive = GC inference kernels take LONGER → fewer SMs, GC is partitioning)"
              << std::endl;
    std::cout << "  (Near zero = GC may NOT be partitioning TRT inference)" << std::endl;
    std::cout << std::endl;
  }

  // Cleanup CUPTI profiler singleton
  auto* cupti_profiler = cupti_timing::CuptiSchedulingProfiler::getInstance();
  if (cupti_profiler) {
    cupti_profiler->cleanup();
  }

  return 0;
}
