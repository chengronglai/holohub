/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// CUPTI headers
#include <cupti.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>

using namespace holoscan;

// CUPTI Scheduling Latency Measurement Infrastructure
namespace cupti_timing {

// Structure to hold kernel launch timing data
struct KernelLaunchData {
  uint64_t launch_timestamp;
  const char* kernel_name;
  cudaStream_t stream;
};

// Global state for CUPTI measurements
class CuptiSchedulingProfiler {
 private:
  static CuptiSchedulingProfiler* instance_;
  static std::mutex mutex_;

  CUpti_SubscriberHandle subscriber_;
  std::unordered_map<uint32_t, KernelLaunchData> launch_map_;  // correlationId -> launch data
  std::unordered_map<uint32_t, double> scheduling_latencies_;  // correlationId -> latency (us)
  std::mutex data_mutex_;
  bool initialized_;
  std::atomic<int> successful_measurements_{0};  // Count successful measurements

 public:
  static CuptiSchedulingProfiler* getInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ == nullptr) {
      instance_ = new CuptiSchedulingProfiler();
    }
    return instance_;
  }

  CuptiSchedulingProfiler() : initialized_(false) {}

  ~CuptiSchedulingProfiler() {
    if (initialized_) {
      cleanup();
    }
  }

  bool initialize() {
    if (initialized_)
      return true;

    try {
      // Subscribe to CUPTI callbacks
      CUptiResult result = cuptiSubscribe(&subscriber_, (CUpti_CallbackFunc)apiCallback, this);
      if (result != CUPTI_SUCCESS) {
        std::cerr << "[CUPTI] Failed to subscribe to callbacks" << std::endl;
        return false;
      }

      // Enable callback domain for CUDA runtime API
      result = cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API);
      if (result != CUPTI_SUCCESS) {
        std::cerr << "[CUPTI] Failed to enable runtime API domain" << std::endl;
        cuptiUnsubscribe(subscriber_);
        return false;
      }

      // Enable activity for kernel execution tracing
      result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
      if (result != CUPTI_SUCCESS) {
        std::cerr << "[CUPTI] Failed to enable concurrent kernel activity" << std::endl;
        cuptiUnsubscribe(subscriber_);
        return false;
      }

      // Register activity flush callback
      result = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
      if (result != CUPTI_SUCCESS) {
        std::cerr << "[CUPTI] Failed to register activity callbacks" << std::endl;
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        cuptiUnsubscribe(subscriber_);
        return false;
      }

      initialized_ = true;
      std::cout << "[CUPTI] Successfully initialized scheduling latency profiler" << std::endl;
      return true;
    } catch (const std::exception& e) {
      std::cerr << "[CUPTI] Initialization error: " << e.what() << std::endl;
      return false;
    }
  }

  void cleanup() {
    if (!initialized_)
      return;

    // Flush remaining activity records
    cuptiActivityFlushAll(0);

    // Disable activities and callbacks
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiUnsubscribe(subscriber_);
    initialized_ = false;
  }

  // Check if measurements are available (thread-safe)
  bool hasMeasurements() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return !scheduling_latencies_.empty();
  }

  // Check if there are pending launches waiting for activity records (thread-safe)
  bool hasPendingLaunches() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return !launch_map_.empty();
  }

  // Get latest scheduling latency measurement and clear others
  double getLatestSchedulingLatency() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (scheduling_latencies_.empty()) {
      if (!launch_map_.empty()) {
        std::cout << "[CUPTI] WARNING: " << launch_map_.size()
                  << " kernel launches detected but no GPU activity records matched!" << std::endl;
      }
      return -1.0;
    }

    // Find the measurement with the highest correlation ID (most recent)
    auto max_it = std::max_element(
        scheduling_latencies_.begin(),
        scheduling_latencies_.end(),
        [](const std::pair<uint32_t, double>& a, const std::pair<uint32_t, double>& b) {
          return a.first < b.first;  // Compare correlation IDs
        });

    double latest_latency = max_it->second;

    // Clear all measurements to prevent accumulation
    scheduling_latencies_.clear();
    launch_map_.clear();

    return latest_latency;
  }

 private:
  // CUPTI API callback for kernel launches
  static void CUPTIAPI apiCallback(void* userdata, CUpti_CallbackDomain domain,
                                   CUpti_CallbackId callbackId, const CUpti_CallbackData* cbdata) {
    CuptiSchedulingProfiler* profiler = (CuptiSchedulingProfiler*)userdata;

    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
        (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
         callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020)) {
      if (cbdata->callbackSite == CUPTI_API_ENTER) {
        // Record kernel launch timestamp
        uint64_t timestamp;
        cuptiGetTimestamp(&timestamp);

        // Store all launch data - we'll filter by kernel name later in activity processing
        // This is safe because we don't access cbdata->symbolName here
        std::lock_guard<std::mutex> lock(profiler->data_mutex_);
        KernelLaunchData data;
        data.launch_timestamp = timestamp;
        data.kernel_name = nullptr;  // Don't store the name to avoid crashes
        data.stream = 0;

        profiler->launch_map_[cbdata->correlationId] = data;
      }
    }
  }

  // CUPTI activity buffer management - increased buffer size for high contention
  static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 64 * 1024;  // Increased to 64KB buffer for better handling under load
    *buffer = (uint8_t*)malloc(*size);
    *maxNumRecords = 0;

    if (*buffer == nullptr) {
      std::cerr << "[CUPTI] ERROR: Failed to allocate activity buffer!" << std::endl;
      *size = 0;
    }
  }

  static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                                       size_t size, size_t validSize) {
    CuptiSchedulingProfiler* profiler = CuptiSchedulingProfiler::getInstance();

    if (validSize > 0) {
      // Check for potential buffer overflow
      if (validSize == size) {
        std::cout
            << "[CUPTI] WARNING: Activity buffer may have overflowed (validSize == bufferSize: "
            << validSize << ")" << std::endl;
      }
      profiler->processActivityBuffer(buffer, validSize);
    } else {
      std::cout << "[CUPTI] WARNING: Empty activity buffer received" << std::endl;
    }
    free(buffer);
  }

  void processActivityBuffer(uint8_t* buffer, size_t validSize) {
    CUpti_Activity* record = nullptr;
    int timing_kernels_found = 0;
    int matches_found = 0;

    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        CUpti_ActivityKernel9* kernelRecord = (CUpti_ActivityKernel9*)record;

        // Filter by kernel name - only process timing benchmark kernels
        bool is_timing_kernel = false;
        if (kernelRecord->name) {
          const char* kernel_name = kernelRecord->name;
          is_timing_kernel = (strstr(kernel_name, "simple_benchmark_kernel") != nullptr);
        }

        if (is_timing_kernel) {
          timing_kernels_found++;

          std::lock_guard<std::mutex> lock(data_mutex_);
          auto it = launch_map_.find(kernelRecord->correlationId);
          if (it != launch_map_.end()) {
            matches_found++;

            // Calculate scheduling latency: GPU execution start - CPU launch time
            double latency_ns = (double)(kernelRecord->start - it->second.launch_timestamp);
            double latency_us = latency_ns / 1000.0;  // Convert to microseconds

            // Sanity check - latency should be reasonable (0.1μs to 10ms)
            if (latency_us >= 0.1 && latency_us <= 10000.0) {
              scheduling_latencies_[kernelRecord->correlationId] = latency_us;
              successful_measurements_++;  // Count successful measurements
            }

            // Remove from launch_map to save memory
            launch_map_.erase(it);
          }
        } else {
          // This was a background kernel, remove from launch_map if present
          std::lock_guard<std::mutex> lock(data_mutex_);
          auto it = launch_map_.find(kernelRecord->correlationId);
          if (it != launch_map_.end()) {
            launch_map_.erase(it);  // Clean up background kernel launch data
          }
        }
      }
    }

    // Optional: Log only if there are issues
    if (matches_found == 0 && timing_kernels_found > 0) {
      std::cout << "[CUPTI] WARNING: Found " << timing_kernels_found
                << " timing kernels but no matches in activity buffer" << std::endl;
    }
  }
};

// Static member definitions
CuptiSchedulingProfiler* CuptiSchedulingProfiler::instance_ = nullptr;
std::mutex CuptiSchedulingProfiler::mutex_;

}  // namespace cupti_timing

// CUDA kernels for benchmarking
// Simple kernel for benchmarking GPU scheduling
__global__ void simple_benchmark_kernel(float* data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size)
    return;

  // Simple computation to create meaningful GPU work
  float value = (float)idx;
  value = value * 1.01f + 0.001f;
  value = sinf(value) + cosf(value);

  data[idx] = value;
}

__global__ void background_load_kernel(float* data, int size, int intensity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float value = data[idx];

  // Create heavy computational load to stress GPU scheduler
  for (int i = 0; i < intensity; i++) {
    value = sinf(value) * cosf(value + idx);
    value += sqrtf(fabsf(value)) * 0.1f;
    value = fmaf(value, 1.01f, 0.001f);

    // Add memory access patterns
    if (i % 10 == 0) {
      value += data[(idx + i) % size] * 0.001f;
    }
  }
  data[idx] = value;
}

struct TimingResults {
  double scheduling_latency_us = 0.0;  // kernel launch → GPU execution start (CUPTI-based)
  bool success = false;
};

// Forward declare calculateStdDev function
double calculateStdDev(const std::vector<double>& data, double mean);

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

class DummyLoadOp : public Operator {
 private:
  Parameter<int> load_intensity_;
  Parameter<int> workload_size_;
  Parameter<int> threads_per_block_;
  Parameter<int> total_iterations_;  // Total iterations for completion tracking
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;
  Parameter<std::shared_ptr<CudaGreenContextPool>> cuda_green_context_pool_;
  Parameter<std::shared_ptr<CudaGreenContext>> cuda_green_context_;
  float* d_load_data_;
  cudaStream_t cached_stream_ = nullptr;
  std::vector<double> execution_times_us_;  // Store execution times for statistics
  mutable std::mutex timing_mutex_;      // Protect timing data access (mutable for const methods)
  std::atomic<int> execution_count_{0};  // Non-static execution counter for this instance

 public:
  std::chrono::steady_clock::time_point completion_timestamp_{};  // Initialize to epoch (default)
  bool has_completed_{false};  // Track if operator actually completed
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyLoadOp)

  DummyLoadOp() = default;

  void setup(OperatorSpec& spec) override {
    // Add dummy output to make operator schedulable with condition
    spec.output<holoscan::gxf::Entity>("dummy_out");
    spec.param(
        load_intensity_, "load_intensity", "Load Intensity", "GPU load intensity factor", 100);
    spec.param(workload_size_, "workload_size", "Workload Size", "Size of GPU workload", 262144);
    spec.param(threads_per_block_,
               "threads_per_block",
               "Threads Per Block",
               "CUDA threads per block for GPU kernel",
               512);
    spec.param(total_iterations_,
               "total_iterations",
               "Total Iterations",
               "Total iterations for completion tracking",
               1000);
    spec.param(cuda_stream_pool_,
               "cuda_stream_pool",
               "CudaStreamPool",
               "Pool of CUDA Streams",
               std::shared_ptr<CudaStreamPool>(nullptr));
    spec.param(cuda_green_context_pool_,
               "cuda_green_context_pool",
               "CudaGreenContextPool",
               "Pool of CUDA Green Contexts",
               std::shared_ptr<CudaGreenContextPool>(nullptr));
    spec.param(cuda_green_context_,
               "cuda_green_context",
               "CudaGreenContext",
               "CUDA Green Context for stream isolation",
               std::shared_ptr<CudaGreenContext>(nullptr));
  }

  void initialize() override {
    Operator::initialize();

    int load_intensity = load_intensity_.get();
    int workload_size = workload_size_.get();

    // Initialize CUDA resources
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&d_load_data_, workload_size * sizeof(float)),
                                   "cudaMalloc failed!");

    // Initialize with random data
    std::vector<float> host_data(workload_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (auto& val : host_data) {
      val = dis(gen);
    }

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(
            d_load_data_, host_data.data(), workload_size * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy failed!");

    std::cout << "[DummyLoadOp] GPU load initialized: " << workload_size << " elements, intensity "
              << load_intensity << std::endl;
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    int current_execution = ++execution_count_;

    auto tick_start_time = std::chrono::steady_clock::now();

    createBackgroundLoad(context);

    // Emit dummy message to satisfy output requirement
    auto dummy_message = holoscan::gxf::Entity::New(&context);
    op_output.emit(dummy_message, "dummy_out");

    auto tick_end_time = std::chrono::steady_clock::now();
    auto tick_duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(tick_end_time - tick_start_time)
            .count();

    {
      std::lock_guard<std::mutex> lock(timing_mutex_);
      execution_times_us_.push_back(static_cast<double>(tick_duration_us));
    }

    // Store completion timestamp when target iterations reached
    int total_iters = total_iterations_.get();
    if (current_execution >= total_iters) {
      completion_timestamp_ = std::chrono::steady_clock::now();
      has_completed_ = true;
    }

    // Log every 1000th execution to avoid spam
    if (current_execution % 1000 == 0) {
      std::cout << "[DummyLoadOp] Execution #" << current_execution << " completed in "
                << tick_duration_us << "μs" << std::endl;
    }
  }

  BenchmarkStats getExecutionStats() const {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    BenchmarkStats stats;

    if (execution_times_us_.empty()) {
      return stats;
    }

    stats.sorted_data = execution_times_us_;
    std::sort(stats.sorted_data.begin(), stats.sorted_data.end());
    stats.sample_count = stats.sorted_data.size();

    stats.avg = std::accumulate(stats.sorted_data.begin(), stats.sorted_data.end(), 0.0) /
                stats.sample_count;

    stats.std_dev = calculateStdDev(stats.sorted_data, stats.avg);

    return stats;
  }

  ~DummyLoadOp() {
    if (d_load_data_) {
      cudaError_t cuda_status = cudaFree(d_load_data_);
      if (cuda_status != cudaSuccess) {
        std::cerr << "[DummyLoadOp] cudaFree failed: " << cudaGetErrorString(cuda_status)
                  << std::endl;
      }
    }
  }

 private:
  void createBackgroundLoad(ExecutionContext& context) {
    int workload_size = workload_size_.get();
    int load_intensity = load_intensity_.get();

    if (cached_stream_ == nullptr) {
      auto maybe_stream = context.allocate_cuda_stream("dummy_load_stream");
      if (maybe_stream) {
        cached_stream_ = maybe_stream.value();
        std::cout << "[DummyLoadOp] Using allocated non-default stream: " << cached_stream_
                  << std::endl;
      } else {
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamCreate(&cached_stream_),
                                       "cudaStreamCreate failed!");
        std::cout << "[DummyLoadOp] Created non-default stream: " << cached_stream_ << std::endl;
      }
    }

    // Launch background load kernel on non-default stream
    int threads_per_block = threads_per_block_.get();
    int blocks = (workload_size + threads_per_block - 1) / threads_per_block;

    // Launch background load to create GPU contention
    background_load_kernel<<<blocks, threads_per_block, 0, cached_stream_>>>(
        d_load_data_, workload_size, load_intensity);

    // Synchronize to ensure background load actually runs and creates GPU contention
    cudaStreamSynchronize(cached_stream_);
  }
};

class TimingBenchmarkOp : public Operator {
 private:
  Parameter<int> timing_workload_;
  Parameter<int> total_iterations_;  // Total iterations for logging control
  Parameter<std::shared_ptr<CudaGreenContextPool>> cuda_green_context_pool_;
  Parameter<std::shared_ptr<CudaGreenContext>> cuda_green_context_;
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;
  float* d_benchmark_data_;
  std::vector<TimingResults> results_;
  cudaStream_t cached_stream_ = nullptr;
  cupti_timing::CuptiSchedulingProfiler* cupti_profiler_;
  std::atomic<int> execution_count_{0};  // Non-static execution counter for this instance

 public:
  std::chrono::steady_clock::time_point completion_timestamp_{};  // Initialize to epoch (default)
  bool has_completed_{false};  // Track if operator actually completed
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimingBenchmarkOp)

  TimingBenchmarkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("out");
    spec.param(
        timing_workload_, "timing_workload", "Timing Workload", "Size of timing workload", 1024);
    spec.param(total_iterations_,
               "total_iterations",
               "Total Iterations",
               "Total iterations for logging control",
               100);
    spec.param(cuda_green_context_pool_,
               "cuda_green_context_pool",
               "CudaGreenContextPool",
               "Pool of CUDA Green Contexts",
               std::shared_ptr<CudaGreenContextPool>(nullptr));
    spec.param(cuda_green_context_,
               "cuda_green_context",
               "CudaGreenContext",
               "CUDA Green Context for stream isolation",
               std::shared_ptr<CudaGreenContext>(nullptr));
    spec.param(cuda_stream_pool_,
               "cuda_stream_pool",
               "CudaStreamPool",
               "Pool of CUDA Streams",
               std::shared_ptr<CudaStreamPool>(nullptr));
  }

  void initialize() override {
    Operator::initialize();

    int timing_workload = timing_workload_.get();

    // Allocate GPU memory for benchmark kernel
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&d_benchmark_data_, timing_workload * sizeof(float)),
                                   "cudaMalloc failed!");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemset(d_benchmark_data_, 0, timing_workload * sizeof(float)), "cudaMemset failed!");

    // Initialize CUPTI profiler for scheduling latency measurement
    cupti_profiler_ = cupti_timing::CuptiSchedulingProfiler::getInstance();
    if (!cupti_profiler_->initialize()) {
      std::cout << "[TimingBenchmarkOp] WARNING: CUPTI initialization failed, no scheduling "
                   "latency measurements will be available"
                << std::endl;
      cupti_profiler_ = nullptr;
    }

    std::cout << "[TimingBenchmarkOp] GPU benchmark initialized: " << timing_workload << " elements"
              << std::endl;
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    int current_execution = ++execution_count_;

    // Log only every 1/10 of total iterations (e.g., 100, 200, 300... for 1000 iterations)
    int total_iters = total_iterations_.get();
    int log_interval = std::max(1, total_iters / 10);
    if (current_execution % log_interval == 0 || current_execution == 1) {
      std::cout << "[TimingBenchmarkOp] Execution #" << current_execution << " starting"
                << std::endl;
    }

    if (cached_stream_ == nullptr) {
      auto maybe_stream = context.allocate_cuda_stream("timing_stream");
      if (maybe_stream) {
        cached_stream_ = maybe_stream.value();
        std::cout << "[TimingBenchmarkOp] Using allocated non-default stream: " << cached_stream_
                  << std::endl;
      } else {
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamCreate(&cached_stream_),
                                       "cudaStreamCreate failed!");
        std::cout << "[TimingBenchmarkOp] Created non-default stream: " << cached_stream_
                  << std::endl;
      }
    }

    // Launch simple benchmark kernel with CUPTI-based scheduling latency measurement
    int timing_workload = timing_workload_.get();
    int threads_per_block = 256;
    int blocks = (timing_workload + threads_per_block - 1) / threads_per_block;

    TimingResults timing_results;

    // Launch kernel - CUPTI will automatically capture launch timestamp and execution start
    simple_benchmark_kernel<<<blocks, threads_per_block, 0, cached_stream_>>>(d_benchmark_data_,
                                                                              timing_workload);

    // Synchronize to ensure kernel completes
    cudaStreamSynchronize(cached_stream_);

    // Get CUPTI-based scheduling latency if available
    if (cupti_profiler_) {
      // Force flush CUPTI activities to trigger processing
      cuptiActivityFlushAll(0);

      // Enhanced polling with adaptive backoff for high contention scenarios
      const int max_poll_attempts = 500;       // Increased from 100 for high contention
      const int initial_poll_interval_ms = 1;  // Start with very short intervals
      int poll_count = 0;
      bool data_ready = false;
      int backoff_factor = 1;

      while (poll_count < max_poll_attempts && !data_ready) {
        // Check if measurements are available
        if (cupti_profiler_->hasMeasurements()) {
          data_ready = true;
          break;
        }

        // If no pending launches, the measurement was likely lost
        if (!cupti_profiler_->hasPendingLaunches()) {
          break;
        }

        // Adaptive sleep with exponential backoff (but capped)
        int current_interval = std::min(initial_poll_interval_ms * backoff_factor, 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(current_interval));

        // More frequent flushes during high contention
        if (poll_count % 5 == 0) {
          cuptiActivityFlushAll(0);
        }

        // Increase backoff every 50 attempts, but cap at 5ms
        if (poll_count % 50 == 0 && backoff_factor < 5) {
          backoff_factor++;
        }

        poll_count++;
      }

      // Get the latest scheduling latency measurement
      timing_results.scheduling_latency_us = cupti_profiler_->getLatestSchedulingLatency();

      // More detailed timeout warning
      if (poll_count >= max_poll_attempts) {
        std::cout << "[CUPTI] WARNING: Data polling timed out after " << poll_count
                  << " attempts (~" << (poll_count * 2)
                  << "ms). Possible buffer overflow or severe contention." << std::endl;
      }
    } else {
      timing_results.scheduling_latency_us = -1.0;  // CUPTI not available
      std::cout << "[WARNING] CUPTI profiler not available!" << std::endl;
    }

    timing_results.success = true;
    results_.push_back(timing_results);

    // Store completion timestamp when target iterations reached
    if (current_execution >= total_iters) {
      completion_timestamp_ = std::chrono::steady_clock::now();
      has_completed_ = true;
    }

    // Create and emit a new message
    auto out_message = holoscan::gxf::Entity::New(&context);
    op_output.emit(out_message, "out");
  }

  const std::vector<TimingResults>& getResults() const { return results_; }

  ~TimingBenchmarkOp() {
    if (d_benchmark_data_) {
      cudaError_t cuda_status = cudaFree(d_benchmark_data_);
      if (cuda_status != cudaSuccess) {
        std::cerr << "[TimingBenchmarkOp] cudaFree failed: " << cudaGetErrorString(cuda_status)
                  << std::endl;
      }
    }
  }
};

class CollectorOp : public Operator {
 private:
  std::vector<holoscan::gxf::Entity> results_;

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CollectorOp)

  CollectorOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<holoscan::gxf::Entity>("in"); }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message = op_input.receive<holoscan::gxf::Entity>("in");
    results_.push_back(in_message.value());
  }

  size_t getResultCount() const { return results_.size(); }
};

class GreenContextBenchmarkApp : public holoscan::Application {
 private:
  bool use_green_context_;
  int iterations_;
  int load_intensity_;
  int load_multiplier_;
  int workload_size_;      // Number of elements for DummyLoadOp GPU kernel
  int threads_per_block_;  // CUDA threads per block for DummyLoadOp kernel
  std::shared_ptr<DummyLoadOp> dummy_load_;
  std::shared_ptr<TimingBenchmarkOp> timing_benchmark_;
  std::shared_ptr<CollectorOp> collector_;

 public:
  explicit GreenContextBenchmarkApp(bool use_green_context = false, int iterations = 100,
                                    int load_intensity = 1000, int load_multiplier = 200,
                                    int workload_size = 2097152, int threads_per_block = 512)
      : use_green_context_(use_green_context),
        iterations_(iterations),
        load_intensity_(load_intensity),
        load_multiplier_(load_multiplier),
        workload_size_(workload_size),
        threads_per_block_(threads_per_block) {}

  void compose() override {
    using namespace holoscan;

    dummy_load_ = make_operator<DummyLoadOp>(
        "dummy_load",
        make_condition<CountCondition>(iterations_ *
                                       load_multiplier_),  // Self-ticking with configurable count
        Arg("load_intensity", load_intensity_),
        Arg("workload_size", workload_size_),
        Arg("threads_per_block", threads_per_block_),
        Arg("total_iterations", iterations_ * load_multiplier_));

    timing_benchmark_ = make_operator<TimingBenchmarkOp>(
        "timing_benchmark",
        make_condition<CountCondition>(iterations_),  // Self-ticking with iteration count
        Arg("timing_workload", 1024),
        Arg("total_iterations", iterations_));

    collector_ = make_operator<CollectorOp>("collector");
    auto dummy_sink = make_operator<CollectorOp>("dummy_sink");

    if (use_green_context_) {
      try {
        // Check GPU properties first
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "[App] GPU: " << prop.name << ", SMs: " << prop.multiProcessorCount
                  << ", Compute: " << prop.major << "." << prop.minor << std::endl;

        std::cout << "[INFO] Attempting Green Context initialization..." << std::endl;

        // Create separate Green Context partitions for proper isolation testing
        // Green Context requires partition size to be multiple of min_sm_count*2 = 4
        int total_sms = prop.multiProcessorCount;
        int sms_per_partition = std::max(4, (total_sms / 2) & ~3);  // Use half total SMs, minimum 4

        // Verify GPU has enough SMs for partitioning
        if (sms_per_partition * 2 > total_sms) {
          throw std::runtime_error("GPU too small for Green Context (need at least 8 SMs)");
        }

        std::vector<uint32_t> partitions = {static_cast<uint32_t>(sms_per_partition),
                                            static_cast<uint32_t>(sms_per_partition)};

        std::cout << "[App] Attempting Green Context with " << partitions.size() << " partitions, "
                  << sms_per_partition << " SMs each (total: " << sms_per_partition * 2 << "/"
                  << total_sms << " SMs)" << std::endl;

        auto cuda_green_context_pool = make_resource<CudaGreenContextPool>(
            "cuda_green_context_pool",
            Arg("dev_id", 0),
            Arg("num_partitions", static_cast<uint32_t>(partitions.size())),
            Arg("sms_per_partition", partitions));

        // Green Context Partition 1 - for DummyLoadOp
        auto dummy_green_context =
            make_resource<CudaGreenContext>("dummy_green_context",
                                            Arg("cuda_green_context_pool", cuda_green_context_pool),
                                            Arg("index", static_cast<int32_t>(0)));

        auto dummy_stream_pool =
            make_resource<CudaStreamPool>("dummy_stream_pool", 0, 0, 0, 1, 5, dummy_green_context);

        // Green Context Partition 2 - for TimingBenchmarkOp
        auto timing_green_context =
            make_resource<CudaGreenContext>("timing_green_context",
                                            Arg("cuda_green_context_pool", cuda_green_context_pool),
                                            Arg("index", static_cast<int32_t>(1)));

        auto timing_stream_pool = make_resource<CudaStreamPool>(
            "timing_stream_pool", 0, 0, 0, 1, 5, timing_green_context);

        dummy_load_->add_arg(Arg("cuda_green_context_pool", cuda_green_context_pool));
        dummy_load_->add_arg(Arg("cuda_green_context", dummy_green_context));
        dummy_load_->add_arg(Arg("cuda_stream_pool", dummy_stream_pool));

        timing_benchmark_->add_arg(Arg("cuda_green_context_pool", cuda_green_context_pool));
        timing_benchmark_->add_arg(Arg("cuda_green_context", timing_green_context));
        timing_benchmark_->add_arg(Arg("cuda_stream_pool", timing_stream_pool));

        std::cout << "[App] Green context enabled with separate partitions:" << std::endl;
        std::cout << "  - DummyLoadOp: Partition 0 (" << sms_per_partition << " SMs)" << std::endl;
        std::cout << "  - TimingBenchmarkOp: Partition 1 (" << sms_per_partition << " SMs)"
                  << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to setup green context: " << e.what() << std::endl;
        std::cerr << "[INFO] Green Context not available - this is common and expected"
                  << std::endl;
        std::cerr << "[INFO] Continuing with stream isolation comparison instead" << std::endl;
        use_green_context_ = false;
      }
    }

    if (!use_green_context_) {
      // Baseline: Both operators use non-default streams but NO Green Context partitions
      try {
        // Create regular (non-Green Context) stream pools for both operators
        auto dummy_stream_pool = make_resource<CudaStreamPool>("dummy_stream_pool", 0, 0, 0, 1, 5);

        auto timing_stream_pool =
            make_resource<CudaStreamPool>("timing_stream_pool", 0, 0, 0, 1, 5);

        dummy_load_->add_arg(Arg("cuda_stream_pool", dummy_stream_pool));
        timing_benchmark_->add_arg(Arg("cuda_stream_pool", timing_stream_pool));

        std::cout << "[App] Baseline mode: Non-default streams WITHOUT Green Context partitions"
                  << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to setup baseline stream pools: " << e.what() << std::endl;
      }
    }

    // Add memory allocator
    auto allocator = make_resource<UnboundedAllocator>("allocator");
    dummy_load_->add_arg(allocator);
    timing_benchmark_->add_arg(allocator);
    dummy_sink->add_arg(allocator);

    // Simplified pipeline - operators are self-ticking with conditions
    // Connect both operators to make them part of the execution graph
    add_flow(timing_benchmark_, collector_, {{"out", "in"}});
    add_flow(dummy_load_, dummy_sink, {{"dummy_out", "in"}});
  }

  const std::vector<TimingResults>& getTimingResults() const {
    return timing_benchmark_->getResults();
  }

  BenchmarkStats getDummyLoadStats() const { return dummy_load_->getExecutionStats(); }

  // Check completion order by comparing timestamps
  void checkCompletionOrder() const {
    // Only compare if both operators actually completed
    if (!dummy_load_->has_completed_ || !timing_benchmark_->has_completed_) {
      std::cout << "[INFO] One or both operators did not complete normally - skipping completion "
                   "order check"
                << std::endl;
      return;
    }

    auto dummy_timestamp = dummy_load_->completion_timestamp_;
    auto timing_timestamp = timing_benchmark_->completion_timestamp_;

    if (dummy_timestamp < timing_timestamp) {
      auto duration_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(timing_timestamp - dummy_timestamp)
              .count();
      std::cout << "\033[1;31m[WARNING] DummyLoadOp finished " << duration_ms
                << "ms before TimingBenchmarkOp"
                << " - background load stopped early!\033[0m" << std::endl;
      std::cout << "  Consider increasing load_multiplier to ensure background load continues "
                   "throughout timing."
                << std::endl;
    } else {
      std::cout << "[INFO] Background load continued appropriately throughout timing benchmark"
                << std::endl;
    }
  }

  size_t getResultCount() const { return collector_->getResultCount(); }

  bool isUsingGreenContext() const { return use_green_context_; }
};

// Calculate percentiles from sorted data
double calculatePercentile(const std::vector<double>& sorted_data, double percentile) {
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

// Calculate standard deviation
double calculateStdDev(const std::vector<double>& data, double mean) {
  if (data.size() <= 1)
    return 0.0;

  double sum_sq_diff = 0.0;
  for (double value : data) {
    double diff = value - mean;
    sum_sq_diff += diff * diff;
  }

  return std::sqrt(sum_sq_diff / (data.size() - 1));
}

// Calculate CUPTI-based scheduling latency statistics
BenchmarkStats calculateSchedulingStats(const std::vector<TimingResults>& results) {
  BenchmarkStats stats;

  // Extract successful CUPTI measurements
  for (const auto& result : results) {
    if (result.success && result.scheduling_latency_us >= 0.0) {
      stats.sorted_data.push_back(result.scheduling_latency_us);
    }
  }

  if (stats.sorted_data.empty())
    return stats;

  std::sort(stats.sorted_data.begin(), stats.sorted_data.end());
  stats.sample_count = stats.sorted_data.size();

  // Calculate basic statistics
  stats.avg =
      std::accumulate(stats.sorted_data.begin(), stats.sorted_data.end(), 0.0) / stats.sample_count;
  stats.std_dev = calculateStdDev(stats.sorted_data, stats.avg);

  // Calculate percentiles
  stats.min_val = stats.sorted_data.front();
  stats.max_val = stats.sorted_data.back();
  stats.p50 = calculatePercentile(stats.sorted_data, 50.0);
  stats.p95 = calculatePercentile(stats.sorted_data, 95.0);
  stats.p99 = calculatePercentile(stats.sorted_data, 99.0);

  return stats;
}

// Calculate empirical CDF at a given value
double calculateCDF(const std::vector<double>& data, double x) {
  if (data.empty())
    return 0.0;

  int count = 0;
  for (double value : data) {
    if (value <= x)
      count++;
  }

  return static_cast<double>(count) / data.size();
}

void printBenchmarkResults(const BenchmarkStats& stats, const std::string& context_type) {
  if (stats.sample_count == 0) {
    std::cout << "\n=== " << context_type << " RESULTS ===" << std::endl;
    std::cout << "CUPTI-based scheduling latency: Not available" << std::endl;
    std::cout << "  (CUPTI initialization may have failed or no measurements captured)"
              << std::endl;
    return;
  }

  std::cout << "\n=== " << context_type << " RESULTS ===" << std::endl;
  std::cout << std::fixed << std::setprecision(2) << std::dec;
  std::cout << "CUPTI-based GPU Scheduling Latency Distribution (microseconds):" << std::endl;
  std::cout << "  Average: " << stats.avg << " μs" << std::endl;
  std::cout << "  Std Dev: " << stats.std_dev << " μs" << std::endl;
  std::cout << "  Min:     " << stats.min_val << " μs" << std::endl;
  std::cout << "  P50:     " << stats.p50 << " μs" << std::endl;
  std::cout << "  P95:     " << stats.p95 << " μs" << std::endl;
  std::cout << "  P99:     " << stats.p99 << " μs" << std::endl;
  std::cout << "  Max:     " << stats.max_val << " μs" << std::endl;
  std::cout << "  Samples: " << stats.sample_count << std::endl;
  std::cout << "  Note: Scheduling latency = GPU execution start - kernel launch time" << std::endl;
}

void print_usage(const char* program_name) {
  std::cout << "Green Context Benchmark - CUPTI-based GPU Scheduling Latency Measurement\n\n";
  std::cout
      << "Measures true scheduling latency: time from kernel launch to GPU execution start\n\n";
  std::cout << "Usage: " << program_name << " [OPTIONS]\n";
  std::cout << "Options:\n";
  std::cout << "  --iterations N        Number of timing iterations (default: 1000)\n";
  std::cout << "  --load-intensity N    GPU load intensity multiplier (default: 10)\n";
  std::cout << "  --load-multiplier N   Background load multiplier (default: 200)\n";
  std::cout
      << "                        Background iterations = timing_iterations * load_multiplier\n";
  std::cout << "  --workload-size N     GPU memory size in MB for DummyLoadOp (default: 8)\n";
  std::cout << "  --threads-per-block N CUDA threads per block for GPU kernels (default: 512)\n";
  std::cout << "  --help               Show this help message\n";
  std::cout << "\nExample:\n";
  std::cout << "  " << program_name
            << " --iterations 100 --load-intensity 20 --load-multiplier 150 --workload-size "
               "4 --threads-per-block 256\n";
  std::cout << "\nNote: Requires CUPTI. May need admin privileges or driver configuration.\n";
}

int main(int argc, char* argv[]) {
  // Default values
  int iterations = 1000;
  int load_intensity = 10;
  int load_multiplier = 200;    // Background iterations = timing_iterations * load_multiplier
  int workload_size_mb = 8;     // GPU memory size in MB for DummyLoadOp
  int threads_per_block = 512;  // CUDA threads per block for GPU kernels

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::atoi(argv[++i]);
      if (iterations <= 0) {
        std::cerr << "Error: iterations must be positive\n";
        return 1;
      }
    } else if (arg == "--load-intensity" && i + 1 < argc) {
      load_intensity = std::atoi(argv[++i]);
      if (load_intensity <= 0) {
        std::cerr << "Error: load-intensity must be positive\n";
        return 1;
      }
    } else if (arg == "--load-multiplier" && i + 1 < argc) {
      load_multiplier = std::atoi(argv[++i]);
      if (load_multiplier <= 0) {
        std::cerr << "Error: load-multiplier must be positive\n";
        return 1;
      }
    } else if (arg == "--workload-size" && i + 1 < argc) {
      workload_size_mb = std::atoi(argv[++i]);
      if (workload_size_mb <= 0) {
        std::cerr << "Error: workload-size must be positive\n";
        return 1;
      }
    } else if (arg == "--threads-per-block" && i + 1 < argc) {
      threads_per_block = std::atoi(argv[++i]);
      if (threads_per_block <= 0) {
        std::cerr << "Error: threads-per-block must be positive\n";
        return 1;
      }
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  // Convert workload size from MB to number of float elements
  int workload_size = (workload_size_mb * 1024 * 1024) / sizeof(float);

  std::cout << "=" << std::string(80, '=') << std::endl;
  std::cout << "CUPTI-ENHANCED GREEN CONTEXT SCHEDULING LATENCY BENCHMARK" << std::endl;
  std::cout << "Measures: True GPU scheduling latency from kernel launch to execution start"
            << std::endl;
  std::cout << "=" << std::string(80, '=') << std::endl;

  std::cout << "\nBenchmark Configuration:" << std::endl;
  std::cout << "  Timing Iterations: " << iterations << std::endl;
  std::cout << "  Load Intensity: " << load_intensity << std::endl;
  std::cout << "  Load Multiplier: " << load_multiplier << std::endl;
  std::cout << "  Workload Size: " << workload_size_mb << " MB (" << workload_size << " elements)"
            << std::endl;
  std::cout << "  Threads Per Block: " << threads_per_block << std::endl;
  std::cout << "  Background Iterations: " << (iterations * load_multiplier) << " ("
            << load_multiplier << "x timing iterations)" << std::endl;

  // Initialize CUDA
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(0), "Failed to set CUDA device");

  // Initialize CUPTI profiler early
  std::cout << "\nInitializing CUPTI scheduling latency profiler..." << std::endl;
  auto* cupti_profiler = cupti_timing::CuptiSchedulingProfiler::getInstance();
  if (!cupti_profiler->initialize()) {
    std::cout << "[WARNING] CUPTI initialization failed - will use host-side timing only"
              << std::endl;
  }

  std::vector<TimingResults> results_without_gc, results_with_gc;
  BenchmarkStats dummy_load_stats_without_gc, dummy_load_stats_with_gc;
  bool green_context_was_used = false;

  // Run WITHOUT green context (baseline: both kernels on separate non-default streams)
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "BASELINE: NON-DEFAULT STREAMS WITHOUT GREEN CONTEXT" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  try {
    auto app_no_gc = std::make_unique<GreenContextBenchmarkApp>(
        false, iterations, load_intensity, load_multiplier, workload_size, threads_per_block);
    app_no_gc->scheduler(app_no_gc->make_scheduler<holoscan::EventBasedScheduler>(
        "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(6))));
    app_no_gc->run();
    results_without_gc = app_no_gc->getTimingResults();
    dummy_load_stats_without_gc = app_no_gc->getDummyLoadStats();
    app_no_gc->checkCompletionOrder();  // Check for timing issues
    std::cout << "[SUCCESS] Benchmark without green context completed (" << std::dec
              << app_no_gc->getResultCount() << " results)" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] Benchmark without green context failed: " << e.what() << std::endl;
  }

  // Run WITH green context attempt
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "ATTEMPTING GREEN CONTEXT: SEPARATE PARTITIONS FOR EACH KERNEL" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  try {
    auto app_with_gc = std::make_unique<GreenContextBenchmarkApp>(
        true, iterations, load_intensity, load_multiplier, workload_size, threads_per_block);
    app_with_gc->scheduler(app_with_gc->make_scheduler<holoscan::EventBasedScheduler>(
        "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(6))));
    app_with_gc->run();
    results_with_gc = app_with_gc->getTimingResults();
    dummy_load_stats_with_gc = app_with_gc->getDummyLoadStats();
    app_with_gc->checkCompletionOrder();  // Check for timing issues

    green_context_was_used = app_with_gc->isUsingGreenContext();
    if (green_context_was_used) {
      std::cout << "[SUCCESS] Green Context benchmark completed (" << std::dec
                << app_with_gc->getResultCount() << " results)" << std::endl;
    } else {
      std::cout << "[INFO] Fell back to baseline configuration - Green Context not available ("
                << std::dec << app_with_gc->getResultCount() << " results)" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] Benchmark with green context failed: " << e.what() << std::endl;
  }

  // Display results
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "BENCHMARK RESULTS" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  // Calculate CUPTI-based scheduling latency statistics
  BenchmarkStats no_gc_stats = calculateSchedulingStats(results_without_gc);
  BenchmarkStats gc_stats = calculateSchedulingStats(results_with_gc);

  // Print CUPTI-based scheduling latency results
  printBenchmarkResults(no_gc_stats, "BASELINE (NON-DEFAULT STREAMS)");

  if (green_context_was_used) {
    printBenchmarkResults(gc_stats, "GREEN CONTEXT (SEPARATE PARTITIONS)");
  } else {
    printBenchmarkResults(gc_stats, "FALLBACK (SAME AS BASELINE)");
  }

  // Display DummyLoadOp execution time statistics
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "DUMMY LOAD EXECUTION TIME STATISTICS" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  if (dummy_load_stats_without_gc.sample_count > 0) {
    std::cout << "\n=== BASELINE DUMMY LOAD EXECUTION TIMES ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Average: " << dummy_load_stats_without_gc.avg << " μs" << std::endl;
    std::cout << "  Std Dev: " << dummy_load_stats_without_gc.std_dev << " μs" << std::endl;
    std::cout << "  Samples: " << dummy_load_stats_without_gc.sample_count << std::endl;
  }

  if (dummy_load_stats_with_gc.sample_count > 0) {
    std::string gc_label = green_context_was_used ? "GREEN CONTEXT" : "FALLBACK";
    std::cout << "\n=== " << gc_label << " DUMMY LOAD EXECUTION TIMES ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Average: " << dummy_load_stats_with_gc.avg << " μs" << std::endl;
    std::cout << "  Std Dev: " << dummy_load_stats_with_gc.std_dev << " μs" << std::endl;
    std::cout << "  Samples: " << dummy_load_stats_with_gc.sample_count << std::endl;
  }

  // Performance Comparison
  if (no_gc_stats.sample_count > 0 && gc_stats.sample_count > 0) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (green_context_was_used) {
      std::cout << "GREEN CONTEXT SCHEDULING LATENCY COMPARISON" << std::endl;
    } else {
      std::cout << "SCHEDULING LATENCY ANALYSIS" << std::endl;
      std::cout << "(Note: Green Context not available - both tests used same configuration)"
                << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;

    // Calculate improvements in scheduling latency
    double avg_improvement = ((no_gc_stats.avg - gc_stats.avg) / no_gc_stats.avg) * 100.0;
    double p95_improvement = ((no_gc_stats.p95 - gc_stats.p95) / no_gc_stats.p95) * 100.0;
    double p99_improvement = ((no_gc_stats.p99 - gc_stats.p99) / no_gc_stats.p99) * 100.0;

    std::cout << std::fixed << std::setprecision(2) << std::dec;

    // Key Performance Metrics
    std::cout << "\n SCHEDULING LATENCY IMPROVEMENTS:" << std::endl;
    std::cout << "  Average Latency:  " << std::setw(8) << no_gc_stats.avg << " μs → "
              << std::setw(8) << gc_stats.avg << " μs  (" << std::showpos << avg_improvement << "%)"
              << std::endl;
    std::cout << "  95th Percentile:  " << std::setw(8) << no_gc_stats.p95 << " μs → "
              << std::setw(8) << gc_stats.p95 << " μs  (" << std::showpos << p95_improvement << "%)"
              << std::endl;
    std::cout << "  99th Percentile:  " << std::setw(8) << no_gc_stats.p99 << " μs → "
              << std::setw(8) << gc_stats.p99 << " μs  (" << std::showpos << p99_improvement << "%)"
              << std::endl;
    std::cout << std::noshowpos;

    // Scheduling latency distribution comparison
    std::cout << "\n SCHEDULING LATENCY DISTRIBUTION:" << std::endl;
    std::cout << "Threshold       Without GC    With GC    Improvement" << std::endl;
    std::cout << "─────────       ──────────    ───────    ───────────" << std::endl;

    std::vector<double> thresholds = {5, 10, 25, 50, 100, 1000};
    for (double threshold : thresholds) {
      double no_gc_cdf = calculateCDF(no_gc_stats.sorted_data, threshold) * 100.0;
      double gc_cdf = calculateCDF(gc_stats.sorted_data, threshold) * 100.0;
      std::cout << std::setw(8) << threshold << " μs     " << std::setw(8) << std::setprecision(1)
                << no_gc_cdf << "%     " << std::setw(6) << gc_cdf << "%     " << std::showpos
                << std::setw(8) << std::setprecision(1) << (gc_cdf - no_gc_cdf) << "%" << std::endl;
    }
    std::cout << std::noshowpos;
  }

  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "BENCHMARK COMPLETE" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  // Cleanup CUPTI profiler
  if (cupti_profiler) {
    cupti_profiler->cleanup();
    std::cout << "\n[CUPTI] Profiler cleaned up." << std::endl;
  }

  return 0;
}
