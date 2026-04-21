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

// Changes vs. original:
//
//  1. TimestampEntry carries a sequence number alongside the timestamp.
//     TimingRxOp validates strict FIFO ordering and drops samples with
//     violations rather than silently mis-attributing latency.
//
//  2. The emit timestamp is captured AFTER entity and tensor preparation,
//     immediately BEFORE op_output.emit(), so it accurately marks pipeline
//     entry rather than the start of the compute() call.
//
//  3. PeriodicTxOp uses double-buffered GPU allocations (kNumBuffers = 2)
//     to prevent the previous inference kernel from reading a buffer while
//     the current tick is wrapping a new tensor into it.
//
//  4. cudaMalloc + cudaMemcpy in PeriodicTxOp::initialize() uses a
//     try-catch RAII guard so a failed memcpy does not leak the allocation.
//
//  5. ContendingSinkOp ready_after_iters default raised from 8 to 50.
//     With TRT, the first several iterations include engine compilation;
//     8 iterations is insufficient for any realistic model.
//
//  6. EBS worker thread count is now configurable via --worker-threads
//     instead of being hardcoded to 16.
//
//  7. --repeat N runs baseline/green-context in alternating pairs so
//     thermal and caching state are similar across conditions. All raw
//     samples are pooled before computing final statistics.
//
//  8. --contending-ready-iters N exposes the readiness gate as a
//     CLI parameter (default 50).
//
//  9. The broken EventBasedScheduler probe used to detect postcheck
//     fastpath support (calling setup() without a fragment) is removed.
//     The argument is applied directly; the SDK will emit an informative
//     error at startup if unsupported.
//
// 10. ensure_model_exists() validates paths for shell metacharacters
//     before constructing the python3 command string.
//
// 11. Measured model defaults raised to input=256, hidden=2048, layers=4
//     so the kernel actually occupies multiple SMs and benefits from
//     green-context SM isolation.
//
// 12. The "TX firing period" output section is labelled as scheduler tick
//     accuracy rather than an inference metric.
//
// 13. All output uses a consistent table format with percentage deltas
//     and plain-language direction labels.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <gxf/std/tensor.hpp>

using namespace holoscan;

// =============================================================================
// Statistics helpers
// =============================================================================

struct BenchmarkStats {
  double avg         = 0.0;
  double std_dev     = 0.0;
  double min_val     = 0.0;
  double p50         = 0.0;
  double p95         = 0.0;
  double p99         = 0.0;
  double max_val     = 0.0;
  size_t sample_count = 0;
};

double calculate_percentile(const std::vector<double>& sorted_data, double percentile) {
  if (sorted_data.empty()) return 0.0;
  double index = (percentile / 100.0) * (sorted_data.size() - 1);
  size_t lower  = static_cast<size_t>(std::floor(index));
  size_t upper  = static_cast<size_t>(std::ceil(index));
  if (lower == upper) return sorted_data[lower];
  double weight = index - lower;
  return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
}

BenchmarkStats calculate_stats(const std::vector<double>& raw) {
  BenchmarkStats s;
  if (raw.empty()) return s;
  std::vector<double> v = raw;
  std::sort(v.begin(), v.end());
  s.sample_count = v.size();
  s.avg     = std::accumulate(v.begin(), v.end(), 0.0) / s.sample_count;
  double sq = 0.0;
  for (double x : v) { double d = x - s.avg; sq += d * d; }
  s.std_dev  = s.sample_count > 1 ? std::sqrt(sq / (s.sample_count - 1)) : 0.0;
  s.min_val  = v.front();
  s.max_val  = v.back();
  s.p50      = calculate_percentile(v, 50.0);
  s.p95      = calculate_percentile(v, 95.0);
  s.p99      = calculate_percentile(v, 99.0);
  return s;
}

// =============================================================================
// Shared pipeline state
//
// TimestampEntry pairs a monotonically increasing sequence number with the
// wall-clock nanosecond timestamp captured by PeriodicTxOp.  The sequence
// number lets TimingRxOp validate that messages arrive in the expected order;
// any ordering violation is logged and the affected sample is dropped rather
// than silently corrupting the measurement.
// =============================================================================

struct TimestampEntry {
  uint64_t seqnum  = 0;
  int64_t  emit_ns = 0;   // steady_clock ns, captured just before op_output.emit()
};

static std::atomic<bool>     g_contending_pipeline_ready{false};
static std::atomic<uint64_t> g_seq_counter{0};
static std::mutex             g_ts_mutex;
static std::deque<TimestampEntry> g_tx_timestamps;

void reset_global_benchmark_state() {
  g_contending_pipeline_ready.store(false, std::memory_order_release);
  g_seq_counter.store(0, std::memory_order_release);
  std::lock_guard<std::mutex> lock(g_ts_mutex);
  g_tx_timestamps.clear();
}

// =============================================================================
// TensorSpec
// =============================================================================

struct TensorSpec {
  std::string          name;
  std::vector<int32_t> shape;
  int total_elements() const {
    int n = 1; for (auto d : shape) n *= d; return n;
  }
};

// =============================================================================
// PeriodicTxOp
//
// Emits pre-allocated GPU tensors at a configurable rate.
//
// Double-buffered GPU allocations (kNumBuffers = 2) ensure that when the
// operator fires a new tick it writes into a buffer that was last used two
// ticks ago, giving the intervening inference kernel at least one full period
// to complete its read before the buffer is reused.
//
// The emit timestamp and sequence number are captured AFTER all entity and
// tensor preparation is complete, immediately before op_output.emit(), so
// they mark the true moment the message enters the downstream pipeline rather
// than the start of the compute() call.
// =============================================================================

class PeriodicTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PeriodicTxOp)
  PeriodicTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("output");
  }

  void set_tensor_specs(const std::vector<TensorSpec>& specs) { tensor_specs_ = specs; }
  void set_record_timestamps(bool record) { record_timestamps_ = record; }

  void initialize() override {
    Operator::initialize();
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (auto& ts : tensor_specs_) {
      int n = ts.total_elements();
      std::vector<float> host_data(n);
      for (auto& v : host_data) v = dis(gen);

      std::array<float*, kNumBuffers> ptrs{};
      for (int b = 0; b < kNumBuffers; ++b) {
        float* d_ptr = nullptr;
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&d_ptr, n * sizeof(float)),
                                       "cudaMalloc failed for tensor '" + ts.name + "'");
        // RAII: free d_ptr if the memcpy throws so we don't leak the allocation.
        try {
          HOLOSCAN_CUDA_CALL_THROW_ERROR(
              cudaMemcpy(d_ptr, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy failed for tensor '" + ts.name + "'");
        } catch (...) {
          cudaFree(d_ptr);
          throw;
        }
        ptrs[b] = d_ptr;
      }
      gpu_buffers_.push_back(ptrs);
    }
    HOLOSCAN_LOG_INFO("[{}] Initialized {} tensor(s) with {} GPU buffers each",
                      name(), tensor_specs_.size(), kNumBuffers);
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override {
    // Rotate to the next buffer slot before building the entity.
    buf_idx_ = (buf_idx_ + 1) % kNumBuffers;

    auto maybe_entity = nvidia::gxf::Entity::New(context.context());
    if (!maybe_entity) throw std::runtime_error("Failed to create GXF entity");
    auto entity = std::move(maybe_entity.value());

    const uint32_t elem_size = sizeof(float);
    for (size_t i = 0; i < tensor_specs_.size(); i++) {
      auto& ts = tensor_specs_[i];
      auto maybe_tensor = entity.add<nvidia::gxf::Tensor>(ts.name.c_str());
      if (!maybe_tensor) throw std::runtime_error("Failed to add tensor: " + ts.name);
      auto tensor = maybe_tensor.value();

      const nvidia::gxf::Shape shape(ts.shape);
      auto result = tensor->wrapMemory(
          shape, nvidia::gxf::PrimitiveType::kFloat32, elem_size,
          nvidia::gxf::ComputeTrivialStrides(shape, elem_size),
          nvidia::gxf::MemoryStorageType::kDevice,
          gpu_buffers_[i][buf_idx_],
          // Null deleter: PeriodicTxOp owns and frees these buffers in its
          // destructor; GXF must not free them when the entity is released.
          [](void*) { return nvidia::gxf::Success; });
      if (!result) throw std::runtime_error("Failed to wrap memory: " + ts.name);
    }

    // Capture timestamp and sequence number HERE — after entity preparation
    // but before emit — so the interval measures actual pipeline transit time.
    if (record_timestamps_) {
      uint64_t seq = g_seq_counter.fetch_add(1, std::memory_order_relaxed);
      int64_t  now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch()).count();
      std::lock_guard<std::mutex> lock(g_ts_mutex);
      g_tx_timestamps.push_back({seq, now});
    }

    holoscan::gxf::Entity out_entity(std::move(entity));
    op_output.emit(out_entity, "output");
  }

  ~PeriodicTxOp() {
    for (auto& ptrs : gpu_buffers_)
      for (auto* p : ptrs)
        if (p) cudaFree(p);
  }

 private:
  static constexpr int kNumBuffers = 2;

  std::vector<TensorSpec>                       tensor_specs_;
  std::vector<std::array<float*, kNumBuffers>>  gpu_buffers_;
  int  buf_idx_           = 0;
  bool record_timestamps_ = true;
};

// =============================================================================
// TimingRxOp
//
// Receives the measured pipeline's inference output and records end-to-end
// latency.
//
// Key design points:
//  - Validates that the sequence number from the timestamp deque matches the
//    expected value on every sample.  A mismatch means the pipeline reordered
//    messages, which would corrupt every subsequent measurement; the sample is
//    dropped and an error is logged so the operator is visible to the user.
//  - cudaStreamSynchronize is intentionally kept.  InferenceOp dispatches GPU
//    kernels asynchronously; without the sync, end_ns captures CPU-side
//    dispatch return rather than actual GPU kernel completion, which would
//    systematically under-report latency.
//  - Exposes raw sample vectors rather than pre-computed stats so the caller
//    can pool samples across multiple repeat runs before computing aggregates.
//  - run_label_ adds context to every log message so interleaved baseline and
//    green-context runs are easy to distinguish in the output.
// =============================================================================

class TimingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimingRxOp)
  TimingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(total_samples_,  "total_samples",  "Total Samples",
               "Number of measurement samples to collect", 1000);
    spec.param(warmup_samples_, "warmup_samples", "Warmup Samples",
               "Samples discarded after contending pipeline is ready", 100);
    spec.input<std::any>("in");
  }

  void set_run_label(const std::string& label) { run_label_ = label; }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    (void)op_input.receive<std::any>("in");

    // Sync the inference CUDA stream so end_ns reflects GPU kernel completion,
    // not just the point at which the CPU received control back from the driver.
    auto streams = op_input.receive_cuda_streams("in");
    if (!streams.empty() && streams[0].has_value()) {
      cudaStreamSynchronize(streams[0].value());
    }
    int64_t end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Pop the matching timestamp entry and validate strict FIFO ordering.
    TimestampEntry entry{};
    {
      std::lock_guard<std::mutex> lock(g_ts_mutex);
      if (!g_tx_timestamps.empty()) {
        entry = g_tx_timestamps.front();
        g_tx_timestamps.pop_front();
      }
    }

    if (entry.emit_ns == 0) {
      HOLOSCAN_LOG_WARN("[{}] No emit timestamp in queue — sample dropped "
                        "(pipeline may have fallen behind TX rate)", run_label_);
      return;
    }

    if (entry.seqnum != expected_seq_) {
      HOLOSCAN_LOG_ERROR(
          "[{}] Pipeline ordering violation detected: expected sequence {} but received {}. "
          "This sample is dropped.  Measurements may be unreliable — consider reducing "
          "--frequency-hz or increasing --worker-threads.",
          run_label_, expected_seq_, entry.seqnum);
      expected_seq_ = entry.seqnum + 1;
      return;
    }
    ++expected_seq_;

    // Phase 1: discard ticks that arrive before the contending pipeline is ready.
    if (!g_contending_pipeline_ready.load(std::memory_order_acquire)) {
      ++warmup_pre_ready_count_;
      if (warmup_pre_ready_count_ == 1 || warmup_pre_ready_count_ % 100 == 0) {
        HOLOSCAN_LOG_INFO("[{}] Waiting for contending pipeline to reach steady state "
                          "({} ticks so far) ...", run_label_, warmup_pre_ready_count_);
      }
      return;
    }

    // Phase 2: discard post-ready warmup samples to let the GPU reach its
    // thermal steady state before recording measurements.
    if (warmup_post_ready_count_ < warmup_samples_.get()) {
      ++warmup_post_ready_count_;
      if (warmup_post_ready_count_ == warmup_samples_.get()) {
        HOLOSCAN_LOG_INFO(
            "[{}] Warmup complete: {} pre-ready ticks + {} post-ready ticks discarded. "
            "Now collecting {} measurement samples ...",
            run_label_, warmup_pre_ready_count_, warmup_post_ready_count_, total_samples_.get());
      }
      return;
    }

    // Record the measurement sample.
    double e2e_us = static_cast<double>(end_ns - entry.emit_ns) / 1000.0;
    e2e_us_.push_back(e2e_us);

    if (prev_emit_ns_ > 0) {
      double period_us = static_cast<double>(entry.emit_ns - prev_emit_ns_) / 1000.0;
      tick_period_us_.push_back(period_us);
    }
    prev_emit_ns_ = entry.emit_ns;

    ++sample_count_;

    // Log progress at ~10 evenly spaced checkpoints.
    int step = std::max(1, total_samples_.get() / 10);
    if (sample_count_ % step == 0 || sample_count_ == total_samples_.get()) {
      int pct = sample_count_ * 100 / total_samples_.get();
      HOLOSCAN_LOG_INFO("[{}] Progress: {}/{} samples  ({:3d}%)",
                        run_label_, sample_count_, total_samples_.get(), pct);
    }

    if (sample_count_ >= total_samples_.get()) {
      fragment()->stop_execution();
    }
  }

  const std::vector<double>& get_e2e_samples()        const { return e2e_us_; }
  const std::vector<double>& get_tick_period_samples() const { return tick_period_us_; }

 private:
  Parameter<int> total_samples_;
  Parameter<int> warmup_samples_;

  std::string run_label_             = "run";
  uint64_t    expected_seq_          = 0;
  int         sample_count_          = 0;
  int         warmup_pre_ready_count_  = 0;
  int         warmup_post_ready_count_ = 0;
  int64_t     prev_emit_ns_          = 0;

  std::vector<double> e2e_us_;
  std::vector<double> tick_period_us_;
};

// =============================================================================
// ContendingSinkOp
//
// Consumes the contending pipeline's inference output and gates the measured
// pipeline's warmup.  The ready_after_iters default is 50 (raised from 8) so
// the contending pipeline has time to exit TRT engine compilation and reach a
// steady GPU workload before measurement begins.
// =============================================================================

class ContendingSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ContendingSinkOp)
  ContendingSinkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(ready_after_iters_, "ready_after_iters",
               "Ready After Iterations",
               "Contending inferences to complete before measurement is allowed to start", 50);
    spec.input<std::any>("in");
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    (void)op_input.receive<std::any>("in");

    if (completed_iters_ == 0) {
      first_iter_time_ = std::chrono::steady_clock::now();
      HOLOSCAN_LOG_INFO("[ContendingSinkOp] Contending inference pipeline started.");
    }
    ++completed_iters_;
    last_iter_time_ = std::chrono::steady_clock::now();

    if (!g_contending_pipeline_ready.load(std::memory_order_acquire) &&
        completed_iters_ >= std::max(1, ready_after_iters_.get())) {
      g_contending_pipeline_ready.store(true, std::memory_order_release);
      HOLOSCAN_LOG_INFO(
          "[ContendingSinkOp] Reached steady state after {} inferences. "
          "Releasing measurement gate.", completed_iters_);
    }
  }

  int    get_completed_iters() const { return completed_iters_; }
  double get_throughput_hz()   const {
    if (completed_iters_ <= 1) return 0.0;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  last_iter_time_ - first_iter_time_).count();
    return us > 0 ? static_cast<double>(completed_iters_ - 1) * 1e6 / us : 0.0;
  }

 private:
  Parameter<int> ready_after_iters_;
  int completed_iters_ = 0;
  std::chrono::steady_clock::time_point first_iter_time_{};
  std::chrono::steady_clock::time_point last_iter_time_{};
};

// =============================================================================
// InferenceSchedulingBenchmarkApp
// =============================================================================

class InferenceSchedulingBenchmarkApp : public holoscan::Application {
 public:
  InferenceSchedulingBenchmarkApp(
      bool use_gc,
      int total_samples, int warmup_samples,
      const std::string& measured_model_path,   int measured_input_size,
      const std::string& contending_model_path, int contending_input_size,
      const std::string& backend,
      int measured_sms, int contending_sms,
      int measured_frequency_hz, int contending_frequency_hz,
      const std::string& periodic_policy,
      bool pin_measured, SchedulingPolicy sched_policy,
      const std::vector<uint32_t>& pin_cores,
      int contending_ready_iters,
      const std::string& run_label)
      : use_gc_(use_gc),
        total_samples_(total_samples), warmup_samples_(warmup_samples),
        measured_model_path_(measured_model_path), measured_input_size_(measured_input_size),
        contending_model_path_(contending_model_path), contending_input_size_(contending_input_size),
        backend_(backend),
        measured_sms_(measured_sms), contending_sms_(contending_sms),
        measured_frequency_hz_(measured_frequency_hz),
        contending_frequency_hz_(contending_frequency_hz),
        periodic_policy_(periodic_policy),
        pin_measured_(pin_measured), sched_policy_(sched_policy), pin_cores_(pin_cores),
        contending_ready_iters_(contending_ready_iters),
        run_label_(run_label) {}

  void compose() override {
    std::shared_ptr<CudaStreamPool>       measured_stream_pool;
    std::shared_ptr<CudaStreamPool>       contending_stream_pool;
    std::shared_ptr<CudaGreenContextPool> gc_pool;
    std::shared_ptr<CudaGreenContext>     measured_gc;
    std::shared_ptr<CudaGreenContext>     contending_gc;

    if (use_gc_) {
      cudaDeviceProp prop;
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDeviceProperties(&prop, 0),
                                     "Failed to query GPU device properties");
      int total_sms     = prop.multiProcessorCount;
      int rounded_total = total_sms & ~3;  // align down to multiple of 4
      if (rounded_total < 4) {
        throw std::runtime_error(
            "GPU has fewer than 4 aligned SMs — green context partitioning is not viable");
      }

      // If not explicitly specified, split evenly.
      int m_sms = measured_sms_   > 0 ? std::max(4, measured_sms_   & ~3)
                                       : std::max(4, (rounded_total / 2) & ~3);
      int c_sms = contending_sms_ > 0 ? std::max(4, contending_sms_ & ~3)
                                       : std::max(4, (rounded_total / 2) & ~3);
      m_sms = std::min(m_sms, rounded_total);
      c_sms = std::min(c_sms, rounded_total);

      if (m_sms + c_sms > rounded_total) {
        HOLOSCAN_LOG_WARN(
            "[compose] SM over-subscription: measured ({}) + contending ({}) = {} > {} available. "
            "Partitions overlap — isolation benefit will be reduced. "
            "Use --measured-sms and --contending-sms to set non-overlapping values.",
            m_sms, c_sms, m_sms + c_sms, rounded_total);
      }
      HOLOSCAN_LOG_INFO(
          "[compose] Green Context: measured pipeline gets {} SMs, "
          "contending pipeline gets {} SMs  (GPU total = {} SMs)",
          m_sms, c_sms, total_sms);

      std::vector<uint32_t> partitions = {static_cast<uint32_t>(m_sms),
                                          static_cast<uint32_t>(c_sms)};
      gc_pool = make_resource<CudaGreenContextPool>(
          "gc_pool",
          Arg("dev_id", 0),
          Arg("num_partitions",    static_cast<uint32_t>(2)),
          Arg("sms_per_partition", partitions));

      measured_gc = make_resource<CudaGreenContext>(
          "measured_gc",
          Arg("cuda_green_context_pool", gc_pool),
          Arg("index", static_cast<int32_t>(0)));
      contending_gc = make_resource<CudaGreenContext>(
          "contending_gc",
          Arg("cuda_green_context_pool", gc_pool),
          Arg("index", static_cast<int32_t>(1)));

      measured_stream_pool = make_resource<CudaStreamPool>(
          "measured_stream_pool", 0, 0, 0, 1, 5, measured_gc);
      contending_stream_pool = make_resource<CudaStreamPool>(
          "contending_stream_pool", 0, 0, 0, 1, 5, contending_gc);
    } else {
      measured_stream_pool   = make_resource<CudaStreamPool>("measured_stream_pool",   0, 0, 0, 1, 5);
      contending_stream_pool = make_resource<CudaStreamPool>("contending_stream_pool", 0, 0, 0, 1, 5);
    }

    // ─── Measured pipeline: PeriodicTxOp → InferenceOp → TimingRxOp ─────

    std::vector<TensorSpec> measured_inputs = {{"input", {1, measured_input_size_}}};

    auto measured_tx = make_operator<PeriodicTxOp>(
        "measured_tx",
        make_condition<PeriodicCondition>("measured_periodic",
            Arg("recess_period") = std::to_string(measured_frequency_hz_) + "hz",
            Arg("policy")        = periodic_policy_));
    measured_tx->set_tensor_specs(measured_inputs);
    measured_tx->set_record_timestamps(true);

    ops::InferenceOp::DataMap    m_model_map;
    ops::InferenceOp::DataVecMap m_pre_map, m_inf_map;
    m_model_map.insert("measured_model", measured_model_path_);
    m_pre_map.insert("measured_model", {"input"});
    m_inf_map.insert("measured_model", {"output"});

    auto measured_alloc = make_resource<UnboundedAllocator>("measured_alloc");
    measured_inference_op_ = make_operator<ops::InferenceOp>(
        "measured_inference",
        from_config("measured_inference"),
        Arg("backend",           backend_),
        Arg("model_path_map",    m_model_map),
        Arg("pre_processor_map", m_pre_map),
        Arg("inference_map",     m_inf_map),
        Arg("allocator")         = measured_alloc,
        Arg("cuda_stream_pool")  = measured_stream_pool);

    timing_rx_ = make_operator<TimingRxOp>(
        "timing_rx",
        Arg("total_samples",  total_samples_),
        Arg("warmup_samples", warmup_samples_));
    timing_rx_->set_run_label(run_label_);

    add_flow(measured_tx,           measured_inference_op_, {{"output", "receivers"}});
    add_flow(measured_inference_op_, timing_rx_,            {{"transmitter", "in"}});

    if (use_gc_) {
      measured_inference_op_->add_arg(measured_gc);
      measured_inference_op_->add_arg(gc_pool);
    }

    if (pin_measured_) {
      auto measured_pool = make_thread_pool("measured_pool", 0);
      auto cores_for = [this](int op_idx) -> std::vector<uint32_t> {
        if (pin_cores_.empty()) return {};
        if (pin_cores_.size() == 1) return {pin_cores_[0]};
        return {pin_cores_[static_cast<size_t>(op_idx)]};
      };
      if (sched_policy_ == SchedulingPolicy::kDeadline) {
        int64_t period_ns  = 1'000'000'000LL / measured_frequency_hz_;
        int64_t deadline_ns = period_ns;
        int64_t runtime_ns  = static_cast<int64_t>(period_ns * 0.90);
        measured_pool->add_realtime(measured_tx,            sched_policy_, true, cores_for(0), 0, runtime_ns, deadline_ns, period_ns);
        measured_pool->add_realtime(measured_inference_op_, sched_policy_, true, cores_for(1), 0, runtime_ns, deadline_ns, period_ns);
        measured_pool->add_realtime(timing_rx_,             sched_policy_, true, cores_for(2), 0, runtime_ns, deadline_ns, period_ns);
      } else {
        measured_pool->add_realtime(measured_tx,            sched_policy_, true, cores_for(0), 99);
        measured_pool->add_realtime(measured_inference_op_, sched_policy_, true, cores_for(1), 99);
        measured_pool->add_realtime(timing_rx_,             sched_policy_, true, cores_for(2), 99);
      }
    }

    // ─── Contending pipeline: PeriodicTxOp → InferenceOp → SinkOp ───────

    std::vector<TensorSpec> contending_inputs = {{"input", {1, contending_input_size_}}};

    std::shared_ptr<PeriodicTxOp> contending_tx;
    if (contending_frequency_hz_ > 0) {
      contending_tx = make_operator<PeriodicTxOp>(
          "contending_tx",
          make_condition<PeriodicCondition>("contending_periodic",
              Arg("recess_period") = std::to_string(contending_frequency_hz_) + "hz",
              Arg("policy")        = periodic_policy_));
    } else {
      contending_tx = make_operator<PeriodicTxOp>("contending_tx");
    }
    contending_tx->set_tensor_specs(contending_inputs);
    contending_tx->set_record_timestamps(false);

    ops::InferenceOp::DataMap    c_model_map;
    ops::InferenceOp::DataVecMap c_pre_map, c_inf_map;
    c_model_map.insert("contending_model", contending_model_path_);
    c_pre_map.insert("contending_model", {"input"});
    c_inf_map.insert("contending_model", {"output"});

    auto contending_alloc = make_resource<UnboundedAllocator>("contending_alloc");
    contending_inference_op_ = make_operator<ops::InferenceOp>(
        "contending_inference",
        from_config("contending_inference"),
        Arg("backend",           backend_),
        Arg("model_path_map",    c_model_map),
        Arg("pre_processor_map", c_pre_map),
        Arg("inference_map",     c_inf_map),
        Arg("allocator")         = contending_alloc,
        Arg("cuda_stream_pool")  = contending_stream_pool);

    contending_sink_ = make_operator<ContendingSinkOp>(
        "contending_sink",
        Arg("ready_after_iters", contending_ready_iters_));

    add_flow(contending_tx,           contending_inference_op_, {{"output", "receivers"}});
    add_flow(contending_inference_op_, contending_sink_,        {{"transmitter", "in"}});

    if (use_gc_) {
      contending_inference_op_->add_arg(contending_gc);
      contending_inference_op_->add_arg(gc_pool);
    }
  }

  const std::vector<double>& get_e2e_samples()        const { return timing_rx_->get_e2e_samples(); }
  const std::vector<double>& get_tick_period_samples() const { return timing_rx_->get_tick_period_samples(); }
  int    get_contending_iters()         const { return contending_sink_->get_completed_iters(); }
  double get_contending_throughput_hz() const { return contending_sink_->get_throughput_hz(); }

 private:
  bool        use_gc_;
  int         total_samples_, warmup_samples_;
  std::string measured_model_path_;
  int         measured_input_size_;
  std::string contending_model_path_;
  int         contending_input_size_;
  std::string backend_;
  int         measured_sms_, contending_sms_;
  int         measured_frequency_hz_, contending_frequency_hz_;
  std::string periodic_policy_;
  bool        pin_measured_;
  SchedulingPolicy      sched_policy_;
  std::vector<uint32_t> pin_cores_;
  int         contending_ready_iters_;
  std::string run_label_;

  std::shared_ptr<ops::InferenceOp> measured_inference_op_;
  std::shared_ptr<ops::InferenceOp> contending_inference_op_;
  std::shared_ptr<TimingRxOp>       timing_rx_;
  std::shared_ptr<ContendingSinkOp> contending_sink_;
};

// =============================================================================
// Output formatting
// =============================================================================

static const std::string kBar(80, '=');
static const std::string kDash(56, '-');
// UTF-8 micro sign + 's'
static const std::string kUs = "\xce\xbc" "s";

// Print a single stat block (one condition, no comparison column).
void print_stats_block(const BenchmarkStats& s, const std::string& label) {
  std::cout << "  " << label << "\n";
  if (s.sample_count == 0) { std::cout << "    No data collected.\n"; return; }
  std::cout << std::fixed << std::setprecision(2);
  auto row = [&](const std::string& name, double val) {
    std::cout << "    " << std::left << std::setw(20) << name
              << std::right << std::setw(10) << val << " " << kUs << "\n";
  };
  row("Average",        s.avg);
  row("Median (P50)",   s.p50);
  row("P95",            s.p95);
  row("P99",            s.p99);
  row("Std deviation",  s.std_dev);
  row("Min",            s.min_val);
  row("Max",            s.max_val);
  std::cout << "    " << std::left << std::setw(20) << "Samples"
            << std::right << std::setw(10) << s.sample_count << "\n";
}

// Print a side-by-side comparison table with percentage deltas.
// For latency metrics, negative delta = green context is faster = better.
void print_comparison_table(const BenchmarkStats& bl, const BenchmarkStats& gc) {
  if (bl.sample_count == 0 || gc.sample_count == 0) return;

  auto delta_pct = [](double baseline, double gc_val) -> double {
    return baseline != 0.0 ? (gc_val - baseline) / baseline * 100.0 : 0.0;
  };
  // For latency, lower is better; negative delta means green context is faster.
  auto label_for = [](double dp) -> const char* {
    if (dp <= -20.0) return "  much better";
    if (dp <=  -5.0) return "  better";
    if (dp >=  20.0) return "  much worse";
    if (dp >=   5.0) return "  worse";
    return "  similar";
  };

  std::cout << "\n";
  std::cout << "  " << std::left << std::setw(18) << "Metric"
            << std::right << std::setw(12) << "Baseline"
            << std::setw(3)  << ""
            << std::setw(13) << "Green Context"
            << std::setw(9)  << "Change"
            << "\n";
  std::cout << "  " << kDash << "\n";

  std::cout << std::fixed << std::setprecision(2);
  auto row = [&](const std::string& name, double a, double b) {
    double dp = delta_pct(a, b);
    std::cout << "  " << std::left << std::setw(18) << name
              << std::right
              << std::setw(10) << a << " " << kUs << "  "
              << std::setw(10) << b << " " << kUs << "  "
              << std::showpos << std::setprecision(1) << std::setw(6) << dp << "%" << std::noshowpos
              << label_for(dp) << "\n";
  };

  row("Average",         bl.avg,     gc.avg);
  row("Median (P50)",    bl.p50,     gc.p50);
  row("P95",             bl.p95,     gc.p95);
  row("P99",             bl.p99,     gc.p99);
  row("Std deviation",   bl.std_dev, gc.std_dev);
  row("Min",             bl.min_val, gc.min_val);
  row("Max",             bl.max_val, gc.max_val);

  std::cout << "  " << std::left << std::setw(18) << "Samples"
            << std::right
            << std::setw(10) << bl.sample_count << "       "
            << std::setw(10) << gc.sample_count << "\n";
}

void print_usage(const char* prog) {
  std::cout
      << "\nGreen Context Inference Scheduling Benchmark\n"
      << "═══════════════════════════════════════════════════════════════════════\n"
      << "Measures end-to-end inference latency for a high-frequency 'measured'\n"
      << "pipeline running alongside a heavy 'contending' inference pipeline,\n"
      << "with and without CUDA Green Context SM partitioning.\n\n"
      << "Usage: " << prog << " [OPTIONS]\n\n"
      << "Core options:\n"
      << "  --samples N                Measurement samples per run      (default: 1000)\n"
      << "  --warmup-samples N         Samples discarded after warmup   (default: 100)\n"
      << "  --repeat N                 Interleaved baseline/GC run pairs(default: 1)\n"
      << "                             Pooling samples across N pairs gives more\n"
      << "                             statistically robust comparisons.\n"
      << "  --backend BACKEND          'trt', 'onnxrt', or 'all'         (default: trt)\n"
      << "                             'all' runs both backends and compares them.\n"
      << "  --frequency-hz N           Measured pipeline rate in Hz     (default: 1000)\n"
      << "  --mode MODE                'baseline', 'green-context', or 'all'\n"
      << "                                                               (default: all)\n\n"
      << "Measured model  (the high-frequency pipeline under test):\n"
      << "  --measured-input-size N    Input/output dimension            (default: 256)\n"
      << "  --measured-hidden-size N   Hidden layer width                (default: 2048)\n"
      << "  --measured-layers N        Number of FC layers               (default: 4)\n"
      << "  Note: the measured model must be large enough to occupy multiple GPU SMs\n"
      << "  for green context isolation to produce a meaningful benefit.\n\n"
      << "Contending model  (the heavy interfering workload):\n"
      << "  --contending-input-size N  Input/output dimension            (default: 1024)\n"
      << "  --contending-hidden-size N Hidden layer width                (default: 4096)\n"
      << "  --contending-layers N      Number of FC layers               (default: 6)\n"
      << "  --contending-frequency-hz N Rate in Hz; 0 = free-running     (default: 0)\n\n"
      << "Scheduling:\n"
      << "  --worker-threads N         EventBasedScheduler worker threads\n"
      << "                             (default: half of hardware_concurrency, min 2)\n"
      << "  --contending-ready-iters N Contending inferences before measurement begins\n"
      << "                             (default: 50 — must exceed TRT compile time)\n"
      << "  --periodic-policy POLICY   CatchUpMissedTicks | MinTimeBetweenTicks |\n"
      << "                             NoCatchUpMissedTicks  (default: CatchUpMissedTicks)\n"
      << "  --pin-measured-pipeline    Pin measured pipeline ops to a real-time thread pool\n"
      << "  --scheduling-policy POL    SCHED_FIFO | SCHED_RR | SCHED_DEADLINE\n"
      << "                             (only with --pin-measured-pipeline)\n"
      << "  --pin-cores C0[,C1,C2]    CPU core IDs for pinned threads (1 or 3 values)\n"
      << "  --enable-postcheck-fastpath\n"
      << "                             Enable EBS worker postcheck fast path.\n"
      << "                             Requires Holoscan SDK with that parameter;\n"
      << "                             startup will fail if the SDK does not support it.\n\n"
      << "Green Context partitioning:\n"
      << "  --measured-sms N           SMs for measured partition; 0 = auto-split\n"
      << "  --contending-sms N         SMs for contending partition; 0 = auto-split\n"
      << "  --sms-per-partition N      Set both partitions to N SMs\n\n"
      << "Paths:\n"
      << "  --model-dir PATH           Directory for generated ONNX models\n"
      << "                             (default: directory of the executable)\n"
      << "  --help                     Show this message\n\n";
}

// =============================================================================
// Model generation
// =============================================================================

// Returns true if the path is safe to embed in a shell command string.
// Rejects characters that can escape double-quote quoting.
static bool path_is_shell_safe(const std::filesystem::path& p) {
  static const std::string kBad = ";|&`$(){}!<>\n\r'\"\\";
  return p.string().find_first_of(kBad) == std::string::npos;
}

bool ensure_model_exists(const std::filesystem::path& gen_script,
                          const std::filesystem::path& model_path,
                          int input_size, int hidden_size, int num_layers,
                          const std::string& label) {
  if (std::filesystem::exists(model_path)) {
    std::cout << "  " << label << " model found: " << model_path << "\n";
    return true;
  }

  if (!path_is_shell_safe(gen_script) || !path_is_shell_safe(model_path)) {
    std::cerr << "Error: script or model path contains shell metacharacters — "
              << "cannot safely invoke python3. Use --model-dir with a plain path.\n";
    return false;
  }

  std::string cmd = "python3 \"" + gen_script.string() + "\""
      " --output \"" + model_path.string() + "\""
      " --input-size "  + std::to_string(input_size)  +
      " --hidden-size " + std::to_string(hidden_size) +
      " --num-layers "  + std::to_string(num_layers);

  std::cout << "  Generating " << label << " model ...\n    " << cmd << "\n";
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "Error: model generation failed (exit code " << ret << ")\n";
    return false;
  }
  return true;
}

// =============================================================================
// Diagnostic analysis
//
// Explains, in plain language, why each metric improved or worsened when
// green context SM partitioning is enabled.  Covers:
//   - Whether the SM partition configuration is valid (no over-subscription)
//   - Whether the measured and contending models are large enough to benefit
//   - Backend-specific behaviour (TRT vs OnnxRT kernel footprints)
//   - Per-metric root causes for improvement, no change, or regression
//   - An overall verdict with actionable next steps
//
// Only called when mode == "all" (both baseline and green-context data exist).
// =============================================================================

static void print_diagnosis(
    const BenchmarkStats& bl_e2e,   const BenchmarkStats& gc_e2e,
    const BenchmarkStats& bl_period, const BenchmarkStats& gc_period,
    double bl_throughput, double gc_throughput,
    const std::string& backend,
    int total_sms, int measured_sms_req, int contending_sms_req,
    int measured_input_size,   int measured_hidden_size,   int measured_layers,
    int contending_input_size, int contending_hidden_size, int contending_layers,
    int frequency_hz) {

  if (bl_e2e.sample_count == 0 || gc_e2e.sample_count == 0) return;

  // ── Replicate compose() SM alignment so reported values match what ran ────
  int rounded_total = total_sms & ~3;
  int m_sms = measured_sms_req > 0
      ? std::min(std::max(4, measured_sms_req   & ~3), rounded_total)
      : std::max(4, (rounded_total / 2) & ~3);
  int c_sms = contending_sms_req > 0
      ? std::min(std::max(4, contending_sms_req & ~3), rounded_total)
      : std::max(4, (rounded_total / 2) & ~3);
  bool over_subscribed = (m_sms + c_sms) > rounded_total;

  // ── FLOPs estimate ────────────────────────────────────────────────────────
  // Architecture: (1 x input) -> [MatMul hidden -> ReLU] x (layers-1) -> (1 x input)
  //   Layer 0        : input  -> hidden  : 2 * input  * hidden  FLOPs
  //   Layers 1..N-2  : hidden -> hidden  : 2 * hidden * hidden  FLOPs each
  //   Layer N-1      : hidden -> input   : 2 * hidden * input   FLOPs
  auto model_flops = [](int in, int hid, int L) -> long long {
    if (L <= 0) return 0LL;
    if (L == 1) return 2LL * in * in;
    long long f = 2LL * in * hid;
    f += 2LL * std::max(0, L - 2) * hid * hid;
    f += 2LL * hid * in;
    return f;
  };
  long long m_flops = model_flops(measured_input_size,   measured_hidden_size,   measured_layers);
  long long c_flops = model_flops(contending_input_size, contending_hidden_size, contending_layers);

  auto flops_str = [](long long f) -> std::string {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if      (f >= 1'000'000'000LL) oss << static_cast<double>(f) / 1e9 << " GFLOP/inf";
    else if (f >= 1'000'000LL)     oss << static_cast<double>(f) / 1e6 << " MFLOP/inf";
    else                           oss << f                            << " FLOP/inf";
    return oss.str();
  };

  // Heuristic: a hidden layer of >= 1024 with >= 32 MFLOPs almost certainly
  // spans multiple SMs.  Below that, the entire workload fits in one or two
  // thread blocks and SM-level isolation makes no difference.
  bool measured_multi_sm = measured_hidden_size >= 1024 && m_flops >= 32'000'000LL;
  bool contending_heavy  = c_flops >= 256'000'000LL;

  // ── Per-metric deltas ─────────────────────────────────────────────────────
  auto dpct = [](double bl, double gc) -> double {
    return bl != 0.0 ? (gc - bl) / bl * 100.0 : 0.0;
  };
  double avg_dp = dpct(bl_e2e.avg,      gc_e2e.avg);
  double p99_dp = dpct(bl_e2e.p99,      gc_e2e.p99);
  double std_dp = dpct(bl_e2e.std_dev,  gc_e2e.std_dev);
  double per_dp = dpct(bl_period.std_dev, gc_period.std_dev);
  double tp_dp  = (bl_throughput > 0.0 && gc_throughput > 0.0)
                  ? dpct(bl_throughput, gc_throughput) : 0.0;

  // ── Header ────────────────────────────────────────────────────────────────
  std::cout << "\n" << kBar << "\n"
            << "  DIAGNOSTIC ANALYSIS\n"
            << "  Backend: " << backend << "  |  GPU: " << total_sms << " SMs total\n"
            << "  Explains why each metric is better or worse with green context,\n"
            << "  and what to tune if results are not as expected.\n"
            << kBar << "\n\n";

  // ── Section 1: SM partition configuration ────────────────────────────────
  std::cout << "  +-- SM Partition Configuration ----------------------------------------+\n";
  if (total_sms > 0) {
    std::cout << "  |  GPU total SMs           : " << total_sms << "\n"
              << "  |  Measured pipeline       : " << m_sms
              << " SMs  (" << (100 * m_sms / total_sms) << "% of GPU)\n"
              << "  |  Contending pipeline     : " << c_sms
              << " SMs  (" << (100 * c_sms / total_sms) << "% of GPU)\n";
    int unalloc = rounded_total - m_sms - c_sms;
    if (!over_subscribed && unalloc > 0) {
      std::cout << "  |  Unallocated SMs         : " << unalloc
                << "  (idle during benchmark — neither pipeline can use them)\n";
    }
  }
  if (over_subscribed) {
    std::cout << "  |  *** OVER-SUBSCRIPTION: " << m_sms << " + " << c_sms
              << " = " << (m_sms + c_sms) << " > " << rounded_total << " aligned SMs ***\n"
              << "  |      Over-lapping partitions mean CUDA cannot guarantee exclusive\n"
              << "  |      SM access for either pipeline.  True isolation requires that\n"
              << "  |      measured_sms + contending_sms <= " << rounded_total << ".\n"
              << "  |      Fix: --measured-sms " << (rounded_total / 2)
              << " --contending-sms " << (rounded_total / 2) << "\n";
  } else {
    std::cout << "  |  Partition validity      : OK — partitions do not overlap\n";
  }
  std::cout << "  +----------------------------------------------------------------------+\n\n";

  // ── Section 2: Model workload assessment ──────────────────────────────────
  std::cout << "  +-- Model Workload Assessment ------------------------------------------+\n"
            << "  |  (Green context only helps if models occupy multiple SMs)\n"
            << "  |\n"
            << "  |  Measured model  : "
            << measured_input_size << " input, " << measured_hidden_size << " hidden, "
            << measured_layers << " layers  [~" << flops_str(m_flops) << "]\n"
            << "  |    SM occupancy  : "
            << (measured_multi_sm
                  ? "MULTI-SM likely — good candidate for GC isolation"
                  : "SINGLE-SM likely — GC partitioning has minimal effect at this size")
            << "\n"
            << "  |  Contending model: "
            << contending_input_size << " input, " << contending_hidden_size << " hidden, "
            << contending_layers << " layers  [~" << flops_str(c_flops) << "]\n"
            << "  |    GPU pressure  : "
            << (contending_heavy
                  ? "HEAVY — creates strong SM contention in baseline"
                  : "MODERATE — baseline may not be fully SM-contended")
            << "\n";
  if (m_flops > 0) {
    std::cout << "  |  Workload ratio  : "
              << std::fixed << std::setprecision(1)
              << static_cast<double>(c_flops) / m_flops
              << "x more FLOPs in contending vs. measured\n";
  }
  if (!measured_multi_sm) {
    std::cout << "  |  ** The measured model is too small to span multiple SMs.  Adding\n"
              << "  |     a GC partition shrinks the SM pool without isolating anything.\n"
              << "  |     Increase: --measured-hidden-size 4096 --measured-layers 6\n";
  }
  if (!contending_heavy) {
    std::cout << "  |  ** The contending model may not saturate the GPU in baseline, so\n"
              << "  |     there may be little SM contention to isolate in the first place.\n"
              << "  |     Increase: --contending-hidden-size 8192 --contending-layers 8\n";
  }
  std::cout << "  +----------------------------------------------------------------------+\n\n";

  // ── Section 3: Backend notes ──────────────────────────────────────────────
  std::cout << "  +-- Inference Backend: " << backend
            << " ---------------------------------------------------+\n";
  if (backend == "trt") {
    std::cout << "  |  TensorRT compiles specialised, fused CUDA kernels tailored to the\n"
              << "  |  exact SM count in the green context partition.  These kernels can\n"
              << "  |  saturate many SMs simultaneously, making SM-level interference from\n"
              << "  |  a concurrent TRT workload severe — and GC isolation highly impactful.\n"
              << "  |\n"
              << "  |  When a green context limits the visible SM count, TRT may recompile\n"
              << "  |  the engine for the smaller SM budget.  If the partition is too small,\n"
              << "  |  the recompiled plan can be LESS efficient than the baseline plan that\n"
              << "  |  had full GPU access, producing higher latency despite isolation.\n"
              << "  |\n"
              << "  |  The --contending-ready-iters warmup gate ensures TRT engine\n"
              << "  |  compilation (which can take several seconds) completes before\n"
              << "  |  latency measurement begins, so compilation time is not measured.\n";
  } else if (backend == "onnxrt") {
    std::cout << "  |  ONNX Runtime dispatches generic cuBLAS / cuDNN kernels rather than\n"
              << "  |  TRT-fused ones.  These kernels typically use fewer SMs per call and\n"
              << "  |  have less aggressive SM occupancy, so SM-level interference between\n"
              << "  |  pipelines is often less severe than with TRT.\n"
              << "  |\n"
              << "  |  For OnnxRT workloads, CUDA stream ordering (ensuring the measured\n"
              << "  |  pipeline's stream is not blocked by the contending stream at the\n"
              << "  |  driver level) often matters more than SM count partitioning.\n"
              << "  |  A green context still provides strict SM isolation, but the\n"
              << "  |  observable latency benefit may be smaller than with TRT.\n";
  }
  std::cout << "  +----------------------------------------------------------------------+\n\n";

  // ── Section 4: Metric-by-metric interpretation ────────────────────────────
  std::cout << "  +-- Metric-by-Metric Interpretation ------------------------------------+\n";
  std::cout << std::fixed;

  // Helper: print a +/-% header for a metric row
  auto metric_header = [&](const std::string& name, double dp) {
    std::cout << "  |\n  |  " << name << "  "
              << std::showpos << std::setprecision(1) << dp << "%" << std::noshowpos << "\n";
  };

  // ── Average latency ───────────────────────────────────────────────────────
  metric_header("Average end-to-end latency:", avg_dp);
  if (avg_dp <= -15.0) {
    std::cout
      << "  |    STRONG IMPROVEMENT.  Green context eliminated most SM queuing delay.\n"
      << "  |    The measured pipeline now has exclusive access to its SM partition;\n"
      << "  |    its kernels no longer stall waiting for SMs that the contending\n"
      << "  |    workload is occupying.  The contending workload's kernels are\n"
      << "  |    dispatched to their own partition and cannot preempt the measured\n"
      << "  |    pipeline's SMs.\n";
  } else if (avg_dp <= -5.0) {
    std::cout
      << "  |    MODERATE IMPROVEMENT.  SM isolation is reducing contention.\n"
      << "  |    Some overhead limits the full benefit:\n"
      << "  |     - TRT may have recompiled the measured kernel for the smaller SM\n"
      << "  |       budget; the new plan is efficient but not as optimal as the\n"
      << "  |       full-GPU baseline plan.\n"
      << "  |     - Green context stream-management adds a small fixed per-inference\n"
      << "  |       CPU cost.  This is usually < 5 us but adds to average latency.\n"
      << "  |    Increase --measured-sms if latency is still above target.\n";
  } else if (avg_dp >= 15.0) {
    std::cout
      << "  |    WORSE.  Average latency INCREASED with green context.  Root causes:\n";
    if (over_subscribed)
      std::cout
        << "  |     - OVER-SUBSCRIPTION: measured+contending SMs overlap.  The measured\n"
        << "  |       pipeline is allocated a paper budget larger than the available pool.\n"
        << "  |       CUDA cannot honour the isolation guarantee.  Fix the partition.\n";
    if (!measured_multi_sm)
      std::cout
        << "  |     - MEASURED MODEL TOO SMALL: the kernel fits in a single thread block\n"
        << "  |       and already runs on one SM.  Restricting the SM pool via GC gives\n"
        << "  |       no isolation benefit but adds stream-management overhead.\n";
    std::cout
      << "  |     - TRT KERNEL REGRESSION: TRT recompiled the measured engine for the\n"
      << "  |       smaller SM count and the new plan is suboptimal for this batch size.\n"
      << "  |       Try running with --backend onnxrt to isolate TRT kernel selection.\n"
      << "  |     - CPU SCHEDULING OVERHEAD: green context management (stream creation,\n"
      << "  |       event barriers across context boundaries) dominates at high frequency.\n"
      << "  |       Fix: --pin-measured-pipeline --scheduling-policy SCHED_FIFO\n";
  } else {
    std::cout
      << "  |    NO SIGNIFICANT CHANGE.  Possible explanations:\n";
    if (!measured_multi_sm)
      std::cout
        << "  |     - MEASURED MODEL TOO SMALL: the kernel runs in one or two thread\n"
        << "  |       blocks and is not SM-contended.  SM partitioning is irrelevant.\n"
        << "  |       Fix: --measured-hidden-size 4096 --measured-layers 6\n";
    if (!contending_heavy)
      std::cout
        << "  |     - CONTENDING WORKLOAD NOT HEAVY: the GPU was not fully SM-saturated\n"
        << "  |       in the baseline, so the measured pipeline already received enough\n"
        << "  |       SMs without isolation.\n"
        << "  |       Fix: --contending-hidden-size 8192 --contending-layers 8\n";
    std::cout
      << "  |     - MEMORY BANDWIDTH BOTTLENECK: green context partitions SMs, not\n"
      << "  |       DRAM bandwidth.  If both pipelines are bandwidth-bound, partitioning\n"
      << "  |       SMs does nothing — both still compete for the same memory bus.\n"
      << "  |     - TICK JITTER DOMINATES: if scheduler tick jitter (see TICK ACCURACY\n"
      << "  |       section) is large relative to inference time, it masks any SM\n"
      << "  |       isolation benefit in the end-to-end measurement.\n";
  }

  // ── P99 tail latency ─────────────────────────────────────────────────────
  metric_header("P99 tail latency:", p99_dp);
  if (p99_dp <= -10.0) {
    std::cout
      << "  |    IMPROVED.  This is the PRIMARY benefit for real-time pipelines.\n"
      << "  |    In the baseline, tail-latency spikes occur when the contending\n"
      << "  |    pipeline's kernels happen to monopolise all shared SMs at the\n"
      << "  |    exact moment the measured pipeline's kernel is dispatched.\n"
      << "  |    Green context makes such monopolisation impossible: the measured\n"
      << "  |    pipeline's SMs are reserved and cannot be taken by the contending\n"
      << "  |    workload under any scheduling conditions.\n";
  } else if (p99_dp >= 10.0) {
    std::cout
      << "  |    WORSE.  Tail latency increased with green context.  This typically\n"
      << "  |    means the measured pipeline's SM partition is too small for its\n"
      << "  |    kernel to schedule without intra-partition queuing.\n"
      << "  |    When a kernel cannot spread across enough SMs, some thread blocks\n"
      << "  |    must wait for earlier blocks to finish — creating tail-latency spikes\n"
      << "  |    within the partition itself.\n"
      << "  |    Fix: increase --measured-sms (and reduce --contending-sms accordingly).\n";
  } else {
    std::cout
      << "  |    SIMILAR in both conditions.  The contending model may not be large\n"
      << "  |    enough to create tail-latency spikes in the baseline, so there is\n"
      << "  |    nothing for GC to protect against.  Use a heavier contending model.\n";
  }

  // ── Jitter (std dev of e2e latency) ─────────────────────────────────────
  metric_header("Latency jitter (std dev):", std_dp);
  if (std_dp <= -15.0) {
    std::cout
      << "  |    REDUCED.  The pipeline is significantly more deterministic with GC.\n"
      << "  |    In the baseline, the contending pipeline randomly wins SM scheduling\n"
      << "  |    races against the measured pipeline — sometimes it gets there first,\n"
      << "  |    sometimes it doesn't.  This non-determinism is the root cause of\n"
      << "  |    latency jitter.  GC eliminates the race entirely: each pipeline has\n"
      << "  |    its own SMs and there is no competition.  Lower jitter means the\n"
      << "  |    pipeline more reliably meets per-frame deadlines.\n";
  } else if (std_dp >= 20.0) {
    std::cout
      << "  |    INCREASED.  Green context management is introducing its own jitter:\n"
      << "  |     - The measured partition may be too small.  Some kernel thread blocks\n"
      << "  |       queue within the partition, producing variable intra-partition wait.\n"
      << "  |     - The EventBasedScheduler has too few worker threads; a worker-thread\n"
      << "  |       stall causes a variable delay before the TX operator can fire.\n"
      << "  |       Fix: increase --worker-threads.\n"
      << "  |     - CUDA stream event synchronisation across green context boundaries\n"
      << "  |       adds a small but variable CPU-side latency per inference.\n";
  } else {
    std::cout
      << "  |    SIMILAR in both conditions.\n";
  }

  // ── Scheduler tick accuracy (period std dev) ─────────────────────────────
  metric_header("Scheduler tick jitter (PeriodicTxOp inter-fire std dev):", per_dp);
  if (std::abs(per_dp) < 10.0) {
    std::cout
      << "  |    SIMILAR in both conditions.  The EventBasedScheduler fires at a\n"
      << "  |    consistent interval regardless of green context configuration.\n"
      << "  |    CPU thread scheduling, not GPU SM allocation, governs tick accuracy.\n"
      << "  |    Note: tick jitter adds directly to end-to-end latency because a late\n"
      << "  |    TX fire shifts the entire measurement window forward.\n";
  } else if (per_dp > 10.0) {
    std::cout
      << "  |    TICK JITTER INCREASED with green context.  Per-tick green context\n"
      << "  |    stream overhead (context switch, event recording) is delaying the\n"
      << "  |    TX operator.  Fix: --pin-measured-pipeline --scheduling-policy SCHED_FIFO\n"
      << "  |    to give the TX operator a real-time CPU thread unaffected by other\n"
      << "  |    software on the system.\n";
  } else {
    std::cout
      << "  |    TICK JITTER IMPROVED slightly.  The measured pipeline's CPU thread\n"
      << "  |    may be less preempted when the contending pipeline's GPU kernels\n"
      << "  |    complete faster (within their dedicated SM partition).\n";
  }

  // ── Contending pipeline throughput ───────────────────────────────────────
  if (bl_throughput > 0.0 && gc_throughput > 0.0) {
    metric_header("Contending pipeline throughput:", tp_dp);
    if (tp_dp <= -10.0) {
      std::cout
        << "  |    REDUCED (expected and desirable).  The contending pipeline's SM\n"
        << "  |    quota was restricted by the green context partition.  Lower contending\n"
        << "  |    throughput CONFIRMS that SM partitioning is active and working:\n"
        << "  |    the contending pipeline has fewer SMs to schedule its kernels onto,\n"
        << "  |    so it completes fewer inferences per second.  This is exactly the\n"
        << "  |    intended trade-off — contending pipeline throughput is sacrificed\n"
        << "  |    to give the measured pipeline guaranteed, uncontested SM access.\n";
    } else if (tp_dp >= 10.0) {
      std::cout
        << "  |    INCREASED (unexpected).  Contending throughput is higher with GC.\n"
        << "  |    This is counterintuitive and suggests:\n"
        << "  |     - --contending-sms may be set larger than the baseline share.\n"
        << "  |     - TRT compiled a more efficient kernel for the contending partition's\n"
        << "  |       SM count than for the shared (unpartitioned) baseline.\n"
        << "  |    Verify partition sizes and check whether over-subscription applies.\n";
    } else {
      std::cout
        << "  |    SIMILAR in both conditions.  The contending pipeline's throughput\n"
        << "  |    did not decrease with green context, which means SM partitioning\n"
        << "  |    may not be creating meaningful isolation for this workload:\n"
        << "  |     - The contending model is memory-bandwidth-bound, not SM-bound.\n"
        << "  |       Restricting its SMs does not slow it down; it was never using\n"
        << "  |       all available SMs simultaneously.\n"
        << "  |     - --contending-frequency-hz is set and caps throughput regardless\n"
        << "  |       of SM availability.\n"
        << "  |    If contending throughput does not drop with GC, the measured pipeline\n"
        << "  |    likely does not benefit from SM isolation for this model size either.\n";
    }
  }

  std::cout << "  |\n"
            << "  +----------------------------------------------------------------------+\n\n";

  // ── Overall verdict ───────────────────────────────────────────────────────
  bool avg_better    = avg_dp  <= -5.0;
  bool p99_better    = p99_dp  <= -5.0;
  bool jitter_better = std_dp  <= -10.0;
  bool avg_worse     = avg_dp  >=  10.0;
  bool p99_worse     = p99_dp  >=  10.0;

  std::cout << "  +-- Overall Verdict ----------------------------------------------------+\n";
  if (avg_better && p99_better && jitter_better) {
    std::cout
      << "  |  GREEN CONTEXT IS STRONGLY BENEFICIAL for this configuration.\n"
      << "  |  Average latency, tail latency, AND jitter all improved.  The measured\n"
      << "  |  pipeline is demonstrably less affected by the contending workload when\n"
      << "  |  SM partitioning is active.  This is the expected outcome when:\n"
      << "  |    - Both models are large enough to occupy multiple SMs\n"
      << "  |    - The contending model creates real GPU pressure\n"
      << "  |    - The partition sizes are set correctly (no over-subscription)\n"
      << "  |  Recommended for production real-time pipelines on CUDA 12.4+ / driver 550+.\n";
  } else if ((avg_better || !avg_worse) && p99_better) {
    std::cout
      << "  |  GREEN CONTEXT IMPROVES TAIL LATENCY — the most important metric for\n"
      << "  |  real-time pipelines.  Even if average latency is unchanged, reducing\n"
      << "  |  worst-case spikes is the core value proposition of SM partitioning:\n"
      << "  |  it bounds the worst case by removing non-deterministic SM scheduling\n"
      << "  |  races between the measured and contending pipelines.\n";
  } else if (jitter_better && !avg_worse && !p99_worse) {
    std::cout
      << "  |  GREEN CONTEXT IMPROVES PIPELINE PREDICTABILITY.  Average and tail\n"
      << "  |  latency are similar, but the pipeline is more deterministic.  Useful\n"
      << "  |  when deadline misses — not average latency — are the primary concern.\n";
  } else if (avg_worse || p99_worse) {
    std::cout
      << "  |  GREEN CONTEXT IS NOT HELPING with this configuration.\n"
      << "  |  See the individual metric explanations above for root causes.\n"
      << "  |\n"
      << "  |  Recommended next steps:\n";
    if (!measured_multi_sm)
      std::cout
        << "  |    1. Increase measured model size:\n"
        << "  |         --measured-hidden-size 4096 --measured-layers 6\n";
    if (!contending_heavy)
      std::cout
        << "  |    2. Increase contending workload intensity:\n"
        << "  |         --contending-hidden-size 8192 --contending-layers 8\n";
    if (over_subscribed)
      std::cout
        << "  |    3. Fix SM over-subscription:\n"
        << "  |         --measured-sms " << (rounded_total / 2)
        << " --contending-sms " << (rounded_total / 2) << "\n";
    std::cout
      << "  |    4. Verify CUDA 12.4+ green context support:\n"
      << "  |         nvidia-smi -q | grep -i 'cuda version'   (need driver 550+)\n";
  } else {
    std::cout
      << "  |  GREEN CONTEXT HAS NO SIGNIFICANT EFFECT with this configuration.\n"
      << "  |  The workload is likely not SM-contended in the baseline, the measured\n"
      << "  |  model is too small, or the contending workload is memory-bandwidth-bound\n"
      << "  |  rather than SM-bound.  Refer to the model workload assessment above.\n";
  }
  std::cout
    << "  +----------------------------------------------------------------------+\n";
}

// =============================================================================
// Backend comparison
//
// Side-by-side analysis of TRT vs ONNX Runtime across three views:
//   1. Baseline (no green context) — which backend is faster on a shared GPU
//   2. Green context — which backend benefits more from SM partitioning
//   3. GC improvement delta — which backend sees a larger latency reduction
//      when green context is enabled
//
// Also provides a plain-language explanation of why each backend behaves
// the way it does for these fully-connected ONNX-model workloads.
// =============================================================================

// Holds final aggregate statistics for one (backend, condition) pair.
struct BackendStats {
  std::string       backend;
  BenchmarkStats    e2e;
  BenchmarkStats    period;
  double            tp_hz    = 0.0;   // contending pipeline throughput
  bool              has_data = false;
};

static void print_backend_comparison(
    const BackendStats& trt_bl, const BackendStats& trt_gc,
    const BackendStats& ort_bl, const BackendStats& ort_gc,
    int total_sms, int measured_sms_req, int contending_sms_req,
    int measured_input_size,   int measured_hidden_size,   int measured_layers,
    int contending_input_size, int contending_hidden_size, int contending_layers) {

  std::cout << "\n" << kBar << "\n"
            << "  BACKEND COMPARISON: TRT vs ONNX Runtime\n"
            << "  GPU: " << total_sms << " SMs total"
            << (measured_sms_req > 0
                  ? "  |  measured partition: " + std::to_string(measured_sms_req) + " SMs"
                  : "  |  GC partitions: auto (half each)")
            << "\n"
            << "  Measured model : " << measured_input_size   << " input, "
            << measured_hidden_size   << " hidden, " << measured_layers   << " layers\n"
            << "  Contending model: " << contending_input_size << " input, "
            << contending_hidden_size << " hidden, " << contending_layers << " layers\n"
            << kBar << "\n\n";

  // ── Helper: "better" label for latency (lower = better) ──────────────────
  auto dpct = [](double a, double b) -> double {  // (b - a) / a * 100
    return a != 0.0 ? (b - a) / a * 100.0 : 0.0;
  };
  // For latency: negative delta means ORT < TRT, so ORT is faster
  auto winner_lat = [](double trt_val, double ort_val) -> const char* {
    double d = trt_val != 0.0 ? (ort_val - trt_val) / trt_val * 100.0 : 0.0;
    if      (d <= -10.0) return "ONNX RT (much faster)";
    else if (d <=  -3.0) return "ONNX RT (faster)";
    else if (d >=  10.0) return "TRT (much faster)";
    else if (d >=   3.0) return "TRT (faster)";
    return "similar";
  };
  // For throughput: higher = better
  auto winner_tp = [](double trt_val, double ort_val) -> const char* {
    double d = trt_val != 0.0 ? (ort_val - trt_val) / trt_val * 100.0 : 0.0;
    if      (d >=  10.0) return "ONNX RT (higher)";
    else if (d >=   3.0) return "ONNX RT (slightly higher)";
    else if (d <= -10.0) return "TRT (higher)";
    else if (d <=  -3.0) return "TRT (slightly higher)";
    return "similar";
  };

  // ── Table printer: two backend columns + winner ───────────────────────────
  auto print_row = [&](const std::string& metric,
                        double trt_v, double ort_v, bool higher_is_better = false) {
    double d = trt_v != 0.0 ? (ort_v - trt_v) / trt_v * 100.0 : 0.0;
    const char* w = higher_is_better ? winner_tp(trt_v, ort_v) : winner_lat(trt_v, ort_v);
    std::cout << "  " << std::left  << std::setw(18) << metric
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << trt_v << " " << kUs << "  "
              << std::setw(10) << ort_v << " " << kUs << "  "
              << std::showpos << std::setprecision(1) << std::setw(7) << d << "% "
              << std::noshowpos << w << "\n";
  };
  auto print_header = [&](const std::string& title) {
    std::cout << "\n  -- " << title << " --\n";
    std::cout << "  " << std::left  << std::setw(18) << "Metric"
              << std::right << std::setw(12) << "TRT"
              << std::setw(14) << "ONNX Runtime"
              << std::setw(10) << "Delta"
              << "  Better backend\n";
    std::cout << "  " << kDash << "\n";
  };

  // ── View 1: Baseline (no GC) ──────────────────────────────────────────────
  if (trt_bl.has_data && ort_bl.has_data) {
    print_header("BASELINE — both pipelines share all GPU SMs");
    print_row("Average",       trt_bl.e2e.avg,     ort_bl.e2e.avg);
    print_row("Median (P50)",  trt_bl.e2e.p50,     ort_bl.e2e.p50);
    print_row("P99",           trt_bl.e2e.p99,     ort_bl.e2e.p99);
    print_row("Std dev",       trt_bl.e2e.std_dev, ort_bl.e2e.std_dev);
    print_row("Min",           trt_bl.e2e.min_val, ort_bl.e2e.min_val);
    print_row("Max",           trt_bl.e2e.max_val, ort_bl.e2e.max_val);
    if (trt_bl.tp_hz > 0.0 && ort_bl.tp_hz > 0.0) {
      // Throughput: print in Hz, positive delta means ORT is higher (better)
      double d = trt_bl.tp_hz != 0.0 ? (ort_bl.tp_hz - trt_bl.tp_hz) / trt_bl.tp_hz * 100.0 : 0.0;
      std::cout << "  " << std::left  << std::setw(18) << "Contend. tp"
                << std::right << std::fixed << std::setprecision(1)
                << std::setw(9) << trt_bl.tp_hz << " Hz  "
                << std::setw(9) << ort_bl.tp_hz << " Hz  "
                << std::showpos << std::setprecision(1) << std::setw(7) << d << "% "
                << std::noshowpos << winner_tp(trt_bl.tp_hz, ort_bl.tp_hz) << "\n";
    }
  }

  // ── View 2: Green context ─────────────────────────────────────────────────
  if (trt_gc.has_data && ort_gc.has_data) {
    print_header("GREEN CONTEXT — each pipeline on its own SM partition");
    print_row("Average",       trt_gc.e2e.avg,     ort_gc.e2e.avg);
    print_row("Median (P50)",  trt_gc.e2e.p50,     ort_gc.e2e.p50);
    print_row("P99",           trt_gc.e2e.p99,     ort_gc.e2e.p99);
    print_row("Std dev",       trt_gc.e2e.std_dev, ort_gc.e2e.std_dev);
    print_row("Min",           trt_gc.e2e.min_val, ort_gc.e2e.min_val);
    print_row("Max",           trt_gc.e2e.max_val, ort_gc.e2e.max_val);
  }

  // ── View 3: GC improvement delta per backend ──────────────────────────────
  if (trt_bl.has_data && trt_gc.has_data && ort_bl.has_data && ort_gc.has_data) {
    std::cout << "\n  -- GREEN CONTEXT IMPROVEMENT: which backend benefits more? --\n";
    std::cout << "  (negative = lower with GC = GC helped that backend)\n";
    std::cout << "  " << std::left << std::setw(18) << "Metric"
              << std::right << std::setw(14) << "TRT GC delta"
              << std::setw(17) << "ONNX RT GC delta"
              << "  More GC benefit\n";
    std::cout << "  " << kDash << "\n";

    auto gc_row = [&](const std::string& metric, double trt_b, double trt_g,
                                                  double ort_b, double ort_g) {
      double trt_d = dpct(trt_b, trt_g);
      double ort_d = dpct(ort_b, ort_g);
      // More negative delta = more improvement from GC
      const char* winner = "similar";
      double diff = trt_d - ort_d;  // negative means TRT improved more
      if      (diff <= -5.0) winner = "TRT benefits more";
      else if (diff >=  5.0) winner = "ONNX RT benefits more";
      std::cout << "  " << std::left << std::setw(18) << metric
                << std::right << std::fixed
                << std::showpos << std::setprecision(1)
                << std::setw(10) << trt_d << "%" << "      "
                << std::setw(10) << ort_d << "%" << "  "
                << std::noshowpos << winner << "\n";
    };

    gc_row("Average",      trt_bl.e2e.avg,     trt_gc.e2e.avg,
                           ort_bl.e2e.avg,     ort_gc.e2e.avg);
    gc_row("P99",          trt_bl.e2e.p99,     trt_gc.e2e.p99,
                           ort_bl.e2e.p99,     ort_gc.e2e.p99);
    gc_row("Std dev",      trt_bl.e2e.std_dev, trt_gc.e2e.std_dev,
                           ort_bl.e2e.std_dev, ort_gc.e2e.std_dev);
    std::cout << std::noshowpos;
  }

  // ── Explanation ───────────────────────────────────────────────────────────
  std::cout << "\n" << kBar << "\n"
            << "  WHY EACH BACKEND BEHAVES THE WAY IT DOES\n"
            << kBar << "\n\n";

  // Determine observed relationships for tailored explanation
  bool trt_faster_bl  = trt_bl.has_data && ort_bl.has_data &&
                         trt_bl.e2e.avg < ort_bl.e2e.avg * 0.97;
  bool ort_faster_bl  = trt_bl.has_data && ort_bl.has_data &&
                         ort_bl.e2e.avg < trt_bl.e2e.avg * 0.97;
  bool trt_faster_gc  = trt_gc.has_data && ort_gc.has_data &&
                         trt_gc.e2e.avg < ort_gc.e2e.avg * 0.97;
  bool trt_more_gc    = (trt_bl.has_data && trt_gc.has_data && ort_bl.has_data && ort_gc.has_data) &&
                         dpct(trt_bl.e2e.avg, trt_gc.e2e.avg) < dpct(ort_bl.e2e.avg, ort_gc.e2e.avg) - 5.0;
  bool ort_more_gc    = (trt_bl.has_data && trt_gc.has_data && ort_bl.has_data && ort_gc.has_data) &&
                         dpct(ort_bl.e2e.avg, ort_gc.e2e.avg) < dpct(trt_bl.e2e.avg, trt_gc.e2e.avg) - 5.0;
  bool trt_lower_jitter_bl = trt_bl.has_data && ort_bl.has_data &&
                              trt_bl.e2e.std_dev < ort_bl.e2e.std_dev * 0.90;
  bool ort_lower_jitter_bl = trt_bl.has_data && ort_bl.has_data &&
                              ort_bl.e2e.std_dev < trt_bl.e2e.std_dev * 0.90;

  std::cout
    << "  TensorRT (TRT)\n"
    << "  ──────────────\n"
    << "  TRT compiles specialised, fused CUDA kernels from the ONNX model graph.\n"
    << "  For fully-connected (MatMul + ReLU) networks like the ones in this\n"
    << "  benchmark, TRT fuses the matrix multiply and activation into a single\n"
    << "  kernel, eliminating intermediate memory round-trips between operations.\n"
    << "  This typically produces LOWER AVERAGE LATENCY than ONNX Runtime.\n"
    << "\n"
    << "  Because TRT kernels are tuned to maximise SM occupancy for the specific\n"
    << "  GPU, they saturate more SMs simultaneously than generic cuBLAS kernels.\n"
    << "  This makes SM-level interference from the contending pipeline more severe\n"
    << "  in the baseline — and therefore makes green context SM partitioning MORE\n"
    << "  IMPACTFUL for TRT workloads: there is more contention to isolate.\n"
    << "\n"
    << "  However, when a green context restricts the SM count below the engine's\n"
    << "  original compile-time assumption, TRT recompiles the engine for the\n"
    << "  smaller SM budget.  If the measured partition is too small, the\n"
    << "  recompiled plan can be SLOWER than the baseline — so TRT latency can\n"
    << "  increase with green context if --measured-sms is set too low.\n";

  if (trt_faster_bl)
    std::cout << "\n  [Observed] TRT was faster in the baseline, consistent with fused kernels.\n";
  else if (ort_faster_bl)
    std::cout << "\n  [Observed] ONNX Runtime was faster in the baseline.  This can happen\n"
              << "  when the batch size (1) is too small for TRT fusion to pay off, or when\n"
              << "  TRT kernel selection overhead amortises poorly at this model size.\n";
  if (trt_more_gc)
    std::cout << "  [Observed] TRT benefited more from green context, consistent with higher\n"
              << "  SM occupancy creating more SM contention to isolate in the baseline.\n";
  std::cout << "\n";

  std::cout
    << "  ONNX Runtime (OnnxRT)\n"
    << "  ─────────────────────\n"
    << "  ONNX Runtime dispatches generic cuBLAS GEMM kernels for MatMul and\n"
    << "  separate cuDNN or custom kernels for ReLU.  These kernels are not\n"
    << "  compiled from the ONNX graph — they are pre-compiled library routines\n"
    << "  that select a GEMM algorithm at runtime based on matrix dimensions.\n"
    << "\n"
    << "  Consequences for this benchmark:\n"
    << "   - HIGHER AVERAGE LATENCY: no op-fusion means extra kernel launches and\n"
    << "     memory round-trips between MatMul output and ReLU input.\n"
    << "   - LOWER SM OCCUPANCY PER CALL: cuBLAS algorithms are tuned for general\n"
    << "     accuracy, not peak occupancy.  A single GEMM for a 1×2048 × 2048×2048\n"
    << "     matrix will not spread across as many SMs as a TRT fused kernel would.\n"
    << "   - LESS SM CONTENTION IN BASELINE: because OnnxRT kernels use fewer SMs\n"
    << "     per call, the measured and contending pipelines interfere less with\n"
    << "     each other even without green context.\n"
    << "   - SMALLER GC BENEFIT: less baseline contention means less room for\n"
    << "     green context SM partitioning to improve things.\n"
    << "   - MORE PREDICTABLE LATENCY: cuBLAS algorithm selection is deterministic\n"
    << "     (no JIT re-compilation), so latency jitter from engine re-optimisation\n"
    << "     is absent.  OnnxRT baseline jitter may be lower than TRT baseline jitter\n"
    << "     for that reason.\n";

  if (ort_lower_jitter_bl)
    std::cout << "\n  [Observed] ONNX Runtime had lower jitter in baseline — consistent with\n"
              << "  deterministic cuBLAS kernel selection and no JIT recompilation.\n";
  else if (trt_lower_jitter_bl)
    std::cout << "\n  [Observed] TRT had lower jitter in baseline.  This can happen when the\n"
              << "  TRT engine is fully compiled and warm, and the contending workload does\n"
              << "  not create significant SM scheduling races at this model size.\n";
  if (ort_more_gc)
    std::cout << "  [Observed] ONNX Runtime benefited more from green context.  This is\n"
              << "  unexpected for a lower-SM-occupancy backend.  Possible reason: CUDA\n"
              << "  stream ordering (not just SM count) was the bottleneck, and the green\n"
              << "  context's dedicated stream helped OnnxRT avoid driver-level serialisation.\n";

  std::cout << "\n"
            << "  Summary for this workload\n"
            << "  ─────────────────────────\n";
  if (trt_faster_gc || trt_faster_bl)
    std::cout << "  For production latency-critical use, prefer TRT when the model is large\n"
              << "  enough for kernel fusion to dominate (hidden >= 2048, layers >= 4).\n";
  std::cout
    << "  Green context provides deterministic SM isolation regardless of backend.\n"
    << "  TRT typically sees a larger absolute latency reduction from GC because\n"
    << "  its higher-occupancy kernels create more SM contention to isolate.\n"
    << "  ONNX Runtime sees a smaller relative benefit but gains predictability\n"
    << "  without the risk of TRT kernel regression from SM-budget recompilation.\n";
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
  // ── Defaults ─────────────────────────────────────────────────────────────
  int         total_samples            = 1000;
  int         warmup_samples           = 100;
  int         repeat                   = 1;
  std::string backend                  = "trt";
  int         frequency_hz             = 1000;
  int         measured_sms             = 0;
  int         contending_sms           = 0;
  std::string mode                     = "all";
  std::string periodic_policy          = "CatchUpMissedTicks";
  bool        pin_measured             = false;
  std::string sched_policy_str         = "SCHED_FIFO";
  std::vector<uint32_t> pin_cores;
  bool        enable_postcheck         = false;
  int         contending_ready_iters   = 50;

  // Measured model: large enough that the kernel actually occupies multiple
  // SMs so green context SM partitioning has a measurable effect.
  int measured_input_size  = 256;
  int measured_hidden_size = 2048;
  int measured_layers      = 4;

  int contending_input_size  = 1024;
  int contending_hidden_size = 4096;
  int contending_layers      = 6;
  int contending_frequency_hz = 0;

  // Default worker threads: half of logical cores, at least 2.
  int worker_threads = std::max(2, static_cast<int>(std::thread::hardware_concurrency()) / 2);

  std::string model_dir_str;

  // ── Argument parsing ─────────────────────────────────────────────────────
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument\n"; std::exit(1);
      }
      return argv[++i];
    };

    if      (arg == "--help"     || arg == "-h")  { print_usage(argv[0]); return 0; }
    else if (arg == "--samples")                  total_samples           = std::atoi(next());
    else if (arg == "--warmup-samples")           warmup_samples          = std::atoi(next());
    else if (arg == "--repeat")                   repeat                  = std::atoi(next());
    else if (arg == "--backend")                  backend                 = next();
    else if (arg == "--frequency-hz")             frequency_hz            = std::atoi(next());
    else if (arg == "--mode")                     mode                    = next();
    else if (arg == "--periodic-policy")          periodic_policy         = next();
    else if (arg == "--pin-measured-pipeline")    pin_measured            = true;
    else if (arg == "--scheduling-policy")        sched_policy_str        = next();
    else if (arg == "--enable-postcheck-fastpath") enable_postcheck       = true;
    else if (arg == "--worker-threads")           worker_threads          = std::atoi(next());
    else if (arg == "--contending-ready-iters")   contending_ready_iters  = std::atoi(next());
    else if (arg == "--measured-input-size")      measured_input_size     = std::atoi(next());
    else if (arg == "--measured-hidden-size")     measured_hidden_size    = std::atoi(next());
    else if (arg == "--measured-layers")          measured_layers         = std::atoi(next());
    else if (arg == "--contending-input-size")    contending_input_size   = std::atoi(next());
    else if (arg == "--contending-hidden-size")   contending_hidden_size  = std::atoi(next());
    else if (arg == "--contending-layers")        contending_layers       = std::atoi(next());
    else if (arg == "--contending-frequency-hz")  contending_frequency_hz = std::atoi(next());
    else if (arg == "--sms-per-partition") {
      measured_sms = contending_sms = std::atoi(next());
    }
    else if (arg == "--measured-sms")             measured_sms   = std::atoi(next());
    else if (arg == "--contending-sms")           contending_sms = std::atoi(next());
    else if (arg == "--model-dir")                model_dir_str  = next();
    else if (arg == "--pin-cores") {
      std::istringstream iss(next());
      std::string token;
      while (std::getline(iss, token, ','))
        pin_cores.push_back(static_cast<uint32_t>(std::atoi(token.c_str())));
    }
    else {
      std::cerr << "Error: unknown argument '" << arg << "'\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  // ── Validation ───────────────────────────────────────────────────────────
  if (mode != "baseline" && mode != "green-context" && mode != "all") {
    std::cerr << "Error: --mode must be baseline | green-context | all\n"; return 1;
  }
  if (backend != "trt" && backend != "onnxrt" && backend != "all") {
    std::cerr << "Error: --backend must be trt | onnxrt | all\n"; return 1;
  }
  if (periodic_policy != "CatchUpMissedTicks" &&
      periodic_policy != "MinTimeBetweenTicks" &&
      periodic_policy != "NoCatchUpMissedTicks") {
    std::cerr << "Error: --periodic-policy must be CatchUpMissedTicks | "
                 "MinTimeBetweenTicks | NoCatchUpMissedTicks\n"; return 1;
  }
  SchedulingPolicy sched_policy{};
  if      (sched_policy_str == "SCHED_FIFO")     sched_policy = SchedulingPolicy::kFirstInFirstOut;
  else if (sched_policy_str == "SCHED_RR")       sched_policy = SchedulingPolicy::kRoundRobin;
  else if (sched_policy_str == "SCHED_DEADLINE") sched_policy = SchedulingPolicy::kDeadline;
  else {
    std::cerr << "Error: --scheduling-policy must be SCHED_FIFO | SCHED_RR | SCHED_DEADLINE\n";
    return 1;
  }
  if (repeat < 1) { std::cerr << "Error: --repeat must be >= 1\n"; return 1; }
  if (worker_threads < 1) { std::cerr << "Error: --worker-threads must be >= 1\n"; return 1; }
  if (contending_ready_iters < 1) {
    std::cerr << "Error: --contending-ready-iters must be >= 1\n"; return 1;
  }
  if (frequency_hz <= 0 || total_samples <= 0 || warmup_samples < 0 ||
      measured_input_size <= 0 || measured_hidden_size <= 0 || measured_layers <= 0 ||
      contending_input_size <= 0 || contending_hidden_size <= 0 || contending_layers <= 0 ||
      contending_frequency_hz < 0) {
    std::cerr << "Error: one or more numeric arguments are invalid (must be > 0)\n"; return 1;
  }
  if (!pin_cores.empty() && pin_cores.size() != 1 && pin_cores.size() != 3) {
    std::cerr << "Error: --pin-cores expects 1 core (all ops share it) or "
                 "3 cores (one per op)\n"; return 1;
  }
  if (!pin_cores.empty() && !pin_measured) {
    std::cerr << "Error: --pin-cores requires --pin-measured-pipeline\n"; return 1;
  }

  // ── Paths ────────────────────────────────────────────────────────────────
  std::filesystem::path exe_dir;
  try { exe_dir = std::filesystem::canonical(argv[0]).parent_path(); }
  catch (...) { exe_dir = std::filesystem::absolute(argv[0]).parent_path(); }

  auto config_path = exe_dir / "inference_scheduling_benchmark.yaml";
  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Error: config not found: " << config_path << "\n"; return 1;
  }

  auto gen_script = exe_dir / "generate_onnx_model.py";
  if (!std::filesystem::exists(gen_script)) {
    std::cerr << "Error: model-generation script not found: " << gen_script << "\n"; return 1;
  }

  std::filesystem::path model_dir = model_dir_str.empty()
      ? exe_dir : std::filesystem::path(model_dir_str);
  std::filesystem::create_directories(model_dir);

  auto measured_model_path = std::filesystem::absolute(
      model_dir / ("measured_i"    + std::to_string(measured_input_size)    +
                   "_h"            + std::to_string(measured_hidden_size)   +
                   "_l"            + std::to_string(measured_layers)        + ".onnx")).string();
  auto contending_model_path = std::filesystem::absolute(
      model_dir / ("contending_i"  + std::to_string(contending_input_size)  +
                   "_h"            + std::to_string(contending_hidden_size) +
                   "_l"            + std::to_string(contending_layers)      + ".onnx")).string();

  // ── GPU properties (queried early so they appear in the Step 2 config block)
  cudaDeviceProp gpu_prop{};
  int total_sms = 0;
  if (cudaGetDeviceProperties(&gpu_prop, 0) == cudaSuccess) {
    total_sms = gpu_prop.multiProcessorCount;
  } else {
    std::strncpy(gpu_prop.name, "unknown", sizeof(gpu_prop.name) - 1);
  }

  // ── Model generation ─────────────────────────────────────────────────────
  std::cout << "\n" << kBar << "\n"
            << "  Step 1 of 3 — Generating / locating synthetic ONNX models\n"
            << kBar << "\n";
  if (!ensure_model_exists(gen_script, measured_model_path,
                            measured_input_size, measured_hidden_size, measured_layers,
                            "Measured")) return 1;
  if (!ensure_model_exists(gen_script, contending_model_path,
                            contending_input_size, contending_hidden_size, contending_layers,
                            "Contending")) return 1;

  // ── Print configuration ───────────────────────────────────────────────────
  std::cout << "\n" << kBar << "\n"
            << "  Step 2 of 3 — Benchmark configuration\n"
            << kBar << "\n";

  auto yn = [](bool b) { return b ? "yes" : "no"; };

  std::cout << "  GPU device                 : " << gpu_prop.name
            << "  (" << total_sms << " SMs)\n"
            << "  Backend                    : " << backend << "\n"
            << "  Mode                       : " << mode << "\n"
            << "  Repeat runs per condition  : " << repeat << "  "
            << "(samples pooled: " << (total_samples * repeat) << " per condition)\n"
            << "  EBS worker threads         : " << worker_threads << "\n"
            << "\n"
            << "  Measured pipeline          : " << measured_input_size << "-dim FC, "
            << measured_layers << " layers, hidden=" << measured_hidden_size
            << "  @" << frequency_hz << " Hz\n"
            << "    Model file               : " << measured_model_path << "\n"
            << "  Contending pipeline        : " << contending_input_size << "-dim FC, "
            << contending_layers << " layers, hidden=" << contending_hidden_size
            << (contending_frequency_hz > 0
                    ? "  @" + std::to_string(contending_frequency_hz) + " Hz"
                    : "  (free-running)") << "\n"
            << "    Model file               : " << contending_model_path << "\n"
            << "\n"
            << "  Measurement samples        : " << total_samples
            << " per run  (" << warmup_samples << " warmup discarded)\n"
            << "  Contending ready threshold : " << contending_ready_iters
            << " inferences before measurement gate opens\n"
            << "  Periodic policy            : " << periodic_policy << "\n"
            << "  Pin measured pipeline      : " << yn(pin_measured) << "\n";
  if (pin_measured) {
    std::cout << "  Scheduling policy          : " << sched_policy_str << "\n";
    if (!pin_cores.empty()) {
      std::cout << "  Pinned CPU cores           : ";
      for (size_t k = 0; k < pin_cores.size(); k++) {
        if (k) std::cout << ", "; std::cout << pin_cores[k];
      }
      std::cout << "\n";
    }
  }
  std::cout << "  Postcheck fast path        : " << yn(enable_postcheck) << "\n";
  if (mode != "baseline") {
    std::cout << "  Green Context — measured SMs   : "
              << (measured_sms   > 0 ? std::to_string(measured_sms)   : "auto (half of GPU)") << "\n"
              << "  Green Context — contending SMs : "
              << (contending_sms > 0 ? std::to_string(contending_sms) : "auto (half of GPU)") << "\n";
  }
  std::cout << "\n";

  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(0), "Failed to select CUDA device 0");

  // ── Determine backends to run ────────────────────────────────────────────
  std::vector<std::string> backends_to_run;
  if (backend == "all") {
    backends_to_run = {"trt", "onnxrt"};
  } else {
    backends_to_run = {backend};
  }

  // ── Per-backend accumulator ───────────────────────────────────────────────
  // Keyed by backend string; each holds raw samples for baseline and GC runs.
  struct BackendAccum {
    std::vector<double> bl_e2e, bl_period;
    std::vector<double> gc_e2e, gc_period;
    double bl_tp_sum = 0.0, gc_tp_sum = 0.0;
    int    bl_runs   = 0,    gc_runs   = 0;
  };
  std::map<std::string, BackendAccum> accum;
  for (const auto& b : backends_to_run) accum[b] = {};

  // ── run_app_once: create and execute one App instance ────────────────────
  struct RunResult {
    std::vector<double> e2e_us, period_us;
    double contending_throughput_hz = 0.0;
  };
  auto run_app_once = [&](bool use_gc, const std::string& run_label,
                           const std::string& b) -> RunResult {
    reset_global_benchmark_state();
    auto app = std::make_unique<InferenceSchedulingBenchmarkApp>(
        use_gc, total_samples, warmup_samples,
        measured_model_path,   measured_input_size,
        contending_model_path, contending_input_size,
        b, measured_sms, contending_sms,
        frequency_hz, contending_frequency_hz, periodic_policy,
        pin_measured, sched_policy, pin_cores,
        contending_ready_iters, run_label);
    app->config(config_path);

    auto sched = app->make_scheduler<holoscan::EventBasedScheduler>(
        "scheduler",
        holoscan::Arg("worker_thread_number", static_cast<int64_t>(worker_threads)));
    if (enable_postcheck)
      sched->add_arg(holoscan::Arg("enable_worker_postcheck_fastpath", true));
    app->scheduler(sched);
    app->run();

    return {app->get_e2e_samples(),
            app->get_tick_period_samples(),
            app->get_contending_throughput_hz()};
  };

  // ── Step 3: run all conditions for all backends ───────────────────────────
  {
    std::string header_suffix;
    if (backend == "all") header_suffix = " (trt and onnxrt interleaved)";
    else                  header_suffix = " (baseline and green-context interleaved)";

    std::cout << "\n" << kBar << "\n"
              << "  Step 3 of 3 — Running benchmark" << header_suffix << "\n"
              << kBar << "\n";
  }

  for (int r = 1; r <= repeat; r++) {
    for (const auto& b : backends_to_run) {
      auto& bd = accum[b];

      if (mode == "baseline" || mode == "all") {
        std::string label = "Baseline run " + std::to_string(r) + "/" +
                            std::to_string(repeat) + " [" + b + "]";
        std::cout << "\n  >> " << label << "\n";
        auto res = run_app_once(false, label, b);
        bd.bl_e2e.insert(bd.bl_e2e.end(), res.e2e_us.begin(), res.e2e_us.end());
        bd.bl_period.insert(bd.bl_period.end(), res.period_us.begin(), res.period_us.end());
        bd.bl_tp_sum += res.contending_throughput_hz;
        ++bd.bl_runs;
        std::cout << "     Collected " << res.e2e_us.size() << " samples. "
                  << "Contending throughput: "
                  << std::fixed << std::setprecision(1) << res.contending_throughput_hz << " Hz\n";
      }

      if (mode == "green-context" || mode == "all") {
        std::string label = "Green Context run " + std::to_string(r) + "/" +
                            std::to_string(repeat) + " [" + b + "]";
        std::cout << "\n  >> " << label << "\n";
        auto res = run_app_once(true, label, b);
        bd.gc_e2e.insert(bd.gc_e2e.end(), res.e2e_us.begin(), res.e2e_us.end());
        bd.gc_period.insert(bd.gc_period.end(), res.period_us.begin(), res.period_us.end());
        bd.gc_tp_sum += res.contending_throughput_hz;
        ++bd.gc_runs;
        std::cout << "     Collected " << res.e2e_us.size() << " samples. "
                  << "Contending throughput: "
                  << std::fixed << std::setprecision(1) << res.contending_throughput_hz << " Hz\n";
      }
    }
  }

  // ── Results — one block per backend ──────────────────────────────────────

  // Helper: print a per-backend latency / tick / throughput section.
  auto print_backend_results = [&](const std::string& b, const BackendAccum& bd) {
    BenchmarkStats bl_e2e    = calculate_stats(bd.bl_e2e);
    BenchmarkStats gc_e2e    = calculate_stats(bd.gc_e2e);
    BenchmarkStats bl_period = calculate_stats(bd.bl_period);
    BenchmarkStats gc_period = calculate_stats(bd.gc_period);

    auto print_section = [&](const std::string& title, const std::string& subtitle,
                              const BenchmarkStats& bl_s, const BenchmarkStats& gc_s) {
      std::cout << "\n" << kBar << "\n  " << title << "\n";
      if (!subtitle.empty()) std::cout << "  " << subtitle << "\n";
      std::cout << kBar << "\n";
      if (mode == "all")           print_comparison_table(bl_s, gc_s);
      else if (mode == "baseline") print_stats_block(bl_s, "Baseline");
      else                         print_stats_block(gc_s, "Green Context");
    };

    // End-to-end latency
    std::ostringstream e2e_sub;
    e2e_sub << "Measured from just before TX emit to after GPU stream sync in RX";
    if (repeat > 1)
      e2e_sub << "  |  " << repeat << " runs pooled ("
              << (bd.bl_runs > 0 ? bd.bl_e2e.size() : 0) << " baseline + "
              << (bd.gc_runs > 0 ? bd.gc_e2e.size() : 0) << " GC samples)";

    print_section("END-TO-END INFERENCE LATENCY  [backend: " + b + "]",
                  e2e_sub.str(), bl_e2e, gc_e2e);

    if (mode == "all" && bl_e2e.sample_count > 0 && gc_e2e.sample_count > 0) {
      auto pct = [](double a, double bv) {
        return a != 0.0 ? (bv - a) / a * 100.0 : 0.0;
      };
      std::cout << "\n  Quick summary — GC vs. baseline [" << b << "]:\n"
                << std::fixed << std::setprecision(1);
      auto qrow = [&](const std::string& name, double bl_v, double gc_v) {
        double dp = pct(bl_v, gc_v);
        std::cout << "    " << std::left << std::setw(20) << name
                  << std::right << std::showpos << std::setw(7) << dp << "%" << std::noshowpos;
        if      (dp <= -10.0) std::cout << "  (reduced — GC is isolating SMs effectively)\n";
        else if (dp >=  10.0) std::cout << "  (increased — check SM partition size)\n";
        else                  std::cout << "  (no significant change)\n";
      };
      qrow("Average latency",  bl_e2e.avg,     gc_e2e.avg);
      qrow("P99 tail latency", bl_e2e.p99,     gc_e2e.p99);
      qrow("Jitter (std dev)", bl_e2e.std_dev, gc_e2e.std_dev);
    }

    // Scheduler tick accuracy
    print_section(
        "SCHEDULER TICK ACCURACY  [backend: " + b + "]",
        "Inter-fire interval of PeriodicTxOp  (nominal = " +
            std::to_string(1'000'000 / frequency_hz) + " " + kUs + ")" +
            "  — lower std dev = more consistent firing",
        bl_period, gc_period);

    // Contending throughput
    double bl_tp = bd.bl_runs > 0 ? bd.bl_tp_sum / bd.bl_runs : 0.0;
    double gc_tp = bd.gc_runs > 0 ? bd.gc_tp_sum / bd.gc_runs : 0.0;
    std::cout << "\n" << kBar << "\n"
              << "  CONTENDING PIPELINE THROUGHPUT  [backend: " << b << "]\n"
              << "  (lower with GC = SM partition restricted it — confirms isolation is active)\n"
              << kBar << "\n"
              << std::fixed << std::setprecision(1);
    if ((mode == "baseline" || mode == "all") && bd.bl_runs > 0)
      std::cout << "  Baseline      (avg over " << bd.bl_runs << " run"
                << (bd.bl_runs > 1 ? "s" : "") << "):  " << bl_tp << " Hz\n";
    if ((mode == "green-context" || mode == "all") && bd.gc_runs > 0) {
      std::cout << "  Green Context (avg over " << bd.gc_runs << " run"
                << (bd.gc_runs > 1 ? "s" : "") << "):  " << gc_tp << " Hz\n";
      if (mode == "all" && bd.bl_runs > 0) {
        double dp = bl_tp != 0.0 ? (gc_tp - bl_tp) / bl_tp * 100.0 : 0.0;
        std::cout << "  Change: " << std::showpos << std::setprecision(1) << dp << "%  "
                  << std::noshowpos;
        if      (dp <= -5.0) std::cout << "(lower — SM partition is restricting contending pipeline)\n";
        else if (dp >=  5.0) std::cout << "(higher — contending partition may be larger than expected)\n";
        else                 std::cout << "(similar — contending model likely memory-bandwidth-bound)\n";
      }
    }

    // Diagnostic analysis
    if (mode == "all" && bl_e2e.sample_count > 0 && gc_e2e.sample_count > 0) {
      print_diagnosis(bl_e2e, gc_e2e, bl_period, gc_period, bl_tp, gc_tp,
                      b, total_sms, measured_sms, contending_sms,
                      measured_input_size, measured_hidden_size, measured_layers,
                      contending_input_size, contending_hidden_size, contending_layers,
                      frequency_hz);
    }
  };

  for (const auto& b : backends_to_run) {
    print_backend_results(b, accum[b]);
  }

  // ── Backend comparison (only when --backend all) ──────────────────────────
  if (backend == "all" && accum.count("trt") && accum.count("onnxrt")) {
    auto make_bs = [&](const std::string& b, bool use_gc) -> BackendStats {
      const auto& bd = accum[b];
      BackendStats s;
      s.backend  = b;
      s.e2e      = use_gc ? calculate_stats(bd.gc_e2e)    : calculate_stats(bd.bl_e2e);
      s.period   = use_gc ? calculate_stats(bd.gc_period)  : calculate_stats(bd.bl_period);
      s.tp_hz    = use_gc ? (bd.gc_runs > 0 ? bd.gc_tp_sum / bd.gc_runs : 0.0)
                           : (bd.bl_runs > 0 ? bd.bl_tp_sum / bd.bl_runs : 0.0);
      s.has_data = s.e2e.sample_count > 0;
      return s;
    };
    print_backend_comparison(
        make_bs("trt",    false), make_bs("trt",    true),
        make_bs("onnxrt", false), make_bs("onnxrt", true),
        total_sms, measured_sms, contending_sms,
        measured_input_size, measured_hidden_size, measured_layers,
        contending_input_size, contending_hidden_size, contending_layers);
  }

  return 0;
}
