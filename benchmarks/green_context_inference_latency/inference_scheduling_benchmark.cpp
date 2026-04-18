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
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <deque>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <gxf/std/tensor.hpp>

using namespace holoscan;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

struct BenchmarkStats {
  double avg = 0.0;
  double std_dev = 0.0;
  double min_val = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max_val = 0.0;
  size_t sample_count = 0;
};

double calculate_percentile(const std::vector<double>& sorted_data, double percentile) {
  if (sorted_data.empty()) return 0.0;
  double index = (percentile / 100.0) * (sorted_data.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(index));
  size_t upper = static_cast<size_t>(std::ceil(index));
  if (lower == upper) return sorted_data[lower];
  double weight = index - lower;
  return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
}

BenchmarkStats calculate_stats(const std::vector<double>& raw_values) {
  BenchmarkStats stats;
  if (raw_values.empty()) return stats;

  std::vector<double> sorted = raw_values;
  std::sort(sorted.begin(), sorted.end());
  stats.sample_count = sorted.size();
  stats.avg = std::accumulate(sorted.begin(), sorted.end(), 0.0) / stats.sample_count;

  double sum_sq = 0.0;
  for (double v : sorted) { double d = v - stats.avg; sum_sq += d * d; }
  stats.std_dev = stats.sample_count > 1 ? std::sqrt(sum_sq / (stats.sample_count - 1)) : 0.0;

  stats.min_val = sorted.front();
  stats.max_val = sorted.back();
  stats.p50 = calculate_percentile(sorted, 50.0);
  stats.p95 = calculate_percentile(sorted, 95.0);
  stats.p99 = calculate_percentile(sorted, 99.0);
  return stats;
}

// ---------------------------------------------------------------------------
// Global state shared between measured-pipeline operators
// ---------------------------------------------------------------------------

static std::atomic<bool> g_contending_pipeline_ready{false};
static std::mutex g_tx_timestamps_mutex;

struct TimestampedSample {
  int64_t emit_ns;
  uint64_t seq;
};
static std::deque<TimestampedSample> g_tx_samples;
static std::atomic<uint64_t> g_tx_seq_counter{0};

void reset_global_benchmark_state() {
  g_contending_pipeline_ready.store(false, std::memory_order_release);
  g_tx_seq_counter.store(0, std::memory_order_release);
  std::lock_guard<std::mutex> lock(g_tx_timestamps_mutex);
  g_tx_samples.clear();
}

// ---------------------------------------------------------------------------
// TensorSpec -- describes a named tensor with shape
// ---------------------------------------------------------------------------

struct TensorSpec {
  std::string name;
  std::vector<int32_t> shape;
  int total_elements() const {
    int n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

// ---------------------------------------------------------------------------
// PeriodicTxOp -- emits pre-allocated GPU tensors each tick
// ---------------------------------------------------------------------------

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

      float* d_ptr = nullptr;
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&d_ptr, n * sizeof(float)),
                                     "cudaMalloc failed for " + ts.name);
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaMemcpy(d_ptr, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy failed for " + ts.name);
      gpu_buffers_.push_back(d_ptr);
    }
    HOLOSCAN_LOG_INFO("[{}] Initialized {} tensors", name(), tensor_specs_.size());
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override {
    if (record_timestamps_) {
      auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      uint64_t seq = g_tx_seq_counter.fetch_add(1, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lock(g_tx_timestamps_mutex);
        g_tx_samples.push_back({now_ns, seq});
      }
    }

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
          nvidia::gxf::MemoryStorageType::kDevice, gpu_buffers_[i],
          [](void*) { return nvidia::gxf::Success; });
      if (!result) throw std::runtime_error("Failed to wrap memory: " + ts.name);
    }

    auto holoscan_entity = holoscan::gxf::Entity(std::move(entity));
    op_output.emit(holoscan_entity, "output");
  }

  ~PeriodicTxOp() {
    for (auto* p : gpu_buffers_) {
      if (p) cudaFree(p);
    }
  }

 private:
  std::vector<TensorSpec> tensor_specs_;
  std::vector<float*> gpu_buffers_;
  bool record_timestamps_ = true;
};

// ---------------------------------------------------------------------------
// TimingRxOp -- measures end-to-end latency of the measured pipeline
// ---------------------------------------------------------------------------

class TimingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimingRxOp)
  TimingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(total_samples_, "total_samples", "Total Samples",
               "Number of samples to collect", 1000);
    spec.param(warmup_samples_, "warmup_samples", "Warmup Samples",
               "Samples to discard before measurement", 100);
    spec.input<std::any>("in");
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    (void)op_input.receive<std::any>("in");

    // Always sync the inference stream immediately after receive, even during
    // warmup. This ensures GPU work is complete before the next cycle.
    auto streams = op_input.receive_cuda_streams("in");
    if (!streams.empty() && streams[0].has_value()) {
      cudaStreamSynchronize(streams[0].value());
    }

    auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    int64_t emit_ns = 0;
    {
      std::lock_guard<std::mutex> lock(g_tx_timestamps_mutex);
      if (!g_tx_samples.empty()) {
        auto sample = g_tx_samples.front();
        g_tx_samples.pop_front();
        emit_ns = sample.emit_ns;
        if (sample.seq != expected_seq_) {
          HOLOSCAN_LOG_ERROR("[TimingRxOp] Sequence mismatch: expected {} got {}. "
                             "Pipeline message ordering violated.",
                             expected_seq_, sample.seq);
          throw std::runtime_error("Pipeline message ordering violated");
        }
        expected_seq_++;
      }
    }
    if (emit_ns <= 0) {
      HOLOSCAN_LOG_WARN("[TimingRxOp] Missing emit timestamp; dropping sample");
      return;
    }

    if (!g_contending_pipeline_ready.load(std::memory_order_acquire)) {
      warmup_pre_ready_count_++;
      if (warmup_pre_ready_count_ == 1 || warmup_pre_ready_count_ % 200 == 0) {
        HOLOSCAN_LOG_INFO("[TimingRxOp] Waiting for contending pipeline ({} ticks)",
                          warmup_pre_ready_count_);
      }
      return;
    }

    if (warmup_post_ready_count_ < warmup_samples_.get()) {
      warmup_post_ready_count_++;
      if (warmup_post_ready_count_ == warmup_samples_.get()) {
        HOLOSCAN_LOG_INFO("[TimingRxOp] Warmup done ({} pre + {} post). Measuring...",
                          warmup_pre_ready_count_, warmup_post_ready_count_);
      }
      return;
    }

    double e2e_us = static_cast<double>(end_ns - emit_ns) / 1000.0;
    e2e_us_.push_back(e2e_us);

    if (prev_emit_ns_ > 0) {
      double tx_period_us = static_cast<double>(emit_ns - prev_emit_ns_) / 1000.0;
      tx_period_us_.push_back(tx_period_us);
    }
    prev_emit_ns_ = emit_ns;

    sample_count_++;
    int log_interval = std::max(1, total_samples_.get() / 10);
    if (sample_count_ % log_interval == 0 || sample_count_ == 1) {
      HOLOSCAN_LOG_INFO("[TimingRxOp] Collected {}/{} samples",
                        sample_count_, total_samples_.get());
    }

    if (sample_count_ >= total_samples_.get()) {
      HOLOSCAN_LOG_INFO("[TimingRxOp] Sequence validation passed: {} samples, "
                        "all in order (seq 0..{}).",
                        sample_count_, expected_seq_ - 1);
      fragment()->stop_execution();
    }
  }

  BenchmarkStats get_e2e_stats() const { return calculate_stats(e2e_us_); }
  BenchmarkStats get_tx_period_stats() const { return calculate_stats(tx_period_us_); }

 private:
  Parameter<int> total_samples_;
  Parameter<int> warmup_samples_;
  int sample_count_ = 0;
  int warmup_pre_ready_count_ = 0;
  int warmup_post_ready_count_ = 0;
  int64_t prev_emit_ns_ = 0;
  uint64_t expected_seq_ = 0;
  std::vector<double> e2e_us_;
  std::vector<double> tx_period_us_;
};

// ---------------------------------------------------------------------------
// ContendingSinkOp -- consumes contending inference output, gates readiness
// ---------------------------------------------------------------------------

class ContendingSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ContendingSinkOp)
  ContendingSinkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(ready_after_iters_, "ready_after_iters", "Ready After Iterations",
               "Mark contending pipeline ready after N completed inferences", 8);
    spec.input<std::any>("in");
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    (void)op_input.receive<std::any>("in");

    if (completed_iters_ == 0) {
      first_iter_time_ = std::chrono::steady_clock::now();
    }
    completed_iters_++;
    last_iter_time_ = std::chrono::steady_clock::now();

    if (!g_contending_pipeline_ready.load(std::memory_order_acquire) &&
        completed_iters_ >= std::max(1, ready_after_iters_.get())) {
      g_contending_pipeline_ready.store(true, std::memory_order_release);
      HOLOSCAN_LOG_INFO(
          "[ContendingSinkOp] Contending inference pipeline ready ({} iterations complete).",
          completed_iters_);
    }
  }

  int get_completed_iters() const { return completed_iters_; }

  double get_throughput_hz() const {
    if (completed_iters_ <= 1) return 0.0;
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        last_iter_time_ - first_iter_time_).count();
    if (duration_us <= 0) return 0.0;
    return static_cast<double>(completed_iters_ - 1) * 1e6 / duration_us;
  }

 private:
  Parameter<int> ready_after_iters_;
  int completed_iters_ = 0;
  std::chrono::steady_clock::time_point first_iter_time_{};
  std::chrono::steady_clock::time_point last_iter_time_{};
};

// ---------------------------------------------------------------------------
// InferenceSchedulingBenchmarkApp
// ---------------------------------------------------------------------------

class InferenceSchedulingBenchmarkApp : public holoscan::Application {
 public:
  InferenceSchedulingBenchmarkApp(bool use_gc, int total_samples, int warmup_samples,
                                  const std::string& measured_model_path,
                                  int measured_input_size,
                                  const std::string& contending_model_path,
                                  int contending_input_size,
                                  const std::string& backend,
                                  int measured_sms, int contending_sms,
                                  int measured_frequency_hz,
                                  int contending_frequency_hz,
                                  const std::string& periodic_policy,
                                  bool pin_measured, SchedulingPolicy sched_policy,
                                  const std::vector<uint32_t>& pin_cores)
      : use_gc_(use_gc), total_samples_(total_samples), warmup_samples_(warmup_samples),
        measured_model_path_(measured_model_path),
        measured_input_size_(measured_input_size),
        contending_model_path_(contending_model_path),
        contending_input_size_(contending_input_size),
        backend_(backend),
        measured_sms_(measured_sms), contending_sms_(contending_sms),
        measured_frequency_hz_(measured_frequency_hz),
        contending_frequency_hz_(contending_frequency_hz),
        periodic_policy_(periodic_policy),
        pin_measured_(pin_measured), sched_policy_(sched_policy),
        pin_cores_(pin_cores) {}

  void compose() override {
    std::shared_ptr<CudaStreamPool> measured_stream_pool;
    std::shared_ptr<CudaStreamPool> contending_stream_pool;
    std::shared_ptr<CudaGreenContextPool> gc_pool;
    std::shared_ptr<CudaGreenContext> measured_gc;
    std::shared_ptr<CudaGreenContext> contending_gc;

    if (use_gc_) {
      cudaDeviceProp prop;
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDeviceProperties(&prop, 0),
                                     "Failed to get device properties");
      int total_sms = prop.multiProcessorCount;
      int rounded_total_sms = total_sms & ~3;
      if (rounded_total_sms < 4) {
        throw std::runtime_error(
            "CUDA Green Context requires at least 4 SMs after alignment");
      }

      int m_sms = measured_sms_ > 0 ? std::max(4, measured_sms_ & ~3)
                                    : std::max(4, (rounded_total_sms / 2) & ~3);
      int c_sms = contending_sms_ > 0 ? std::max(4, contending_sms_ & ~3)
                                      : std::max(4, (rounded_total_sms / 2) & ~3);
      m_sms = std::min(m_sms, rounded_total_sms);
      c_sms = std::min(c_sms, rounded_total_sms);

      if (m_sms + c_sms > rounded_total_sms) {
        HOLOSCAN_LOG_WARN("GC partitions overlap: measured ({}) + contending ({}) = {} > {} total SMs. "
                          "Overlapping partitions reduce isolation.",
                          m_sms, c_sms, m_sms + c_sms, rounded_total_sms);
      }
      HOLOSCAN_LOG_INFO("GC: measured={} SMs, contending={} SMs ({} total)",
                        m_sms, c_sms, rounded_total_sms);

      std::vector<uint32_t> partitions = {static_cast<uint32_t>(m_sms),
                                          static_cast<uint32_t>(c_sms)};
      gc_pool = make_resource<CudaGreenContextPool>(
          "gc_pool", Arg("dev_id", 0),
          Arg("num_partitions", static_cast<uint32_t>(2)),
          Arg("sms_per_partition", partitions));

      measured_gc = make_resource<CudaGreenContext>(
          "measured_gc", Arg("cuda_green_context_pool", gc_pool),
          Arg("index", static_cast<int32_t>(0)));
      contending_gc = make_resource<CudaGreenContext>(
          "contending_gc", Arg("cuda_green_context_pool", gc_pool),
          Arg("index", static_cast<int32_t>(1)));

      measured_stream_pool = make_resource<CudaStreamPool>(
          "measured_stream_pool", 0, 0, 0, 1, 5, measured_gc);
      contending_stream_pool = make_resource<CudaStreamPool>(
          "contending_stream_pool", 0, 0, 0, 1, 5, contending_gc);
    } else {
      measured_stream_pool = make_resource<CudaStreamPool>("measured_stream_pool", 0, 0, 0, 1, 5);
      contending_stream_pool = make_resource<CudaStreamPool>("contending_stream_pool", 0, 0, 0, 1, 5);
    }

    // --- Measured pipeline: PeriodicTxOp → InferenceOp → TimingRxOp ---

    std::vector<TensorSpec> measured_inputs = {{"input", {1, measured_input_size_}}};

    auto measured_tx = make_operator<PeriodicTxOp>(
        "measured_tx",
        make_condition<PeriodicCondition>("measured_periodic",
            Arg("recess_period") = std::to_string(measured_frequency_hz_) + "hz",
            Arg("policy") = periodic_policy_));
    measured_tx->set_tensor_specs(measured_inputs);
    measured_tx->set_record_timestamps(true);

    ops::InferenceOp::DataMap m_model_map;
    m_model_map.insert("measured_model", measured_model_path_);
    ops::InferenceOp::DataVecMap m_pre_map;
    m_pre_map.insert("measured_model", {"input"});
    ops::InferenceOp::DataVecMap m_inf_map;
    m_inf_map.insert("measured_model", {"output"});

    auto measured_alloc = make_resource<UnboundedAllocator>("measured_alloc");
    measured_inference_op_ = make_operator<ops::InferenceOp>(
        "measured_inference",
        from_config("measured_inference"),
        Arg("backend", backend_),
        Arg("model_path_map", m_model_map),
        Arg("pre_processor_map", m_pre_map),
        Arg("inference_map", m_inf_map),
        Arg("allocator") = measured_alloc,
        Arg("cuda_stream_pool") = measured_stream_pool);

    timing_rx_ = make_operator<TimingRxOp>(
        "timing_rx",
        Arg("total_samples", total_samples_),
        Arg("warmup_samples", warmup_samples_));

    add_flow(measured_tx, measured_inference_op_, {{"output", "receivers"}});
    add_flow(measured_inference_op_, timing_rx_, {{"transmitter", "in"}});

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
        int64_t period_ns = 1'000'000'000LL / measured_frequency_hz_;
        int64_t deadline_ns = period_ns;
        int64_t runtime_ns = static_cast<int64_t>(period_ns * 0.90);
        measured_pool->add_realtime(measured_tx, sched_policy_, true, cores_for(0), 0,
                                   runtime_ns, deadline_ns, period_ns);
        measured_pool->add_realtime(measured_inference_op_, sched_policy_, true, cores_for(1), 0,
                                   runtime_ns, deadline_ns, period_ns);
        measured_pool->add_realtime(timing_rx_, sched_policy_, true, cores_for(2), 0,
                                   runtime_ns, deadline_ns, period_ns);
      } else {
        measured_pool->add_realtime(measured_tx, sched_policy_, true, cores_for(0), 99);
        measured_pool->add_realtime(measured_inference_op_, sched_policy_, true, cores_for(1), 99);
        measured_pool->add_realtime(timing_rx_, sched_policy_, true, cores_for(2), 99);
      }
    }

    // --- Contending pipeline: PeriodicTxOp → InferenceOp → ContendingSinkOp ---

    std::vector<TensorSpec> contending_inputs = {{"input", {1, contending_input_size_}}};

    std::shared_ptr<PeriodicTxOp> contending_tx;
    if (contending_frequency_hz_ > 0) {
      contending_tx = make_operator<PeriodicTxOp>(
          "contending_tx",
          make_condition<PeriodicCondition>("contending_periodic",
              Arg("recess_period") = std::to_string(contending_frequency_hz_) + "hz",
              Arg("policy") = periodic_policy_));
    } else {
      contending_tx = make_operator<PeriodicTxOp>("contending_tx");
    }
    contending_tx->set_tensor_specs(contending_inputs);
    contending_tx->set_record_timestamps(false);

    ops::InferenceOp::DataMap c_model_map;
    c_model_map.insert("contending_model", contending_model_path_);
    ops::InferenceOp::DataVecMap c_pre_map;
    c_pre_map.insert("contending_model", {"input"});
    ops::InferenceOp::DataVecMap c_inf_map;
    c_inf_map.insert("contending_model", {"output"});

    auto contending_alloc = make_resource<UnboundedAllocator>("contending_alloc");
    contending_inference_op_ = make_operator<ops::InferenceOp>(
        "contending_inference",
        from_config("contending_inference"),
        Arg("backend", backend_),
        Arg("model_path_map", c_model_map),
        Arg("pre_processor_map", c_pre_map),
        Arg("inference_map", c_inf_map),
        Arg("allocator") = contending_alloc,
        Arg("cuda_stream_pool") = contending_stream_pool);

    contending_sink_ = make_operator<ContendingSinkOp>("contending_sink");

    add_flow(contending_tx, contending_inference_op_, {{"output", "receivers"}});
    add_flow(contending_inference_op_, contending_sink_, {{"transmitter", "in"}});

    if (use_gc_) {
      contending_inference_op_->add_arg(contending_gc);
      contending_inference_op_->add_arg(gc_pool);
    }
  }

  BenchmarkStats get_e2e_stats() const { return timing_rx_->get_e2e_stats(); }
  BenchmarkStats get_tx_period_stats() const { return timing_rx_->get_tx_period_stats(); }

  int get_contending_iters() const {
    return contending_sink_->get_completed_iters();
  }

  double get_contending_throughput_hz() const {
    return contending_sink_->get_throughput_hz();
  }

 private:
  bool use_gc_;
  int total_samples_;
  int warmup_samples_;
  std::string measured_model_path_;
  int measured_input_size_;
  std::string contending_model_path_;
  int contending_input_size_;
  std::string backend_;
  int measured_sms_;
  int contending_sms_;
  int measured_frequency_hz_;
  int contending_frequency_hz_;
  std::string periodic_policy_;
  bool pin_measured_;
  SchedulingPolicy sched_policy_;
  std::vector<uint32_t> pin_cores_;
  std::shared_ptr<ops::InferenceOp> measured_inference_op_;
  std::shared_ptr<ops::InferenceOp> contending_inference_op_;
  std::shared_ptr<TimingRxOp> timing_rx_;
  std::shared_ptr<ContendingSinkOp> contending_sink_;
};

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

void print_stats(const BenchmarkStats& stats, const std::string& label) {
  std::cout << "=== " << label << " ===" << std::endl;
  if (stats.sample_count == 0) {
    std::cout << "  No data" << std::endl;
    return;
  }
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  Average: " << stats.avg << " \xce\xbcs" << std::endl;
  std::cout << "  Std Dev: " << stats.std_dev << " \xce\xbcs" << std::endl;
  std::cout << "  Min:     " << stats.min_val << " \xce\xbcs" << std::endl;
  std::cout << "  P50:     " << stats.p50 << " \xce\xbcs" << std::endl;
  std::cout << "  P95:     " << stats.p95 << " \xce\xbcs" << std::endl;
  std::cout << "  P99:     " << stats.p99 << " \xce\xbcs" << std::endl;
  std::cout << "  Max:     " << stats.max_val << " \xce\xbcs" << std::endl;
  std::cout << "  Samples: " << stats.sample_count << std::endl;
}

void print_usage(const char* prog) {
  std::cout << "Green Context Inference Latency Benchmark\n\n"
            << "Measures end-to-end inference pipeline latency and determinism\n"
            << "under contending TRT inference workload, with and without Green Context.\n"
            << "Both measured and contending pipelines use synthetic ONNX models\n"
            << "auto-generated at startup (no external model files required).\n\n"
            << "Usage: " << prog << " [OPTIONS]\n"
            << "Options:\n"
            << "  --samples N                Measurement samples (default: 1000)\n"
            << "  --warmup-samples N         Warmup iterations (default: 100)\n"
            << "  --backend BACKEND          'trt' or 'onnxrt' (default: trt)\n"
            << "  --frequency-hz N           Measured pipeline frequency in Hz (default: 1000)\n"
            << "  --mode MODE                'baseline', 'green-context', or 'all' (default: all)\n"
            << "\n  Measured model (high-frequency control loop):\n"
            << "  --measured-input-size N     Input/output dimension (default: 64)\n"
            << "  --measured-hidden-size N    Hidden layer width (default: 256)\n"
            << "  --measured-layers N         Number of FC layers (default: 3)\n"
            << "\n  Contending model (heavy lower-priority workload):\n"
            << "  --contending-input-size N   Input/output dimension (default: 1024)\n"
            << "  --contending-hidden-size N  Hidden layer width (default: 4096)\n"
            << "  --contending-layers N       Number of FC layers (default: 6)\n"
            << "  --contending-frequency-hz N Contending pipeline Hz; 0=free-running (default: 0)\n"
            << "\n  Scheduling:\n"
            << "  --periodic-policy POLICY    PeriodicCondition policy (default: CatchUpMissedTicks)\n"
            << "                              CatchUpMissedTicks | MinTimeBetweenTicks | NoCatchUpMissedTicks\n"
            << "  --pin-measured-pipeline     Pin measured pipeline ops to a dedicated RT thread pool\n"
            << "  --scheduling-policy POL     RT scheduling: SCHED_FIFO (default), SCHED_RR, SCHED_DEADLINE\n"
            << "                              Only applies when --pin-measured-pipeline is set\n"
            << "  --pin-cores C0[,C1,C2]     Pin RT threads to specific CPU cores (comma-separated).\n"
            << "                              1 core: all ops on same core. 3 cores: one per op.\n"
            << "                              Only applies when --pin-measured-pipeline is set.\n"
            << "  --enable-postcheck-fastpath Enable EventBasedScheduler worker postcheck fast path.\n"
            << "                              Reduces scheduler dispatch overhead by letting workers\n"
            << "                              bypass the dispatcher for READY/WAIT_TIME transitions.\n"
            << "                              Requires SDK with enable_worker_postcheck_fastpath support.\n"
            << "  --worker-threads N          EventBasedScheduler worker thread count (default: 16).\n"
            << "                              Set lower when using --pin-measured-pipeline to reduce\n"
            << "                              CPU contention between default workers and RT threads.\n"
            << "\n  Green Context partitioning:\n"
            << "  --sms-per-partition N       SMs for both partitions, 0=auto (default: 0)\n"
            << "  --measured-sms N            SMs for measured partition only, 0=auto (default: 0)\n"
            << "  --contending-sms N          SMs for contending partition only, 0=auto (default: 0)\n"
            << "\n  Paths:\n"
            << "  --model-dir PATH           Directory for generated models (default: exe dir)\n"
            << "  --help                     Show this message\n";
}

// ---------------------------------------------------------------------------
// Model generation helper -- calls generate_onnx_model.py if model is missing
// ---------------------------------------------------------------------------

bool ensure_model_exists(const std::filesystem::path& gen_script,
                         const std::filesystem::path& model_path,
                         int input_size, int hidden_size, int num_layers,
                         const std::string& label) {
  if (std::filesystem::exists(model_path)) {
    std::cout << "  " << label << " model exists: " << model_path << std::endl;
    return true;
  }
  std::string cmd = "python3 \"" + gen_script.string() + "\""
      " --output \"" + model_path.string() + "\""
      " --input-size " + std::to_string(input_size) +
      " --hidden-size " + std::to_string(hidden_size) +
      " --num-layers " + std::to_string(num_layers);
  std::cout << "  Generating " << label << " model..." << std::endl;
  std::cout << "    " << cmd << std::endl;
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "Error: Failed to generate " << label << " model (exit code " << ret << ")\n";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  int total_samples = 1000;
  int warmup_samples = 100;
  std::string backend = "trt";
  int frequency_hz = 1000;
  int measured_sms = 0;
  int contending_sms = 0;
  std::string mode = "all";
  std::string periodic_policy = "CatchUpMissedTicks";
  bool pin_measured = false;
  std::string sched_policy_str = "SCHED_FIFO";
  std::vector<uint32_t> pin_cores;
  bool enable_postcheck_fastpath = false;
  int worker_threads = 16;

  int measured_input_size = 64;
  int measured_hidden_size = 256;
  int measured_layers = 3;

  int contending_input_size = 1024;
  int contending_hidden_size = 4096;
  int contending_layers = 6;
  int contending_frequency_hz = 0;

  std::string model_dir_str;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    else if (arg == "--samples" && i + 1 < argc) total_samples = std::atoi(argv[++i]);
    else if (arg == "--warmup-samples" && i + 1 < argc) warmup_samples = std::atoi(argv[++i]);
    else if (arg == "--backend" && i + 1 < argc) backend = argv[++i];
    else if (arg == "--frequency-hz" && i + 1 < argc) frequency_hz = std::atoi(argv[++i]);
    else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (arg == "--periodic-policy" && i + 1 < argc) periodic_policy = argv[++i];
    else if (arg == "--pin-measured-pipeline") pin_measured = true;
    else if (arg == "--scheduling-policy" && i + 1 < argc) sched_policy_str = argv[++i];
    else if (arg == "--pin-cores" && i + 1 < argc) {
      std::istringstream iss(argv[++i]);
      std::string token;
      while (std::getline(iss, token, ','))
        pin_cores.push_back(static_cast<uint32_t>(std::atoi(token.c_str())));
    }
    else if (arg == "--enable-postcheck-fastpath") enable_postcheck_fastpath = true;
    else if (arg == "--worker-threads" && i + 1 < argc) worker_threads = std::atoi(argv[++i]);
    else if (arg == "--measured-input-size" && i + 1 < argc) measured_input_size = std::atoi(argv[++i]);
    else if (arg == "--measured-hidden-size" && i + 1 < argc) measured_hidden_size = std::atoi(argv[++i]);
    else if (arg == "--measured-layers" && i + 1 < argc) measured_layers = std::atoi(argv[++i]);
    else if (arg == "--contending-input-size" && i + 1 < argc) contending_input_size = std::atoi(argv[++i]);
    else if (arg == "--contending-hidden-size" && i + 1 < argc) contending_hidden_size = std::atoi(argv[++i]);
    else if (arg == "--contending-layers" && i + 1 < argc) contending_layers = std::atoi(argv[++i]);
    else if (arg == "--contending-frequency-hz" && i + 1 < argc) contending_frequency_hz = std::atoi(argv[++i]);
    else if (arg == "--sms-per-partition" && i + 1 < argc) {
      measured_sms = std::atoi(argv[++i]); contending_sms = measured_sms;
    }
    else if (arg == "--measured-sms" && i + 1 < argc) measured_sms = std::atoi(argv[++i]);
    else if (arg == "--contending-sms" && i + 1 < argc) contending_sms = std::atoi(argv[++i]);
    else if (arg == "--model-dir" && i + 1 < argc) model_dir_str = argv[++i];
    else { std::cerr << "Unknown argument: " << arg << "\n"; print_usage(argv[0]); return 1; }
  }

  if (mode != "baseline" && mode != "green-context" && mode != "all") {
    std::cerr << "Error: --mode must be baseline|green-context|all\n";
    return 1;
  }
  if (backend != "trt" && backend != "onnxrt") {
    std::cerr << "Error: --backend must be trt|onnxrt\n";
    return 1;
  }
  if (periodic_policy != "CatchUpMissedTicks" &&
      periodic_policy != "MinTimeBetweenTicks" &&
      periodic_policy != "NoCatchUpMissedTicks") {
    std::cerr << "Error: --periodic-policy must be "
                 "CatchUpMissedTicks|MinTimeBetweenTicks|NoCatchUpMissedTicks\n";
    return 1;
  }
  SchedulingPolicy sched_policy{};
  if (sched_policy_str == "SCHED_DEADLINE") {
    sched_policy = SchedulingPolicy::kDeadline;
  } else if (sched_policy_str == "SCHED_FIFO") {
    sched_policy = SchedulingPolicy::kFirstInFirstOut;
  } else if (sched_policy_str == "SCHED_RR") {
    sched_policy = SchedulingPolicy::kRoundRobin;
  } else {
    std::cerr << "Error: --scheduling-policy must be SCHED_DEADLINE|SCHED_FIFO|SCHED_RR\n";
    return 1;
  }
  if (frequency_hz <= 0 || total_samples <= 0 || warmup_samples < 0 ||
      measured_input_size <= 0 || measured_hidden_size <= 0 || measured_layers <= 0 ||
      contending_input_size <= 0 || contending_hidden_size <= 0 || contending_layers <= 0 ||
      contending_frequency_hz < 0 || worker_threads <= 0) {
    std::cerr << "Error: invalid numeric argument(s)\n";
    return 1;
  }
  if (!pin_cores.empty() && pin_cores.size() != 1 && pin_cores.size() != 3) {
    std::cerr << "Error: --pin-cores expects 1 or 3 comma-separated core IDs\n";
    return 1;
  }
  if (!pin_cores.empty() && !pin_measured) {
    std::cerr << "Error: --pin-cores requires --pin-measured-pipeline\n";
    return 1;
  }

  if (enable_postcheck_fastpath) {
    bool supported = false;
    try {
      holoscan::ComponentSpec probe_spec;
      holoscan::EventBasedScheduler probe_sched;
      probe_sched.setup(probe_spec);
      supported = probe_spec.params().count("enable_worker_postcheck_fastpath") > 0;
    } catch (...) {
      supported = false;
    }
    if (!supported) {
      std::cerr << "Error: --enable-postcheck-fastpath requires a Holoscan SDK version with\n"
                << "enable_worker_postcheck_fastpath support on EventBasedScheduler.\n"
                << "Remove --enable-postcheck-fastpath or upgrade your SDK.\n";
      return 1;
    }
  }

  // Resolve paths relative to executable directory
  std::filesystem::path exe_dir;
  try { exe_dir = std::filesystem::canonical(argv[0]).parent_path(); }
  catch (const std::filesystem::filesystem_error&) {
    exe_dir = std::filesystem::absolute(argv[0]).parent_path();
  }

  auto config_path = exe_dir / "inference_scheduling_benchmark.yaml";
  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Error: Config not found: " << config_path << "\n";
    return 1;
  }

  auto gen_script = exe_dir / "generate_onnx_model.py";
  if (!std::filesystem::exists(gen_script)) {
    std::cerr << "Error: Model generation script not found: " << gen_script << "\n";
    return 1;
  }

  std::filesystem::path model_dir = model_dir_str.empty()
      ? exe_dir : std::filesystem::path(model_dir_str);
  std::filesystem::create_directories(model_dir);

  // Build parameter-encoded model filenames
  std::string measured_model_name = "measured_i" + std::to_string(measured_input_size) +
      "_h" + std::to_string(measured_hidden_size) +
      "_l" + std::to_string(measured_layers) + ".onnx";
  std::string contending_model_name = "contending_i" + std::to_string(contending_input_size) +
      "_h" + std::to_string(contending_hidden_size) +
      "_l" + std::to_string(contending_layers) + ".onnx";

  auto measured_model_path = std::filesystem::absolute(model_dir / measured_model_name).string();
  auto contending_model_path = std::filesystem::absolute(model_dir / contending_model_name).string();

  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Generating / locating synthetic ONNX models" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  if (!ensure_model_exists(gen_script, measured_model_path,
                           measured_input_size, measured_hidden_size, measured_layers,
                           "Measured"))
    return 1;
  if (!ensure_model_exists(gen_script, contending_model_path,
                           contending_input_size, contending_hidden_size, contending_layers,
                           "Contending"))
    return 1;

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Green Context Inference Latency Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "  Mode:                    " << mode << std::endl;
  std::cout << "  Backend:                 " << backend << std::endl;
  std::cout << "  Measured freq:           " << frequency_hz << " Hz" << std::endl;
  std::cout << "  Contending freq:         "
            << (contending_frequency_hz > 0
                ? std::to_string(contending_frequency_hz) + " Hz"
                : "free-running") << std::endl;
  std::cout << "  Periodic policy:         " << periodic_policy << std::endl;
  std::cout << "  Pin measured pipeline:   " << (pin_measured ? "yes" : "no") << std::endl;
  if (pin_measured) {
    std::cout << "  Scheduling policy:       " << sched_policy_str << std::endl;
    if (!pin_cores.empty()) {
      std::cout << "  Pin cores:               ";
      for (size_t i = 0; i < pin_cores.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << pin_cores[i];
      }
      std::cout << std::endl;
    }
  }
  std::cout << "  Postcheck fastpath:      "
            << (enable_postcheck_fastpath ? "yes" : "no") << std::endl;
  std::cout << "  Worker threads:          " << worker_threads << std::endl;
  std::cout << "  Samples:                 " << total_samples << std::endl;
  std::cout << "  Warmup:                  " << warmup_samples << std::endl;
  std::cout << "  Measured model:          " << measured_model_path
            << " (i=" << measured_input_size << " h=" << measured_hidden_size
            << " l=" << measured_layers << ")" << std::endl;
  std::cout << "  Contending model:        " << contending_model_path
            << " (i=" << contending_input_size << " h=" << contending_hidden_size
            << " l=" << contending_layers << ")" << std::endl;
  std::cout << "  Measured SMs:            "
            << (measured_sms > 0 ? std::to_string(measured_sms) : "auto") << std::endl;
  std::cout << "  Contending SMs:          "
            << (contending_sms > 0 ? std::to_string(contending_sms) : "auto") << std::endl;
  std::cout << std::endl;

  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(0), "Failed to set CUDA device");

  struct RunResult {
    BenchmarkStats e2e;
    BenchmarkStats tx_period;
    int contending_iters = 0;
    double contending_throughput_hz = 0.0;
  };

  auto run_app_once = [&](bool use_gc) -> RunResult {
    reset_global_benchmark_state();
    auto app = std::make_unique<InferenceSchedulingBenchmarkApp>(
        use_gc, total_samples, warmup_samples,
        measured_model_path, measured_input_size,
        contending_model_path, contending_input_size,
        backend, measured_sms, contending_sms,
        frequency_hz, contending_frequency_hz, periodic_policy,
        pin_measured, sched_policy, pin_cores);
    app->config(config_path);
    auto sched = app->make_scheduler<holoscan::EventBasedScheduler>(
        "scheduler", holoscan::Arg("worker_thread_number", static_cast<int64_t>(worker_threads)));
    if (enable_postcheck_fastpath) {
      sched->add_arg(holoscan::Arg("enable_worker_postcheck_fastpath", true));
    }
    app->scheduler(sched);
    app->run();
    return {app->get_e2e_stats(),
            app->get_tx_period_stats(),
            app->get_contending_iters(),
            app->get_contending_throughput_hz()};
  };

  RunResult bl{}, gc{};

  if (mode == "baseline" || mode == "all") {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Running BASELINE (no Green Context)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    bl = run_app_once(false);
    std::cout << "Baseline complete." << std::endl;
  }

  if (mode == "green-context" || mode == "all") {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Running GREEN CONTEXT" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    gc = run_app_once(true);
    std::cout << "Green Context complete." << std::endl;
  }

  // --- Results ---

  auto pct = [](double a, double b) { return a != 0.0 ? (a - b) / a * 100.0 : 0.0; };

  auto print_comparison = [&](const BenchmarkStats& a, const BenchmarkStats& b) {
    if (a.sample_count == 0 || b.sample_count == 0) return;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Avg:     " << a.avg     << " \xe2\x86\x92 " << b.avg     << " \xce\xbcs  (" << std::showpos << pct(a.avg,     b.avg)     << "%)" << std::noshowpos << std::endl;
    std::cout << "  P95:     " << a.p95     << " \xe2\x86\x92 " << b.p95     << " \xce\xbcs  (" << std::showpos << pct(a.p95,     b.p95)     << "%)" << std::noshowpos << std::endl;
    std::cout << "  P99:     " << a.p99     << " \xe2\x86\x92 " << b.p99     << " \xce\xbcs  (" << std::showpos << pct(a.p99,     b.p99)     << "%)" << std::noshowpos << std::endl;
    std::cout << "  Std Dev: " << a.std_dev << " \xe2\x86\x92 " << b.std_dev << " \xce\xbcs  (" << std::showpos << pct(a.std_dev, b.std_dev) << "%)" << std::noshowpos << std::endl;
  };

  auto print_section = [&](const std::string& title, const std::string& subtitle,
                           const BenchmarkStats& bl_s, const BenchmarkStats& gc_s) {
    std::cout << std::endl << std::string(80, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << subtitle << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    if (mode == "baseline" || mode == "all") { print_stats(bl_s, "Baseline"); std::cout << std::endl; }
    if (mode == "green-context" || mode == "all") { print_stats(gc_s, "Green Context"); std::cout << std::endl; }
    if (mode == "all") {
      std::cout << "=== Comparison (BL \xe2\x86\x92 GC) ===" << std::endl;
      print_comparison(bl_s, gc_s);
      std::cout << std::endl;
    }
  };

  print_section(
      "End-to-End Pipeline Latency (PeriodicTxOp \xe2\x86\x92 TimingRxOp)",
      "(steady_clock at TxOp::compute() \xe2\x86\x92 after cudaStreamSynchronize in RxOp)",
      bl.e2e, gc.e2e);

  print_section(
      "TxOp Firing Period (inter-fire interval, nominal = "
          + std::to_string(1'000'000 / frequency_hz) + " \xce\xbcs)",
      "(time between consecutive PeriodicTxOp fires -- scheduler frequency accuracy)",
      bl.tx_period, gc.tx_period);

  std::cout << std::endl << std::string(80, '=') << std::endl;
  std::cout << "Contending Inference Pipeline Throughput" << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << std::fixed << std::setprecision(1);
  if (mode == "baseline" || mode == "all") {
    std::cout << "=== Baseline ===" << std::endl;
    std::cout << "  Iterations: " << bl.contending_iters << std::endl;
    std::cout << "  Throughput: " << bl.contending_throughput_hz << " Hz" << std::endl << std::endl;
  }
  if (mode == "green-context" || mode == "all") {
    std::cout << "=== Green Context ===" << std::endl;
    std::cout << "  Iterations: " << gc.contending_iters << std::endl;
    std::cout << "  Throughput: " << gc.contending_throughput_hz << " Hz" << std::endl << std::endl;
  }

  return 0;
}
