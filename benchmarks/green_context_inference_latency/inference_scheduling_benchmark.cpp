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
#include <stdexcept>
#include <vector>

#include <holoscan/holoscan.hpp>
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
static std::deque<int64_t> g_tx_emit_timestamps_ns;

void reset_global_benchmark_state() {
  g_contending_pipeline_ready.store(false, std::memory_order_release);
  std::lock_guard<std::mutex> lock(g_tx_timestamps_mutex);
  g_tx_emit_timestamps_ns.clear();
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

    if (record_timestamps_) {
      auto now = std::chrono::steady_clock::now().time_since_epoch();
      {
        std::lock_guard<std::mutex> lock(g_tx_timestamps_mutex);
        g_tx_emit_timestamps_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
      }
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
    int64_t emit_ns = 0;
    {
      std::lock_guard<std::mutex> lock(g_tx_timestamps_mutex);
      if (!g_tx_emit_timestamps_ns.empty()) {
        emit_ns = g_tx_emit_timestamps_ns.front();
        g_tx_emit_timestamps_ns.pop_front();
      }
    }
    if (emit_ns <= 0) {
      HOLOSCAN_LOG_WARN("[TimingRxOp] Missing emit timestamp for received tick; dropping sample");
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

    auto streams = op_input.receive_cuda_streams("in");
    if (!streams.empty() && streams[0].has_value()) {
      cudaStreamSynchronize(streams[0].value());
    }

    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now_ns < emit_ns) {
      HOLOSCAN_LOG_WARN("[TimingRxOp] Non-monotonic latency sample (now < emit); dropping sample");
      return;
    }
    double latency_us = static_cast<double>(now_ns - emit_ns) / 1000.0;
    latencies_us_.push_back(latency_us);

    sample_count_++;
    int log_interval = std::max(1, total_samples_.get() / 10);
    if (sample_count_ % log_interval == 0 || sample_count_ == 1) {
      HOLOSCAN_LOG_INFO("[TimingRxOp] Collected {}/{} samples",
                        sample_count_, total_samples_.get());
    }

    if (sample_count_ >= total_samples_.get()) {
      fragment()->stop_execution();
    }
  }

  BenchmarkStats get_latency_stats() const {
    return calculate_stats(latencies_us_);
  }

 private:
  Parameter<int> total_samples_;
  Parameter<int> warmup_samples_;
  int sample_count_ = 0;
  int warmup_pre_ready_count_ = 0;
  int warmup_post_ready_count_ = 0;
  std::vector<double> latencies_us_;
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
                                  int64_t measured_period_ns,
                                  int64_t contending_period_ns)
      : use_gc_(use_gc), total_samples_(total_samples), warmup_samples_(warmup_samples),
        measured_model_path_(measured_model_path),
        measured_input_size_(measured_input_size),
        contending_model_path_(contending_model_path),
        contending_input_size_(contending_input_size),
        backend_(backend),
        measured_sms_(measured_sms), contending_sms_(contending_sms),
        measured_period_ns_(measured_period_ns),
        contending_period_ns_(contending_period_ns) {}

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
    std::vector<TensorSpec> measured_outputs = {{"output", {1, measured_input_size_}}};

    auto measured_tx = make_operator<PeriodicTxOp>(
        "measured_tx",
        make_condition<PeriodicCondition>("measured_periodic",
            std::chrono::nanoseconds(measured_period_ns_)));
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

    // --- Contending pipeline: PeriodicTxOp → InferenceOp → ContendingSinkOp ---

    std::vector<TensorSpec> contending_inputs = {{"input", {1, contending_input_size_}}};

    std::shared_ptr<PeriodicTxOp> contending_tx;
    if (contending_period_ns_ > 0) {
      contending_tx = make_operator<PeriodicTxOp>(
          "contending_tx",
          make_condition<PeriodicCondition>("contending_periodic",
              std::chrono::nanoseconds(contending_period_ns_)));
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

  BenchmarkStats get_latency_stats() const {
    return timing_rx_->get_latency_stats();
  }

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
  int64_t measured_period_ns_;
  int64_t contending_period_ns_;
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
  if (frequency_hz <= 0 || total_samples <= 0 || warmup_samples < 0 ||
      measured_input_size <= 0 || measured_hidden_size <= 0 || measured_layers <= 0 ||
      contending_input_size <= 0 || contending_hidden_size <= 0 || contending_layers <= 0 ||
      contending_frequency_hz < 0) {
    std::cerr << "Error: invalid numeric argument(s)\n";
    return 1;
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

  int64_t measured_period_ns = static_cast<int64_t>(1e9 / frequency_hz);
  int64_t contending_period_ns = contending_frequency_hz > 0
      ? static_cast<int64_t>(1e9 / contending_frequency_hz) : 0;

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Green Context Inference Latency Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "  Mode:                    " << mode << std::endl;
  std::cout << "  Backend:                 " << backend << std::endl;
  std::cout << "  Measured freq:           " << frequency_hz << " Hz ("
            << (measured_period_ns / 1000) << " \xce\xbcs period)" << std::endl;
  std::cout << "  Contending freq:         "
            << (contending_frequency_hz > 0
                ? std::to_string(contending_frequency_hz) + " Hz"
                : "free-running") << std::endl;
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

  BenchmarkStats stats_baseline, stats_gc;
  int contending_iters_bl = 0, contending_iters_gc = 0;
  double contending_hz_bl = 0.0, contending_hz_gc = 0.0;

  struct RunResult {
    BenchmarkStats latency;
    int contending_iters;
    double contending_throughput_hz;
  };

  auto run_app_once = [&](bool use_gc) -> RunResult {
    reset_global_benchmark_state();
    auto app = std::make_unique<InferenceSchedulingBenchmarkApp>(
        use_gc, total_samples, warmup_samples,
        measured_model_path, measured_input_size,
        contending_model_path, contending_input_size,
        backend, measured_sms, contending_sms,
        measured_period_ns, contending_period_ns);
    app->config(config_path);
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
        "scheduler", holoscan::Arg("worker_thread_number", static_cast<int64_t>(16))));
    app->run();
    return {app->get_latency_stats(),
            app->get_contending_iters(),
            app->get_contending_throughput_hz()};
  };

  if (mode == "baseline" || mode == "all") {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Running BASELINE (no Green Context)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    auto result = run_app_once(false);
    stats_baseline = result.latency;
    contending_iters_bl = result.contending_iters;
    contending_hz_bl = result.contending_throughput_hz;
    std::cout << "Baseline complete." << std::endl;
  }

  if (mode == "green-context" || mode == "all") {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Running GREEN CONTEXT" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    auto result = run_app_once(true);
    stats_gc = result.latency;
    contending_iters_gc = result.contending_iters;
    contending_hz_gc = result.contending_throughput_hz;
    std::cout << "Green Context complete." << std::endl;
  }

  // --- Results ---

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "End-to-End Pipeline Latency (TxOp \xe2\x86\x92 InferenceOp \xe2\x86\x92 RxOp)" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  if (mode == "baseline" || mode == "all") {
    print_stats(stats_baseline, "Baseline");
    std::cout << std::endl;
  }
  if (mode == "green-context" || mode == "all") {
    print_stats(stats_gc, "Green Context");
    std::cout << std::endl;
  }

  auto pct_improvement = [](double bl, double gc) { return (bl - gc) / bl * 100.0; };

  if (mode == "all" && stats_baseline.sample_count > 0 && stats_gc.sample_count > 0) {
    std::cout << "=== Comparison (BL \xe2\x86\x92 GC) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "  Avg:        " << stats_baseline.avg << " \xe2\x86\x92 " << stats_gc.avg
              << " \xce\xbcs  (" << std::showpos << pct_improvement(stats_baseline.avg, stats_gc.avg) << "%)" << std::endl;
    std::cout << std::noshowpos;
    std::cout << "  P95:        " << stats_baseline.p95 << " \xe2\x86\x92 " << stats_gc.p95
              << " \xce\xbcs  (" << std::showpos << pct_improvement(stats_baseline.p95, stats_gc.p95) << "%)" << std::endl;
    std::cout << std::noshowpos;
    std::cout << "  P99:        " << stats_baseline.p99 << " \xe2\x86\x92 " << stats_gc.p99
              << " \xce\xbcs  (" << std::showpos << pct_improvement(stats_baseline.p99, stats_gc.p99) << "%)" << std::endl;
    std::cout << std::noshowpos;
    std::cout << "  Std Dev:    " << stats_baseline.std_dev << " \xe2\x86\x92 " << stats_gc.std_dev
              << " \xce\xbcs  (" << std::showpos << pct_improvement(stats_baseline.std_dev, stats_gc.std_dev) << "%)" << std::endl;
    std::cout << std::noshowpos;
  }

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Contending Inference Pipeline Throughput" << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << std::fixed << std::setprecision(1);

  if (mode == "baseline" || mode == "all") {
    std::cout << "=== Baseline ===" << std::endl;
    std::cout << "  Iterations: " << contending_iters_bl << std::endl;
    std::cout << "  Throughput: " << contending_hz_bl << " Hz" << std::endl;
    std::cout << std::endl;
  }
  if (mode == "green-context" || mode == "all") {
    std::cout << "=== Green Context ===" << std::endl;
    std::cout << "  Iterations: " << contending_iters_gc << std::endl;
    std::cout << "  Throughput: " << contending_hz_gc << " Hz" << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
