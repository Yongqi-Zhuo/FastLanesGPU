#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#define CASDEC_BENCHMARK_CHECK_CUDA(call)                                      \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(status) << std::endl;                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace casdec::benchmark {

namespace detail {

class L2Flush {
  int l2Size{};
  int *l2Buffer{};

public:
  L2Flush() {
    int deviceId{};
    CASDEC_BENCHMARK_CHECK_CUDA(cudaGetDevice(&deviceId));
    CASDEC_BENCHMARK_CHECK_CUDA(
        cudaDeviceGetAttribute(&l2Size, cudaDevAttrL2CacheSize, deviceId));
    if (l2Size > 0) {
      void *buffer = l2Buffer;
      CASDEC_BENCHMARK_CHECK_CUDA(
          cudaMalloc(&buffer, static_cast<std::size_t>(l2Size)));
      l2Buffer = reinterpret_cast<int *>(buffer);
    }
  }

  ~L2Flush() noexcept {
    if (l2Buffer) {
      CASDEC_BENCHMARK_CHECK_CUDA(cudaFree(l2Buffer));
    }
  }

  void flush(cudaStream_t stream) {
    if (l2Size > 0) {
      if (stream == nullptr) {
        CASDEC_BENCHMARK_CHECK_CUDA(
            cudaMemset(l2Buffer, 0, static_cast<std::size_t>(l2Size)));
      } else {
        CASDEC_BENCHMARK_CHECK_CUDA(cudaMemsetAsync(
            l2Buffer, 0, static_cast<std::size_t>(l2Size), stream));
      }
    }
  }
};

template <typename T>
inline T parseEnv(const char *envVar, T defaultValue) {
  T value = defaultValue;
  const auto *envVarStr = std::getenv(envVar);
  if (envVarStr) {
    if (auto [ptr, ec] = std::from_chars(
            envVarStr, envVarStr + std::strlen(envVarStr), value);
        ec == std::errc()) {
      // Successfully parsed.
    } else {
      std::cerr << "Warning: Invalid " << envVar << " value '" << envVarStr
                << "'. Using default value: " << defaultValue << "."
                << std::endl;
    }
  }
  return value;
}

} // namespace detail

class Stream {
  cudaStream_t stream = nullptr;

public:
  Stream() { CASDEC_BENCHMARK_CHECK_CUDA(cudaStreamCreate(&stream)); }
  Stream(std::nullptr_t) : stream(nullptr) {}
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  Stream(Stream &&other) noexcept : stream(other.stream) {
    other.stream = nullptr;
  }
  Stream &operator=(Stream &&other) noexcept {
    if (this != &other) {
      std::swap(stream, other.stream);
    }
    return *this;
  }
  ~Stream() noexcept {
    if (stream) {
      CASDEC_BENCHMARK_CHECK_CUDA(cudaStreamDestroy(stream));
    }
  }

  operator cudaStream_t() const { return stream; }

  void synchronize() const {
    if (stream) {
      CASDEC_BENCHMARK_CHECK_CUDA(cudaStreamSynchronize(stream));
    } else {
      CASDEC_BENCHMARK_CHECK_CUDA(cudaDeviceSynchronize());
    }
  }
};

struct Event {
  cudaEvent_t event;
  Event() { CASDEC_BENCHMARK_CHECK_CUDA(cudaEventCreate(&event)); }
  ~Event() noexcept { CASDEC_BENCHMARK_CHECK_CUDA(cudaEventDestroy(event)); }
  void record(cudaStream_t stream = nullptr) {
    CASDEC_BENCHMARK_CHECK_CUDA(cudaEventRecord(event, stream));
  }
  friend double operator-(const Event &lhs, const Event &rhs) {
    float ms;
    CASDEC_BENCHMARK_CHECK_CUDA(
        cudaEventElapsedTime(&ms, rhs.event, lhs.event));
    return static_cast<double>(ms);
  }
};

struct ValueWithError {
  double average;
  double deviation;
  double min;
  double max;

  ValueWithError() : average(0.0), deviation(0.0), min(0.0), max(0.0) {}

  ValueWithError(double average, double deviation, double min, double max)
      : average(average), deviation(deviation), min(min), max(max) {}

  template <typename It>
  ValueWithError(It begin, It end) {
    std::vector<double> values(begin, end);
    // Debug: print all values
    std::cerr << "Got values: ";
    for (double v : values) {
      std::cerr << v << " ";
    }
    std::cerr << std::endl;
    if (values.empty()) {
      std::cerr << "Error: No values to compute ValueWithError." << std::endl;
      std::exit(1);
    }
    average = std::accumulate(values.begin(), values.end(), 0.0) /
              static_cast<double>(values.size());
    deviation = std::sqrt(
        std::accumulate(values.begin(), values.end(), 0.0,
                        [average = average](double acc, double value) {
                          return acc + (value - average) * (value - average);
                        }) /
        static_cast<double>(values.size() - 1));
    min = *std::min_element(values.begin(), values.end());
    max = *std::max_element(values.begin(), values.end());
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const ValueWithError &result) {
    os << result.average << " ± " << result.deviation << " (min: " << result.min
       << ", max: " << result.max << ")";
    return os;
  }

  ValueWithError &operator*=(double rhs) {
    average *= rhs;
    deviation *= rhs;
    min *= rhs;
    max *= rhs;
    return *this;
  }

  friend ValueWithError operator*(const ValueWithError &lhs, double rhs) {
    auto temp = lhs;
    temp *= rhs;
    return temp;
  }

  friend ValueWithError operator*(double lhs, const ValueWithError &rhs) {
    auto temp = rhs;
    temp *= lhs;
    return temp;
  }

  ValueWithError &operator/=(double rhs) {
    average /= rhs;
    deviation /= rhs;
    min /= rhs;
    max /= rhs;
    return *this;
  }

  friend ValueWithError operator/(const ValueWithError &lhs, double rhs) {
    auto temp = lhs;
    temp /= rhs;
    return temp;
  }

  friend ValueWithError operator/(double lhs, const ValueWithError &rhs) {
    return ValueWithError(lhs / rhs.average,
                          lhs * rhs.deviation / (rhs.average * rhs.average),
                          lhs / rhs.max, lhs / rhs.min);
  }
};

inline unsigned getDefaultNumTotalRuns() {
  return detail::parseEnv("CASDEC_BENCHMARK_TOTAL_RUNS", 20u);
}

// Accepted kernel function signatures:
//  void kernel();
//  void kernel(unsigned runIndex);
template <typename F>
inline ValueWithError benchmarkKernel(F &&kernel, unsigned totalRuns,
                                      cudaStream_t stream) {
  auto call = [&](unsigned i) {
    if constexpr (std::is_invocable_v<F, unsigned>) {
      std::invoke(kernel, i);
    } else if constexpr (std::is_invocable_v<F>) {
      std::invoke(kernel);
    } else {
      static_assert(std::is_invocable_v<F>,
                    "The kernel function must have the signature "
                    "'void kernel()' or "
                    "'void kernel(unsigned)'.");
    }
  };

  detail::L2Flush l2Flush;

  auto numWarmupRuns = detail::parseEnv("CASDEC_BENCHMARK_WARMUP_RUNS", 3u);
  numWarmupRuns = std::min(numWarmupRuns, totalRuns);

  // Warm up.
  for (unsigned i = 0; i < numWarmupRuns; ++i) {
    call(i);
  }
  if (stream) {
    CASDEC_BENCHMARK_CHECK_CUDA(cudaStreamSynchronize(stream));
  } else {
    CASDEC_BENCHMARK_CHECK_CUDA(cudaDeviceSynchronize());
  }

  unsigned runs = totalRuns - numWarmupRuns;

  if (runs == 0) {
    return ValueWithError();
  }

  std::vector<Event> start(runs), stop(runs);
  for (unsigned i = 0; i < runs; ++i) {
    l2Flush.flush(stream);
    start[i].record(stream);
    call(i + numWarmupRuns);
    stop[i].record(stream);
  }
  if (stream) {
    CASDEC_BENCHMARK_CHECK_CUDA(cudaStreamSynchronize(stream));
  } else {
    CASDEC_BENCHMARK_CHECK_CUDA(cudaDeviceSynchronize());
  }

  std::vector<double> durations(runs);
  for (unsigned i = 0; i < runs; ++i) {
    durations[i] = stop[i] - start[i];
  }

  return ValueWithError(durations.begin(), durations.end());
}

inline unsigned getDesiredNumVectors() {
  // To eliminate overhead and saturate the GPU, we require at least
  // kDefaultNumVectors vectors in a grid. If the column has fewer tuples, it
  // needs to be topped up by repeating the work.
  // The most powerful GPU has about 128 SMs, so in total there are 512
  // sub-partitions. Each sub-partition should handle 32 warps to reduce tail
  // effects. Also, in our current model, each warp handles 64 vectors (each
  // vector is 1024 tuples).
  constexpr unsigned kDefaultNumVectors = 64 * 512 * 32;
  return detail::parseEnv("CASDEC_BENCHMARK_DESIRED_VECTORS",
                          kDefaultNumVectors);
}

} // namespace casdec::benchmark
