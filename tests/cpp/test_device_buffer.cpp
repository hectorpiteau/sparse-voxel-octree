#include <svo/DeviceBuffer.hpp>
#include <svo/Error.hpp>
#include <svo/Octree.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void test_zero_size_cpu_buffer() {
  svo::DeviceBuffer<int> buffer;
  require(buffer.empty(), "default buffer should be empty");
  require(buffer.size() == 0, "default buffer size should be zero");
  require(buffer.size_bytes() == 0, "default buffer byte size should be zero");
  require(buffer.device() == svo::Device::CPU, "default buffer should be CPU");

  buffer.allocate(0, svo::Device::CPU);
  require(buffer.empty(), "zero-size CPU allocation should be empty");
}

void test_cpu_allocate_and_copy() {
  const std::vector<int> source{1, 2, 3, 5, 8};
  svo::DeviceBuffer<int> buffer(source.size(), svo::Device::CPU);

  require(buffer.size() == source.size(), "CPU buffer size mismatch");
  require(buffer.size_bytes() == source.size() * sizeof(int), "CPU buffer byte size mismatch");
  require(buffer.data() != nullptr, "non-empty CPU buffer should expose data");

  buffer.copy_from_host(source.data(), source.size());
  require(buffer.to_host() == source, "CPU buffer roundtrip mismatch");
}

void test_move_buffer() {
  const std::vector<std::uint32_t> source{10u, 20u, 30u};
  svo::DeviceBuffer<std::uint32_t> buffer =
      svo::DeviceBuffer<std::uint32_t>::from_host(source, svo::Device::CPU);

  svo::DeviceBuffer<std::uint32_t> moved(std::move(buffer));
  require(buffer.empty(), "moved-from buffer should be empty");
  require(moved.to_host() == source, "move constructor should preserve contents");

  svo::DeviceBuffer<std::uint32_t> assigned;
  assigned = std::move(moved);
  require(moved.empty(), "move-assigned source buffer should be empty");
  require(assigned.to_host() == source, "move assignment should preserve contents");
}

void test_failed_cpu_allocate_preserves_existing_buffer() {
  const std::vector<int> source{4, 2, 0};
  svo::DeviceBuffer<int> buffer = svo::DeviceBuffer<int>::from_host(source, svo::Device::CPU);

  try {
    buffer.allocate(std::numeric_limits<std::size_t>::max(), svo::Device::CPU);
    require(false, "oversized CPU allocation should fail");
  } catch (const std::exception&) {
  }

  require(buffer.device() == svo::Device::CPU, "failed CPU allocation should preserve device");
  require(buffer.size() == source.size(), "failed CPU allocation should preserve size");
  require(buffer.to_host() == source, "failed CPU allocation should preserve contents");
}

void test_bounds_checks() {
  svo::DeviceBuffer<int> buffer(2, svo::Device::CPU);
  const int value = 1;

  try {
    buffer.copy_from_host(&value, 3);
    require(false, "oversized host copy should fail");
  } catch (const svo::ValidationError&) {
  }

  try {
    buffer.copy_from_host(nullptr, 1);
    require(false, "non-zero null host copy should fail");
  } catch (const svo::ValidationError&) {
  }
}

void test_large_cpu_buffer() {
  constexpr std::size_t kCount = 1u << 20u;
  std::vector<std::uint32_t> source(kCount);
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<std::uint32_t>(index);
  }

  svo::DeviceBuffer<std::uint32_t> buffer =
      svo::DeviceBuffer<std::uint32_t>::from_host(source, svo::Device::CPU);
  const std::vector<std::uint32_t> roundtrip = buffer.to_host();

  require(roundtrip.front() == 0u, "large buffer first value mismatch");
  require(roundtrip.back() == static_cast<std::uint32_t>(kCount - 1u), "large buffer last value mismatch");
}

void test_cuda_buffer_if_enabled() {
#if SVO_ENABLE_CUDA
  const std::vector<int> source{3, 1, 4, 1, 5, 9};
  svo::DeviceBuffer<int> buffer =
      svo::DeviceBuffer<int>::from_host(source, svo::Device::CUDA);

  require(buffer.device() == svo::Device::CUDA, "CUDA buffer should report CUDA device");
  require(buffer.size() == source.size(), "CUDA buffer size mismatch");
  require(buffer.to_host() == source, "CUDA buffer roundtrip mismatch");
#else
  try {
    svo::DeviceBuffer<int> buffer(1, svo::Device::CUDA);
    (void)buffer;
    require(false, "CUDA allocation should fail when CUDA is disabled");
  } catch (const svo::ValidationError&) {
  }
#endif
}

void test_cuda_zero_size_buffer_if_enabled() {
#if SVO_ENABLE_CUDA
  svo::DeviceBuffer<int> buffer(0, svo::Device::CUDA);
  require(buffer.empty(), "zero-size CUDA allocation should be empty");
  require(buffer.size_bytes() == 0, "zero-size CUDA allocation should have no bytes");
  require(buffer.device() == svo::Device::CUDA, "zero-size CUDA allocation should preserve requested device");
  require(buffer.data() == nullptr, "zero-size CUDA allocation should not expose data");
#endif
}

void test_cuda_move_buffer_if_enabled() {
#if SVO_ENABLE_CUDA
  const std::vector<std::uint32_t> source{13u, 21u, 34u, 55u};
  svo::DeviceBuffer<std::uint32_t> buffer =
      svo::DeviceBuffer<std::uint32_t>::from_host(source, svo::Device::CUDA);

  svo::DeviceBuffer<std::uint32_t> moved(std::move(buffer));
  require(buffer.empty(), "moved-from CUDA buffer should be empty");
  require(moved.device() == svo::Device::CUDA, "moved CUDA buffer should preserve device");
  require(moved.to_host() == source, "CUDA move constructor should preserve contents");

  svo::DeviceBuffer<std::uint32_t> assigned;
  assigned = std::move(moved);
  require(moved.empty(), "move-assigned CUDA source buffer should be empty");
  require(assigned.device() == svo::Device::CUDA, "move-assigned CUDA buffer should preserve device");
  require(assigned.to_host() == source, "CUDA move assignment should preserve contents");
#endif
}

void test_cuda_partial_copy_if_enabled() {
#if SVO_ENABLE_CUDA
  const std::vector<int> zeroes{0, 0, 0, 0, 0, 0};
  const std::vector<int> prefix{8, 6, 7};
  const std::vector<int> expected{8, 6, 7, 0, 0, 0};

  svo::DeviceBuffer<int> buffer = svo::DeviceBuffer<int>::from_host(zeroes, svo::Device::CUDA);
  buffer.copy_from_host(prefix.data(), prefix.size());

  require(buffer.to_host() == expected, "partial CUDA host-to-device copy mismatch");
#endif
}

void test_cuda_async_stream_ordering_if_enabled() {
#if SVO_ENABLE_CUDA
  const std::vector<int> source{2, 7, 1, 8, 2, 8};
  std::vector<int> destination(source.size(), 0);

  cudaStream_t stream = nullptr;
  cudaError_t result = cudaStreamCreate(&stream);
  require(result == cudaSuccess, std::string("cudaStreamCreate failed: ") + cudaGetErrorString(result));

  {
    svo::DeviceBuffer<int> buffer(source.size(), svo::Device::CUDA);
    buffer.copy_from_host(source.data(), source.size(), stream);
    buffer.copy_to_host(destination.data(), destination.size(), stream);
    result = cudaStreamSynchronize(stream);
    require(result == cudaSuccess, std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(result));
  }

  result = cudaStreamDestroy(stream);
  require(result == cudaSuccess, std::string("cudaStreamDestroy failed: ") + cudaGetErrorString(result));
  require(destination == source, "CUDA async stream-ordered roundtrip mismatch");
#endif
}

}  // namespace

int main() {
  static_assert(!std::is_copy_constructible<svo::DeviceBuffer<int>>::value, "DeviceBuffer must not be copyable");
  static_assert(!std::is_copy_assignable<svo::DeviceBuffer<int>>::value, "DeviceBuffer must not be copy-assignable");
  static_assert(std::is_move_constructible<svo::DeviceBuffer<int>>::value, "DeviceBuffer must be movable");
  static_assert(std::is_move_assignable<svo::DeviceBuffer<int>>::value, "DeviceBuffer must be move-assignable");

  test_zero_size_cpu_buffer();
  test_cpu_allocate_and_copy();
  test_move_buffer();
  test_failed_cpu_allocate_preserves_existing_buffer();
  test_bounds_checks();
  test_large_cpu_buffer();
  test_cuda_buffer_if_enabled();
  test_cuda_zero_size_buffer_if_enabled();
  test_cuda_move_buffer_if_enabled();
  test_cuda_partial_copy_if_enabled();
  test_cuda_async_stream_ordering_if_enabled();
  return 0;
}
