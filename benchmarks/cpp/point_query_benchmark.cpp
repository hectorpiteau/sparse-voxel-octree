#include <svo/DeviceBuffer.hpp>
#include <svo/Octree.hpp>
#include <svo/Query.hpp>

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_int3.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() {
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class CudaStream {
 public:
  CudaStream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
  ~CudaStream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
  float milliseconds = 0.0f;
  check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()), "cudaEventElapsedTime");
  return milliseconds;
}

template <typename F>
float time_cuda_ms(cudaStream_t stream, F&& function) {
  CudaEvent start;
  CudaEvent stop;
  check_cuda(cudaEventRecord(start.get(), stream), "cudaEventRecord start");
  function();
  check_cuda(cudaEventRecord(stop.get(), stream), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize stop");
  return elapsed_ms(start, stop);
}

glm::vec3 voxel_center_point(int grid_size, const glm::ivec3& coordinate) {
  const float inv_grid = 1.0f / static_cast<float>(grid_size);
  return glm::vec3{
      (static_cast<float>(coordinate.x) + 0.5f) * inv_grid,
      (static_cast<float>(coordinate.y) + 0.5f) * inv_grid,
      (static_cast<float>(coordinate.z) + 0.5f) * inv_grid};
}

std::string today_local() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &now);
#else
  localtime_r(&now, &tm);
#endif
  std::ostringstream stream;
  stream << std::put_time(&tm, "%Y-%m-%d");
  return stream.str();
}

std::string gpu_name() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  return properties.name;
}

std::vector<glm::ivec3> make_scene(int grid_size) {
  std::vector<glm::ivec3> coordinates;
  coordinates.reserve(24000);
  for (int z = 0; z < grid_size; ++z) {
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        const int dx = x - grid_size / 2;
        const int dy = y - grid_size / 2;
        const int dz = z - grid_size / 2;
        const bool inside_sphere = dx * dx + dy * dy + dz * dz <= (grid_size / 3) * (grid_size / 3);
        if (inside_sphere && ((x * 17 + y * 31 + z * 43) % 7) == 0) {
          coordinates.emplace_back(x, y, z);
        }
      }
    }
  }
  return coordinates;
}

std::vector<glm::vec3> make_points(std::size_t count, int grid_size) {
  std::mt19937 rng(123456u);
  std::uniform_int_distribution<int> coord_dist(0, grid_size - 1);
  std::vector<glm::vec3> points;
  points.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if ((index % 23u) == 0u) {
      points.push_back({1.0f, 0.5f, 0.5f});
    } else {
      points.push_back(voxel_center_point(grid_size, {coord_dist(rng), coord_dist(rng), coord_dist(rng)}));
    }
  }
  return points;
}

std::size_t parse_point_count(int argc, char** argv) {
  if (argc < 2) {
    return 1u << 20u;
  }
  const long long value = std::atoll(argv[1]);
  if (value <= 0) {
    throw std::runtime_error("point count argument must be positive");
  }
  return static_cast<std::size_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
  constexpr int max_depth = 6;
  constexpr int grid_size = 1 << max_depth;
  const std::size_t point_count = parse_point_count(argc, argv);

  svo::BuildOptions build_options;
  build_options.max_depth = max_depth;

  const std::vector<glm::ivec3> coordinates = make_scene(grid_size);
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, build_options);
  const std::vector<glm::vec3> points = make_points(point_count, grid_size);

  const auto cpu_start = std::chrono::steady_clock::now();
  const std::vector<std::int32_t> cpu_results = svo::query_points(octree, points);
  const auto cpu_stop = std::chrono::steady_clock::now();
  const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

  CudaStream stream;
  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes(octree.nodes().size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices(
      octree.leaf_payload_indices().size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_points(points.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_results(points.size(), svo::Device::CUDA);
  std::vector<std::int32_t> cuda_results(points.size());

  const float h2d_ms = time_cuda_ms(stream.get(), [&]() {
    device_nodes.copy_from_host(octree.nodes().data(), octree.nodes().size(), stream.get());
    device_leaf_payload_indices.copy_from_host(
        octree.leaf_payload_indices().data(), octree.leaf_payload_indices().size(), stream.get());
    device_points.copy_from_host(points.data(), points.size(), stream.get());
  });

  const float kernel_ms = time_cuda_ms(stream.get(), [&]() {
    svo::query_points_cuda(
        device_nodes.data(),
        device_nodes.size(),
        device_leaf_payload_indices.data(),
        device_leaf_payload_indices.size(),
        octree.max_depth(),
        octree.root_bounds(),
        device_points.data(),
        device_results.data(),
        device_results.size(),
        {},
        stream.get());
  });

  const float d2h_ms = time_cuda_ms(stream.get(), [&]() {
    device_results.copy_to_host(cuda_results.data(), cuda_results.size(), stream.get());
  });

  if (cuda_results != cpu_results) {
    std::cerr << "CUDA results do not match CPU reference\n";
    return 1;
  }

#ifdef NDEBUG
  const char* build_type = "Release";
#else
  const char* build_type = "Debug";
#endif
  const double total_cuda_ms = static_cast<double>(h2d_ms + kernel_ms + d2h_ms);
  const double speedup = total_cuda_ms > 0.0 ? cpu_ms / total_cuda_ms : 0.0;

  std::cout << "| " << today_local()
            << " | " << gpu_name()
            << " | unknown"
            << " | " << build_type << "/CUDA"
            << " | sparse sphere depth-" << max_depth
            << " | " << point_count
            << " | " << std::fixed << std::setprecision(3) << cpu_ms
            << " | " << h2d_ms
            << " | " << kernel_ms
            << " | " << d2h_ms
            << " | " << total_cuda_ms
            << " | " << speedup
            << " | generated by svo_point_query_benchmark |\n";
  return 0;
}
