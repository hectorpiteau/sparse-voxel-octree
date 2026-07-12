#include <svo/Builder.hpp>
#include <svo/DeviceBuffer.hpp>
#include <svo/Renderer.hpp>

#include <cuda_runtime_api.h>
#include <glm/geometric.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
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

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float milliseconds = 0.0f;
  check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
  return milliseconds;
}

std::vector<glm::ivec3> make_sparse_scene(int max_depth) {
  const int grid_size = 1 << max_depth;
  std::vector<glm::ivec3> coordinates;
  coordinates.reserve(24000);
  for (int z = 0; z < grid_size; ++z) {
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
        const float ny = (static_cast<float>(y) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
        const float nz = (static_cast<float>(z) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
        const bool inside_sphere = nx * nx + ny * ny + nz * nz <= 0.24f * 0.24f;
        if (inside_sphere && ((x * 13 + y * 17 + z * 19) % 5) == 0) {
          coordinates.emplace_back(x, y, z);
        }
      }
    }
  }
  return coordinates;
}

void make_rays(std::size_t count, std::vector<glm::vec3>& origins, std::vector<glm::vec3>& directions) {
  std::mt19937 rng(20260712u);
  std::uniform_real_distribution<float> uv_dist(0.1f, 0.9f);

  origins.clear();
  directions.clear();
  origins.reserve(count);
  directions.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const float y = uv_dist(rng);
    const float z = uv_dist(rng);
    origins.push_back({-0.5f, y, z});
    directions.push_back(glm::normalize(glm::vec3{1.0f, 0.5f - y, 0.5f - z}));
  }
}

void make_payloads(const svo::Octree& octree, std::vector<float>& sigma, std::vector<float>& color) {
  sigma.resize(octree.num_leaves());
  color.resize(octree.num_leaves() * 3u);
  for (std::size_t index = 0; index < octree.num_leaves(); ++index) {
    const float t = static_cast<float>(index % 97u) / 96.0f;
    sigma[index] = 0.25f + 2.0f * t;
    color[index * 3u + 0u] = 0.5f + 0.5f * std::sin(6.2831853f * t);
    color[index * 3u + 1u] = 0.5f + 0.5f * std::sin(6.2831853f * (t + 0.3333333f));
    color[index * 3u + 2u] = 0.5f + 0.5f * std::sin(6.2831853f * (t + 0.6666667f));
  }
}

struct RenderMetrics {
  float transfer_ms = 0.0f;
  float forward_ms = 0.0f;
  float backward_ms = 0.0f;
};

template <typename LaunchForward, typename LaunchBackward>
RenderMetrics time_render_pair(cudaStream_t stream, int iterations, LaunchForward&& forward, LaunchBackward&& backward) {
  CudaEvent forward_start;
  CudaEvent forward_stop;
  CudaEvent backward_start;
  CudaEvent backward_stop;

  forward();
  backward();
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

  check_cuda(cudaEventRecord(forward_start.get(), stream), "cudaEventRecord forward_start");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    forward();
  }
  check_cuda(cudaEventRecord(forward_stop.get(), stream), "cudaEventRecord forward_stop");
  check_cuda(cudaEventSynchronize(forward_stop.get()), "cudaEventSynchronize forward_stop");

  check_cuda(cudaEventRecord(backward_start.get(), stream), "cudaEventRecord backward_start");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    backward();
  }
  check_cuda(cudaEventRecord(backward_stop.get(), stream), "cudaEventRecord backward_stop");
  check_cuda(cudaEventSynchronize(backward_stop.get()), "cudaEventSynchronize backward_stop");

  RenderMetrics metrics;
  metrics.forward_ms = elapsed_ms(forward_start.get(), forward_stop.get()) / static_cast<float>(iterations);
  metrics.backward_ms = elapsed_ms(backward_start.get(), backward_stop.get()) / static_cast<float>(iterations);
  return metrics;
}

RenderMetrics benchmark_octree8(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const std::vector<float>& sigma,
    const std::vector<float>& color,
    cudaStream_t stream,
    int iterations) {
  CudaEvent transfer_start;
  CudaEvent transfer_stop;
  auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>(octree.nodes().size(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>(octree.leaf_payload_indices().size(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>(origins.size(), svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>(directions.size(), svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>(sigma.size(), svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>(color.size(), svo::Device::CUDA);
  auto device_rgb = svo::DeviceBuffer<glm::vec3>(origins.size(), svo::Device::CUDA);
  auto device_depth = svo::DeviceBuffer<float>(origins.size(), svo::Device::CUDA);
  auto device_opacity = svo::DeviceBuffer<float>(origins.size(), svo::Device::CUDA);
  auto device_grad_rgb = svo::DeviceBuffer<glm::vec3>(origins.size(), svo::Device::CUDA);
  auto device_grad_opacity = svo::DeviceBuffer<float>(origins.size(), svo::Device::CUDA);
  auto device_grad_sigma = svo::DeviceBuffer<float>(sigma.size(), svo::Device::CUDA);
  auto device_grad_color = svo::DeviceBuffer<float>(color.size(), svo::Device::CUDA);

  const std::vector<glm::vec3> grad_rgb(origins.size(), glm::vec3{1.0f, 1.0f, 1.0f});
  const std::vector<float> grad_opacity(origins.size(), 1.0f);

  check_cuda(cudaEventRecord(transfer_start.get(), stream), "cudaEventRecord transfer_start");
  device_nodes.copy_from_host(octree.nodes().data(), octree.nodes().size(), stream);
  device_payload_indices.copy_from_host(
      octree.leaf_payload_indices().data(), octree.leaf_payload_indices().size(), stream);
  device_origins.copy_from_host(origins.data(), origins.size(), stream);
  device_directions.copy_from_host(directions.data(), directions.size(), stream);
  device_sigma.copy_from_host(sigma.data(), sigma.size(), stream);
  device_color.copy_from_host(color.data(), color.size(), stream);
  device_grad_rgb.copy_from_host(grad_rgb.data(), grad_rgb.size(), stream);
  device_grad_opacity.copy_from_host(grad_opacity.data(), grad_opacity.size(), stream);
  check_cuda(cudaMemsetAsync(device_grad_sigma.data(), 0, sigma.size() * sizeof(float), stream), "cudaMemsetAsync grad_sigma");
  check_cuda(cudaMemsetAsync(device_grad_color.data(), 0, color.size() * sizeof(float), stream), "cudaMemsetAsync grad_color");
  check_cuda(cudaEventRecord(transfer_stop.get(), stream), "cudaEventRecord transfer_stop");
  check_cuda(cudaEventSynchronize(transfer_stop.get()), "cudaEventSynchronize transfer_stop");

  RenderMetrics metrics = time_render_pair(
      stream,
      iterations,
      [&]() {
        svo::render_volume_cuda(
            device_nodes.data(),
            device_nodes.size(),
            device_payload_indices.data(),
            device_payload_indices.size(),
            octree.max_depth(),
            octree.root_bounds(),
            device_origins.data(),
            device_directions.data(),
            device_sigma.data(),
            device_color.data(),
            device_rgb.data(),
            device_depth.data(),
            device_opacity.data(),
            origins.size(),
            sigma.size());
      },
      [&]() {
        svo::render_volume_backward_cuda(
            device_nodes.data(),
            device_nodes.size(),
            device_payload_indices.data(),
            device_payload_indices.size(),
            octree.max_depth(),
            octree.root_bounds(),
            device_origins.data(),
            device_directions.data(),
            device_sigma.data(),
            device_color.data(),
            device_grad_rgb.data(),
            device_grad_opacity.data(),
            device_grad_sigma.data(),
            device_grad_color.data(),
            origins.size(),
            sigma.size());
      });
  metrics.transfer_ms = elapsed_ms(transfer_start.get(), transfer_stop.get());
  return metrics;
}

RenderMetrics benchmark_wide4(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const std::vector<float>& sigma,
    const std::vector<float>& color,
    cudaStream_t stream,
    int iterations) {
  CudaEvent transfer_start;
  CudaEvent transfer_stop;
  auto device_nodes = svo::DeviceBuffer<svo::WideNodeDescriptor>(octree.wide_nodes().size(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>(octree.leaf_payload_indices().size(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>(origins.size(), svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>(directions.size(), svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>(sigma.size(), svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>(color.size(), svo::Device::CUDA);
  auto device_rgb = svo::DeviceBuffer<glm::vec3>(origins.size(), svo::Device::CUDA);
  auto device_depth = svo::DeviceBuffer<float>(origins.size(), svo::Device::CUDA);
  auto device_opacity = svo::DeviceBuffer<float>(origins.size(), svo::Device::CUDA);
  auto device_grad_rgb = svo::DeviceBuffer<glm::vec3>(origins.size(), svo::Device::CUDA);
  auto device_grad_opacity = svo::DeviceBuffer<float>(origins.size(), svo::Device::CUDA);
  auto device_grad_sigma = svo::DeviceBuffer<float>(sigma.size(), svo::Device::CUDA);
  auto device_grad_color = svo::DeviceBuffer<float>(color.size(), svo::Device::CUDA);

  const std::vector<glm::vec3> grad_rgb(origins.size(), glm::vec3{1.0f, 1.0f, 1.0f});
  const std::vector<float> grad_opacity(origins.size(), 1.0f);

  check_cuda(cudaEventRecord(transfer_start.get(), stream), "cudaEventRecord transfer_start");
  device_nodes.copy_from_host(octree.wide_nodes().data(), octree.wide_nodes().size(), stream);
  device_payload_indices.copy_from_host(
      octree.leaf_payload_indices().data(), octree.leaf_payload_indices().size(), stream);
  device_origins.copy_from_host(origins.data(), origins.size(), stream);
  device_directions.copy_from_host(directions.data(), directions.size(), stream);
  device_sigma.copy_from_host(sigma.data(), sigma.size(), stream);
  device_color.copy_from_host(color.data(), color.size(), stream);
  device_grad_rgb.copy_from_host(grad_rgb.data(), grad_rgb.size(), stream);
  device_grad_opacity.copy_from_host(grad_opacity.data(), grad_opacity.size(), stream);
  check_cuda(cudaMemsetAsync(device_grad_sigma.data(), 0, sigma.size() * sizeof(float), stream), "cudaMemsetAsync grad_sigma");
  check_cuda(cudaMemsetAsync(device_grad_color.data(), 0, color.size() * sizeof(float), stream), "cudaMemsetAsync grad_color");
  check_cuda(cudaEventRecord(transfer_stop.get(), stream), "cudaEventRecord transfer_stop");
  check_cuda(cudaEventSynchronize(transfer_stop.get()), "cudaEventSynchronize transfer_stop");

  RenderMetrics metrics = time_render_pair(
      stream,
      iterations,
      [&]() {
        svo::render_volume_wide_cuda(
            device_nodes.data(),
            device_nodes.size(),
            device_payload_indices.data(),
            device_payload_indices.size(),
            octree.max_depth(),
            octree.root_bounds(),
            device_origins.data(),
            device_directions.data(),
            device_sigma.data(),
            device_color.data(),
            device_rgb.data(),
            device_depth.data(),
            device_opacity.data(),
            origins.size(),
            sigma.size());
      },
      [&]() {
        svo::render_volume_backward_wide_cuda(
            device_nodes.data(),
            device_nodes.size(),
            device_payload_indices.data(),
            device_payload_indices.size(),
            octree.max_depth(),
            octree.root_bounds(),
            device_origins.data(),
            device_directions.data(),
            device_sigma.data(),
            device_color.data(),
            device_grad_rgb.data(),
            device_grad_opacity.data(),
            device_grad_sigma.data(),
            device_grad_color.data(),
            origins.size(),
            sigma.size());
      });
  metrics.transfer_ms = elapsed_ms(transfer_start.get(), transfer_stop.get());
  return metrics;
}

void print_metrics(const char* topology, const svo::Octree& octree, std::size_t ray_count, const RenderMetrics& metrics) {
  std::cout << "render benchmark " << topology << '\n';
  std::cout << "  max_depth: " << octree.max_depth() << '\n';
  std::cout << "  nodes: " << octree.num_nodes() << '\n';
  std::cout << "  leaves: " << octree.num_leaves() << '\n';
  std::cout << "  rays: " << ray_count << '\n';
  std::cout << "  transfer_ms: " << metrics.transfer_ms << '\n';
  std::cout << "  forward_kernel_ms: " << metrics.forward_ms << '\n';
  std::cout << "  backward_kernel_ms: " << metrics.backward_ms << '\n';
}

}  // namespace

int main() {
  constexpr int kMaxDepth = 6;
  constexpr std::size_t kRayCount = 1u << 18u;
  constexpr int kIterations = 20;

  svo::BuildOptions build_options;
  build_options.max_depth = kMaxDepth;
  const std::vector<glm::ivec3> coordinates = make_sparse_scene(kMaxDepth);
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, build_options);
  build_options.branching = svo::BranchingMode::Wide4;
  const svo::Octree wide = svo::Octree::from_voxels_cpu(coordinates, build_options);

  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> directions;
  make_rays(kRayCount, origins, directions);

  std::vector<float> sigma;
  std::vector<float> color;
  make_payloads(octree, sigma, color);

  CudaStream stream;
  print_metrics("octree8", octree, origins.size(), benchmark_octree8(octree, origins, directions, sigma, color, stream.get(), kIterations));
  print_metrics("wide4", wide, origins.size(), benchmark_wide4(wide, origins, directions, sigma, color, stream.get(), kIterations));
  return 0;
}
