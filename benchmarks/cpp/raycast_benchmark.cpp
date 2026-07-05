#include <svo/Builder.hpp>
#include <svo/DeviceBuffer.hpp>
#include <svo/Raycast.hpp>

#include <cuda_runtime_api.h>
#include <glm/geometric.hpp>

#include <chrono>
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
  CudaEvent() {
    check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
  }

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
  CudaStream() {
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  }

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
  std::mt19937 rng(20260709u);
  std::bernoulli_distribution keep_dist(0.035);

  std::vector<glm::ivec3> coordinates;
  for (int z = 0; z < grid_size; ++z) {
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        const bool in_box = x >= grid_size / 4 && x < (3 * grid_size) / 4 && y >= grid_size / 4 &&
            y < (3 * grid_size) / 4 && z >= grid_size / 4 && z < (3 * grid_size) / 4;
        if (in_box && keep_dist(rng)) {
          coordinates.emplace_back(x, y, z);
        }
      }
    }
  }
  return coordinates;
}

void make_rays(std::size_t count, std::vector<glm::vec3>& origins, std::vector<glm::vec3>& directions) {
  std::mt19937 rng(20260710u);
  std::uniform_real_distribution<float> uv_dist(0.05f, 0.95f);
  std::uniform_real_distribution<float> jitter_dist(-0.15f, 0.15f);

  origins.clear();
  directions.clear();
  origins.reserve(count);
  directions.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const float y = uv_dist(rng);
    const float z = uv_dist(rng);
    origins.push_back({-0.25f, y, z});
    directions.push_back({1.0f, jitter_dist(rng), jitter_dist(rng)});
  }
}

}  // namespace

int main() {
  constexpr int kMaxDepth = 6;
  constexpr std::size_t kRayCount = 1u << 20u;
  constexpr int kIterations = 20;

  svo::BuildOptions build_options;
  build_options.max_depth = kMaxDepth;
  const std::vector<glm::ivec3> coordinates = make_sparse_scene(kMaxDepth);
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, build_options);

  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> directions;
  make_rays(kRayCount, origins, directions);

  CudaStream stream;
  CudaEvent transfer_start;
  CudaEvent transfer_stop;
  CudaEvent kernel_start;
  CudaEvent kernel_stop;

  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes(octree.nodes().size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices(
      octree.leaf_payload_indices().size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_origins(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_directions(directions.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint8_t> device_hit_mask(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_leaf_ids(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_t(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_positions(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_depths(origins.size(), svo::Device::CUDA);

  check_cuda(cudaEventRecord(transfer_start.get(), stream.get()), "cudaEventRecord transfer_start");
  device_nodes.copy_from_host(octree.nodes().data(), octree.nodes().size(), stream.get());
  device_leaf_payload_indices.copy_from_host(
      octree.leaf_payload_indices().data(), octree.leaf_payload_indices().size(), stream.get());
  device_origins.copy_from_host(origins.data(), origins.size(), stream.get());
  device_directions.copy_from_host(directions.data(), directions.size(), stream.get());
  check_cuda(cudaEventRecord(transfer_stop.get(), stream.get()), "cudaEventRecord transfer_stop");
  check_cuda(cudaEventSynchronize(transfer_stop.get()), "cudaEventSynchronize transfer_stop");

  svo::raycast_cuda(
      device_nodes.data(),
      device_nodes.size(),
      device_leaf_payload_indices.data(),
      device_leaf_payload_indices.size(),
      octree.max_depth(),
      octree.root_bounds(),
      device_origins.data(),
      device_directions.data(),
      device_hit_mask.data(),
      device_leaf_ids.data(),
      device_t.data(),
      device_positions.data(),
      device_depths.data(),
      origins.size(),
      {},
      stream.get());
  check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize warmup");

  check_cuda(cudaEventRecord(kernel_start.get(), stream.get()), "cudaEventRecord kernel_start");
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    svo::raycast_cuda(
        device_nodes.data(),
        device_nodes.size(),
        device_leaf_payload_indices.data(),
        device_leaf_payload_indices.size(),
        octree.max_depth(),
        octree.root_bounds(),
        device_origins.data(),
        device_directions.data(),
        device_hit_mask.data(),
        device_leaf_ids.data(),
        device_t.data(),
        device_positions.data(),
        device_depths.data(),
        origins.size(),
        {},
        stream.get());
  }
  check_cuda(cudaEventRecord(kernel_stop.get(), stream.get()), "cudaEventRecord kernel_stop");
  check_cuda(cudaEventSynchronize(kernel_stop.get()), "cudaEventSynchronize kernel_stop");

  std::vector<std::uint8_t> hit_mask = device_hit_mask.to_host(stream.get());
  check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize hit copy");

  std::size_t hits = 0;
  for (std::uint8_t value : hit_mask) {
    hits += value != 0u ? 1u : 0u;
  }

  const float transfer_ms_value = elapsed_ms(transfer_start.get(), transfer_stop.get());
  const float kernel_ms_total = elapsed_ms(kernel_start.get(), kernel_stop.get());
  const float kernel_ms = kernel_ms_total / static_cast<float>(kIterations);
  const double rays_per_second =
      (static_cast<double>(origins.size()) / static_cast<double>(kernel_ms)) * 1000.0;

  std::cout << "raycast benchmark\n";
  std::cout << "  max_depth: " << kMaxDepth << '\n';
  std::cout << "  nodes: " << octree.num_nodes() << '\n';
  std::cout << "  leaves: " << octree.num_leaves() << '\n';
  std::cout << "  rays: " << origins.size() << '\n';
  std::cout << "  hit_rate: " << (static_cast<double>(hits) / static_cast<double>(origins.size())) << '\n';
  std::cout << "  transfer_ms: " << transfer_ms_value << '\n';
  std::cout << "  kernel_ms: " << kernel_ms << '\n';
  std::cout << "  rays_per_second: " << rays_per_second << '\n';

  return 0;
}
