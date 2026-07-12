#include <svo/DeviceBuffer.hpp>
#include <svo/Octree.hpp>
#include <svo/Query.hpp>

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_int3.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if SVO_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}


#if SVO_ENABLE_CUDA
class TestCudaStream {
 public:
  TestCudaStream() {
    const cudaError_t result = cudaStreamCreate(&stream_);
    require(result == cudaSuccess, std::string("cudaStreamCreate failed: ") + cudaGetErrorString(result));
  }

  ~TestCudaStream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  TestCudaStream(const TestCudaStream&) = delete;
  TestCudaStream& operator=(const TestCudaStream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};
#endif

glm::vec3 voxel_center_point(int grid_size, const glm::ivec3& coordinate) {
  const float inv_grid = 1.0f / static_cast<float>(grid_size);
  return glm::vec3{
      (static_cast<float>(coordinate.x) + 0.5f) * inv_grid,
      (static_cast<float>(coordinate.y) + 0.5f) * inv_grid,
      (static_cast<float>(coordinate.z) + 0.5f) * inv_grid};
}

std::vector<std::int32_t> query_points_cuda_reference(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& points,
    const svo::QueryOptions& options = {}) {
#if SVO_ENABLE_CUDA
  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes =
      svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_points =
      svo::DeviceBuffer<glm::vec3>::from_host(points, svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_results(points.size(), svo::Device::CUDA);

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
      options);

  return device_results.to_host();
#else
  (void)octree;
  (void)points;
  (void)options;
  return {};
#endif
}

std::vector<std::int32_t> query_points_wide_cuda_reference(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& points,
    const svo::QueryOptions& options = {}) {
#if SVO_ENABLE_CUDA
  svo::DeviceBuffer<svo::WideNodeDescriptor> device_nodes =
      svo::DeviceBuffer<svo::WideNodeDescriptor>::from_host(octree.wide_nodes(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_points =
      svo::DeviceBuffer<glm::vec3>::from_host(points, svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_results(points.size(), svo::Device::CUDA);

  svo::query_points_wide_cuda(
      device_nodes.data(),
      device_nodes.size(),
      device_leaf_payload_indices.data(),
      device_leaf_payload_indices.size(),
      octree.max_depth(),
      octree.root_bounds(),
      device_points.data(),
      device_results.data(),
      device_results.size(),
      options);

  return device_results.to_host();
#else
  (void)octree;
  (void)points;
  (void)options;
  return {};
#endif
}


std::vector<std::int32_t> query_points_cuda_on_stream(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& points,
    const svo::QueryOptions& options = {}) {
#if SVO_ENABLE_CUDA
  TestCudaStream stream;
  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes(octree.nodes().size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices(
      octree.leaf_payload_indices().size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_points(points.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_results(points.size(), svo::Device::CUDA);
  std::vector<std::int32_t> results(points.size());

  device_nodes.copy_from_host(octree.nodes().data(), octree.nodes().size(), stream.get());
  device_leaf_payload_indices.copy_from_host(
      octree.leaf_payload_indices().data(), octree.leaf_payload_indices().size(), stream.get());
  device_points.copy_from_host(points.data(), points.size(), stream.get());

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
      options,
      stream.get());

  device_results.copy_to_host(results.data(), results.size(), stream.get());
  const cudaError_t result = cudaStreamSynchronize(stream.get());
  require(result == cudaSuccess, std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(result));
  return results;
#else
  (void)octree;
  (void)points;
  (void)options;
  return {};
#endif
}

void require_cuda_matches_cpu(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& points,
    const svo::QueryOptions& options,
    const std::string& message) {
#if SVO_ENABLE_CUDA
  const std::vector<std::int32_t> cpu_results = svo::query_points(octree, points, options);
  const std::vector<std::int32_t> cuda_results = query_points_cuda_reference(octree, points, options);
  require(cuda_results == cpu_results, message);
#else
  (void)octree;
  (void)points;
  (void)options;
  (void)message;
#endif
}

void test_small_deterministic_tree() {
#if SVO_ENABLE_CUDA
  svo::BuildOptions build_options;
  build_options.max_depth = 3;

  const std::vector<glm::ivec3> coordinates{
      {0, 0, 0},
      {4, 4, 4},
      {7, 7, 7},
  };
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, build_options);

  const std::vector<glm::vec3> points{
      voxel_center_point(8, {0, 0, 0}),
      voxel_center_point(8, {4, 4, 4}),
      voxel_center_point(8, {1, 1, 1}),
      {-0.1f, 0.2f, 0.2f},
      {1.0f, 0.5f, 0.5f},
      {0.0f, 0.0f, 0.0f},
      {0.999999f, 0.999999f, 0.999999f},
      {0.5f, 0.5f, 0.5f},
  };

  require_cuda_matches_cpu(octree, points, {}, "CUDA deterministic query should match CPU");
  require(
      query_points_cuda_on_stream(octree, points) == svo::query_points(octree, points),
      "CUDA deterministic query on explicit stream should match CPU");

  svo::QueryOptions payload_options;
  payload_options.return_payload_indices = true;
  require_cuda_matches_cpu(octree, points, payload_options, "CUDA payload query should match CPU");

  const svo::Octree remapped_octree = svo::Octree::from_voxels_cpu(
      {glm::ivec3{3, 3, 3}, glm::ivec3{0, 0, 0}, glm::ivec3{2, 1, 0}},
      {42u, 7u, 99u},
      build_options);
  const std::vector<glm::vec3> remapped_points{
      voxel_center_point(8, {0, 0, 0}),
      voxel_center_point(8, {2, 1, 0}),
      voxel_center_point(8, {3, 3, 3}),
      voxel_center_point(8, {1, 1, 1}),
  };
  require_cuda_matches_cpu(
      remapped_octree,
      remapped_points,
      payload_options,
      "CUDA custom payload query should match CPU");
#endif
}

void test_grid_shaped_batch() {
#if SVO_ENABLE_CUDA
  constexpr int max_depth = 5;
  constexpr int grid_size = 1 << max_depth;
  constexpr int image_width = 48;
  constexpr int image_height = 32;

  svo::BuildOptions build_options;
  build_options.max_depth = max_depth;

  std::vector<glm::ivec3> coordinates;
  for (int z = 8; z < 24; ++z) {
    for (int y = 6; y < 26; ++y) {
      for (int x = 10; x < 22; ++x) {
        if (((x + y + z) % 5) == 0) {
          coordinates.emplace_back(x, y, z);
        }
      }
    }
  }

  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, build_options);

  std::vector<glm::vec3> points;
  points.reserve(static_cast<std::size_t>(image_width * image_height));
  for (int row = 0; row < image_height; ++row) {
    for (int col = 0; col < image_width; ++col) {
      const float x = (static_cast<float>(col) + 0.5f) / static_cast<float>(image_width);
      const float y = (static_cast<float>(row) + 0.5f) / static_cast<float>(image_height);
      points.push_back({x, y, 0.5f});
    }
  }

  require_cuda_matches_cpu(octree, points, {}, "CUDA grid-shaped point batch should match CPU");
#endif
}

void test_random_tree_batch() {
#if SVO_ENABLE_CUDA
  constexpr int max_depth = 5;
  constexpr int grid_size = 1 << max_depth;

  svo::BuildOptions build_options;
  build_options.max_depth = max_depth;

  std::mt19937 rng(20260701u);
  std::uniform_int_distribution<int> coord_dist(0, grid_size - 1);
  std::bernoulli_distribution keep_dist(0.08);

  std::vector<glm::ivec3> coordinates;
  for (int z = 0; z < grid_size; ++z) {
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        if (keep_dist(rng)) {
          coordinates.emplace_back(x, y, z);
        }
      }
    }
  }

  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, build_options);

  std::vector<glm::vec3> points;
  points.reserve(2048);
  for (int index = 0; index < 2048; ++index) {
    points.push_back(voxel_center_point(grid_size, {coord_dist(rng), coord_dist(rng), coord_dist(rng)}));
  }
  points.push_back({-0.01f, 0.5f, 0.5f});
  points.push_back({1.0f, 0.5f, 0.5f});

  require_cuda_matches_cpu(octree, points, {}, "CUDA random batch should match CPU");
#endif
}

void test_empty_and_zero_count_queries() {
#if SVO_ENABLE_CUDA
  svo::BuildOptions build_options;
  build_options.max_depth = 4;
  const svo::Octree empty_octree = svo::Octree::from_voxels_cpu({}, build_options);

  const std::vector<glm::vec3> points{
      {0.25f, 0.25f, 0.25f},
      {0.75f, 0.75f, 0.75f},
  };
  require_cuda_matches_cpu(empty_octree, points, {}, "CUDA empty tree query should match CPU");

  const std::vector<glm::vec3> no_points;
  require_cuda_matches_cpu(empty_octree, no_points, {}, "CUDA zero-count query should match CPU");
#endif
}

void test_wide_query_matches_cpu() {
#if SVO_ENABLE_CUDA
  svo::BuildOptions build_options;
  build_options.max_depth = 4;
  build_options.branching = svo::BranchingMode::Wide4;
  const svo::Octree octree = svo::Octree::from_voxels_cpu(
      {glm::ivec3{0, 0, 0}, glm::ivec3{1, 2, 3}, glm::ivec3{4, 4, 4}, glm::ivec3{15, 15, 15}},
      {10u, 11u, 12u, 13u},
      build_options);

  const std::vector<glm::vec3> points{
      voxel_center_point(16, {0, 0, 0}),
      voxel_center_point(16, {1, 2, 3}),
      voxel_center_point(16, {4, 4, 4}),
      voxel_center_point(16, {15, 15, 15}),
      voxel_center_point(16, {2, 2, 2}),
      {1.0f, 0.5f, 0.5f},
  };

  require(
      query_points_wide_cuda_reference(octree, points) == svo::query_points(octree, points),
      "wide CUDA query should match wide CPU query");

  svo::QueryOptions payload_options;
  payload_options.return_payload_indices = true;
  require(
      query_points_wide_cuda_reference(octree, points, payload_options) ==
          svo::query_points(octree, points, payload_options),
      "wide CUDA payload query should match wide CPU query");
#endif
}

}  // namespace

int main() {
  test_small_deterministic_tree();
  test_grid_shaped_batch();
  test_random_tree_batch();
  test_empty_and_zero_count_queries();
  test_wide_query_matches_cpu();
  return 0;
}
