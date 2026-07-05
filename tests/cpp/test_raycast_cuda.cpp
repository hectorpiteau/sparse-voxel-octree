#include <svo/Builder.hpp>
#include <svo/DeviceBuffer.hpp>
#include <svo/Error.hpp>
#include <svo/Raycast.hpp>

#include <glm/geometric.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#if SVO_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {

constexpr float kTolerance = 1.0e-4f;

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

void require_float_matches(float actual, float expected, const std::string& message) {
  if (std::isnan(expected)) {
    require(std::isnan(actual), message + " should be NaN");
    return;
  }
  if (std::isinf(expected)) {
    require(std::isinf(actual) && std::signbit(actual) == std::signbit(expected), message + " should be inf");
    return;
  }
  require(std::fabs(actual - expected) <= kTolerance, message);
}

void require_vec_matches(const glm::vec3& actual, const glm::vec3& expected, const std::string& message) {
  require_float_matches(actual.x, expected.x, message + " x");
  require_float_matches(actual.y, expected.y, message + " y");
  require_float_matches(actual.z, expected.z, message + " z");
}

struct CudaRaycastResults {
  std::vector<std::uint8_t> hit_mask;
  std::vector<std::int32_t> leaf_ids;
  std::vector<float> t;
  std::vector<glm::vec3> positions;
  std::vector<std::int32_t> depths;
};

CudaRaycastResults raycast_cuda_reference(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const svo::RaycastOptions& options = {}) {
#if SVO_ENABLE_CUDA
  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes =
      svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_origins =
      svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_directions =
      svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  svo::DeviceBuffer<std::uint8_t> device_hit_mask(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_leaf_ids(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_t(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_positions(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_depths(origins.size(), svo::Device::CUDA);

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
      options);

  return {
      device_hit_mask.to_host(),
      device_leaf_ids.to_host(),
      device_t.to_host(),
      device_positions.to_host(),
      device_depths.to_host()};
#else
  (void)octree;
  (void)origins;
  (void)directions;
  (void)options;
  return {};
#endif
}

CudaRaycastResults raycast_cuda_on_stream(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const svo::RaycastOptions& options = {}) {
#if SVO_ENABLE_CUDA
  TestCudaStream stream;
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

  device_nodes.copy_from_host(octree.nodes().data(), octree.nodes().size(), stream.get());
  device_leaf_payload_indices.copy_from_host(
      octree.leaf_payload_indices().data(), octree.leaf_payload_indices().size(), stream.get());
  device_origins.copy_from_host(origins.data(), origins.size(), stream.get());
  device_directions.copy_from_host(directions.data(), directions.size(), stream.get());

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
      options,
      stream.get());

  CudaRaycastResults results;
  results.hit_mask.resize(origins.size());
  results.leaf_ids.resize(origins.size());
  results.t.resize(origins.size());
  results.positions.resize(origins.size());
  results.depths.resize(origins.size());
  device_hit_mask.copy_to_host(results.hit_mask.data(), results.hit_mask.size(), stream.get());
  device_leaf_ids.copy_to_host(results.leaf_ids.data(), results.leaf_ids.size(), stream.get());
  device_t.copy_to_host(results.t.data(), results.t.size(), stream.get());
  device_positions.copy_to_host(results.positions.data(), results.positions.size(), stream.get());
  device_depths.copy_to_host(results.depths.data(), results.depths.size(), stream.get());
  const cudaError_t result = cudaStreamSynchronize(stream.get());
  require(result == cudaSuccess, std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(result));
  return results;
#else
  (void)octree;
  (void)origins;
  (void)directions;
  (void)options;
  return {};
#endif
}

void require_cuda_matches_cpu(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const svo::RaycastOptions& options,
    const std::string& message) {
#if SVO_ENABLE_CUDA
  const svo::RaycastBatch cpu = svo::raycast_cpu(octree, origins, directions, options);
  const CudaRaycastResults cuda = raycast_cuda_reference(octree, origins, directions, options);
  require(cuda.hit_mask.size() == cpu.hit_mask.size(), message + " hit size");
  require(cuda.leaf_ids.size() == cpu.leaf_ids.size(), message + " leaf size");

  for (std::size_t index = 0; index < origins.size(); ++index) {
    require(cuda.hit_mask[index] == cpu.hit_mask[index], message + " hit mask");
    require(cuda.leaf_ids[index] == cpu.leaf_ids[index], message + " leaf id");
    require_float_matches(cuda.t[index], cpu.t[index], message + " t");
    require_vec_matches(cuda.positions[index], cpu.positions[index], message + " position");
    require(cuda.depths[index] == cpu.depths[index], message + " depth");
  }
#else
  (void)octree;
  (void)origins;
  (void)directions;
  (void)options;
  (void)message;
#endif
}

void test_tiny_scenes_and_boundaries() {
#if SVO_ENABLE_CUDA
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu(
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}},
      options);

  const std::vector<glm::vec3> origins{
      {-1.0f, 0.25f, 0.25f},
      {-1.0f, 0.25f, 0.75f},
      {0.75f, 0.25f, 0.25f},
      {-1.0f, -1.0f, -1.0f},
      {-1.0f, 0.5f, 0.25f},
  };
  const std::vector<glm::vec3> directions{
      {1.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {1.0f, 1.0f, 1.0f},
      {1.0f, 0.0f, 0.0f},
  };
  require_cuda_matches_cpu(octree, origins, directions, {}, "tiny scene");

  svo::RaycastOptions payload_options;
  payload_options.return_payload_indices = true;
  require_cuda_matches_cpu(octree, origins, directions, payload_options, "tiny scene payload");
  const CudaRaycastResults stream_results = raycast_cuda_on_stream(octree, origins, directions);
  const svo::RaycastBatch cpu = svo::raycast_cpu(octree, origins, directions);
  require(stream_results.leaf_ids == cpu.leaf_ids, "explicit stream leaf ids should match CPU");
#endif
}

void test_empty_zero_count_custom_bounds_and_depth_zero() {
#if SVO_ENABLE_CUDA
  {
    svo::BuildOptions options;
    options.max_depth = 3;
    const svo::Octree empty = svo::Octree::from_voxels_cpu({}, options);
    require_cuda_matches_cpu(
        empty,
        {{-1.0f, 0.5f, 0.5f}},
        {{1.0f, 0.0f, 0.0f}},
        {},
        "empty tree");
    require_cuda_matches_cpu(empty, {}, {}, {}, "zero ray count");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 1;
    options.root_bounds = {glm::vec3{-1.0f, -1.0f, -1.0f}, glm::vec3{1.0f, 1.0f, 1.0f}};
    const svo::Octree octree = svo::Octree::from_voxels_cpu({{1, 0, 0}}, options);
    require_cuda_matches_cpu(
        octree,
        {{-2.0f, -0.5f, -0.5f}},
        {{1.0f, 0.0f, 0.0f}},
        {},
        "custom root bounds");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 0;
    const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, options);
    require_cuda_matches_cpu(
        octree,
        {{-1.0f, 0.5f, 0.5f}},
        {{2.0f, 0.0f, 0.0f}},
        {},
        "max depth zero");
  }
#endif
}

void test_random_sparse_scene_and_large_batches() {
#if SVO_ENABLE_CUDA
  constexpr int max_depth = 5;
  constexpr int grid_size = 1 << max_depth;
  svo::BuildOptions options;
  options.max_depth = max_depth;

  std::mt19937 rng(20260709u);
  std::bernoulli_distribution keep_dist(0.08);
  std::uniform_real_distribution<float> origin_dist(-0.5f, 1.5f);
  std::uniform_real_distribution<float> direction_dist(-1.0f, 1.0f);

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
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, options);

  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> directions;
  origins.reserve(4096);
  directions.reserve(4096);
  for (int index = 0; index < 2048; ++index) {
    glm::vec3 direction{direction_dist(rng), direction_dist(rng), direction_dist(rng)};
    if (glm::length(direction) <= 1.0e-3f) {
      direction = {1.0f, 0.0f, 0.0f};
    }
    origins.push_back({origin_dist(rng), origin_dist(rng), origin_dist(rng)});
    directions.push_back(direction);
  }
  for (int index = 0; index < 1024; ++index) {
    origins.push_back({-1.0f, 0.5f, 0.5f});
    directions.push_back({-1.0f, 0.0f, 0.0f});
  }
  for (int index = 0; index < 1024; ++index) {
    origins.push_back({0.5f, 0.5f, 0.5f});
    directions.push_back({1.0f, 0.0f, 0.0f});
  }

  require_cuda_matches_cpu(octree, origins, directions, {}, "random large batch");
#endif
}

void test_invalid_directions_write_misses() {
#if SVO_ENABLE_CUDA
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, options);
  const CudaRaycastResults cuda = raycast_cuda_reference(
      octree,
      {{0.25f, 0.25f, 0.25f}, {0.25f, 0.25f, 0.25f}},
      {{0.0f, 0.0f, 0.0f}, {std::numeric_limits<float>::infinity(), 0.0f, 0.0f}});
  require(cuda.hit_mask == std::vector<std::uint8_t>({0u, 0u}), "invalid directions should miss");
  require(cuda.leaf_ids == std::vector<std::int32_t>({-1, -1}), "invalid directions leaf ids");
  require(std::isinf(cuda.t[0]) && std::isinf(cuda.t[1]), "invalid directions t should be inf");
#endif
}

void test_bad_launcher_inputs_fail() {
#if SVO_ENABLE_CUDA
  bool saw_error = false;
  try {
    svo::raycast_cuda(
        nullptr,
        1,
        nullptr,
        0,
        1,
        svo::default_root_bounds(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        1);
  } catch (const svo::ValidationError&) {
    saw_error = true;
  }
  require(saw_error, "bad launcher inputs should fail before launch");
#endif
}

}  // namespace

int main() {
  test_tiny_scenes_and_boundaries();
  test_empty_zero_count_custom_bounds_and_depth_zero();
  test_random_sparse_scene_and_large_batches();
  test_invalid_directions_write_misses();
  test_bad_launcher_inputs_fail();
  return 0;
}
