#include <svo/DeviceBuffer.hpp>
#include <svo/Octree.hpp>
#include <svo/Renderer.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void require_close(float actual, float expected, float tolerance, const std::string& message) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
    std::exit(1);
  }
}

void require_vec_close(const glm::vec3& actual, const glm::vec3& expected, float tolerance, const std::string& message) {
  require_close(actual.x, expected.x, tolerance, message + " x");
  require_close(actual.y, expected.y, tolerance, message + " y");
  require_close(actual.z, expected.z, tolerance, message + " z");
}

void compare_cuda_to_cpu(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const std::vector<float>& sigma,
    const std::vector<float>& color,
    const svo::RenderOptions& options = {}) {
  const svo::RenderBatch cpu = svo::render_volume_cpu(
      octree, origins, directions, sigma.data(), color.data(), sigma.size(), options);

  auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>::from_host(sigma, svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>::from_host(color, svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_depth(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_opacity(origins.size(), svo::Device::CUDA);

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
      sigma.size(),
      options);

  const std::vector<glm::vec3> cuda_rgb = device_rgb.to_host();
  const std::vector<float> cuda_depth = device_depth.to_host();
  const std::vector<float> cuda_opacity = device_opacity.to_host();
  require(cuda_rgb.size() == cpu.rgb.size(), "CUDA rgb size");
  for (std::size_t index = 0; index < origins.size(); ++index) {
    require_vec_close(cuda_rgb[index], cpu.rgb[index], 2.0e-5f, "CUDA rgb should match CPU");
    if (std::isinf(cpu.depth[index])) {
      require(std::isinf(cuda_depth[index]), "CUDA depth should be inf");
    } else {
      require_close(cuda_depth[index], cpu.depth[index], 2.0e-5f, "CUDA depth should match CPU");
    }
    require_close(cuda_opacity[index], cpu.opacity[index], 2.0e-5f, "CUDA opacity should match CPU");
  }
}

void test_cuda_matches_cpu_for_dense_tree() {
  svo::BuildOptions build_options;
  build_options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu(
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}},
      build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.25f, 0.25f}, {-1.0f, 0.75f, 0.75f}, {0.25f, 0.25f, -1.0f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
  const std::vector<float> sigma{0.2f, 0.5f, 0.8f, 1.1f, 1.4f, 1.7f, 2.0f, 2.3f};
  const std::vector<float> color{
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f,
      1.0f, 1.0f, 0.0f,
      1.0f, 0.0f, 1.0f,
      0.0f, 1.0f, 1.0f,
      0.5f, 0.5f, 0.5f,
      1.0f, 1.0f, 1.0f};
  svo::RenderOptions options;
  options.background_color = {0.01f, 0.02f, 0.03f};
  compare_cuda_to_cpu(octree, origins, directions, sigma, color, options);
}

void test_cuda_matches_cpu_for_sparse_tree_and_clipping() {
  svo::BuildOptions build_options;
  build_options.max_depth = 2;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {3, 0, 0}, {3, 3, 3}}, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.125f, 0.125f}, {-1.0f, 0.875f, 0.875f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{1.0f, 2.0f, 3.0f};
  const std::vector<float> color{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  svo::RenderOptions options;
  options.near_plane = 0.5f;
  options.far_plane = 2.0f;
  compare_cuda_to_cpu(octree, origins, directions, sigma, color, options);
}

}  // namespace

int main() {
  test_cuda_matches_cpu_for_dense_tree();
  test_cuda_matches_cpu_for_sparse_tree_and_clipping();
  return 0;
}
