#include <svo/DeviceBuffer.hpp>
#include <svo/Octree.hpp>
#include <svo/Renderer.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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

void compare_wide_cuda_to_cpu(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const std::vector<float>& sigma,
    const std::vector<float>& color,
    const svo::RenderOptions& options = {}) {
  const svo::RenderBatch cpu = svo::render_volume_cpu(
      octree, origins, directions, sigma.data(), color.data(), sigma.size(), options);

  auto device_nodes = svo::DeviceBuffer<svo::WideNodeDescriptor>::from_host(octree.wide_nodes(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>::from_host(sigma, svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>::from_host(color, svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_depth(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_opacity(origins.size(), svo::Device::CUDA);

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
      sigma.size(),
      options);

  const std::vector<glm::vec3> cuda_rgb = device_rgb.to_host();
  const std::vector<float> cuda_depth = device_depth.to_host();
  const std::vector<float> cuda_opacity = device_opacity.to_host();
  require(cuda_rgb.size() == cpu.rgb.size(), "wide CUDA rgb size");
  for (std::size_t index = 0; index < origins.size(); ++index) {
    require_vec_close(cuda_rgb[index], cpu.rgb[index], 2.0e-5f, "wide CUDA rgb should match CPU");
    if (std::isinf(cpu.depth[index])) {
      require(std::isinf(cuda_depth[index]), "wide CUDA depth should be inf");
    } else {
      require_close(cuda_depth[index], cpu.depth[index], 2.0e-5f, "wide CUDA depth should match CPU");
    }
    require_close(cuda_opacity[index], cpu.opacity[index], 2.0e-5f, "wide CUDA opacity should match CPU");
  }
}

void compare_intervals_to_direct(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const std::vector<float>& sigma,
    const std::vector<float>& color,
    const svo::RenderOptions& options = {}) {
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>::from_host(sigma, svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>::from_host(color, svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> direct_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> direct_depth(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> direct_opacity(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> interval_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> interval_depth(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> interval_opacity(origins.size(), svo::Device::CUDA);
  svo::RenderIntervalBuffer intervals;

  if (octree.branching() == svo::BranchingMode::Wide4) {
    auto device_nodes = svo::DeviceBuffer<svo::WideNodeDescriptor>::from_host(octree.wide_nodes(), svo::Device::CUDA);
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
        direct_rgb.data(),
        direct_depth.data(),
        direct_opacity.data(),
        origins.size(),
        sigma.size(),
        options);
    svo::build_render_intervals_wide_cuda(
        device_nodes.data(),
        device_nodes.size(),
        device_payload_indices.data(),
        device_payload_indices.size(),
        octree.max_depth(),
        octree.root_bounds(),
        device_origins.data(),
        device_directions.data(),
        origins.size(),
        options,
        intervals);
  } else {
    auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
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
        direct_rgb.data(),
        direct_depth.data(),
        direct_opacity.data(),
        origins.size(),
        sigma.size(),
        options);
    svo::build_render_intervals_cuda(
        device_nodes.data(),
        device_nodes.size(),
        device_payload_indices.data(),
        device_payload_indices.size(),
        octree.max_depth(),
        octree.root_bounds(),
        device_origins.data(),
        device_directions.data(),
        origins.size(),
        options,
        intervals);
  }

  svo::render_volume_from_intervals_cuda(
      intervals,
      device_sigma.data(),
      device_color.data(),
      interval_rgb.data(),
      interval_depth.data(),
      interval_opacity.data(),
      origins.size(),
      sigma.size(),
      options);

  const std::vector<glm::vec3> direct_rgb_host = direct_rgb.to_host();
  const std::vector<float> direct_depth_host = direct_depth.to_host();
  const std::vector<float> direct_opacity_host = direct_opacity.to_host();
  const std::vector<glm::vec3> interval_rgb_host = interval_rgb.to_host();
  const std::vector<float> interval_depth_host = interval_depth.to_host();
  const std::vector<float> interval_opacity_host = interval_opacity.to_host();
  for (std::size_t index = 0; index < origins.size(); ++index) {
    require_vec_close(interval_rgb_host[index], direct_rgb_host[index], 2.0e-5f, "interval rgb should match direct");
    if (std::isinf(direct_depth_host[index])) {
      require(std::isinf(interval_depth_host[index]), "interval depth should be inf");
    } else {
      require_close(interval_depth_host[index], direct_depth_host[index], 2.0e-5f, "interval depth should match direct");
    }
    require_close(interval_opacity_host[index], direct_opacity_host[index], 2.0e-5f, "interval opacity should match direct");
  }
}

void compare_interval_backward_to_direct(
    const svo::Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const std::vector<float>& sigma,
    const std::vector<float>& color,
    const std::vector<glm::vec3>& grad_rgb,
    const std::vector<float>& grad_opacity,
    const svo::RenderOptions& options = {}) {
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>::from_host(sigma, svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>::from_host(color, svo::Device::CUDA);
  auto device_grad_rgb = svo::DeviceBuffer<glm::vec3>::from_host(grad_rgb, svo::Device::CUDA);
  auto device_grad_opacity = svo::DeviceBuffer<float>::from_host(grad_opacity, svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> interval_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> interval_depth(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> interval_opacity(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> direct_grad_sigma(sigma.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> direct_grad_color(color.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> interval_grad_sigma(sigma.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> interval_grad_color(color.size(), svo::Device::CUDA);
  const std::vector<float> zero_sigma(sigma.size(), 0.0f);
  const std::vector<float> zero_color(color.size(), 0.0f);
  direct_grad_sigma.copy_from_host(zero_sigma.data(), zero_sigma.size());
  direct_grad_color.copy_from_host(zero_color.data(), zero_color.size());
  interval_grad_sigma.copy_from_host(zero_sigma.data(), zero_sigma.size());
  interval_grad_color.copy_from_host(zero_color.data(), zero_color.size());
  svo::RenderIntervalBuffer intervals;

  if (octree.branching() == svo::BranchingMode::Wide4) {
    auto device_nodes = svo::DeviceBuffer<svo::WideNodeDescriptor>::from_host(octree.wide_nodes(), svo::Device::CUDA);
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
        direct_grad_sigma.data(),
        direct_grad_color.data(),
        origins.size(),
        sigma.size(),
        options);
    svo::build_render_intervals_wide_cuda(
        device_nodes.data(),
        device_nodes.size(),
        device_payload_indices.data(),
        device_payload_indices.size(),
        octree.max_depth(),
        octree.root_bounds(),
        device_origins.data(),
        device_directions.data(),
        origins.size(),
        options,
        intervals);
  } else {
    auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
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
        direct_grad_sigma.data(),
        direct_grad_color.data(),
        origins.size(),
        sigma.size(),
        options);
    svo::build_render_intervals_cuda(
        device_nodes.data(),
        device_nodes.size(),
        device_payload_indices.data(),
        device_payload_indices.size(),
        octree.max_depth(),
        octree.root_bounds(),
        device_origins.data(),
        device_directions.data(),
        origins.size(),
        options,
        intervals);
  }

  svo::render_volume_from_intervals_cuda(
      intervals,
      device_sigma.data(),
      device_color.data(),
      interval_rgb.data(),
      interval_depth.data(),
      interval_opacity.data(),
      origins.size(),
      sigma.size(),
      options);
  svo::render_volume_backward_from_intervals_cuda(
      intervals,
      device_sigma.data(),
      device_color.data(),
      device_grad_rgb.data(),
      device_grad_opacity.data(),
      interval_grad_sigma.data(),
      interval_grad_color.data(),
      origins.size(),
      sigma.size(),
      options);

  const std::vector<float> direct_sigma = direct_grad_sigma.to_host();
  const std::vector<float> direct_color = direct_grad_color.to_host();
  const std::vector<float> interval_sigma = interval_grad_sigma.to_host();
  const std::vector<float> interval_color = interval_grad_color.to_host();
  for (std::size_t index = 0; index < sigma.size(); ++index) {
    require_close(interval_sigma[index], direct_sigma[index], 3.0e-5f, "interval backward sigma should match direct");
  }
  for (std::size_t index = 0; index < color.size(); ++index) {
    require_close(interval_color[index], direct_color[index], 3.0e-5f, "interval backward color should match direct");
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

void test_wide_cuda_matches_cpu() {
  svo::BuildOptions build_options;
  build_options.max_depth = 4;
  build_options.branching = svo::BranchingMode::Wide4;
  const svo::Octree octree = svo::Octree::from_voxels_cpu(
      {{0, 0, 0}, {4, 4, 4}, {15, 15, 15}},
      build_options);
  const std::vector<glm::vec3> origins{
      {-1.0f, 0.03125f, 0.03125f},
      {-1.0f, 0.28125f, 0.28125f},
      {2.0f, 0.96875f, 0.96875f},
      {-1.0f, 0.5f, 0.5f}};
  const std::vector<glm::vec3> directions{
      {1.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {-1.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{1.0f, 2.0f, 3.0f};
  const std::vector<float> color{
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  svo::RenderOptions options;
  options.background_color = {0.01f, 0.02f, 0.03f};
  compare_wide_cuda_to_cpu(octree, origins, directions, sigma, color, options);
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

void test_interval_forward_matches_direct_octree8() {
  svo::BuildOptions build_options;
  build_options.max_depth = 2;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {1, 1, 1}, {3, 3, 3}}, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.125f, 0.125f}, {-1.0f, 0.375f, 0.375f}, {2.0f, 0.875f, 0.875f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{1.0f, 2.0f, 3.0f};
  const std::vector<float> color{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  svo::RenderOptions options;
  options.background_color = {0.01f, 0.02f, 0.03f};
  compare_intervals_to_direct(octree, origins, directions, sigma, color, options);
}

void test_interval_forward_matches_direct_wide4() {
  svo::BuildOptions build_options;
  build_options.max_depth = 4;
  build_options.branching = svo::BranchingMode::Wide4;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {4, 4, 4}, {15, 15, 15}}, build_options);
  const std::vector<glm::vec3> origins{
      {-1.0f, 0.03125f, 0.03125f},
      {-1.0f, 0.28125f, 0.28125f},
      {2.0f, 0.96875f, 0.96875f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{1.0f, 2.0f, 3.0f};
  const std::vector<float> color{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  compare_intervals_to_direct(octree, origins, directions, sigma, color);
}

void test_interval_forward_handles_more_than_fixed_segment_cache() {
  svo::BuildOptions build_options;
  build_options.max_depth = 10;
  std::vector<glm::ivec3> coords;
  coords.reserve(1024);
  for (int x = 0; x < 1024; ++x) {
    coords.push_back({x, 512, 512});
  }
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coords, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 512.5f / 1024.0f, 512.5f / 1024.0f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}};
  std::vector<float> sigma(coords.size(), 0.01f);
  std::vector<float> color(coords.size() * 3u, 0.25f);
  compare_intervals_to_direct(octree, origins, directions, sigma, color);
}

void test_interval_backward_matches_direct_octree8() {
  svo::BuildOptions build_options;
  build_options.max_depth = 2;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {1, 1, 1}, {3, 3, 3}}, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.125f, 0.125f}, {-1.0f, 0.375f, 0.375f}, {2.0f, 0.875f, 0.875f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{1.0f, 2.0f, 3.0f};
  const std::vector<float> color{1.0f, 0.2f, 0.0f, 0.0f, 1.0f, 0.4f, 0.0f, 0.0f, 1.0f};
  const std::vector<glm::vec3> grad_rgb{{0.7f, 0.2f, 0.1f}, {0.3f, 0.8f, 0.5f}, {0.4f, 0.1f, 0.9f}};
  const std::vector<float> grad_opacity{0.4f, 0.5f, 0.6f};
  svo::RenderOptions options;
  options.background_color = {0.01f, 0.02f, 0.03f};
  compare_interval_backward_to_direct(octree, origins, directions, sigma, color, grad_rgb, grad_opacity, options);
}

void test_interval_backward_matches_direct_wide4() {
  svo::BuildOptions build_options;
  build_options.max_depth = 4;
  build_options.branching = svo::BranchingMode::Wide4;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {4, 4, 4}, {15, 15, 15}}, build_options);
  const std::vector<glm::vec3> origins{
      {-1.0f, 0.03125f, 0.03125f},
      {-1.0f, 0.28125f, 0.28125f},
      {2.0f, 0.96875f, 0.96875f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{1.0f, 2.0f, 3.0f};
  const std::vector<float> color{1.0f, 0.2f, 0.0f, 0.0f, 1.0f, 0.4f, 0.0f, 0.0f, 1.0f};
  const std::vector<glm::vec3> grad_rgb{{0.7f, 0.2f, 0.1f}, {0.3f, 0.8f, 0.5f}, {0.4f, 0.1f, 0.9f}};
  const std::vector<float> grad_opacity{0.4f, 0.5f, 0.6f};
  compare_interval_backward_to_direct(octree, origins, directions, sigma, color, grad_rgb, grad_opacity);
}

void test_interval_backward_matches_direct_with_early_stop_and_negative_sigma() {
  svo::BuildOptions build_options;
  build_options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {1, 0, 0}}, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.25f, 0.25f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{10.0f, -1.0f};
  const std::vector<float> color{1.0f, 0.2f, 0.0f, 0.0f, 1.0f, 0.4f};
  const std::vector<glm::vec3> grad_rgb{{0.7f, 0.2f, 0.1f}};
  const std::vector<float> grad_opacity{0.4f};
  svo::RenderOptions options;
  options.early_stop_transmittance = 0.1f;
  compare_interval_backward_to_direct(octree, origins, directions, sigma, color, grad_rgb, grad_opacity, options);
}

void test_interval_backward_matches_direct_empty_and_zero_rays() {
  svo::BuildOptions build_options;
  build_options.max_depth = 1;
  const svo::Octree empty = svo::Octree::from_voxels_cpu({}, build_options);
  compare_interval_backward_to_direct(
      empty,
      {{-1.0f, 0.25f, 0.25f}},
      {{1.0f, 0.0f, 0.0f}},
      {},
      {},
      {{0.7f, 0.2f, 0.1f}},
      {0.4f});

  const svo::Octree single = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);
  compare_interval_backward_to_direct(single, {}, {}, {1.0f}, {1.0f, 0.0f, 0.0f}, {}, {});
}

void test_interval_backward_before_forward_fails_clearly() {
  svo::BuildOptions build_options;
  build_options.max_depth = 0;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.5f, 0.5f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{1.0f};
  const std::vector<float> color{1.0f, 0.0f, 0.0f};
  const std::vector<glm::vec3> grad_rgb{{1.0f, 1.0f, 1.0f}};
  const std::vector<float> grad_opacity{1.0f};
  const std::vector<float> zero_sigma{0.0f};
  const std::vector<float> zero_color{0.0f, 0.0f, 0.0f};

  auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>::from_host(sigma, svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>::from_host(color, svo::Device::CUDA);
  auto device_grad_rgb = svo::DeviceBuffer<glm::vec3>::from_host(grad_rgb, svo::Device::CUDA);
  auto device_grad_opacity = svo::DeviceBuffer<float>::from_host(grad_opacity, svo::Device::CUDA);
  auto device_grad_sigma = svo::DeviceBuffer<float>::from_host(zero_sigma, svo::Device::CUDA);
  auto device_grad_color = svo::DeviceBuffer<float>::from_host(zero_color, svo::Device::CUDA);
  svo::RenderIntervalBuffer intervals;
  svo::build_render_intervals_cuda(
      device_nodes.data(),
      device_nodes.size(),
      device_payload_indices.data(),
      device_payload_indices.size(),
      octree.max_depth(),
      octree.root_bounds(),
      device_origins.data(),
      device_directions.data(),
      origins.size(),
      {},
      intervals);

  bool failed = false;
  try {
    svo::render_volume_backward_from_intervals_cuda(
        intervals,
        device_sigma.data(),
        device_color.data(),
        device_grad_rgb.data(),
        device_grad_opacity.data(),
        device_grad_sigma.data(),
        device_grad_color.data(),
        origins.size(),
        sigma.size());
  } catch (const svo::ValidationError& error) {
    failed = std::string(error.what()).find("run interval forward before backward") != std::string::npos;
  }
  require(failed, "interval backward before forward should fail clearly");
}


void test_cuda_backward_single_leaf_matches_analytic_gradient() {
  svo::BuildOptions build_options;
  build_options.max_depth = 0;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.5f, 0.5f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{2.0f};
  const std::vector<float> color{0.8f, 0.2f, 0.1f};
  const std::vector<glm::vec3> grad_rgb{{1.0f, 1.0f, 1.0f}};
  const std::vector<float> grad_opacity{1.0f};
  const std::vector<float> zeros_sigma{0.0f};
  const std::vector<float> zeros_color{0.0f, 0.0f, 0.0f};

  auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>::from_host(sigma, svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>::from_host(color, svo::Device::CUDA);
  auto device_grad_rgb = svo::DeviceBuffer<glm::vec3>::from_host(grad_rgb, svo::Device::CUDA);
  auto device_grad_opacity = svo::DeviceBuffer<float>::from_host(grad_opacity, svo::Device::CUDA);
  auto device_grad_sigma = svo::DeviceBuffer<float>::from_host(zeros_sigma, svo::Device::CUDA);
  auto device_grad_color = svo::DeviceBuffer<float>::from_host(zeros_color, svo::Device::CUDA);

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

  const std::vector<float> actual_sigma = device_grad_sigma.to_host();
  const std::vector<float> actual_color = device_grad_color.to_host();
  const float alpha = 1.0f - std::exp(-2.0f);
  const float expected_sigma = (1.0f + color[0] + color[1] + color[2]) * (1.0f - alpha);
  require_close(actual_sigma[0], expected_sigma, 2.0e-5f, "backward sigma gradient");
  require_close(actual_color[0], alpha, 2.0e-5f, "backward color r gradient");
  require_close(actual_color[1], alpha, 2.0e-5f, "backward color g gradient");
  require_close(actual_color[2], alpha, 2.0e-5f, "backward color b gradient");
}

void test_cuda_backward_negative_sigma_has_zero_gradient() {
  svo::BuildOptions build_options;
  build_options.max_depth = 0;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.5f, 0.5f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{-1.0f};
  const std::vector<float> color{0.8f, 0.2f, 0.1f};
  const std::vector<glm::vec3> grad_rgb{{1.0f, 1.0f, 1.0f}};
  const std::vector<float> grad_opacity{1.0f};
  const std::vector<float> zeros_sigma{0.0f};
  const std::vector<float> zeros_color{0.0f, 0.0f, 0.0f};

  auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>::from_host(sigma, svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>::from_host(color, svo::Device::CUDA);
  auto device_grad_rgb = svo::DeviceBuffer<glm::vec3>::from_host(grad_rgb, svo::Device::CUDA);
  auto device_grad_opacity = svo::DeviceBuffer<float>::from_host(grad_opacity, svo::Device::CUDA);
  auto device_grad_sigma = svo::DeviceBuffer<float>::from_host(zeros_sigma, svo::Device::CUDA);
  auto device_grad_color = svo::DeviceBuffer<float>::from_host(zeros_color, svo::Device::CUDA);

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

  const std::vector<float> actual_sigma = device_grad_sigma.to_host();
  const std::vector<float> actual_color = device_grad_color.to_host();
  require_close(actual_sigma[0], 0.0f, 1.0e-7f, "negative sigma gradient");
  require_close(actual_color[0], 0.0f, 1.0e-7f, "negative color r gradient");
  require_close(actual_color[1], 0.0f, 1.0e-7f, "negative color g gradient");
  require_close(actual_color[2], 0.0f, 1.0e-7f, "negative color b gradient");
}

void test_wide_cuda_backward_single_leaf_matches_analytic_gradient() {
  svo::BuildOptions build_options;
  build_options.max_depth = 0;
  build_options.branching = svo::BranchingMode::Wide4;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);
  const std::vector<glm::vec3> origins{{-1.0f, 0.5f, 0.5f}};
  const std::vector<glm::vec3> directions{{1.0f, 0.0f, 0.0f}};
  const std::vector<float> sigma{2.0f};
  const std::vector<float> color{0.8f, 0.2f, 0.1f};
  const std::vector<glm::vec3> grad_rgb{{1.0f, 1.0f, 1.0f}};
  const std::vector<float> grad_opacity{1.0f};
  const std::vector<float> zeros_sigma{0.0f};
  const std::vector<float> zeros_color{0.0f, 0.0f, 0.0f};

  auto device_nodes = svo::DeviceBuffer<svo::WideNodeDescriptor>::from_host(octree.wide_nodes(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_origins = svo::DeviceBuffer<glm::vec3>::from_host(origins, svo::Device::CUDA);
  auto device_directions = svo::DeviceBuffer<glm::vec3>::from_host(directions, svo::Device::CUDA);
  auto device_sigma = svo::DeviceBuffer<float>::from_host(sigma, svo::Device::CUDA);
  auto device_color = svo::DeviceBuffer<float>::from_host(color, svo::Device::CUDA);
  auto device_grad_rgb = svo::DeviceBuffer<glm::vec3>::from_host(grad_rgb, svo::Device::CUDA);
  auto device_grad_opacity = svo::DeviceBuffer<float>::from_host(grad_opacity, svo::Device::CUDA);
  auto device_grad_sigma = svo::DeviceBuffer<float>::from_host(zeros_sigma, svo::Device::CUDA);
  auto device_grad_color = svo::DeviceBuffer<float>::from_host(zeros_color, svo::Device::CUDA);

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

  const std::vector<float> actual_sigma = device_grad_sigma.to_host();
  const std::vector<float> actual_color = device_grad_color.to_host();
  const float alpha = 1.0f - std::exp(-2.0f);
  const float expected_sigma = (1.0f + color[0] + color[1] + color[2]) * (1.0f - alpha);
  require_close(actual_sigma[0], expected_sigma, 2.0e-5f, "wide backward sigma gradient");
  require_close(actual_color[0], alpha, 2.0e-5f, "wide backward color r gradient");
  require_close(actual_color[1], alpha, 2.0e-5f, "wide backward color g gradient");
  require_close(actual_color[2], alpha, 2.0e-5f, "wide backward color b gradient");
}

}  // namespace

int main() {
  test_cuda_matches_cpu_for_dense_tree();
  test_wide_cuda_matches_cpu();
  test_cuda_matches_cpu_for_sparse_tree_and_clipping();
  test_interval_forward_matches_direct_octree8();
  test_interval_forward_matches_direct_wide4();
  test_interval_forward_handles_more_than_fixed_segment_cache();
  test_interval_backward_matches_direct_octree8();
  test_interval_backward_matches_direct_wide4();
  test_interval_backward_matches_direct_with_early_stop_and_negative_sigma();
  test_interval_backward_matches_direct_empty_and_zero_rays();
  test_interval_backward_before_forward_fails_clearly();
  test_cuda_backward_single_leaf_matches_analytic_gradient();
  test_cuda_backward_negative_sigma_has_zero_gradient();
  test_wide_cuda_backward_single_leaf_matches_analytic_gradient();
  return 0;
}
