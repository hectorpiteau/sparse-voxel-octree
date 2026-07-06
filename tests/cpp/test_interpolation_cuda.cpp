#include <svo/DeviceBuffer.hpp>
#include <svo/Interpolation.hpp>
#include <svo/Octree.hpp>

#include <glm/ext/vector_int3.hpp>

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

void require_close(double actual, double expected, double tolerance, const std::string& message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
    std::exit(1);
  }
}

svo::Octree dense_depth_one_tree() {
  svo::BuildOptions options;
  options.max_depth = 1;
  return svo::Octree::from_voxels_cpu(
      {
          {0, 0, 0},
          {1, 0, 0},
          {0, 1, 0},
          {1, 1, 0},
          {0, 0, 1},
          {1, 0, 1},
          {0, 1, 1},
          {1, 1, 1},
      },
      options);
}

void test_cuda_forward_matches_cpu() {
  const svo::Octree octree = dense_depth_one_tree();
  const std::vector<glm::vec3> points{{0.25f, 0.25f, 0.25f}, {0.5f, 0.5f, 0.5f}};
  const std::vector<float> payload{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  const std::vector<float> cpu = svo::sample_trilinear_float(
      octree, points, payload.data(), payload.size(), 1, 0.0f);

  auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_points = svo::DeviceBuffer<glm::vec3>::from_host(points, svo::Device::CUDA);
  auto device_payload = svo::DeviceBuffer<float>::from_host(payload, svo::Device::CUDA);
  svo::DeviceBuffer<float> device_outputs(points.size(), svo::Device::CUDA);

  svo::sample_trilinear_cuda_float(
      device_nodes.data(),
      device_nodes.size(),
      device_payload_indices.data(),
      device_payload_indices.size(),
      octree.max_depth(),
      octree.root_bounds(),
      device_points.data(),
      device_payload.data(),
      device_outputs.data(),
      points.size(),
      payload.size(),
      1,
      0.0f);

  const std::vector<float> cuda = device_outputs.to_host();
  require(cuda.size() == cpu.size(), "CUDA output size should match CPU output size");
  for (std::size_t index = 0; index < cuda.size(); ++index) {
    require_close(cuda[index], cpu[index], 1e-6, "CUDA interpolation should match CPU");
  }
}

void test_cuda_backward_matches_cpu() {
  const svo::Octree octree = dense_depth_one_tree();
  const std::vector<glm::vec3> points{{0.5f, 0.5f, 0.5f}};
  const std::vector<float> grad_outputs{2.0f};
  const std::vector<float> cpu = svo::sample_trilinear_backward_float(
      octree, points, grad_outputs.data(), 8, 1, 0.0f);

  auto device_nodes = svo::DeviceBuffer<svo::NodeDescriptor>::from_host(octree.nodes(), svo::Device::CUDA);
  auto device_payload_indices =
      svo::DeviceBuffer<std::uint32_t>::from_host(octree.leaf_payload_indices(), svo::Device::CUDA);
  auto device_points = svo::DeviceBuffer<glm::vec3>::from_host(points, svo::Device::CUDA);
  auto device_grad_outputs = svo::DeviceBuffer<float>::from_host(grad_outputs, svo::Device::CUDA);
  svo::DeviceBuffer<float> device_grad_payload(8, svo::Device::CUDA);
  const std::vector<float> zeros(8, 0.0f);
  device_grad_payload.copy_from_host(zeros.data(), zeros.size());

  svo::sample_trilinear_backward_cuda_float(
      device_nodes.data(),
      device_nodes.size(),
      device_payload_indices.data(),
      device_payload_indices.size(),
      octree.max_depth(),
      octree.root_bounds(),
      device_points.data(),
      device_grad_outputs.data(),
      device_grad_payload.data(),
      points.size(),
      8,
      1,
      0.0f);

  const std::vector<float> cuda = device_grad_payload.to_host();
  for (std::size_t index = 0; index < cuda.size(); ++index) {
    require_close(cuda[index], cpu[index], 1e-6, "CUDA backward should match CPU");
  }
}

}  // namespace

int main() {
  test_cuda_forward_matches_cpu();
  test_cuda_backward_matches_cpu();
  return 0;
}
