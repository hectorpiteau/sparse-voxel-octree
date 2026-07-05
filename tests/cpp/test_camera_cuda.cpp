#include <svo/Camera.hpp>
#include <svo/DeviceBuffer.hpp>

#include <glm/geometric.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
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

void require_vec_near(
    const glm::vec3& actual,
    const glm::vec3& expected,
    float tolerance,
    const std::string& message) {
  const glm::vec3 delta = actual - expected;
  if (std::fabs(delta.x) > tolerance || std::fabs(delta.y) > tolerance || std::fabs(delta.z) > tolerance) {
    std::cerr << message << ": expected (" << expected.x << ", " << expected.y << ", " << expected.z
              << "), got (" << actual.x << ", " << actual.y << ", " << actual.z << ")\n";
    std::exit(1);
  }
}

#if SVO_ENABLE_CUDA
void require_cuda_matches_cpu(const svo::Camera& camera, const std::string& message) {
  const svo::RayBatch cpu_rays = svo::generate_rays_cpu(camera);
  svo::DeviceBuffer<glm::vec3> device_origins(cpu_rays.origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_directions(cpu_rays.directions.size(), svo::Device::CUDA);

  svo::generate_rays_cuda(
      camera,
      device_origins.data(),
      device_directions.data(),
      device_directions.size());

  const std::vector<glm::vec3> cuda_origins = device_origins.to_host();
  const std::vector<glm::vec3> cuda_directions = device_directions.to_host();
  require(cuda_origins.size() == cpu_rays.origins.size(), message + " origin size");
  require(cuda_directions.size() == cpu_rays.directions.size(), message + " direction size");

  for (std::size_t index = 0; index < cpu_rays.directions.size(); ++index) {
    require_vec_near(cuda_origins[index], cpu_rays.origins[index], 1.0e-6f, message + " origin");
    require_vec_near(cuda_directions[index], cpu_rays.directions[index], 1.0e-5f, message + " direction");
  }
}
#endif

void test_odd_and_even_image_sizes() {
#if SVO_ENABLE_CUDA
  const svo::Camera odd = svo::Camera::look_at(
      {0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f, 0.0f},
      5,
      3,
      60.0f,
      svo::CameraConvention::OpenGL);
  require_cuda_matches_cpu(odd, "odd camera");

  const svo::CameraIntrinsics intrinsics{6, 4, 4.0f, 5.0f, 3.0f, 2.0f};
  const svo::Camera even = svo::Camera::from_intrinsics(
      {1.0f, 2.0f, 3.0f},
      {2.0f, 2.0f, 1.0f},
      {0.0f, 1.0f, 0.0f},
      intrinsics,
      svo::CameraConvention::ComputerVision);
  require_cuda_matches_cpu(even, "even camera");
#endif
}

}  // namespace

int main() {
  test_odd_and_even_image_sizes();
  return 0;
}
