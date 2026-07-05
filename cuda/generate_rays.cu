#include <svo/Camera.hpp>

#include <cstddef>
#include <string>

#include <cuda_runtime_api.h>

#include <svo/Error.hpp>

namespace svo {
namespace {

__device__ glm::vec3 normalize_device(const glm::vec3& value) noexcept {
  const float inv_length = rsqrtf(value.x * value.x + value.y * value.y + value.z * value.z);
  return value * inv_length;
}

__global__ void generate_rays_kernel(
    glm::vec3 origin,
    glm::vec3 right,
    glm::vec3 up,
    glm::vec3 forward,
    CameraIntrinsics intrinsics,
    float vertical_sign,
    glm::vec3* origins,
    glm::vec3* directions,
    std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (index >= count) {
    return;
  }

  const int x = static_cast<int>(index % static_cast<std::size_t>(intrinsics.width));
  const int y = static_cast<int>(index / static_cast<std::size_t>(intrinsics.width));
  const float pixel_x = (static_cast<float>(x) + 0.5f - intrinsics.cx) / intrinsics.fx;
  const float pixel_y = (static_cast<float>(y) + 0.5f - intrinsics.cy) / intrinsics.fy;

  origins[index] = origin;
  directions[index] = normalize_device(forward + right * pixel_x + up * (vertical_sign * pixel_y));
}

void check_not_null(const void* pointer, std::size_t count, const char* name) {
  if (count != 0 && pointer == nullptr) {
    throw ValidationError(std::string(name) + " cannot be null when count is non-zero");
  }
}

void check_cuda_launch(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw Error(std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

}  // namespace

void generate_rays_cuda(
    const Camera& camera,
    glm::vec3* origins,
    glm::vec3* directions,
    std::size_t count,
    CudaStreamHandle stream) {
  const std::size_t expected_count =
      static_cast<std::size_t>(camera.width()) * static_cast<std::size_t>(camera.height());
  if (count != expected_count) {
    throw ValidationError("ray buffer count must equal camera width * height");
  }
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const float vertical_sign = camera.convention() == CameraConvention::OpenGL ? -1.0f : 1.0f;

  generate_rays_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      camera.origin(),
      camera.right(),
      camera.up(),
      camera.forward(),
      camera.intrinsics(),
      vertical_sign,
      origins,
      directions,
      count);

  check_cuda_launch(cudaGetLastError(), "generate_rays_kernel launch");
}

}  // namespace svo
