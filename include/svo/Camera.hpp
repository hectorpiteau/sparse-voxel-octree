#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <svo/Math.hpp>
#include <svo/Octree.hpp>

namespace svo {

enum class CameraConvention : std::uint8_t {
  OpenGL = 0,
  ComputerVision = 1,
};

struct CameraIntrinsics {
  int width = 0;
  int height = 0;
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
};

struct RayBatch {
  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> directions;
  int width = 0;
  int height = 0;
};

class Camera {
 public:
  Camera() = default;

  static Camera look_at(
      const glm::vec3& origin,
      const glm::vec3& target,
      const glm::vec3& up,
      int width,
      int height,
      float vertical_fov_y_degrees,
      CameraConvention convention = CameraConvention::OpenGL);

  static Camera from_intrinsics(
      const glm::vec3& origin,
      const glm::vec3& target,
      const glm::vec3& up,
      const CameraIntrinsics& intrinsics,
      CameraConvention convention = CameraConvention::OpenGL);

  const glm::vec3& origin() const noexcept { return origin_; }
  const glm::vec3& right() const noexcept { return right_; }
  const glm::vec3& up() const noexcept { return up_; }
  const glm::vec3& forward() const noexcept { return forward_; }
  const CameraIntrinsics& intrinsics() const noexcept { return intrinsics_; }
  CameraConvention convention() const noexcept { return convention_; }
  int width() const noexcept { return intrinsics_.width; }
  int height() const noexcept { return intrinsics_.height; }

 private:
  Camera(
      const glm::vec3& origin,
      const glm::vec3& right,
      const glm::vec3& up,
      const glm::vec3& forward,
      const CameraIntrinsics& intrinsics,
      CameraConvention convention);

  glm::vec3 origin_{0.0f, 0.0f, 0.0f};
  glm::vec3 right_{1.0f, 0.0f, 0.0f};
  glm::vec3 up_{0.0f, 1.0f, 0.0f};
  glm::vec3 forward_{0.0f, 0.0f, -1.0f};
  CameraIntrinsics intrinsics_;
  CameraConvention convention_ = CameraConvention::OpenGL;
};

RayBatch generate_rays_cpu(const Camera& camera);

#if SVO_ENABLE_CUDA
void generate_rays_cuda(
    const Camera& camera,
    glm::vec3* origins,
    glm::vec3* directions,
    std::size_t count,
    CudaStreamHandle stream = nullptr);
#endif

}  // namespace svo
