#include <svo/Camera.hpp>

#include <cmath>
#include <string>

#include <glm/geometric.hpp>

#include <svo/Error.hpp>

namespace svo {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1.0e-6f;

bool is_finite(float value) noexcept {
  return std::isfinite(value);
}

bool is_finite_vec3(const glm::vec3& value) noexcept {
  return is_finite(value.x) && is_finite(value.y) && is_finite(value.z);
}

void validate_intrinsics(const CameraIntrinsics& intrinsics) {
  if (intrinsics.width <= 0 || intrinsics.height <= 0) {
    throw ValidationError("camera width and height must be positive");
  }
  if (!is_finite(intrinsics.fx) || !is_finite(intrinsics.fy) || intrinsics.fx <= 0.0f ||
      intrinsics.fy <= 0.0f) {
    throw ValidationError("camera fx and fy must be positive finite values");
  }
  if (!is_finite(intrinsics.cx) || !is_finite(intrinsics.cy)) {
    throw ValidationError("camera cx and cy must be finite values");
  }
}

CameraIntrinsics intrinsics_from_fov(int width, int height, float vertical_fov_y_degrees) {
  if (width <= 0 || height <= 0) {
    throw ValidationError("camera width and height must be positive");
  }
  if (!is_finite(vertical_fov_y_degrees) || vertical_fov_y_degrees <= 0.0f ||
      vertical_fov_y_degrees >= 180.0f) {
    throw ValidationError("vertical_fov_y_degrees must be in the range (0, 180)");
  }

  const float radians = vertical_fov_y_degrees * kPi / 180.0f;
  const float focal = 0.5f * static_cast<float>(height) / std::tan(0.5f * radians);
  return CameraIntrinsics{
      width,
      height,
      focal,
      focal,
      0.5f * static_cast<float>(width),
      0.5f * static_cast<float>(height)};
}

glm::vec3 normalized_or_throw(const glm::vec3& value, const char* message) {
  if (!is_finite_vec3(value)) {
    throw ValidationError(std::string(message) + " must be finite");
  }
  const float length = glm::length(value);
  if (!is_finite(length) || length <= kEpsilon) {
    throw ValidationError(message);
  }
  return value / length;
}

glm::vec3 ray_direction_for_pixel(const Camera& camera, int x, int y) {
  const CameraIntrinsics& intrinsics = camera.intrinsics();
  const float pixel_x = (static_cast<float>(x) + 0.5f - intrinsics.cx) / intrinsics.fx;
  const float pixel_y = (static_cast<float>(y) + 0.5f - intrinsics.cy) / intrinsics.fy;
  const float vertical_sign = camera.convention() == CameraConvention::OpenGL ? -1.0f : 1.0f;
  return glm::normalize(
      camera.forward() + camera.right() * pixel_x + camera.up() * (vertical_sign * pixel_y));
}

}  // namespace

Camera::Camera(
    const glm::vec3& origin,
    const glm::vec3& right,
    const glm::vec3& up,
    const glm::vec3& forward,
    const CameraIntrinsics& intrinsics,
    CameraConvention convention)
    : origin_(origin),
      right_(right),
      up_(up),
      forward_(forward),
      intrinsics_(intrinsics),
      convention_(convention) {}

Camera Camera::look_at(
    const glm::vec3& origin,
    const glm::vec3& target,
    const glm::vec3& up,
    int width,
    int height,
    float vertical_fov_y_degrees,
    CameraConvention convention) {
  return from_intrinsics(
      origin,
      target,
      up,
      intrinsics_from_fov(width, height, vertical_fov_y_degrees),
      convention);
}

Camera Camera::from_intrinsics(
    const glm::vec3& origin,
    const glm::vec3& target,
    const glm::vec3& up,
    const CameraIntrinsics& intrinsics,
    CameraConvention convention) {
  validate_intrinsics(intrinsics);
  if (!is_finite_vec3(origin) || !is_finite_vec3(target)) {
    throw ValidationError("camera origin and target must be finite");
  }

  const glm::vec3 forward = normalized_or_throw(target - origin, "camera forward vector cannot be zero-length");
  const glm::vec3 up_hint = normalized_or_throw(up, "camera up vector cannot be zero-length");
  const glm::vec3 right = normalized_or_throw(glm::cross(forward, up_hint), "camera up vector is degenerate");
  const glm::vec3 orthonormal_up = glm::normalize(glm::cross(right, forward));

  return Camera(origin, right, orthonormal_up, forward, intrinsics, convention);
}

RayBatch generate_rays_cpu(const Camera& camera) {
  const std::size_t count =
      static_cast<std::size_t>(camera.width()) * static_cast<std::size_t>(camera.height());

  RayBatch batch;
  batch.width = camera.width();
  batch.height = camera.height();
  batch.origins.assign(count, camera.origin());
  batch.directions.reserve(count);

  for (int y = 0; y < camera.height(); ++y) {
    for (int x = 0; x < camera.width(); ++x) {
      batch.directions.push_back(ray_direction_for_pixel(camera, x, y));
    }
  }

  return batch;
}

}  // namespace svo
