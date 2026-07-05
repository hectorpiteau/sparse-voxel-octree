#include <svo/Camera.hpp>
#include <svo/Error.hpp>

#include <glm/geometric.hpp>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void require_near(float actual, float expected, float tolerance, const std::string& message) {
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << message << ": expected " << expected << ", got " << actual << '\n';
    std::exit(1);
  }
}

void require_vec_near(
    const glm::vec3& actual,
    const glm::vec3& expected,
    float tolerance,
    const std::string& message) {
  require_near(actual.x, expected.x, tolerance, message + " x");
  require_near(actual.y, expected.y, tolerance, message + " y");
  require_near(actual.z, expected.z, tolerance, message + " z");
}

template <typename Func>
void require_validation_error(Func&& func, const std::string& message) {
  try {
    func();
  } catch (const svo::ValidationError&) {
    return;
  } catch (const std::exception& error) {
    std::cerr << message << ": expected ValidationError, got " << error.what() << '\n';
    std::exit(1);
  }
  std::cerr << message << ": expected ValidationError\n";
  std::exit(1);
}

std::size_t ray_index(int width, int x, int y) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

void test_center_pixel_points_at_target() {
  const svo::Camera camera = svo::Camera::look_at(
      {0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f, 0.0f},
      5,
      5,
      60.0f,
      svo::CameraConvention::OpenGL);
  const svo::RayBatch rays = svo::generate_rays_cpu(camera);

  require(rays.width == 5 && rays.height == 5, "ray batch should preserve dimensions");
  require(rays.origins.size() == 25 && rays.directions.size() == 25, "ray batch should be flattened");
  require_vec_near(rays.origins[ray_index(5, 2, 2)], {0.0f, 0.0f, 0.0f}, 1.0e-6f, "origin mismatch");
  require_vec_near(
      rays.directions[ray_index(5, 2, 2)],
      {0.0f, 0.0f, -1.0f},
      1.0e-6f,
      "center ray should point at target");
}

void test_basis_is_orthonormal() {
  const svo::Camera camera = svo::Camera::look_at(
      {1.0f, 2.0f, 3.0f},
      {3.0f, 2.5f, 1.0f},
      {0.0f, 1.0f, 0.0f},
      7,
      5,
      55.0f);

  require_near(glm::length(camera.right()), 1.0f, 1.0e-5f, "right length");
  require_near(glm::length(camera.up()), 1.0f, 1.0e-5f, "up length");
  require_near(glm::length(camera.forward()), 1.0f, 1.0e-5f, "forward length");
  require_near(glm::dot(camera.right(), camera.up()), 0.0f, 1.0e-5f, "right/up dot");
  require_near(glm::dot(camera.right(), camera.forward()), 0.0f, 1.0e-5f, "right/forward dot");
  require_near(glm::dot(camera.up(), camera.forward()), 0.0f, 1.0e-5f, "up/forward dot");
}

void test_fov_changes_corner_spread() {
  const svo::Camera narrow = svo::Camera::look_at(
      {0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f, 0.0f},
      5,
      5,
      30.0f);
  const svo::Camera wide = svo::Camera::look_at(
      {0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f, 0.0f},
      5,
      5,
      90.0f);

  const glm::vec3 narrow_corner = svo::generate_rays_cpu(narrow).directions.front();
  const glm::vec3 wide_corner = svo::generate_rays_cpu(wide).directions.front();
  require(
      std::fabs(wide_corner.x) > std::fabs(narrow_corner.x),
      "wider FOV should increase horizontal corner spread");
  require(
      std::fabs(wide_corner.y) > std::fabs(narrow_corner.y),
      "wider FOV should increase vertical corner spread");
}

void test_conventions_share_center_and_flip_vertical() {
  const svo::Camera opengl = svo::Camera::look_at(
      {0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f, 0.0f},
      3,
      3,
      60.0f,
      svo::CameraConvention::OpenGL);
  const svo::Camera cv = svo::Camera::look_at(
      {0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f, 0.0f},
      3,
      3,
      60.0f,
      svo::CameraConvention::ComputerVision);

  const svo::RayBatch gl_rays = svo::generate_rays_cpu(opengl);
  const svo::RayBatch cv_rays = svo::generate_rays_cpu(cv);
  require_vec_near(
      gl_rays.directions[ray_index(3, 1, 1)],
      cv_rays.directions[ray_index(3, 1, 1)],
      1.0e-6f,
      "center conventions should match");
  require_near(
      gl_rays.directions[ray_index(3, 1, 0)].y,
      -cv_rays.directions[ray_index(3, 1, 0)].y,
      1.0e-6f,
      "vertical convention should flip");
}

void test_from_intrinsics() {
  const svo::CameraIntrinsics intrinsics{3, 3, 2.0f, 2.0f, 1.5f, 1.5f};
  const svo::Camera camera = svo::Camera::from_intrinsics(
      {1.0f, 2.0f, 3.0f},
      {1.0f, 2.0f, 2.0f},
      {0.0f, 1.0f, 0.0f},
      intrinsics,
      svo::CameraConvention::OpenGL);
  const svo::RayBatch rays = svo::generate_rays_cpu(camera);
  require_vec_near(
      rays.directions[ray_index(3, 1, 1)],
      {0.0f, 0.0f, -1.0f},
      1.0e-6f,
      "intrinsics center ray");
}

void test_invalid_inputs() {
  require_validation_error(
      [] {
        (void)svo::Camera::look_at({0, 0, 0}, {0, 0, -1}, {0, 1, 0}, 0, 3, 60.0f);
      },
      "zero width");
  require_validation_error(
      [] {
        (void)svo::Camera::look_at({0, 0, 0}, {0, 0, -1}, {0, 1, 0}, 3, -1, 60.0f);
      },
      "negative height");
  require_validation_error(
      [] {
        (void)svo::Camera::look_at({0, 0, 0}, {0, 0, -1}, {0, 1, 0}, 3, 3, 0.0f);
      },
      "invalid fov");
  require_validation_error(
      [] {
        (void)svo::Camera::look_at({0, 0, 0}, {0, 0, 0}, {0, 1, 0}, 3, 3, 60.0f);
      },
      "zero forward");
  require_validation_error(
      [] {
        (void)svo::Camera::look_at({0, 0, 0}, {0, 0, -1}, {0, 0, 0}, 3, 3, 60.0f);
      },
      "zero up");
  require_validation_error(
      [] {
        (void)svo::Camera::look_at({0, 0, 0}, {0, 0, -1}, {0, 0, -1}, 3, 3, 60.0f);
      },
      "degenerate up");
  require_validation_error(
      [] {
        const svo::CameraIntrinsics intrinsics{3, 3, 0.0f, 2.0f, 1.5f, 1.5f};
        (void)svo::Camera::from_intrinsics({0, 0, 0}, {0, 0, -1}, {0, 1, 0}, intrinsics);
      },
      "non-positive fx");
}

}  // namespace

int main() {
  test_center_pixel_points_at_target();
  test_basis_is_orthonormal();
  test_fov_changes_corner_spread();
  test_conventions_share_center_and_flip_vertical();
  test_from_intrinsics();
  test_invalid_inputs();
  return 0;
}
