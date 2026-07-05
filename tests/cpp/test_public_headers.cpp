#include <svo/Builder.hpp>
#include <svo/Camera.hpp>
#include <svo/DeviceBuffer.hpp>
#include <svo/Error.hpp>
#include <svo/Math.hpp>
#include <svo/Octree.hpp>
#include <svo/Query.hpp>
#include <svo/Renderer.hpp>
#include <svo/Version.hpp>

#include <vector>

int main() {
  std::vector<glm::ivec3> coordinates{{0, 0, 0}};
  std::vector<glm::vec3> points{{0.25f, 0.25f, 0.25f}};

  svo::BuildOptions build_options;
  svo::QueryOptions query_options;
  svo::RenderOptions render_options;
  svo::CameraIntrinsics intrinsics{3, 3, 3.0f, 3.0f, 1.5f, 1.5f};
  svo::Camera camera = svo::Camera::from_intrinsics(
      {0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, 1.0f, 0.0f},
      intrinsics,
      svo::CameraConvention::OpenGL);
  svo::RayBatch rays = svo::generate_rays_cpu(camera);
  svo::DeviceBuffer<int> buffer(1);
  svo::NodeDescriptor descriptor =
      svo::NodeDescriptor::pack(0b00000001u, 0b00000001u, 0u, 0u);
  svo::Octree octree{
      1, svo::Device::CPU, svo::default_root_bounds(), {descriptor}, {0u}};

  octree.validate();

  (void)coordinates;
  (void)points;
  (void)build_options;
  (void)query_options;
  (void)render_options;
  (void)intrinsics;
  (void)camera;
  (void)rays;
  (void)buffer;
  (void)descriptor;
  (void)octree;

  return 0;
}
