#include <svo/Error.hpp>
#include <svo/Octree.hpp>
#include <svo/Renderer.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr float kTolerance = 1.0e-5f;

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
  }
  std::cerr << message << ": expected ValidationError\n";
  std::exit(1);
}

void test_empty_tree_returns_background() {
  svo::BuildOptions build_options;
  build_options.max_depth = 2;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({}, build_options);

  svo::RenderOptions render_options;
  render_options.background_color = {0.2f, 0.3f, 0.4f};
  const std::vector<float> sigma;
  const std::vector<float> color;
  const svo::RenderBatch result = svo::render_volume_cpu(
      octree,
      {{-1.0f, 0.5f, 0.5f}},
      {{1.0f, 0.0f, 0.0f}},
      sigma.data(),
      color.data(),
      0,
      render_options);

  require(result.rgb.size() == 1, "empty render size");
  require_vec_near(result.rgb[0], render_options.background_color, kTolerance, "empty render background");
  require_near(result.opacity[0], 0.0f, kTolerance, "empty render opacity");
  require(std::isinf(result.depth[0]) && result.depth[0] > 0.0f, "empty render depth should be +inf");
}

void test_single_root_leaf_composites_density_and_color() {
  svo::BuildOptions build_options;
  build_options.max_depth = 0;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);

  const std::vector<float> sigma{2.0f};
  const std::vector<float> color{0.8f, 0.2f, 0.1f};
  const svo::RenderBatch result = svo::render_volume_cpu(
      octree,
      {{-1.0f, 0.5f, 0.5f}},
      {{1.0f, 0.0f, 0.0f}},
      sigma.data(),
      color.data(),
      sigma.size());

  const float alpha = 1.0f - std::exp(-2.0f);
  require_vec_near(result.rgb[0], alpha * glm::vec3{0.8f, 0.2f, 0.1f}, kTolerance, "root rgb");
  require_near(result.opacity[0], alpha, kTolerance, "root opacity");
  require_near(result.depth[0], 1.5f, kTolerance, "root expected depth");
}

void test_front_to_back_two_leaf_compositing() {
  svo::BuildOptions build_options;
  build_options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {1, 0, 0}}, build_options);

  const std::vector<float> sigma{1.0f, 1.0f};
  const std::vector<float> color{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  const svo::RenderBatch result = svo::render_volume_cpu(
      octree,
      {{-1.0f, 0.25f, 0.25f}},
      {{1.0f, 0.0f, 0.0f}},
      sigma.data(),
      color.data(),
      sigma.size());

  const float alpha = 1.0f - std::exp(-0.5f);
  const float front_weight = alpha;
  const float back_weight = (1.0f - alpha) * alpha;
  const float opacity = front_weight + back_weight;
  require_vec_near(result.rgb[0], {front_weight, back_weight, 0.0f}, kTolerance, "two leaf rgb");
  require_near(result.opacity[0], opacity, kTolerance, "two leaf opacity");
  require_near(result.depth[0], (front_weight * 1.25f + back_weight * 1.75f) / opacity, kTolerance, "two leaf depth");
}

void test_options_clip_segment() {
  svo::BuildOptions build_options;
  build_options.max_depth = 0;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);
  const std::vector<float> sigma{1.0f};
  const std::vector<float> color{1.0f, 1.0f, 1.0f};
  svo::RenderOptions options;
  options.near_plane = 1.25f;
  options.far_plane = 1.75f;

  const svo::RenderBatch result = svo::render_volume_cpu(
      octree,
      {{-1.0f, 0.5f, 0.5f}},
      {{1.0f, 0.0f, 0.0f}},
      sigma.data(),
      color.data(),
      sigma.size(),
      options);

  const float alpha = 1.0f - std::exp(-0.5f);
  require_vec_near(result.rgb[0], {alpha, alpha, alpha}, kTolerance, "clipped rgb");
  require_near(result.opacity[0], alpha, kTolerance, "clipped opacity");
  require_near(result.depth[0], 1.5f, kTolerance, "clipped depth");
}

void test_ray_batch_shape_is_preserved() {
  svo::BuildOptions build_options;
  build_options.max_depth = 0;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);
  const std::vector<float> sigma{0.0f};
  const std::vector<float> color{0.0f, 0.0f, 0.0f};
  svo::RayBatch rays;
  rays.width = 2;
  rays.height = 1;
  rays.origins = {{-1.0f, 0.5f, 0.5f}, {-1.0f, 2.0f, 0.5f}};
  rays.directions = {{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};

  const svo::RenderBatch result = svo::render_volume_cpu(octree, rays, sigma.data(), color.data(), sigma.size());
  require(result.width == 2 && result.height == 1, "render should preserve RayBatch shape");
}

void test_invalid_inputs_fail_clearly() {
  svo::BuildOptions build_options;
  build_options.max_depth = 0;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, build_options);
  const std::vector<float> sigma{1.0f};
  const std::vector<float> color{1.0f, 1.0f, 1.0f};

  require_validation_error(
      [&]() { (void)svo::render_volume_cpu(octree, {{0.0f, 0.0f, 0.0f}}, {}, sigma.data(), color.data(), 1); },
      "mismatched ray counts");
  require_validation_error(
      [&]() { (void)svo::render_volume_cpu(octree, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}, sigma.data(), color.data(), 1); },
      "zero direction");
  require_validation_error(
      [&]() { (void)svo::render_volume_cpu(octree, {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}}, nullptr, color.data(), 1); },
      "null sigma");
  require_validation_error(
      [&]() { (void)svo::render_volume_cpu(octree, {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}}, sigma.data(), color.data(), 0); },
      "payload rows too small");

  svo::RenderOptions options;
  options.near_plane = 2.0f;
  options.far_plane = 1.0f;
  require_validation_error(
      [&]() { (void)svo::render_volume_cpu(octree, {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}}, sigma.data(), color.data(), 1, options); },
      "invalid clip range");
}

void test_wide_render_matches_octree8() {
  svo::BuildOptions build_options;
  build_options.max_depth = 4;
  const std::vector<glm::ivec3> coordinates{
      {0, 0, 0},
      {4, 4, 4},
      {8, 8, 8},
      {15, 15, 15},
  };
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, build_options);
  build_options.branching = svo::BranchingMode::Wide4;
  const svo::Octree wide = svo::Octree::from_voxels_cpu(coordinates, build_options);

  const std::vector<float> sigma{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> color{
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f,
      1.0f, 1.0f, 0.0f};
  const std::vector<glm::vec3> origins{
      {-1.0f, 0.03125f, 0.03125f},
      {-1.0f, 0.53125f, 0.53125f},
      {2.0f, 0.96875f, 0.96875f},
      {-1.0f, 0.2f, 0.2f},
  };
  const std::vector<glm::vec3> directions{
      {1.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {-1.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
  };

  const svo::RenderBatch expected =
      svo::render_volume_cpu(octree, origins, directions, sigma.data(), color.data(), sigma.size());
  const svo::RenderBatch actual =
      svo::render_volume_cpu(wide, origins, directions, sigma.data(), color.data(), sigma.size());
  for (std::size_t index = 0; index < actual.rgb.size(); ++index) {
    require_vec_near(actual.rgb[index], expected.rgb[index], 2.0e-5f, "wide render rgb");
    require_near(actual.opacity[index], expected.opacity[index], 2.0e-5f, "wide render opacity");
    if (std::isinf(expected.depth[index])) {
      require(std::isinf(actual.depth[index]), "wide render infinite depth");
    } else {
      require_near(actual.depth[index], expected.depth[index], 2.0e-5f, "wide render depth");
    }
  }
}

}  // namespace

int main() {
  test_empty_tree_returns_background();
  test_single_root_leaf_composites_density_and_color();
  test_front_to_back_two_leaf_compositing();
  test_options_clip_segment();
  test_ray_batch_shape_is_preserved();
  test_invalid_inputs_fail_clearly();
  test_wide_render_matches_octree8();
  return 0;
}
