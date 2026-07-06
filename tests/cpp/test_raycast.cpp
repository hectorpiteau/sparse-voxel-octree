#include <svo/Builder.hpp>
#include <svo/Error.hpp>
#include <svo/Octree.hpp>
#include <svo/Query.hpp>
#include <svo/Raycast.hpp>

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
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

glm::vec3 voxel_center(int grid_size, const glm::ivec3& coordinate, const svo::RootBounds& bounds) {
  const glm::vec3 extent = bounds[1] - bounds[0];
  return bounds[0] + extent *
      glm::vec3{
          (static_cast<float>(coordinate.x) + 0.5f) / static_cast<float>(grid_size),
          (static_cast<float>(coordinate.y) + 0.5f) / static_cast<float>(grid_size),
          (static_cast<float>(coordinate.z) + 0.5f) / static_cast<float>(grid_size)};
}

struct DenseHit {
  bool hit = false;
  std::int32_t leaf_id = -1;
  float t = std::numeric_limits<float>::infinity();
  glm::vec3 position{
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  std::int32_t depth = -1;
};

bool intersect_axis(float origin, float direction, float min_bound, float max_bound, float& t_min, float& t_max) {
  if (std::fabs(direction) <= 1.0e-6f) {
    return origin >= min_bound && origin <= max_bound;
  }

  float t0 = (min_bound - origin) / direction;
  float t1 = (max_bound - origin) / direction;
  if (t0 > t1) {
    std::swap(t0, t1);
  }
  t_min = std::max(t_min, t0);
  t_max = std::min(t_max, t1);
  return t_min <= t_max;
}

bool intersect_bounds(
    const glm::vec3& min_bound,
    const glm::vec3& max_bound,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float& t_near) {
  float t_min = -std::numeric_limits<float>::infinity();
  float t_max = std::numeric_limits<float>::infinity();
  if (!intersect_axis(origin.x, direction.x, min_bound.x, max_bound.x, t_min, t_max) ||
      !intersect_axis(origin.y, direction.y, min_bound.y, max_bound.y, t_min, t_max) ||
      !intersect_axis(origin.z, direction.z, min_bound.z, max_bound.z, t_min, t_max) ||
      t_max < 0.0f) {
    return false;
  }
  t_near = std::max(t_min, 0.0f);
  return true;
}

DenseHit dense_reference_raycast(
    const svo::Octree& octree,
    const std::vector<glm::ivec3>& coordinates,
    const glm::vec3& origin,
    const glm::vec3& direction) {
  const glm::vec3 normalized = glm::normalize(direction);
  const int grid_size = 1 << octree.max_depth();
  const svo::RootBounds& bounds = octree.root_bounds();
  const glm::vec3 extent = bounds[1] - bounds[0];
  DenseHit best;

  for (const glm::ivec3& coordinate : coordinates) {
    const glm::vec3 voxel_min = bounds[0] + extent *
        glm::vec3{
            static_cast<float>(coordinate.x) / static_cast<float>(grid_size),
            static_cast<float>(coordinate.y) / static_cast<float>(grid_size),
            static_cast<float>(coordinate.z) / static_cast<float>(grid_size)};
    const glm::vec3 voxel_max = bounds[0] + extent *
        glm::vec3{
            static_cast<float>(coordinate.x + 1) / static_cast<float>(grid_size),
            static_cast<float>(coordinate.y + 1) / static_cast<float>(grid_size),
            static_cast<float>(coordinate.z + 1) / static_cast<float>(grid_size)};

    float t = 0.0f;
    if (!intersect_bounds(voxel_min, voxel_max, origin, normalized, t)) {
      continue;
    }

    const std::int32_t leaf_id =
        svo::query_points(octree, {voxel_center(grid_size, coordinate, bounds)}).front();
    const bool better = !best.hit || t < best.t - kTolerance ||
        (std::fabs(t - best.t) <= kTolerance && leaf_id < best.leaf_id);
    if (better) {
      best.hit = true;
      best.leaf_id = leaf_id;
      best.t = t;
      best.position = origin + normalized * t;
      best.depth = octree.max_depth();
    }
  }

  return best;
}

void require_hit(
    const svo::RaycastBatch& results,
    std::size_t index,
    std::int32_t leaf_id,
    float t,
    const glm::vec3& position,
    std::int32_t depth,
    const std::string& message) {
  require(results.hit_mask[index] == 1u, message + " should hit");
  require(results.leaf_ids[index] == leaf_id, message + " leaf id");
  require_near(results.t[index], t, kTolerance, message + " t");
  require_vec_near(results.positions[index], position, kTolerance, message + " position");
  require(results.depths[index] == depth, message + " depth");
}

void require_miss(const svo::RaycastBatch& results, std::size_t index, const std::string& message) {
  require(results.hit_mask[index] == 0u, message + " should miss");
  require(results.leaf_ids[index] == -1, message + " leaf id");
  require(std::isinf(results.t[index]) && results.t[index] > 0.0f, message + " t");
  require(std::isnan(results.positions[index].x), message + " position x");
  require(std::isnan(results.positions[index].y), message + " position y");
  require(std::isnan(results.positions[index].z), message + " position z");
  require(results.depths[index] == -1, message + " depth");
}

void test_empty_tree_misses() {
  svo::BuildOptions options;
  options.max_depth = 3;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({}, options);
  const svo::RaycastBatch results =
      svo::raycast_cpu(octree, {{-1.0f, 0.5f, 0.5f}}, {{1.0f, 0.0f, 0.0f}});
  require_miss(results, 0, "empty tree");
}

void test_single_voxel_hit_and_miss() {
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, options);

  const svo::RaycastBatch results = svo::raycast_cpu(
      octree,
      {
          {-1.0f, 0.25f, 0.25f},
          {-1.0f, 0.25f, 0.75f},
      },
      {
          {1.0f, 0.0f, 0.0f},
          {1.0f, 0.0f, 0.0f},
      });

  require_hit(results, 0, 0, 1.0f, {0.0f, 0.25f, 0.25f}, 1, "single voxel");
  require_miss(results, 1, "single voxel miss");
}

void test_cube_hit() {
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu(
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

  const svo::RaycastBatch results =
      svo::raycast_cpu(octree, {{-1.0f, 0.75f, 0.75f}}, {{1.0f, 0.0f, 0.0f}});
  require_hit(results, 0, 6, 1.0f, {0.0f, 0.75f, 0.75f}, 1, "cube hit");
}

void test_ray_starts_inside_root_and_voxel() {
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {1, 0, 0}}, options);

  const svo::RaycastBatch results = svo::raycast_cpu(
      octree,
      {
          {0.75f, 0.25f, 0.25f},
          {0.25f, 0.25f, 0.25f},
      },
      {
          {1.0f, 0.0f, 0.0f},
          {1.0f, 0.0f, 0.0f},
      });

  require_hit(results, 0, 1, 0.0f, {0.75f, 0.25f, 0.25f}, 1, "inside root");
  require_hit(results, 1, 0, 0.0f, {0.25f, 0.25f, 0.25f}, 1, "inside voxel");
}

void test_axis_diagonal_boundary_and_payload() {
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}, {0, 1, 0}, {1, 1, 1}}, options);

  svo::RaycastOptions payload_options;
  payload_options.return_payload_indices = true;

  const svo::RaycastBatch results = svo::raycast_cpu(
      octree,
      {
          {-1.0f, 0.5f, 0.25f},
          {-1.0f, -1.0f, -1.0f},
          {-1.0f, 0.25f, 0.75f},
      },
      {
          {1.0f, 0.0f, 0.0f},
          {1.0f, 1.0f, 1.0f},
          {1.0f, 0.0f, 0.0f},
      },
      payload_options);

  require_hit(results, 0, 0, 1.0f, {0.0f, 0.5f, 0.25f}, 1, "boundary tie");
  require_hit(
      results,
      1,
      0,
      std::sqrt(3.0f),
      {0.0f, 0.0f, 0.0f},
      1,
      "diagonal ray");
  require_miss(results, 2, "axis miss");

  const svo::Octree remapped_octree = svo::Octree::from_voxels_cpu(
      {{0, 0, 0}, {0, 1, 0}, {1, 1, 1}},
      {10u, 11u, 12u},
      options);
  const svo::RaycastBatch remapped_results = svo::raycast_cpu(
      remapped_octree,
      {{-1.0f, 0.5f, 0.25f}},
      {{1.0f, 0.0f, 0.0f}},
      payload_options);
  require_hit(remapped_results, 0, 10, 1.0f, {0.0f, 0.5f, 0.25f}, 1, "remapped payload hit");
}

void test_custom_root_bounds() {
  svo::BuildOptions options;
  options.max_depth = 1;
  options.root_bounds = {glm::vec3{-1.0f, -1.0f, -1.0f}, glm::vec3{1.0f, 1.0f, 1.0f}};
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{1, 0, 0}}, options);

  const svo::RaycastBatch results =
      svo::raycast_cpu(octree, {{-2.0f, -0.5f, -0.5f}}, {{1.0f, 0.0f, 0.0f}});
  require_hit(results, 0, 0, 2.0f, {0.0f, -0.5f, -0.5f}, 1, "custom bounds");
}

void test_max_depth_zero() {
  svo::BuildOptions options;
  options.max_depth = 0;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, options);
  const svo::RaycastBatch results =
      svo::raycast_cpu(octree, {{-1.0f, 0.5f, 0.5f}}, {{2.0f, 0.0f, 0.0f}});
  require_hit(results, 0, 0, 1.0f, {0.0f, 0.5f, 0.5f}, 0, "max depth zero");
}

void test_dense_reference_comparison() {
  constexpr int max_depth = 3;
  constexpr int grid_size = 1 << max_depth;
  svo::BuildOptions options;
  options.max_depth = max_depth;

  std::mt19937 rng(20260705u);
  std::bernoulli_distribution keep_dist(0.12);
  std::uniform_real_distribution<float> origin_dist(-0.5f, 1.5f);
  std::uniform_real_distribution<float> direction_dist(-1.0f, 1.0f);

  std::vector<glm::ivec3> coordinates;
  for (int z = 0; z < grid_size; ++z) {
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        if (keep_dist(rng)) {
          coordinates.emplace_back(x, y, z);
        }
      }
    }
  }

  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, options);
  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> directions;
  for (int index = 0; index < 128; ++index) {
    glm::vec3 direction{direction_dist(rng), direction_dist(rng), direction_dist(rng)};
    if (glm::length(direction) <= 1.0e-3f) {
      direction = {1.0f, 0.0f, 0.0f};
    }
    origins.push_back({origin_dist(rng), origin_dist(rng), origin_dist(rng)});
    directions.push_back(direction);
  }

  const svo::RaycastBatch results = svo::raycast_cpu(octree, origins, directions);
  for (std::size_t index = 0; index < origins.size(); ++index) {
    const DenseHit expected = dense_reference_raycast(octree, coordinates, origins[index], directions[index]);
    if (!expected.hit) {
      require_miss(results, index, "dense reference miss");
      continue;
    }
    require_hit(
        results,
        index,
        expected.leaf_id,
        expected.t,
        expected.position,
        expected.depth,
        "dense reference hit");
  }
}

void test_ray_batch_dimensions() {
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, options);

  svo::RayBatch rays;
  rays.width = 2;
  rays.height = 1;
  rays.origins = {{-1.0f, 0.25f, 0.25f}, {-1.0f, 0.75f, 0.75f}};
  rays.directions = {{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};

  const svo::RaycastBatch results = svo::raycast_cpu(octree, rays);
  require(results.width == 2 && results.height == 1, "RayBatch dimensions should be preserved");
  require_hit(results, 0, 0, 1.0f, {0.0f, 0.25f, 0.25f}, 1, "RayBatch hit");
  require_miss(results, 1, "RayBatch miss");
}

void test_invalid_inputs() {
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, options);

  require_validation_error(
      [&] {
        (void)svo::raycast_cpu(octree, {{0.0f, 0.0f, 0.0f}}, {});
      },
      "mismatched counts");
  require_validation_error(
      [&] {
        (void)svo::raycast_cpu(octree, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}});
      },
      "zero direction");
  require_validation_error(
      [&] {
        (void)svo::raycast_cpu(
            octree,
            {{std::numeric_limits<float>::infinity(), 0.0f, 0.0f}},
            {{1.0f, 0.0f, 0.0f}});
      },
      "non-finite origin");
}

}  // namespace

int main() {
  test_empty_tree_misses();
  test_single_voxel_hit_and_miss();
  test_cube_hit();
  test_ray_starts_inside_root_and_voxel();
  test_axis_diagonal_boundary_and_payload();
  test_custom_root_bounds();
  test_max_depth_zero();
  test_dense_reference_comparison();
  test_ray_batch_dimensions();
  test_invalid_inputs();
  return 0;
}
