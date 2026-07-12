#include <svo/Octree.hpp>
#include <svo/Query.hpp>

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_int3.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

std::size_t flat_index(int grid_size, int x, int y, int z) {
  return static_cast<std::size_t>(x) +
      static_cast<std::size_t>(grid_size) *
          (static_cast<std::size_t>(y) + static_cast<std::size_t>(grid_size) * static_cast<std::size_t>(z));
}

glm::vec3 voxel_center_point(int grid_size, const glm::ivec3& coordinate) {
  const float inv_grid = 1.0f / static_cast<float>(grid_size);
  return glm::vec3{
      (static_cast<float>(coordinate.x) + 0.5f) * inv_grid,
      (static_cast<float>(coordinate.y) + 0.5f) * inv_grid,
      (static_cast<float>(coordinate.z) + 0.5f) * inv_grid};
}

bool sphere_contains_voxel_center(
    const glm::ivec3& coordinate,
    const glm::vec3& center,
    float radius) {
  const glm::vec3 voxel_center{
      static_cast<float>(coordinate.x) + 0.5f,
      static_cast<float>(coordinate.y) + 0.5f,
      static_cast<float>(coordinate.z) + 0.5f};
  const glm::vec3 delta = voxel_center - center;
  return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z <= radius * radius;
}

void test_basic_cpu_query() {
  svo::BuildOptions options;
  options.max_depth = 3;

  const std::vector<glm::ivec3> coordinates{
      glm::ivec3{0, 0, 0},
      glm::ivec3{4, 4, 4},
      glm::ivec3{7, 7, 7},
  };
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, options);

  const std::vector<glm::vec3> points{
      voxel_center_point(8, {0, 0, 0}),
      voxel_center_point(8, {4, 4, 4}),
      voxel_center_point(8, {1, 1, 1}),
      {-0.1f, 0.2f, 0.2f},
      {1.0f, 0.5f, 0.5f},
      {0.0f, 0.0f, 0.0f},
      {0.999999f, 0.999999f, 0.999999f},
      {0.5f, 0.5f, 0.5f},
  };

  const std::vector<std::int32_t> leaf_ids = svo::query_points(octree, points);
  require(leaf_ids[0] == 0, "occupied voxel center should return leaf 0");
  require(leaf_ids[1] == 1, "occupied voxel center should return leaf 1");
  require(leaf_ids[2] == -1, "empty voxel center should miss");
  require(leaf_ids[3] == -1, "point below root bounds should miss");
  require(leaf_ids[4] == -1, "point on max boundary should miss");
  require(leaf_ids[5] == 0, "point exactly at root min should map into voxel 0");
  require(leaf_ids[6] == 2, "point near root max should hit the last voxel");
  require(leaf_ids[7] == 1, "boundary point should use floor mapping into voxel 4");

  svo::QueryOptions payload_options;
  payload_options.return_payload_indices = true;
  const std::vector<std::int32_t> payload_indices =
      svo::query_points(octree, {voxel_center_point(8, {7, 7, 7})}, payload_options);
  require(payload_indices.size() == 1 && payload_indices[0] == 2, "payload query should return payload index 2");
}

void test_custom_payload_index_query() {
  svo::BuildOptions options;
  options.max_depth = 2;

  const std::vector<glm::ivec3> coordinates{
      glm::ivec3{3, 3, 3},
      glm::ivec3{0, 0, 0},
      glm::ivec3{2, 1, 0},
  };
  const std::vector<std::uint32_t> payload_indices{42u, 7u, 99u};
  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, payload_indices, options);

  const std::vector<glm::vec3> points{
      voxel_center_point(4, {0, 0, 0}),
      voxel_center_point(4, {2, 1, 0}),
      voxel_center_point(4, {3, 3, 3}),
      voxel_center_point(4, {1, 1, 1}),
  };

  svo::QueryOptions payload_options;
  payload_options.return_payload_indices = true;
  const std::vector<std::int32_t> via_option = svo::query_points(octree, points, payload_options);
  const std::vector<std::int32_t> via_helper = svo::query_payload_indices(octree, points);

  require(via_option == std::vector<std::int32_t>{7, 99, 42, -1}, "custom payload query should remap hits");
  require(via_helper == via_option, "query_payload_indices should match the explicit query option");
  require(svo::query_points(octree, points) == std::vector<std::int32_t>{0, 1, 2, -1}, "leaf id query should stay unchanged");
}

void test_random_reference_query() {
  constexpr int max_depth = 5;
  constexpr int grid_size = 1 << max_depth;
  constexpr int total_voxels = grid_size * grid_size * grid_size;

  svo::BuildOptions options;
  options.max_depth = max_depth;

  std::mt19937 rng(12345u);
  std::uniform_int_distribution<int> coord_dist(0, grid_size - 1);
  std::bernoulli_distribution keep_dist(0.07);

  std::vector<std::uint8_t> occupied(total_voxels, 0u);
  std::vector<glm::ivec3> coordinates;

  for (int z = 0; z < grid_size; ++z) {
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        if (!keep_dist(rng)) {
          continue;
        }
        occupied[flat_index(grid_size, x, y, z)] = 1u;
        coordinates.emplace_back(x, y, z);
      }
    }
  }

  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, options);

  for (int sample = 0; sample < 256; ++sample) {
    const glm::ivec3 coordinate{coord_dist(rng), coord_dist(rng), coord_dist(rng)};
    const bool expected = occupied[flat_index(grid_size, coordinate.x, coordinate.y, coordinate.z)] != 0u;
    const std::int32_t leaf_id =
        svo::query_points(octree, {voxel_center_point(grid_size, coordinate)}).front();
    require((leaf_id != -1) == expected, "random reference query disagrees with dense occupancy");
  }
}

void test_sphere_occupancy_query() {
  constexpr int max_depth = 6;
  constexpr int grid_size = 1 << max_depth;
  constexpr int total_voxels = grid_size * grid_size * grid_size;
  const glm::vec3 center{32.0f, 32.0f, 32.0f};
  constexpr float radius = 18.0f;

  svo::BuildOptions options;
  options.max_depth = max_depth;

  std::vector<std::uint8_t> dense_values(total_voxels, 0u);
  std::vector<glm::ivec3> sphere_voxels;

  for (int z = 0; z < grid_size; ++z) {
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        const glm::ivec3 coordinate{x, y, z};
        if (!sphere_contains_voxel_center(coordinate, center, radius)) {
          continue;
        }
        dense_values[flat_index(grid_size, x, y, z)] = 1u;
        sphere_voxels.push_back(coordinate);
      }
    }
  }

  const svo::Octree octree = svo::Octree::from_voxels_cpu(sphere_voxels, options);
  require(
      static_cast<std::size_t>(octree.num_leaves()) == sphere_voxels.size(),
      "sphere leaf count must match occupied voxel count");

  std::vector<int> payload_values(static_cast<std::size_t>(octree.num_leaves()), 1);

  const std::vector<glm::vec3> probe_points{
      voxel_center_point(grid_size, {32, 32, 32}),
      voxel_center_point(grid_size, {32, 32, 49}),
      voxel_center_point(grid_size, {0, 0, 0}),
      voxel_center_point(grid_size, {63, 63, 63}),
      {-0.01f, 0.5f, 0.5f},
      {1.01f, 0.5f, 0.5f},
  };

  const std::vector<std::int32_t> probe_hits = svo::query_points(octree, probe_points);
  require(probe_hits[0] >= 0, "sphere center should be inside the sphere");
  require(probe_hits[1] >= 0, "point near the sphere surface should still be inside");
  require(probe_hits[2] == -1, "far corner should be outside the sphere");
  require(probe_hits[3] == -1, "opposite far corner should be outside the sphere");
  require(probe_hits[4] == -1, "point outside root bounds should miss");
  require(probe_hits[5] == -1, "point outside root max should miss");
  require(payload_values[static_cast<std::size_t>(probe_hits[0])] == 1, "inside sphere leaf value must be 1");
  require(payload_values[static_cast<std::size_t>(probe_hits[1])] == 1, "inside sphere leaf value must be 1");

  for (int z = 0; z < grid_size; ++z) {
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        const glm::ivec3 coordinate{x, y, z};
        const glm::vec3 point = voxel_center_point(grid_size, coordinate);
        const std::int32_t hit = svo::query_points(octree, {point}).front();
        const int queried_value =
            hit >= 0 ? payload_values[static_cast<std::size_t>(hit)] : 0;
        const int expected_value =
            static_cast<int>(dense_values[flat_index(grid_size, x, y, z)]);
        require(
            queried_value == expected_value,
            "sphere occupancy query disagrees with dense reference");
      }
    }
  }
}

void test_wide_query_matches_octree8() {
  svo::BuildOptions options;
  options.max_depth = 4;

  const std::vector<glm::ivec3> coordinates{
      {0, 0, 0},
      {1, 2, 3},
      {4, 4, 4},
      {7, 8, 9},
      {15, 15, 15},
  };
  const std::vector<std::uint32_t> payload_indices{10u, 11u, 12u, 13u, 14u};

  const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, payload_indices, options);
  options.branching = svo::BranchingMode::Wide4;
  const svo::Octree wide = svo::Octree::from_voxels_cpu(coordinates, payload_indices, options);

  const std::vector<glm::vec3> points{
      voxel_center_point(16, {0, 0, 0}),
      voxel_center_point(16, {1, 2, 3}),
      voxel_center_point(16, {4, 4, 4}),
      voxel_center_point(16, {7, 8, 9}),
      voxel_center_point(16, {15, 15, 15}),
      voxel_center_point(16, {2, 2, 2}),
      {0.25f, 0.25f, 0.25f},
      {1.0f, 0.5f, 0.5f},
  };

  require(svo::query_points(wide, points) == svo::query_points(octree, points), "wide query leaf ids should match octree8");
  svo::QueryOptions payload_options;
  payload_options.return_payload_indices = true;
  require(
      svo::query_points(wide, points, payload_options) == svo::query_points(octree, points, payload_options),
      "wide query payload ids should match octree8");
}

}  // namespace

int main() {
  test_basic_cpu_query();
  test_custom_payload_index_query();
  test_random_reference_query();
  test_sphere_occupancy_query();
  test_wide_query_matches_octree8();
  return 0;
}
