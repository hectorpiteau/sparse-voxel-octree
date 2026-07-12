#include <svo/Query.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace svo {
namespace {

int count_bits(std::uint8_t value) noexcept {
  return std::popcount(value);
}

int count_bits(std::uint64_t value) noexcept {
  return std::popcount(value);
}

int child_index_for_depth(const glm::ivec3& coordinate, int child_depth) noexcept {
  const int x_bit = (coordinate.x >> child_depth) & 1;
  const int y_bit = (coordinate.y >> child_depth) & 1;
  const int z_bit = (coordinate.z >> child_depth) & 1;
  return x_bit | (y_bit << 1) | (z_bit << 2);
}

int wide_child_index_for_depth(const glm::ivec3& coordinate, int child_depth) noexcept {
  const int x_bits = (coordinate.x >> child_depth) & 3;
  const int y_bits = (coordinate.y >> child_depth) & 3;
  const int z_bits = (coordinate.z >> child_depth) & 3;
  return x_bits | (y_bits << 2) | (z_bits << 4);
}

int prefix_rank(std::uint8_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint8_t lower_mask = static_cast<std::uint8_t>((1u << child_index) - 1u);
  return count_bits(static_cast<std::uint8_t>(mask & lower_mask));
}

int prefix_rank(std::uint64_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint64_t lower_mask = (1ull << child_index) - 1ull;
  return count_bits(mask & lower_mask);
}

bool contains_point(const RootBounds& bounds, const glm::vec3& point) noexcept {
  return point.x >= bounds[0].x && point.y >= bounds[0].y && point.z >= bounds[0].z &&
         point.x < bounds[1].x && point.y < bounds[1].y && point.z < bounds[1].z;
}

glm::ivec3 point_to_voxel_coord(const Octree& octree, const glm::vec3& point) noexcept {
  const RootBounds& bounds = octree.root_bounds();
  const glm::vec3 extent = bounds[1] - bounds[0];
  const glm::vec3 normalized = (point - bounds[0]) / extent;
  const float scale = static_cast<float>(static_cast<std::uint32_t>(1u) << octree.max_depth());

  return glm::ivec3{
      static_cast<int>(std::floor(normalized.x * scale)),
      static_cast<int>(std::floor(normalized.y * scale)),
      static_cast<int>(std::floor(normalized.z * scale))};
}

std::int32_t query_single_point(
    const Octree& octree,
    const glm::vec3& point,
    const QueryOptions& options) noexcept {
  if (!contains_point(octree.root_bounds(), point) || octree.nodes().empty()) {
    return -1;
  }

  if (octree.max_depth() == 0) {
    if (octree.leaf_payload_indices().empty()) {
      return -1;
    }
    return options.return_payload_indices
        ? static_cast<std::int32_t>(octree.leaf_payload_indices().front())
        : 0;
  }

  const glm::ivec3 voxel = point_to_voxel_coord(octree, point);
  std::size_t node_index = 0;
  int depth_remaining = octree.max_depth();

  while (true) {
    const NodeDescriptor descriptor = octree.nodes()[node_index];
    const int child_index = child_index_for_depth(voxel, depth_remaining - 1);
    const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);

    if ((descriptor.child_mask() & child_bit) == 0u) {
      return -1;
    }

    if ((descriptor.leaf_mask() & child_bit) != 0u) {
      const std::size_t leaf_id = static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank(descriptor.leaf_mask(), child_index));

      if (options.return_payload_indices) {
        return static_cast<std::int32_t>(octree.leaf_payload_indices()[leaf_id]);
      }
      return static_cast<std::int32_t>(leaf_id);
    }

    const std::uint8_t internal_mask = descriptor.internal_child_mask();
    node_index = static_cast<std::size_t>(descriptor.child_base()) +
        static_cast<std::size_t>(prefix_rank(internal_mask, child_index));
    --depth_remaining;

    if (depth_remaining <= 0) {
      return -1;
    }
  }
}

std::int32_t query_single_point_wide(
    const Octree& octree,
    const glm::vec3& point,
    const QueryOptions& options) noexcept {
  if (!contains_point(octree.root_bounds(), point) || octree.wide_nodes().empty()) {
    return -1;
  }

  if (octree.max_depth() == 0) {
    if (octree.leaf_payload_indices().empty()) {
      return -1;
    }
    return options.return_payload_indices
        ? static_cast<std::int32_t>(octree.leaf_payload_indices().front())
        : 0;
  }

  const glm::ivec3 voxel = point_to_voxel_coord(octree, point);
  std::size_t node_index = 0;
  int depth_remaining = octree.max_depth();

  while (true) {
    if (node_index >= octree.wide_nodes().size()) {
      return -1;
    }

    const WideNodeDescriptor descriptor = octree.wide_nodes()[node_index];
    const int child_index = wide_child_index_for_depth(voxel, depth_remaining - 2);
    const std::uint64_t child_bit = 1ull << child_index;

    if ((descriptor.child_mask() & child_bit) == 0u) {
      return -1;
    }

    if ((descriptor.leaf_mask() & child_bit) != 0u) {
      const std::size_t leaf_id = static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank(descriptor.leaf_mask(), child_index));

      if (options.return_payload_indices) {
        return static_cast<std::int32_t>(octree.leaf_payload_indices()[leaf_id]);
      }
      return static_cast<std::int32_t>(leaf_id);
    }

    const std::uint64_t internal_mask = descriptor.internal_child_mask();
    node_index = static_cast<std::size_t>(descriptor.child_base()) +
        static_cast<std::size_t>(prefix_rank(internal_mask, child_index));
    depth_remaining -= 2;

    if (depth_remaining <= 0) {
      return -1;
    }
  }
}

}  // namespace

std::vector<std::int32_t> query_points(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const QueryOptions& options) {
  std::vector<std::int32_t> results;
  results.reserve(points.size());

  for (const glm::vec3& point : points) {
    results.push_back(octree.branching() == BranchingMode::Wide4
        ? query_single_point_wide(octree, point, options)
        : query_single_point(octree, point, options));
  }

  return results;
}

std::vector<std::int32_t> query_payload_indices(
    const Octree& octree,
    const std::vector<glm::vec3>& points) {
  QueryOptions options;
  options.return_payload_indices = true;
  return query_points(octree, points, options);
}

}  // namespace svo
