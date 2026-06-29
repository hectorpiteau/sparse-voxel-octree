#include <svo/Builder.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <svo/Error.hpp>

namespace svo {
namespace {

constexpr std::uint32_t kDescriptorFieldLimit =
    static_cast<std::uint32_t>((1u << NodeDescriptor::kChildBaseBits) - 1u);

bool lexicographic_less(const glm::ivec3& lhs, const glm::ivec3& rhs) noexcept {
  if (lhs.x != rhs.x) {
    return lhs.x < rhs.x;
  }
  if (lhs.y != rhs.y) {
    return lhs.y < rhs.y;
  }
  return lhs.z < rhs.z;
}

bool lexicographic_equal(const glm::ivec3& lhs, const glm::ivec3& rhs) noexcept {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

int child_index_for_depth(const glm::ivec3& coordinate, int child_depth) noexcept {
  const int x_bit = (coordinate.x >> child_depth) & 1;
  const int y_bit = (coordinate.y >> child_depth) & 1;
  const int z_bit = (coordinate.z >> child_depth) & 1;
  return x_bit | (y_bit << 1) | (z_bit << 2);
}

void check_descriptor_capacity(std::size_t value, const char* field_name) {
  if (value > static_cast<std::size_t>(kDescriptorFieldLimit)) {
    throw ValidationError(std::string(field_name) + " exceeds descriptor capacity");
  }
}

void emit_node_at(
    std::size_t node_index,
    const std::vector<glm::ivec3>& coordinates,
    int depth_remaining,
    std::vector<NodeDescriptor>& nodes,
    std::vector<std::uint32_t>& leaf_payload_indices) {
  std::array<std::vector<glm::ivec3>, 8> buckets;
  std::uint8_t child_mask = 0u;
  std::uint8_t leaf_mask = 0u;

  if (depth_remaining <= 0) {
    throw ValidationError("builder encountered a non-positive depth while emitting a node");
  }

  for (const glm::ivec3& coordinate : coordinates) {
    const int child_index = child_index_for_depth(coordinate, depth_remaining - 1);
    buckets[static_cast<std::size_t>(child_index)].push_back(coordinate);
  }

  if (depth_remaining == 1) {
    const std::size_t payload_base = leaf_payload_indices.size();
    check_descriptor_capacity(payload_base, "payload_base");

    for (std::size_t child_index = 0; child_index < buckets.size(); ++child_index) {
      if (buckets[child_index].empty()) {
        continue;
      }

      child_mask = static_cast<std::uint8_t>(child_mask | (1u << child_index));
      leaf_mask = static_cast<std::uint8_t>(leaf_mask | (1u << child_index));

      check_descriptor_capacity(leaf_payload_indices.size(), "payload index");
      leaf_payload_indices.push_back(static_cast<std::uint32_t>(leaf_payload_indices.size()));
    }

    nodes[node_index] = NodeDescriptor::pack(child_mask, leaf_mask, 0u, static_cast<std::uint32_t>(payload_base));
    return;
  }

  std::array<std::size_t, 8> child_root_indices{};
  std::size_t internal_child_count = 0;

  for (std::size_t child_index = 0; child_index < buckets.size(); ++child_index) {
    if (buckets[child_index].empty()) {
      continue;
    }

    child_mask = static_cast<std::uint8_t>(child_mask | (1u << child_index));
    child_root_indices[child_index] = internal_child_count;
    ++internal_child_count;
  }

  const std::size_t child_base = nodes.size();
  check_descriptor_capacity(child_base, "child_base");
  check_descriptor_capacity(child_base + internal_child_count, "child span");

  nodes.resize(nodes.size() + internal_child_count);
  nodes[node_index] = NodeDescriptor::pack(child_mask, leaf_mask, static_cast<std::uint32_t>(child_base), 0u);

  for (std::size_t child_index = 0; child_index < buckets.size(); ++child_index) {
    if (buckets[child_index].empty()) {
      continue;
    }

    const std::size_t child_node_index = child_base + child_root_indices[child_index];
    emit_node_at(child_node_index, buckets[child_index], depth_remaining - 1, nodes, leaf_payload_indices);
  }
}

std::vector<glm::ivec3> validate_and_normalize_coordinates(
    const std::vector<glm::ivec3>& coordinates,
    const BuildOptions& options) {
  if (options.max_depth < 0 || options.max_depth > 30) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }

  const std::int64_t upper_bound = static_cast<std::int64_t>(1) << options.max_depth;
  std::vector<glm::ivec3> unique_coordinates = coordinates;

  for (const glm::ivec3& coordinate : unique_coordinates) {
    if (coordinate.x < 0 || coordinate.y < 0 || coordinate.z < 0) {
      throw ValidationError("voxel coordinates must be non-negative");
    }

    if (coordinate.x >= upper_bound || coordinate.y >= upper_bound || coordinate.z >= upper_bound) {
      throw ValidationError("voxel coordinates must be inside [0, 2^max_depth)");
    }
  }

  std::sort(unique_coordinates.begin(), unique_coordinates.end(), lexicographic_less);
  unique_coordinates.erase(
      std::unique(unique_coordinates.begin(), unique_coordinates.end(), lexicographic_equal),
      unique_coordinates.end());

  check_descriptor_capacity(unique_coordinates.size(), "leaf count");
  return unique_coordinates;
}

}  // namespace

Octree build_octree_cpu(const std::vector<glm::ivec3>& coordinates, const BuildOptions& options) {
  std::vector<glm::ivec3> unique_coordinates = validate_and_normalize_coordinates(coordinates, options);

  if (unique_coordinates.empty()) {
    Octree octree{options.max_depth, Device::CPU, options.root_bounds, {}, {}};
    octree.validate();
    return octree;
  }

  if (options.max_depth == 0) {
    std::vector<NodeDescriptor> nodes{
        NodeDescriptor::pack(0b00000001u, 0b00000001u, 0u, 0u)};
    std::vector<std::uint32_t> leaf_payload_indices{0u};

    Octree octree{
        options.max_depth,
        Device::CPU,
        options.root_bounds,
        std::move(nodes),
        std::move(leaf_payload_indices)};
    octree.validate();
    return octree;
  }

  std::vector<NodeDescriptor> nodes(1);
  std::vector<std::uint32_t> leaf_payload_indices;
  leaf_payload_indices.reserve(unique_coordinates.size());

  emit_node_at(0, unique_coordinates, options.max_depth, nodes, leaf_payload_indices);

  Octree octree{
      options.max_depth,
      Device::CPU,
      options.root_bounds,
      std::move(nodes),
      std::move(leaf_payload_indices)};
  octree.validate();
  return octree;
}

Octree Octree::from_voxels_cpu(
    const std::vector<glm::ivec3>& coordinates,
    const BuildOptions& options) {
  return build_octree_cpu(coordinates, options);
}

}  // namespace svo
