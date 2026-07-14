#include <svo/Builder.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <svo/Error.hpp>

namespace svo {
namespace {

constexpr std::uint32_t kDescriptorFieldLimit =
    static_cast<std::uint32_t>((1u << NodeDescriptor::kChildBaseBits) - 1u);
constexpr std::uint64_t kWideChildCount = 64;

struct BuildVoxel {
  glm::ivec3 coordinate{};
  std::uint32_t payload_index = 0u;
};

bool lexicographic_less(const glm::ivec3& lhs, const glm::ivec3& rhs) noexcept;
bool lexicographic_equal(const glm::ivec3& lhs, const glm::ivec3& rhs) noexcept;
void validate_build_options_and_coordinate_bounds(
    const std::vector<glm::ivec3>& coordinates,
    const BuildOptions& options);

bool leaf_spec_less(const LeafSpec& lhs, const LeafSpec& rhs) noexcept {
  if (!lexicographic_equal(lhs.coord_min, rhs.coord_min)) {
    return lexicographic_less(lhs.coord_min, rhs.coord_min);
  }
  if (lhs.depth != rhs.depth) {
    return lhs.depth < rhs.depth;
  }
  return lhs.payload_index < rhs.payload_index;
}

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

bool build_voxel_less(const BuildVoxel& lhs, const BuildVoxel& rhs) noexcept {
  return lexicographic_less(lhs.coordinate, rhs.coordinate);
}

bool build_voxel_same_coordinate(const BuildVoxel& lhs, const BuildVoxel& rhs) noexcept {
  return lexicographic_equal(lhs.coordinate, rhs.coordinate);
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

void check_descriptor_capacity(std::size_t value, const char* field_name) {
  if (value > static_cast<std::size_t>(kDescriptorFieldLimit)) {
    throw ValidationError(std::string(field_name) + " exceeds descriptor capacity");
  }
}

void check_wide_descriptor_capacity(std::size_t value, const char* field_name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw ValidationError(std::string(field_name) + " exceeds wide descriptor capacity");
  }
}

int cell_size_for_depth(int max_depth, int depth) {
  return 1 << (max_depth - depth);
}

void validate_leaf_spec_bounds_and_alignment(const LeafSpec& spec, const BuildOptions& options) {
  if (spec.depth < 0 || spec.depth > options.max_depth) {
    throw ValidationError("leaf spec depth must be in [0, max_depth]");
  }
  if (options.max_depth > 0 && spec.depth == 0) {
    throw ValidationError("root leaf specs are only supported when max_depth is zero");
  }
  if (spec.payload_index > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    throw ValidationError("leaf spec payload_index must be representable as int32");
  }

  const int cell_size = cell_size_for_depth(options.max_depth, spec.depth);
  const std::int64_t upper_bound = static_cast<std::int64_t>(1) << options.max_depth;
  const glm::ivec3 max_corner = spec.coord_min + glm::ivec3{cell_size, cell_size, cell_size};

  if (spec.coord_min.x < 0 || spec.coord_min.y < 0 || spec.coord_min.z < 0) {
    throw ValidationError("leaf spec coord_min must be non-negative");
  }
  if (max_corner.x > upper_bound || max_corner.y > upper_bound || max_corner.z > upper_bound) {
    throw ValidationError("leaf spec bounds must be inside [0, 2^max_depth)");
  }
  if ((spec.coord_min.x % cell_size) != 0 || (spec.coord_min.y % cell_size) != 0 ||
      (spec.coord_min.z % cell_size) != 0) {
    throw ValidationError("leaf spec coord_min must be aligned to its depth");
  }
}

std::vector<LeafSpec> validate_and_normalize_leaf_specs(
    const std::vector<LeafSpec>& input_specs,
    const BuildOptions& options) {
  validate_build_options_and_coordinate_bounds({}, options);
  if (options.branching != BranchingMode::Octree8) {
    throw ValidationError("adaptive leaf specs currently support only octree8 branching");
  }

  std::vector<LeafSpec> specs = input_specs;
  for (const LeafSpec& spec : specs) {
    validate_leaf_spec_bounds_and_alignment(spec, options);
  }

  std::sort(specs.begin(), specs.end(), leaf_spec_less);
  check_descriptor_capacity(specs.size(), "leaf count");
  return specs;
}

void emit_leaf_spec_node_at(
    std::size_t node_index,
    const std::vector<LeafSpec>& specs,
    int depth,
    const BuildOptions& options,
    std::vector<NodeDescriptor>& nodes,
    std::vector<std::uint32_t>& leaf_payload_indices,
    std::vector<LeafSpec>& ordered_leaf_specs) {
  if (depth >= options.max_depth) {
    throw ValidationError("leaf spec builder encountered a non-leaf at max depth");
  }

  std::array<std::vector<LeafSpec>, 8> buckets;
  std::uint8_t child_mask = 0u;
  std::uint8_t leaf_mask = 0u;

  const int child_depth = depth + 1;
  const int child_bit_depth = options.max_depth - child_depth;
  for (const LeafSpec& spec : specs) {
    const int child_index = child_index_for_depth(spec.coord_min, child_bit_depth);
    buckets[static_cast<std::size_t>(child_index)].push_back(spec);
  }

  std::array<std::size_t, 8> child_root_indices{};
  std::size_t internal_child_count = 0;
  const std::size_t payload_base = leaf_payload_indices.size();
  check_descriptor_capacity(payload_base, "payload_base");

  for (std::size_t child_index = 0; child_index < buckets.size(); ++child_index) {
    if (buckets[child_index].empty()) {
      continue;
    }

    const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);
    child_mask = static_cast<std::uint8_t>(child_mask | child_bit);

    std::optional<LeafSpec> leaf_spec;
    std::vector<LeafSpec> child_specs;
    child_specs.reserve(buckets[child_index].size());
    for (const LeafSpec& spec : buckets[child_index]) {
      if (spec.depth == child_depth) {
        if (leaf_spec.has_value()) {
          throw ValidationError("duplicate leaf specs are not allowed");
        }
        leaf_spec = spec;
      } else {
        child_specs.push_back(spec);
      }
    }

    if (leaf_spec.has_value() && !child_specs.empty()) {
      throw ValidationError("leaf specs cannot overlap parent and child cells");
    }
    if (leaf_spec.has_value()) {
      leaf_mask = static_cast<std::uint8_t>(leaf_mask | child_bit);
      check_descriptor_capacity(leaf_payload_indices.size(), "payload index");
      leaf_payload_indices.push_back(leaf_spec->payload_index);
      ordered_leaf_specs.push_back(*leaf_spec);
      continue;
    }

    child_root_indices[child_index] = internal_child_count;
    ++internal_child_count;
  }

  const std::size_t child_base = nodes.size();
  check_descriptor_capacity(child_base, "child_base");
  check_descriptor_capacity(child_base + internal_child_count, "child span");
  nodes.resize(nodes.size() + internal_child_count);
  nodes[node_index] = NodeDescriptor::pack(child_mask, leaf_mask, static_cast<std::uint32_t>(child_base), static_cast<std::uint32_t>(payload_base));

  for (std::size_t child_index = 0; child_index < buckets.size(); ++child_index) {
    if (buckets[child_index].empty()) {
      continue;
    }
    bool has_leaf = false;
    std::vector<LeafSpec> child_specs;
    child_specs.reserve(buckets[child_index].size());
    for (const LeafSpec& spec : buckets[child_index]) {
      if (spec.depth == child_depth) {
        has_leaf = true;
      } else {
        child_specs.push_back(spec);
      }
    }
    if (has_leaf) {
      continue;
    }
    const std::size_t child_node_index = child_base + child_root_indices[child_index];
    emit_leaf_spec_node_at(
        child_node_index,
        child_specs,
        child_depth,
        options,
        nodes,
        leaf_payload_indices,
        ordered_leaf_specs);
  }
}

void emit_node_at(
    std::size_t node_index,
    const std::vector<BuildVoxel>& voxels,
    int depth_remaining,
    bool use_custom_payload_indices,
    std::vector<NodeDescriptor>& nodes,
    std::vector<std::uint32_t>& leaf_payload_indices,
    std::vector<LeafSpec>* ordered_leaf_specs) {
  std::array<std::vector<BuildVoxel>, 8> buckets;
  std::uint8_t child_mask = 0u;
  std::uint8_t leaf_mask = 0u;

  if (depth_remaining <= 0) {
    throw ValidationError("builder encountered a non-positive depth while emitting a node");
  }

  for (const BuildVoxel& voxel : voxels) {
    const int child_index = child_index_for_depth(voxel.coordinate, depth_remaining - 1);
    buckets[static_cast<std::size_t>(child_index)].push_back(voxel);
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
      const std::uint32_t payload_index = use_custom_payload_indices
          ? buckets[child_index].front().payload_index
          : static_cast<std::uint32_t>(leaf_payload_indices.size());
      leaf_payload_indices.push_back(payload_index);
      if (ordered_leaf_specs != nullptr) {
        ordered_leaf_specs->push_back(LeafSpec{buckets[child_index].front().coordinate, 0, payload_index});
      }
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
    emit_node_at(
        child_node_index,
        buckets[child_index],
        depth_remaining - 1,
        use_custom_payload_indices,
        nodes,
        leaf_payload_indices,
        ordered_leaf_specs);
  }
}

void emit_wide_node_at(
    std::size_t node_index,
    const std::vector<BuildVoxel>& voxels,
    int depth_remaining,
    bool use_custom_payload_indices,
    std::vector<WideNodeDescriptor>& nodes,
    std::vector<std::uint32_t>& leaf_payload_indices) {
  std::array<std::vector<BuildVoxel>, kWideChildCount> buckets;
  std::uint64_t child_mask = 0u;
  std::uint64_t leaf_mask = 0u;

  if (depth_remaining <= 0 || (depth_remaining % 2) != 0) {
    throw ValidationError("wide4 builder encountered an invalid depth while emitting a node");
  }

  for (const BuildVoxel& voxel : voxels) {
    const int child_index = wide_child_index_for_depth(voxel.coordinate, depth_remaining - 2);
    buckets[static_cast<std::size_t>(child_index)].push_back(voxel);
  }

  if (depth_remaining == 2) {
    const std::size_t payload_base = leaf_payload_indices.size();
    check_wide_descriptor_capacity(payload_base, "payload_base");

    for (std::size_t child_index = 0; child_index < buckets.size(); ++child_index) {
      if (buckets[child_index].empty()) {
        continue;
      }

      const std::uint64_t child_bit = 1ull << child_index;
      child_mask |= child_bit;
      leaf_mask |= child_bit;

      check_wide_descriptor_capacity(leaf_payload_indices.size(), "payload index");
      const std::uint32_t payload_index = use_custom_payload_indices
          ? buckets[child_index].front().payload_index
          : static_cast<std::uint32_t>(leaf_payload_indices.size());
      leaf_payload_indices.push_back(payload_index);
    }

    nodes[node_index] =
        WideNodeDescriptor::pack(child_mask, leaf_mask, 0u, static_cast<std::uint32_t>(payload_base));
    return;
  }

  std::array<std::size_t, kWideChildCount> child_root_indices{};
  std::size_t internal_child_count = 0;

  for (std::size_t child_index = 0; child_index < buckets.size(); ++child_index) {
    if (buckets[child_index].empty()) {
      continue;
    }

    child_mask |= 1ull << child_index;
    child_root_indices[child_index] = internal_child_count;
    ++internal_child_count;
  }

  const std::size_t child_base = nodes.size();
  check_wide_descriptor_capacity(child_base, "child_base");
  check_wide_descriptor_capacity(child_base + internal_child_count, "child span");

  nodes.resize(nodes.size() + internal_child_count);
  nodes[node_index] = WideNodeDescriptor::pack(child_mask, leaf_mask, static_cast<std::uint32_t>(child_base), 0u);

  for (std::size_t child_index = 0; child_index < buckets.size(); ++child_index) {
    if (buckets[child_index].empty()) {
      continue;
    }

    const std::size_t child_node_index = child_base + child_root_indices[child_index];
    emit_wide_node_at(
        child_node_index,
        buckets[child_index],
        depth_remaining - 2,
        use_custom_payload_indices,
        nodes,
        leaf_payload_indices);
  }
}

void validate_build_options_and_coordinate_bounds(
    const std::vector<glm::ivec3>& coordinates,
    const BuildOptions& options) {
  if (options.max_depth < 0 || options.max_depth > 30) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }
  if (options.branching == BranchingMode::Wide4 && (options.max_depth % 2) != 0) {
    throw ValidationError("wide4 trees require an even max_depth");
  }

  const std::int64_t upper_bound = static_cast<std::int64_t>(1) << options.max_depth;

  for (const glm::ivec3& coordinate : coordinates) {
    if (coordinate.x < 0 || coordinate.y < 0 || coordinate.z < 0) {
      throw ValidationError("voxel coordinates must be non-negative");
    }

    if (coordinate.x >= upper_bound || coordinate.y >= upper_bound || coordinate.z >= upper_bound) {
      throw ValidationError("voxel coordinates must be inside [0, 2^max_depth)");
    }
  }
}

std::vector<BuildVoxel> validate_and_normalize_identity_voxels(
    const std::vector<glm::ivec3>& coordinates,
    const BuildOptions& options) {
  validate_build_options_and_coordinate_bounds(coordinates, options);

  std::vector<glm::ivec3> unique_coordinates = coordinates;
  std::sort(unique_coordinates.begin(), unique_coordinates.end(), lexicographic_less);
  unique_coordinates.erase(
      std::unique(unique_coordinates.begin(), unique_coordinates.end(), lexicographic_equal),
      unique_coordinates.end());

  check_descriptor_capacity(unique_coordinates.size(), "leaf count");

  std::vector<BuildVoxel> voxels;
  voxels.reserve(unique_coordinates.size());
  for (const glm::ivec3& coordinate : unique_coordinates) {
    voxels.push_back(BuildVoxel{coordinate, 0u});
  }
  return voxels;
}

std::vector<BuildVoxel> validate_and_normalize_payload_voxels(
    const std::vector<glm::ivec3>& coordinates,
    const std::vector<std::uint32_t>& payload_indices,
    const BuildOptions& options) {
  validate_build_options_and_coordinate_bounds(coordinates, options);

  if (payload_indices.size() != coordinates.size()) {
    throw ValidationError("payload_indices must have the same length as coordinates");
  }

  constexpr std::uint32_t kMaxReturnedIndex = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
  std::vector<BuildVoxel> voxels;
  voxels.reserve(coordinates.size());

  for (std::size_t index = 0; index < coordinates.size(); ++index) {
    if (payload_indices[index] > kMaxReturnedIndex) {
      throw ValidationError("payload_indices must be representable as int32");
    }
    voxels.push_back(BuildVoxel{coordinates[index], payload_indices[index]});
  }

  std::sort(voxels.begin(), voxels.end(), build_voxel_less);

  for (std::size_t index = 1; index < voxels.size(); ++index) {
    if (build_voxel_same_coordinate(voxels[index - 1], voxels[index]) &&
        voxels[index - 1].payload_index != voxels[index].payload_index) {
      throw ValidationError("duplicate voxel coordinates must have matching payload indices");
    }
  }

  voxels.erase(
      std::unique(voxels.begin(), voxels.end(), build_voxel_same_coordinate),
      voxels.end());

  check_descriptor_capacity(voxels.size(), "leaf count");
  return voxels;
}

Octree build_octree_from_voxels(
    std::vector<BuildVoxel> voxels,
    const BuildOptions& options,
    bool use_custom_payload_indices) {
  if (voxels.empty()) {
    Octree octree{
        options.max_depth,
        Device::CPU,
        options.root_bounds,
        options.branching,
        {},
        {},
        {},
        {}};
    octree.validate();
    return octree;
  }

  if (options.branching == BranchingMode::Wide4) {
    if (options.max_depth == 0) {
      const std::uint32_t root_payload_index = use_custom_payload_indices ? voxels.front().payload_index : 0u;
      std::vector<WideNodeDescriptor> nodes{
          WideNodeDescriptor::pack(1ull, 1ull, 0u, 0u)};
      std::vector<std::uint32_t> leaf_payload_indices{root_payload_index};

      Octree octree{
          options.max_depth,
          Device::CPU,
          options.root_bounds,
          options.branching,
          {},
          std::move(nodes),
          std::move(leaf_payload_indices),
          {LeafSpec{glm::ivec3{0, 0, 0}, 0, root_payload_index}}};
      octree.validate();
      return octree;
    }

    std::vector<WideNodeDescriptor> nodes(1);
    std::vector<std::uint32_t> leaf_payload_indices;
    leaf_payload_indices.reserve(voxels.size());

    emit_wide_node_at(0, voxels, options.max_depth, use_custom_payload_indices, nodes, leaf_payload_indices);

    Octree octree{
        options.max_depth,
        Device::CPU,
        options.root_bounds,
        options.branching,
        {},
        std::move(nodes),
        std::move(leaf_payload_indices),
        {}};
    octree.validate();
    return octree;
  }

  if (options.max_depth == 0) {
    const std::uint32_t root_payload_index = use_custom_payload_indices ? voxels.front().payload_index : 0u;
    std::vector<NodeDescriptor> nodes{
        NodeDescriptor::pack(0b00000001u, 0b00000001u, 0u, 0u)};
    std::vector<std::uint32_t> leaf_payload_indices{root_payload_index};

    Octree octree{
        options.max_depth,
        Device::CPU,
        options.root_bounds,
        std::move(nodes),
        std::move(leaf_payload_indices),
        {LeafSpec{glm::ivec3{0, 0, 0}, 0, root_payload_index}}};
    octree.validate();
    return octree;
  }

  std::vector<NodeDescriptor> nodes(1);
  std::vector<std::uint32_t> leaf_payload_indices;
  leaf_payload_indices.reserve(voxels.size());

  std::vector<LeafSpec> leaf_specs;
  leaf_specs.reserve(voxels.size());
  emit_node_at(
      0,
      voxels,
      options.max_depth,
      use_custom_payload_indices,
      nodes,
      leaf_payload_indices,
      &leaf_specs);
  for (LeafSpec& spec : leaf_specs) {
    spec.depth = options.max_depth;
  }

  Octree octree{
      options.max_depth,
      Device::CPU,
      options.root_bounds,
      std::move(nodes),
      std::move(leaf_payload_indices),
      std::move(leaf_specs)};
  octree.validate();
  return octree;
}

}  // namespace

Octree build_octree_cpu(const std::vector<glm::ivec3>& coordinates, const BuildOptions& options) {
  return build_octree_from_voxels(
      validate_and_normalize_identity_voxels(coordinates, options),
      options,
      false);
}

Octree build_octree_cpu(
    const std::vector<glm::ivec3>& coordinates,
    const std::vector<std::uint32_t>& payload_indices,
    const BuildOptions& options) {
  return build_octree_from_voxels(
      validate_and_normalize_payload_voxels(coordinates, payload_indices, options),
      options,
      true);
}

Octree build_octree_from_leaf_specs_cpu(
    const std::vector<LeafSpec>& leaf_specs,
    const BuildOptions& options) {
  const std::vector<LeafSpec> specs = validate_and_normalize_leaf_specs(leaf_specs, options);
  if (specs.empty()) {
    Octree octree{
        options.max_depth,
        Device::CPU,
        options.root_bounds,
        options.branching,
        {},
        {},
        {},
        {}};
    octree.validate();
    return octree;
  }

  if (options.max_depth == 0) {
    if (specs.size() != 1 || specs.front().depth != 0 ||
        !lexicographic_equal(specs.front().coord_min, glm::ivec3{0, 0, 0})) {
      throw ValidationError("max_depth zero trees require exactly one root leaf spec");
    }
    std::vector<NodeDescriptor> nodes{
        NodeDescriptor::pack(0b00000001u, 0b00000001u, 0u, 0u)};
    std::vector<std::uint32_t> leaf_payload_indices{specs.front().payload_index};
    Octree octree{
        options.max_depth,
        Device::CPU,
        options.root_bounds,
        std::move(nodes),
        std::move(leaf_payload_indices),
        specs};
    octree.validate();
    return octree;
  }

  std::vector<NodeDescriptor> nodes(1);
  std::vector<std::uint32_t> leaf_payload_indices;
  std::vector<LeafSpec> ordered_leaf_specs;
  leaf_payload_indices.reserve(specs.size());
  ordered_leaf_specs.reserve(specs.size());
  emit_leaf_spec_node_at(0, specs, 0, options, nodes, leaf_payload_indices, ordered_leaf_specs);

  Octree octree{
      options.max_depth,
      Device::CPU,
      options.root_bounds,
      std::move(nodes),
      std::move(leaf_payload_indices),
      std::move(ordered_leaf_specs)};
  octree.validate();
  return octree;
}

Octree Octree::from_voxels_cpu(
    const std::vector<glm::ivec3>& coordinates,
    const BuildOptions& options) {
  return build_octree_cpu(coordinates, options);
}

Octree Octree::from_voxels_cpu(
    const std::vector<glm::ivec3>& coordinates,
    const std::vector<std::uint32_t>& payload_indices,
    const BuildOptions& options) {
  return build_octree_cpu(coordinates, payload_indices, options);
}

Octree Octree::from_leaf_specs_cpu(
    const std::vector<LeafSpec>& leaf_specs,
    const BuildOptions& options) {
  return build_octree_from_leaf_specs_cpu(leaf_specs, options);
}

}  // namespace svo
