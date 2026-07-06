#include <svo/Octree.hpp>

#include <bit>
#include <cmath>
#include <string>
#include <utility>

#include <svo/Error.hpp>

namespace svo {
namespace {

int count_bits(std::uint8_t value) noexcept {
  return std::popcount(value);
}

bool is_finite_vec3(const glm::vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void validate_root_bounds(const RootBounds& bounds) {
  if (!is_finite_vec3(bounds[0]) || !is_finite_vec3(bounds[1])) {
    throw ValidationError("root bounds must contain only finite values");
  }

  if (!(bounds[0].x < bounds[1].x && bounds[0].y < bounds[1].y && bounds[0].z < bounds[1].z)) {
    throw ValidationError("root bounds min corner must be strictly less than max corner");
  }
}

}  // namespace

const char* device_name(Device device) noexcept {
  switch (device) {
    case Device::CPU:
      return "cpu";
    case Device::CUDA:
      return "cuda";
  }

  return "unknown";
}

Octree::Octree(
    int max_depth,
    Device device,
    RootBounds root_bounds,
    std::vector<NodeDescriptor> nodes,
    std::vector<std::uint32_t> leaf_payload_indices)
    : max_depth_(max_depth),
      device_(device),
      root_bounds_(std::move(root_bounds)),
      nodes_(std::move(nodes)),
      leaf_payload_indices_(std::move(leaf_payload_indices)) {}

void Octree::validate() const {
  validate_octree(*this);
}

void validate_octree(const Octree& octree) {
  if (octree.max_depth() < 0) {
    throw ValidationError("max_depth must be non-negative");
  }

  validate_root_bounds(octree.root_bounds());

  if (octree.nodes().empty()) {
    if (!octree.leaf_payload_indices().empty()) {
      throw ValidationError("empty tree cannot have leaf payload indices");
    }
    return;
  }

  const std::size_t num_nodes = octree.nodes().size();
  const std::size_t num_payloads = octree.leaf_payload_indices().size();

  for (std::size_t node_index = 0; node_index < num_nodes; ++node_index) {
    const NodeDescriptor descriptor = octree.nodes()[node_index];
    const std::uint8_t child_mask = descriptor.child_mask();
    const std::uint8_t leaf_mask = descriptor.leaf_mask();

    if ((leaf_mask & child_mask) != leaf_mask) {
      throw ValidationError(
          "node " + std::to_string(node_index) + " leaf mask must be a subset of child mask");
    }

    const std::size_t child_span = static_cast<std::size_t>(count_bits(descriptor.internal_child_mask()));
    const std::size_t payload_span = static_cast<std::size_t>(count_bits(leaf_mask));

    if (static_cast<std::size_t>(descriptor.child_base()) + child_span > num_nodes) {
      throw ValidationError(
          "node " + std::to_string(node_index) + " child span exceeds node buffer");
    }

    if (static_cast<std::size_t>(descriptor.payload_base()) + payload_span > num_payloads) {
      throw ValidationError(
          "node " + std::to_string(node_index) + " payload span exceeds payload buffer");
    }
  }
}

}  // namespace svo
