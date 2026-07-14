#include <svo/CoarseOccupancy.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <svo/Error.hpp>

namespace svo {
namespace {

struct AabbHost {
  glm::vec3 min;
  glm::vec3 max;
};

std::size_t cell_count_for_resolution(int resolution) {
  const auto cells = static_cast<std::size_t>(resolution) * static_cast<std::size_t>(resolution) *
      static_cast<std::size_t>(resolution);
  return cells;
}

std::size_t word_index_for_cell(int resolution, int x, int y, int z) {
  const std::size_t linear = static_cast<std::size_t>(x) +
      static_cast<std::size_t>(resolution) *
          (static_cast<std::size_t>(y) + static_cast<std::size_t>(resolution) * static_cast<std::size_t>(z));
  return linear / 32u;
}

std::uint32_t bit_for_cell(int resolution, int x, int y, int z) {
  const std::size_t linear = static_cast<std::size_t>(x) +
      static_cast<std::size_t>(resolution) *
          (static_cast<std::size_t>(y) + static_cast<std::size_t>(resolution) * static_cast<std::size_t>(z));
  return std::uint32_t{1u} << static_cast<std::uint32_t>(linear & 31u);
}

void validate_resolution(int resolution) {
  if (!is_valid_coarse_occupancy_resolution(resolution)) {
    throw ValidationError("coarse occupancy resolution must be 16, 32, or 64");
  }
}

int clamp_cell(int value, int resolution) noexcept {
  return std::max(0, std::min(resolution - 1, value));
}

int cell_min_for_axis(float value, float root_min, float cell_size, int resolution) noexcept {
  return clamp_cell(static_cast<int>(std::floor((value - root_min) / cell_size)), resolution);
}

int cell_max_for_axis(float value, float root_min, float cell_size, int resolution) noexcept {
  constexpr float kBoundaryEpsilon = 1.0e-6f;
  return clamp_cell(static_cast<int>(std::ceil((value - root_min) / cell_size - kBoundaryEpsilon)) - 1, resolution);
}

AabbHost octree_child_bounds(const AabbHost& bounds, int child_index) noexcept {
  const glm::vec3 mid = (bounds.min + bounds.max) * 0.5f;
  const bool high_x = (child_index & 1) != 0;
  const bool high_y = (child_index & 2) != 0;
  const bool high_z = (child_index & 4) != 0;
  return {
      glm::vec3{
          high_x ? mid.x : bounds.min.x,
          high_y ? mid.y : bounds.min.y,
          high_z ? mid.z : bounds.min.z},
      glm::vec3{
          high_x ? bounds.max.x : mid.x,
          high_y ? bounds.max.y : mid.y,
          high_z ? bounds.max.z : mid.z}};
}

AabbHost wide_child_bounds(const AabbHost& bounds, int child_index) noexcept {
  const glm::vec3 cell_size = (bounds.max - bounds.min) * 0.25f;
  const int x = child_index & 3;
  const int y = (child_index >> 2) & 3;
  const int z = (child_index >> 4) & 3;
  const glm::vec3 child_min =
      bounds.min + cell_size * glm::vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
  return {child_min, child_min + cell_size};
}

void mark_bounds(CoarseOccupancyGrid& grid, const AabbHost& bounds) {
  const RootBounds& root = grid.root_bounds();
  const int resolution = grid.resolution();
  const glm::vec3 cell_size = (root[1] - root[0]) / static_cast<float>(resolution);

  const int min_x = cell_min_for_axis(bounds.min.x, root[0].x, cell_size.x, resolution);
  const int min_y = cell_min_for_axis(bounds.min.y, root[0].y, cell_size.y, resolution);
  const int min_z = cell_min_for_axis(bounds.min.z, root[0].z, cell_size.z, resolution);
  const int max_x = cell_max_for_axis(bounds.max.x, root[0].x, cell_size.x, resolution);
  const int max_y = cell_max_for_axis(bounds.max.y, root[0].y, cell_size.y, resolution);
  const int max_z = cell_max_for_axis(bounds.max.z, root[0].z, cell_size.z, resolution);

  for (int z = min_z; z <= max_z; ++z) {
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        grid.mark(x, y, z);
      }
    }
  }
}

void mark_octree8_node(
    const Octree& octree,
    std::size_t node_index,
    int depth_remaining,
    const AabbHost& bounds,
    CoarseOccupancyGrid& grid) {
  if (node_index >= octree.nodes().size() || depth_remaining <= 0) {
    return;
  }
  const NodeDescriptor descriptor = octree.nodes()[node_index];
  const std::uint8_t child_mask = descriptor.child_mask();
  const std::uint8_t leaf_mask = descriptor.leaf_mask();
  const std::uint8_t internal_mask = descriptor.internal_child_mask();
  for (int child_index = 0; child_index < 8; ++child_index) {
    const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);
    if ((child_mask & child_bit) == 0u) {
      continue;
    }
    const AabbHost child = octree_child_bounds(bounds, child_index);
    if ((leaf_mask & child_bit) != 0u) {
      mark_bounds(grid, child);
      continue;
    }
    const std::size_t child_node_index = static_cast<std::size_t>(descriptor.child_base()) +
        static_cast<std::size_t>(__builtin_popcount(static_cast<unsigned>(internal_mask & static_cast<std::uint8_t>(child_bit - 1u))));
    mark_octree8_node(octree, child_node_index, depth_remaining - 1, child, grid);
  }
}

void mark_wide_node(
    const Octree& octree,
    std::size_t node_index,
    int depth_remaining,
    const AabbHost& bounds,
    CoarseOccupancyGrid& grid) {
  if (node_index >= octree.wide_nodes().size() || depth_remaining <= 0) {
    return;
  }
  const WideNodeDescriptor descriptor = octree.wide_nodes()[node_index];
  const std::uint64_t child_mask = descriptor.child_mask();
  const std::uint64_t leaf_mask = descriptor.leaf_mask();
  const std::uint64_t internal_mask = descriptor.internal_child_mask();
  for (int child_index = 0; child_index < 64; ++child_index) {
    const std::uint64_t child_bit = 1ull << child_index;
    if ((child_mask & child_bit) == 0u) {
      continue;
    }
    const AabbHost child = wide_child_bounds(bounds, child_index);
    if ((leaf_mask & child_bit) != 0u) {
      mark_bounds(grid, child);
      continue;
    }
    const std::size_t child_node_index = static_cast<std::size_t>(descriptor.child_base()) +
        static_cast<std::size_t>(__builtin_popcountll(internal_mask & (child_bit - 1ull)));
    mark_wide_node(octree, child_node_index, depth_remaining - 2, child, grid);
  }
}

}  // namespace

bool is_valid_coarse_occupancy_resolution(int resolution) noexcept {
  return resolution == 16 || resolution == 32 || resolution == 64;
}

CoarseOccupancyGrid::CoarseOccupancyGrid(int resolution, RootBounds root_bounds)
    : resolution_(resolution),
      root_bounds_(root_bounds) {
  validate_resolution(resolution);
  const glm::vec3 extent = root_bounds_[1] - root_bounds_[0];
  if (extent.x <= 0.0f || extent.y <= 0.0f || extent.z <= 0.0f) {
    throw ValidationError("coarse occupancy root bounds must have positive extent");
  }
  words_.assign((cell_count_for_resolution(resolution) + 31u) / 32u, 0u);
}

CoarseOccupancyGrid CoarseOccupancyGrid::from_octree(const Octree& octree, int resolution) {
  CoarseOccupancyGrid grid(resolution, octree.root_bounds());
  if (octree.num_leaves() == 0) {
    return grid;
  }
  const AabbHost root{octree.root_bounds()[0], octree.root_bounds()[1]};
  if (octree.max_depth() == 0) {
    mark_bounds(grid, root);
    return grid;
  }
  if (octree.branching() == BranchingMode::Wide4) {
    mark_wide_node(octree, 0u, octree.max_depth(), root, grid);
  } else {
    mark_octree8_node(octree, 0u, octree.max_depth(), root, grid);
  }
  return grid;
}

bool CoarseOccupancyGrid::occupied(int x, int y, int z) const {
  if (x < 0 || x >= resolution_ || y < 0 || y >= resolution_ || z < 0 || z >= resolution_) {
    throw ValidationError("coarse occupancy cell index is out of range");
  }
  return (words_[word_index_for_cell(resolution_, x, y, z)] & bit_for_cell(resolution_, x, y, z)) != 0u;
}

void CoarseOccupancyGrid::mark(int x, int y, int z) {
  if (x < 0 || x >= resolution_ || y < 0 || y >= resolution_ || z < 0 || z >= resolution_) {
    throw ValidationError("coarse occupancy cell index is out of range");
  }
  words_[word_index_for_cell(resolution_, x, y, z)] |= bit_for_cell(resolution_, x, y, z);
}

CoarseOccupancyDeviceView CoarseOccupancyGrid::device_view(const std::uint32_t* device_words) const noexcept {
  return {device_words, resolution_, root_bounds_[0], root_bounds_[1]};
}

#if SVO_ENABLE_CUDA
DeviceCoarseOccupancyGrid::DeviceCoarseOccupancyGrid(
    const CoarseOccupancyGrid& host_grid,
    CudaStreamHandle stream) {
  upload(host_grid, stream);
}

void DeviceCoarseOccupancyGrid::upload(const CoarseOccupancyGrid& host_grid, CudaStreamHandle stream) {
  resolution_ = host_grid.resolution();
  root_bounds_ = host_grid.root_bounds();
  words_ = DeviceBuffer<std::uint32_t>::from_host(host_grid.words(), Device::CUDA, stream);
}

CoarseOccupancyDeviceView DeviceCoarseOccupancyGrid::view() const noexcept {
  return {words_.data(), resolution_, root_bounds_[0], root_bounds_[1]};
}
#endif

}  // namespace svo
