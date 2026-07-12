#include <svo/Interpolation.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <svo/Error.hpp>

namespace svo {
namespace {

int count_bits(std::uint8_t value) noexcept {
  return std::popcount(value);
}

int child_index_for_depth(const glm::ivec3& coordinate, int child_depth) noexcept {
  const int x_bit = (coordinate.x >> child_depth) & 1;
  const int y_bit = (coordinate.y >> child_depth) & 1;
  const int z_bit = (coordinate.z >> child_depth) & 1;
  return x_bit | (y_bit << 1) | (z_bit << 2);
}

int prefix_rank(std::uint8_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint8_t lower_mask = static_cast<std::uint8_t>((1u << child_index) - 1u);
  return count_bits(static_cast<std::uint8_t>(mask & lower_mask));
}

bool contains_point(const RootBounds& bounds, const glm::vec3& point) noexcept {
  return point.x >= bounds[0].x && point.y >= bounds[0].y && point.z >= bounds[0].z &&
      point.x < bounds[1].x && point.y < bounds[1].y && point.z < bounds[1].z;
}

bool voxel_inside_grid(const glm::ivec3& coordinate, int grid_size) noexcept {
  return coordinate.x >= 0 && coordinate.y >= 0 && coordinate.z >= 0 &&
      coordinate.x < grid_size && coordinate.y < grid_size && coordinate.z < grid_size;
}

std::int32_t lookup_payload_index_by_voxel(
    const Octree& octree,
    const glm::ivec3& voxel,
    std::size_t payload_rows) {
  if (octree.nodes().empty()) {
    return -1;
  }

  if (octree.max_depth() == 0) {
    if (octree.leaf_payload_indices().empty()) {
      return -1;
    }
    const std::uint32_t payload_index = octree.leaf_payload_indices().front();
    if (payload_index >= payload_rows) {
      throw ValidationError("leaf payload index is outside the payload row range");
    }
    return static_cast<std::int32_t>(payload_index);
  }

  const int grid_size = 1 << octree.max_depth();
  if (!voxel_inside_grid(voxel, grid_size)) {
    return -1;
  }

  std::size_t node_index = 0;
  int depth_remaining = octree.max_depth();
  while (true) {
    if (node_index >= octree.nodes().size()) {
      return -1;
    }

    const NodeDescriptor descriptor = octree.nodes()[node_index];
    const int child_index = child_index_for_depth(voxel, depth_remaining - 1);
    const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);

    if ((descriptor.child_mask() & child_bit) == 0u) {
      return -1;
    }

    if ((descriptor.leaf_mask() & child_bit) != 0u) {
      const std::size_t leaf_id = static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank(descriptor.leaf_mask(), child_index));
      if (leaf_id >= octree.leaf_payload_indices().size()) {
        return -1;
      }
      const std::uint32_t payload_index = octree.leaf_payload_indices()[leaf_id];
      if (payload_index >= payload_rows) {
        throw ValidationError("leaf payload index is outside the payload row range");
      }
      return static_cast<std::int32_t>(payload_index);
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

glm::vec3 point_to_leaf_center_grid(const Octree& octree, const glm::vec3& point) noexcept {
  const RootBounds& bounds = octree.root_bounds();
  const glm::vec3 extent = bounds[1] - bounds[0];
  const glm::vec3 normalized = (point - bounds[0]) / extent;
  const float scale = static_cast<float>(static_cast<std::uint32_t>(1u) << octree.max_depth());
  return normalized * scale - glm::vec3{0.5f, 0.5f, 0.5f};
}

void validate_inputs(
    const Octree& octree,
    const void* payload,
    std::size_t payload_rows,
    std::size_t channels) {
  if (octree.branching() == BranchingMode::Wide4) {
    throw ValidationError("trilinear interpolation does not support wide4 trees yet");
  }
  if (octree.max_depth() < 0 || octree.max_depth() > 30) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }
  if (channels == 0) {
    throw ValidationError("payload must have at least one channel");
  }
  if (payload_rows != 0 && payload == nullptr) {
    throw ValidationError("payload pointer cannot be null when payload rows are non-zero");
  }
  if (payload_rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw ValidationError("payload row count must be representable as int32");
  }
  for (std::uint32_t payload_index : octree.leaf_payload_indices()) {
    if (payload_index >= payload_rows) {
      throw ValidationError("leaf payload index is outside the payload row range");
    }
  }
}

template <typename T>
std::vector<T> sample_trilinear_impl(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const T* payload,
    std::size_t payload_rows,
    std::size_t channels,
    T fill_value) {
  validate_inputs(octree, payload, payload_rows, channels);

  std::vector<T> outputs(points.size() * channels, fill_value);
  const int grid_size = 1 << octree.max_depth();

  for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
    const glm::vec3& point = points[point_index];
    if (!contains_point(octree.root_bounds(), point)) {
      continue;
    }

    const glm::vec3 grid = point_to_leaf_center_grid(octree, point);
    const glm::ivec3 base{
        static_cast<int>(std::floor(grid.x)),
        static_cast<int>(std::floor(grid.y)),
        static_cast<int>(std::floor(grid.z))};
    const glm::vec3 frac = grid - glm::vec3{base};

    for (int dz = 0; dz <= 1; ++dz) {
      const T wz = static_cast<T>(dz == 0 ? 1.0f - frac.z : frac.z);
      for (int dy = 0; dy <= 1; ++dy) {
        const T wy = static_cast<T>(dy == 0 ? 1.0f - frac.y : frac.y);
        for (int dx = 0; dx <= 1; ++dx) {
          const T wx = static_cast<T>(dx == 0 ? 1.0f - frac.x : frac.x);
          const T weight = wx * wy * wz;
          if (weight == T{0}) {
            continue;
          }

          const glm::ivec3 neighbor = base + glm::ivec3{dx, dy, dz};
          T const* neighbor_payload = nullptr;
          const std::int32_t payload_index = voxel_inside_grid(neighbor, grid_size)
              ? lookup_payload_index_by_voxel(octree, neighbor, payload_rows)
              : -1;
          if (payload_index >= 0) {
            neighbor_payload = payload + static_cast<std::size_t>(payload_index) * channels;
          }

          for (std::size_t channel = 0; channel < channels; ++channel) {
            const T value = neighbor_payload == nullptr ? fill_value : neighbor_payload[channel];
            outputs[point_index * channels + channel] += weight * (value - fill_value);
          }
        }
      }
    }
  }

  return outputs;
}

template <typename T>
std::vector<T> sample_trilinear_backward_impl(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const T* grad_outputs,
    std::size_t payload_rows,
    std::size_t channels) {
  validate_inputs(octree, grad_outputs, payload_rows, channels);

  std::vector<T> grad_payload(payload_rows * channels, T{0});
  const int grid_size = 1 << octree.max_depth();

  for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
    const glm::vec3& point = points[point_index];
    if (!contains_point(octree.root_bounds(), point)) {
      continue;
    }

    const glm::vec3 grid = point_to_leaf_center_grid(octree, point);
    const glm::ivec3 base{
        static_cast<int>(std::floor(grid.x)),
        static_cast<int>(std::floor(grid.y)),
        static_cast<int>(std::floor(grid.z))};
    const glm::vec3 frac = grid - glm::vec3{base};

    for (int dz = 0; dz <= 1; ++dz) {
      const T wz = static_cast<T>(dz == 0 ? 1.0f - frac.z : frac.z);
      for (int dy = 0; dy <= 1; ++dy) {
        const T wy = static_cast<T>(dy == 0 ? 1.0f - frac.y : frac.y);
        for (int dx = 0; dx <= 1; ++dx) {
          const T wx = static_cast<T>(dx == 0 ? 1.0f - frac.x : frac.x);
          const T weight = wx * wy * wz;
          if (weight == T{0}) {
            continue;
          }

          const glm::ivec3 neighbor = base + glm::ivec3{dx, dy, dz};
          const std::int32_t payload_index = voxel_inside_grid(neighbor, grid_size)
              ? lookup_payload_index_by_voxel(octree, neighbor, payload_rows)
              : -1;
          if (payload_index < 0) {
            continue;
          }

          T* grad_row = grad_payload.data() + static_cast<std::size_t>(payload_index) * channels;
          for (std::size_t channel = 0; channel < channels; ++channel) {
            grad_row[channel] += weight * grad_outputs[point_index * channels + channel];
          }
        }
      }
    }
  }

  return grad_payload;
}

}  // namespace

std::vector<float> sample_trilinear_float(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const float* payload,
    std::size_t payload_rows,
    std::size_t channels,
    float fill_value) {
  return sample_trilinear_impl(octree, points, payload, payload_rows, channels, fill_value);
}

std::vector<double> sample_trilinear_double(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const double* payload,
    std::size_t payload_rows,
    std::size_t channels,
    double fill_value) {
  return sample_trilinear_impl(octree, points, payload, payload_rows, channels, fill_value);
}

std::vector<float> sample_trilinear_backward_float(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const float* grad_outputs,
    std::size_t payload_rows,
    std::size_t channels,
    float fill_value) {
  (void)fill_value;
  return sample_trilinear_backward_impl(octree, points, grad_outputs, payload_rows, channels);
}

std::vector<double> sample_trilinear_backward_double(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const double* grad_outputs,
    std::size_t payload_rows,
    std::size_t channels,
    double fill_value) {
  (void)fill_value;
  return sample_trilinear_backward_impl(octree, points, grad_outputs, payload_rows, channels);
}

}  // namespace svo
