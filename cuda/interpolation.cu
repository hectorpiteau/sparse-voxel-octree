#include <svo/Interpolation.hpp>

#include <cstdint>
#include <limits>
#include <string>

#include <cuda_runtime_api.h>

#include <svo/Error.hpp>

namespace svo {
namespace {

__device__ int count_bits_device(std::uint8_t value) noexcept {
  int count = 0;
  while (value != 0u) {
    count += static_cast<int>(value & 1u);
    value = static_cast<std::uint8_t>(value >> 1u);
  }
  return count;
}

__device__ int prefix_rank_device(std::uint8_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint8_t lower_mask = static_cast<std::uint8_t>((1u << child_index) - 1u);
  return count_bits_device(static_cast<std::uint8_t>(mask & lower_mask));
}

__device__ int child_index_for_depth_device(int x, int y, int z, int child_depth) noexcept {
  const int x_bit = (x >> child_depth) & 1;
  const int y_bit = (y >> child_depth) & 1;
  const int z_bit = (z >> child_depth) & 1;
  return x_bit | (y_bit << 1) | (z_bit << 2);
}

__device__ bool contains_point_device(
    const glm::vec3& min_bound,
    const glm::vec3& max_bound,
    const glm::vec3& point) noexcept {
  return point.x >= min_bound.x && point.y >= min_bound.y && point.z >= min_bound.z &&
      point.x < max_bound.x && point.y < max_bound.y && point.z < max_bound.z;
}

__device__ bool voxel_inside_grid_device(int x, int y, int z, int grid_size) noexcept {
  return x >= 0 && y >= 0 && z >= 0 && x < grid_size && y < grid_size && z < grid_size;
}

__device__ void atomic_add_value(float* address, float value) noexcept {
  atomicAdd(address, value);
}

__device__ void atomic_add_value(double* address, double value) noexcept {
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __double_as_longlong(value + __longlong_as_double(static_cast<long long int>(assumed))));
  } while (assumed != old);
}

__device__ std::int32_t lookup_payload_index_by_voxel_device(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    int voxel_x,
    int voxel_y,
    int voxel_z,
    std::size_t payload_rows) noexcept {
  if (num_nodes == 0) {
    return -1;
  }

  if (max_depth == 0) {
    if (num_leaves == 0) {
      return -1;
    }
    const std::uint32_t payload_index = leaf_payload_indices[0];
    return payload_index < payload_rows ? static_cast<std::int32_t>(payload_index) : -1;
  }

  const int grid_size = 1 << max_depth;
  if (!voxel_inside_grid_device(voxel_x, voxel_y, voxel_z, grid_size)) {
    return -1;
  }

  std::size_t node_index = 0;
  int depth_remaining = max_depth;
  while (true) {
    if (node_index >= num_nodes) {
      return -1;
    }

    const NodeDescriptor descriptor = nodes[node_index];
    const int child_index = child_index_for_depth_device(voxel_x, voxel_y, voxel_z, depth_remaining - 1);
    const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);

    if ((descriptor.child_mask() & child_bit) == 0u) {
      return -1;
    }

    if ((descriptor.leaf_mask() & child_bit) != 0u) {
      const std::size_t leaf_id = static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank_device(descriptor.leaf_mask(), child_index));
      if (leaf_id >= num_leaves) {
        return -1;
      }
      const std::uint32_t payload_index = leaf_payload_indices[leaf_id];
      return payload_index < payload_rows ? static_cast<std::int32_t>(payload_index) : -1;
    }

    const std::uint8_t internal_mask = descriptor.internal_child_mask();
    node_index = static_cast<std::size_t>(descriptor.child_base()) +
        static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
    --depth_remaining;
    if (depth_remaining <= 0) {
      return -1;
    }
  }
}

template <typename T>
__global__ void sample_trilinear_kernel(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* points,
    const T* payload,
    T* outputs,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    T fill_value) {
  const std::size_t point_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (point_index >= count) {
    return;
  }

  T* output_row = outputs + point_index * channels;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    output_row[channel] = fill_value;
  }

  const glm::vec3 point = points[point_index];
  if (!contains_point_device(min_bound, max_bound, point)) {
    return;
  }

  const glm::vec3 extent = max_bound - min_bound;
  const glm::vec3 normalized = (point - min_bound) / extent;
  const float scale = static_cast<float>(static_cast<std::uint32_t>(1u) << max_depth);
  const glm::vec3 grid = normalized * scale - glm::vec3{0.5f, 0.5f, 0.5f};
  const int base_x = static_cast<int>(floorf(grid.x));
  const int base_y = static_cast<int>(floorf(grid.y));
  const int base_z = static_cast<int>(floorf(grid.z));
  const float frac_x = grid.x - static_cast<float>(base_x);
  const float frac_y = grid.y - static_cast<float>(base_y);
  const float frac_z = grid.z - static_cast<float>(base_z);

  for (int dz = 0; dz <= 1; ++dz) {
    const T wz = static_cast<T>(dz == 0 ? 1.0f - frac_z : frac_z);
    for (int dy = 0; dy <= 1; ++dy) {
      const T wy = static_cast<T>(dy == 0 ? 1.0f - frac_y : frac_y);
      for (int dx = 0; dx <= 1; ++dx) {
        const T wx = static_cast<T>(dx == 0 ? 1.0f - frac_x : frac_x);
        const T weight = wx * wy * wz;
        if (weight == T{0}) {
          continue;
        }

        const std::int32_t payload_index = lookup_payload_index_by_voxel_device(
            nodes,
            num_nodes,
            leaf_payload_indices,
            num_leaves,
            max_depth,
            base_x + dx,
            base_y + dy,
            base_z + dz,
            payload_rows);
        const T* payload_row = payload_index >= 0
            ? payload + static_cast<std::size_t>(payload_index) * channels
            : nullptr;
        for (std::size_t channel = 0; channel < channels; ++channel) {
          const T value = payload_row == nullptr ? fill_value : payload_row[channel];
          output_row[channel] += weight * (value - fill_value);
        }
      }
    }
  }
}

template <typename T>
__global__ void sample_trilinear_backward_kernel(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* points,
    const T* grad_outputs,
    T* grad_payload,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels) {
  const std::size_t point_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (point_index >= count) {
    return;
  }

  const glm::vec3 point = points[point_index];
  if (!contains_point_device(min_bound, max_bound, point)) {
    return;
  }

  const glm::vec3 extent = max_bound - min_bound;
  const glm::vec3 normalized = (point - min_bound) / extent;
  const float scale = static_cast<float>(static_cast<std::uint32_t>(1u) << max_depth);
  const glm::vec3 grid = normalized * scale - glm::vec3{0.5f, 0.5f, 0.5f};
  const int base_x = static_cast<int>(floorf(grid.x));
  const int base_y = static_cast<int>(floorf(grid.y));
  const int base_z = static_cast<int>(floorf(grid.z));
  const float frac_x = grid.x - static_cast<float>(base_x);
  const float frac_y = grid.y - static_cast<float>(base_y);
  const float frac_z = grid.z - static_cast<float>(base_z);

  for (int dz = 0; dz <= 1; ++dz) {
    const T wz = static_cast<T>(dz == 0 ? 1.0f - frac_z : frac_z);
    for (int dy = 0; dy <= 1; ++dy) {
      const T wy = static_cast<T>(dy == 0 ? 1.0f - frac_y : frac_y);
      for (int dx = 0; dx <= 1; ++dx) {
        const T wx = static_cast<T>(dx == 0 ? 1.0f - frac_x : frac_x);
        const T weight = wx * wy * wz;
        if (weight == T{0}) {
          continue;
        }

        const std::int32_t payload_index = lookup_payload_index_by_voxel_device(
            nodes,
            num_nodes,
            leaf_payload_indices,
            num_leaves,
            max_depth,
            base_x + dx,
            base_y + dy,
            base_z + dz,
            payload_rows);
        if (payload_index < 0) {
          continue;
        }

        T* grad_row = grad_payload + static_cast<std::size_t>(payload_index) * channels;
        const T* grad_output_row = grad_outputs + point_index * channels;
        for (std::size_t channel = 0; channel < channels; ++channel) {
          atomic_add_value(grad_row + channel, weight * grad_output_row[channel]);
        }
      }
    }
  }
}

void check_not_null(const void* pointer, std::size_t count, const char* name) {
  if (count != 0 && pointer == nullptr) {
    throw ValidationError(std::string(name) + " cannot be null when count is non-zero");
  }
}

void check_cuda_launch(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw Error(std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

void validate_common(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const glm::vec3* points,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels) {
  if (max_depth < 0 || max_depth > 30) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }
  if (channels == 0) {
    throw ValidationError("payload must have at least one channel");
  }
  check_not_null(nodes, num_nodes, "nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(points, count, "points");
  if (payload_rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw ValidationError("payload row count must be representable as int32");
  }
}

template <typename T>
void launch_forward(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const T* payload,
    T* outputs,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    T fill_value,
    CudaStreamHandle stream) {
  validate_common(
      nodes, num_nodes, leaf_payload_indices, num_leaves, max_depth, points, count, payload_rows, channels);
  check_not_null(payload, payload_rows * channels, "payload");
  check_not_null(outputs, count * channels, "outputs");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  sample_trilinear_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      points,
      payload,
      outputs,
      count,
      payload_rows,
      channels,
      fill_value);
  check_cuda_launch(cudaGetLastError(), "sample_trilinear_kernel launch");
}

template <typename T>
void launch_backward(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const T* grad_outputs,
    T* grad_payload,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    CudaStreamHandle stream) {
  validate_common(
      nodes, num_nodes, leaf_payload_indices, num_leaves, max_depth, points, count, payload_rows, channels);
  check_not_null(grad_outputs, count * channels, "grad_outputs");
  check_not_null(grad_payload, payload_rows * channels, "grad_payload");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  sample_trilinear_backward_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      points,
      grad_outputs,
      grad_payload,
      count,
      payload_rows,
      channels);
  check_cuda_launch(cudaGetLastError(), "sample_trilinear_backward_kernel launch");
}

}  // namespace

void sample_trilinear_cuda_float(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const float* payload,
    float* outputs,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    float fill_value,
    CudaStreamHandle stream) {
  launch_forward(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds,
      points,
      payload,
      outputs,
      count,
      payload_rows,
      channels,
      fill_value,
      stream);
}

void sample_trilinear_cuda_double(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const double* payload,
    double* outputs,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    double fill_value,
    CudaStreamHandle stream) {
  launch_forward(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds,
      points,
      payload,
      outputs,
      count,
      payload_rows,
      channels,
      fill_value,
      stream);
}

void sample_trilinear_backward_cuda_float(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const float* grad_outputs,
    float* grad_payload,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    float fill_value,
    CudaStreamHandle stream) {
  (void)fill_value;
  launch_backward(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds,
      points,
      grad_outputs,
      grad_payload,
      count,
      payload_rows,
      channels,
      stream);
}

void sample_trilinear_backward_cuda_double(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const double* grad_outputs,
    double* grad_payload,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    double fill_value,
    CudaStreamHandle stream) {
  (void)fill_value;
  launch_backward(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds,
      points,
      grad_outputs,
      grad_payload,
      count,
      payload_rows,
      channels,
      stream);
}

}  // namespace svo
