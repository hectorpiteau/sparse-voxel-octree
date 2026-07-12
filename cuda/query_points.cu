#include <svo/Query.hpp>

#include <cstdint>
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

__device__ int count_bits_device(std::uint64_t value) noexcept {
  return __popcll(static_cast<unsigned long long>(value));
}

__device__ int prefix_rank_device(std::uint8_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint8_t lower_mask = static_cast<std::uint8_t>((1u << child_index) - 1u);
  return count_bits_device(static_cast<std::uint8_t>(mask & lower_mask));
}

__device__ int prefix_rank_device(std::uint64_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint64_t lower_mask = (1ull << child_index) - 1ull;
  return count_bits_device(mask & lower_mask);
}

__device__ int child_index_for_depth_device(int x, int y, int z, int child_depth) noexcept {
  const int x_bit = (x >> child_depth) & 1;
  const int y_bit = (y >> child_depth) & 1;
  const int z_bit = (z >> child_depth) & 1;
  return x_bit | (y_bit << 1) | (z_bit << 2);
}

__device__ int wide_child_index_for_depth_device(int x, int y, int z, int child_depth) noexcept {
  const int x_bits = (x >> child_depth) & 3;
  const int y_bits = (y >> child_depth) & 3;
  const int z_bits = (z >> child_depth) & 3;
  return x_bits | (y_bits << 2) | (z_bits << 4);
}

__device__ bool contains_point_device(
    const glm::vec3& min_bound,
    const glm::vec3& max_bound,
    const glm::vec3& point) noexcept {
  return point.x >= min_bound.x && point.y >= min_bound.y && point.z >= min_bound.z &&
      point.x < max_bound.x && point.y < max_bound.y && point.z < max_bound.z;
}

__device__ int floor_to_int_device(float value) noexcept {
  return static_cast<int>(floorf(value));
}

__device__ std::int32_t query_single_point_device(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const glm::vec3& min_bound,
    const glm::vec3& max_bound,
    const glm::vec3& point,
    bool return_payload_indices) noexcept {
  if (num_nodes == 0 || !contains_point_device(min_bound, max_bound, point)) {
    return -1;
  }

  if (max_depth == 0) {
    if (num_leaves == 0) {
      return -1;
    }
    return return_payload_indices ? static_cast<std::int32_t>(leaf_payload_indices[0]) : 0;
  }

  const glm::vec3 extent = max_bound - min_bound;
  const float scale = static_cast<float>(static_cast<std::uint32_t>(1u) << max_depth);
  const int voxel_x = floor_to_int_device(((point.x - min_bound.x) / extent.x) * scale);
  const int voxel_y = floor_to_int_device(((point.y - min_bound.y) / extent.y) * scale);
  const int voxel_z = floor_to_int_device(((point.z - min_bound.z) / extent.z) * scale);

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
      return return_payload_indices ? static_cast<std::int32_t>(leaf_payload_indices[leaf_id])
                                    : static_cast<std::int32_t>(leaf_id);
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

__device__ std::int32_t query_single_point_wide_device(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const glm::vec3& min_bound,
    const glm::vec3& max_bound,
    const glm::vec3& point,
    bool return_payload_indices) noexcept {
  if (num_nodes == 0 || !contains_point_device(min_bound, max_bound, point)) {
    return -1;
  }

  if (max_depth == 0) {
    if (num_leaves == 0) {
      return -1;
    }
    return return_payload_indices ? static_cast<std::int32_t>(leaf_payload_indices[0]) : 0;
  }

  const glm::vec3 extent = max_bound - min_bound;
  const float scale = static_cast<float>(static_cast<std::uint32_t>(1u) << max_depth);
  const int voxel_x = floor_to_int_device(((point.x - min_bound.x) / extent.x) * scale);
  const int voxel_y = floor_to_int_device(((point.y - min_bound.y) / extent.y) * scale);
  const int voxel_z = floor_to_int_device(((point.z - min_bound.z) / extent.z) * scale);

  std::size_t node_index = 0;
  int depth_remaining = max_depth;

  while (true) {
    if (node_index >= num_nodes) {
      return -1;
    }

    const WideNodeDescriptor descriptor = nodes[node_index];
    const int child_index = wide_child_index_for_depth_device(voxel_x, voxel_y, voxel_z, depth_remaining - 2);
    const std::uint64_t child_bit = 1ull << child_index;

    if ((descriptor.child_mask() & child_bit) == 0u) {
      return -1;
    }

    if ((descriptor.leaf_mask() & child_bit) != 0u) {
      const std::size_t leaf_id = static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank_device(descriptor.leaf_mask(), child_index));
      if (leaf_id >= num_leaves) {
        return -1;
      }
      return return_payload_indices ? static_cast<std::int32_t>(leaf_payload_indices[leaf_id])
                                    : static_cast<std::int32_t>(leaf_id);
    }

    const std::uint64_t internal_mask = descriptor.internal_child_mask();
    node_index = static_cast<std::size_t>(descriptor.child_base()) +
        static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
    depth_remaining -= 2;

    if (depth_remaining <= 0) {
      return -1;
    }
  }
}

__global__ void query_points_kernel(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* points,
    std::int32_t* results,
    std::size_t count,
    bool return_payload_indices) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (index >= count) {
    return;
  }

  results[index] = query_single_point_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      min_bound,
      max_bound,
      points[index],
      return_payload_indices);
}

__global__ void query_points_wide_kernel(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* points,
    std::int32_t* results,
    std::size_t count,
    bool return_payload_indices) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (index >= count) {
    return;
  }

  results[index] = query_single_point_wide_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      min_bound,
      max_bound,
      points[index],
      return_payload_indices);
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

}  // namespace

void query_points_cuda(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    std::int32_t* results,
    std::size_t count,
    const QueryOptions& options,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > 30) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }
  check_not_null(nodes, num_nodes, "nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(points, count, "points");
  check_not_null(results, count, "results");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  query_points_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      points,
      results,
      count,
      options.return_payload_indices);

  check_cuda_launch(cudaGetLastError(), "query_points_kernel launch");
}

void query_points_wide_cuda(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    std::int32_t* results,
    std::size_t count,
    const QueryOptions& options,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > 30 || (max_depth % 2) != 0) {
    throw ValidationError("wide4 max_depth must be even and in the range [0, 30]");
  }
  check_not_null(nodes, num_nodes, "wide_nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(points, count, "points");
  check_not_null(results, count, "results");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  query_points_wide_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      points,
      results,
      count,
      options.return_payload_indices);

  check_cuda_launch(cudaGetLastError(), "query_points_wide_kernel launch");
}

}  // namespace svo
