#include <svo/Raycast.hpp>

#include <cstdint>
#include <limits>
#include <string>

#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <svo/Error.hpp>

namespace svo {
namespace {

constexpr float kEpsilon = 1.0e-6f;
constexpr int kMaxDepth = 30;
constexpr int kMaxStackEntries = 8 * (kMaxDepth + 1);
constexpr int kMaxWideStackEntries = 64 * ((kMaxDepth / 2) + 1);

struct AabbDevice {
  glm::vec3 min;
  glm::vec3 max;
};

struct StackEntry {
  std::size_t node_index;
  int depth_remaining;
  AabbDevice bounds;
};

struct HitCandidateDevice {
  bool hit = false;
  std::int32_t leaf_id = -1;
  float t = std::numeric_limits<float>::infinity();
  glm::vec3 position{
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  std::int32_t depth = -1;
};

__device__ bool finite_vec3_device(const glm::vec3& value) noexcept {
  return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

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

__device__ void add_stat_device(std::uint64_t* field, std::uint64_t value = 1u) noexcept {
  if (field != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(field),
        static_cast<unsigned long long>(value));
  }
}

__device__ void max_stat_device(std::uint64_t* field, std::uint64_t value) noexcept {
  if (field != nullptr) {
    atomicMax(
        reinterpret_cast<unsigned long long*>(field),
        static_cast<unsigned long long>(value));
  }
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

__device__ AabbDevice child_bounds_device(const AabbDevice& bounds, int child_index) noexcept {
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

__device__ AabbDevice wide_child_bounds_device(const AabbDevice& bounds, int child_index) noexcept {
  const int x = child_index & 3;
  const int y = (child_index >> 2) & 3;
  const int z = (child_index >> 4) & 3;
  const glm::vec3 step = (bounds.max - bounds.min) * 0.25f;
  const glm::vec3 child_min = bounds.min + step * glm::vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
  return {child_min, child_min + step};
}

__device__ bool intersect_axis_device(
    float origin,
    float direction,
    float min_bound,
    float max_bound,
    float& t_min,
    float& t_max) noexcept {
  if (fabsf(direction) <= kEpsilon) {
    return origin >= min_bound && origin <= max_bound;
  }

  float t0 = (min_bound - origin) / direction;
  float t1 = (max_bound - origin) / direction;
  if (t0 > t1) {
    const float tmp = t0;
    t0 = t1;
    t1 = tmp;
  }

  t_min = fmaxf(t_min, t0);
  t_max = fminf(t_max, t1);
  return t_min <= t_max;
}

__device__ bool intersect_aabb_device(
    const AabbDevice& bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float& t_near,
    float& t_far) noexcept {
  t_near = -CUDART_INF_F;
  t_far = CUDART_INF_F;

  return intersect_axis_device(origin.x, direction.x, bounds.min.x, bounds.max.x, t_near, t_far) &&
      intersect_axis_device(origin.y, direction.y, bounds.min.y, bounds.max.y, t_near, t_far) &&
      intersect_axis_device(origin.z, direction.z, bounds.min.z, bounds.max.z, t_near, t_far) &&
      t_far >= 0.0f;
}

__device__ bool better_hit_device(
    const HitCandidateDevice& candidate,
    const HitCandidateDevice& best) noexcept {
  if (!candidate.hit) {
    return false;
  }
  if (!best.hit) {
    return true;
  }
  if (candidate.t < best.t - kEpsilon) {
    return true;
  }
  return fabsf(candidate.t - best.t) <= kEpsilon && candidate.leaf_id < best.leaf_id;
}

__device__ void write_miss_device(
    std::uint8_t* hit_mask,
    std::int32_t* leaf_ids,
    float* t,
    glm::vec3* positions,
    std::int32_t* depths,
    std::size_t index) noexcept {
  hit_mask[index] = 0u;
  leaf_ids[index] = -1;
  t[index] = CUDART_INF_F;
  positions[index] = {CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F};
  depths[index] = -1;
}

__device__ void write_hit_device(
    const HitCandidateDevice& hit,
    std::uint8_t* hit_mask,
    std::int32_t* leaf_ids,
    float* t,
    glm::vec3* positions,
    std::int32_t* depths,
    std::size_t index) noexcept {
  hit_mask[index] = hit.hit ? 1u : 0u;
  leaf_ids[index] = hit.leaf_id;
  t[index] = hit.t;
  positions[index] = hit.position;
  depths[index] = hit.depth;
}

__device__ glm::vec3 normalize_or_zero_device(const glm::vec3& direction, bool& valid) noexcept {
  valid = false;
  if (!finite_vec3_device(direction)) {
    return {0.0f, 0.0f, 0.0f};
  }
  const float squared_length =
      direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
  if (!isfinite(squared_length) || squared_length <= kEpsilon * kEpsilon) {
    return {0.0f, 0.0f, 0.0f};
  }
  valid = true;
  return direction * rsqrtf(squared_length);
}

__device__ void consider_leaf_device(
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const AabbDevice& bounds,
    std::int32_t leaf_id,
    std::int32_t depth,
    bool return_payload_indices,
    TraversalStats* stats,
    HitCandidateDevice& best) noexcept {
  if (leaf_id < 0 || static_cast<std::size_t>(leaf_id) >= num_leaves) {
    return;
  }

  float t_near = 0.0f;
  float t_far = 0.0f;
  if (!intersect_aabb_device(bounds, origin, direction, t_near, t_far)) {
    return;
  }
  add_stat_device(stats != nullptr ? &stats->leaf_segments : nullptr);

  HitCandidateDevice candidate;
  candidate.hit = true;
  candidate.leaf_id = return_payload_indices
      ? static_cast<std::int32_t>(leaf_payload_indices[static_cast<std::size_t>(leaf_id)])
      : leaf_id;
  candidate.t = fmaxf(t_near, 0.0f);
  candidate.position = origin + direction * candidate.t;
  candidate.depth = depth;

  if (better_hit_device(candidate, best)) {
    best = candidate;
  }
}

__device__ HitCandidateDevice raycast_single_device(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const AabbDevice& root_bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    bool return_payload_indices,
    TraversalStats* stats) noexcept {
  HitCandidateDevice best;
  if (num_nodes == 0 || !finite_vec3_device(origin)) {
    return best;
  }

  bool valid_direction = false;
  const glm::vec3 normalized_direction = normalize_or_zero_device(direction, valid_direction);
  if (!valid_direction) {
    return best;
  }

  float root_t_near = 0.0f;
  float root_t_far = 0.0f;
  if (!intersect_aabb_device(root_bounds, origin, normalized_direction, root_t_near, root_t_far)) {
    return best;
  }

  if (max_depth == 0) {
    consider_leaf_device(
        leaf_payload_indices,
        num_leaves,
        origin,
        normalized_direction,
        root_bounds,
        0,
        0,
        return_payload_indices,
        stats,
        best);
    return best;
  }

  StackEntry stack[kMaxStackEntries];
  int stack_size = 0;
  stack[stack_size++] = StackEntry{0u, max_depth, root_bounds};
  add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
  max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));

  while (stack_size > 0) {
    const StackEntry entry = stack[--stack_size];
    if (entry.node_index >= num_nodes || entry.depth_remaining <= 0) {
      continue;
    }
    add_stat_device(stats != nullptr ? &stats->nodes_visited : nullptr);
    add_stat_device(stats != nullptr ? &stats->stack_pops : nullptr);

    const NodeDescriptor descriptor = nodes[entry.node_index];
    const std::uint8_t child_mask = descriptor.child_mask();
    const std::uint8_t leaf_mask = descriptor.leaf_mask();
    const std::uint8_t internal_mask = descriptor.internal_child_mask();

    for (int child_index = 0; child_index < 8; ++child_index) {
      const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);
      if ((child_mask & child_bit) == 0u) {
        continue;
      }
      add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);

      const AabbDevice child = child_bounds_device(entry.bounds, child_index);
      float t_near = 0.0f;
      float t_far = 0.0f;
      if (!intersect_aabb_device(child, origin, normalized_direction, t_near, t_far)) {
        continue;
      }
      if (best.hit && fmaxf(t_near, 0.0f) > best.t + kEpsilon) {
        continue;
      }

      if ((leaf_mask & child_bit) != 0u) {
        const std::int32_t leaf_id = static_cast<std::int32_t>(
            static_cast<std::size_t>(descriptor.payload_base()) +
            static_cast<std::size_t>(prefix_rank_device(leaf_mask, child_index)));
        const std::int32_t depth = max_depth - entry.depth_remaining + 1;
        consider_leaf_device(
            leaf_payload_indices,
            num_leaves,
            origin,
            normalized_direction,
            child,
            leaf_id,
            depth,
            return_payload_indices,
            stats,
            best);
        continue;
      }

      const std::size_t child_node_index = static_cast<std::size_t>(descriptor.child_base()) +
          static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
      if (stack_size >= kMaxStackEntries) {
        return HitCandidateDevice{};
      }
      stack[stack_size++] = StackEntry{child_node_index, entry.depth_remaining - 1, child};
      add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
      max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));
    }
  }

  return best;
}

__device__ HitCandidateDevice raycast_single_wide_device(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const AabbDevice& root_bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    bool return_payload_indices,
    TraversalStats* stats) noexcept {
  HitCandidateDevice best;
  if (num_nodes == 0 || !finite_vec3_device(origin)) {
    return best;
  }

  bool valid_direction = false;
  const glm::vec3 normalized_direction = normalize_or_zero_device(direction, valid_direction);
  if (!valid_direction) {
    return best;
  }

  float root_t_near = 0.0f;
  float root_t_far = 0.0f;
  if (!intersect_aabb_device(root_bounds, origin, normalized_direction, root_t_near, root_t_far)) {
    return best;
  }

  if (max_depth == 0) {
    consider_leaf_device(
        leaf_payload_indices,
        num_leaves,
        origin,
        normalized_direction,
        root_bounds,
        0,
        0,
        return_payload_indices,
        stats,
        best);
    return best;
  }

  StackEntry stack[kMaxWideStackEntries];
  int stack_size = 0;
  stack[stack_size++] = StackEntry{0u, max_depth, root_bounds};
  add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
  max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));

  while (stack_size > 0) {
    const StackEntry entry = stack[--stack_size];
    if (entry.node_index >= num_nodes || entry.depth_remaining <= 0) {
      continue;
    }
    add_stat_device(stats != nullptr ? &stats->nodes_visited : nullptr);
    add_stat_device(stats != nullptr ? &stats->stack_pops : nullptr);

    const WideNodeDescriptor descriptor = nodes[entry.node_index];
    const std::uint64_t child_mask = descriptor.child_mask();
    const std::uint64_t leaf_mask = descriptor.leaf_mask();
    const std::uint64_t internal_mask = descriptor.internal_child_mask();

    for (std::uint64_t active_mask = child_mask; active_mask != 0u; active_mask &= active_mask - 1ull) {
      const int child_index = __ffsll(static_cast<unsigned long long>(active_mask)) - 1;
      const std::uint64_t child_bit = 1ull << child_index;
      add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);

      const AabbDevice child = wide_child_bounds_device(entry.bounds, child_index);
      float t_near = 0.0f;
      float t_far = 0.0f;
      if (!intersect_aabb_device(child, origin, normalized_direction, t_near, t_far)) {
        continue;
      }
      if (best.hit && fmaxf(t_near, 0.0f) > best.t + kEpsilon) {
        continue;
      }

      if ((leaf_mask & child_bit) != 0u) {
        const std::int32_t leaf_id = static_cast<std::int32_t>(
            static_cast<std::size_t>(descriptor.payload_base()) +
            static_cast<std::size_t>(prefix_rank_device(leaf_mask, child_index)));
        const std::int32_t depth = max_depth - entry.depth_remaining + 2;
        consider_leaf_device(
            leaf_payload_indices,
            num_leaves,
            origin,
            normalized_direction,
            child,
            leaf_id,
            depth,
            return_payload_indices,
            stats,
            best);
        continue;
      }

      const std::size_t child_node_index = static_cast<std::size_t>(descriptor.child_base()) +
          static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
      if (stack_size < kMaxWideStackEntries) {
        stack[stack_size++] = StackEntry{child_node_index, entry.depth_remaining - 2, child};
        add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
        max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));
      }
    }
  }

  return best;
}

__global__ void raycast_kernel(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    std::uint8_t* hit_mask,
    std::int32_t* leaf_ids,
    float* t,
    glm::vec3* positions,
    std::int32_t* depths,
    std::size_t count,
    bool return_payload_indices,
    TraversalStats* stats) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (index >= count) {
    return;
  }

  const AabbDevice root_bounds{min_bound, max_bound};
  const HitCandidateDevice hit = raycast_single_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds,
      origins[index],
      directions[index],
      return_payload_indices,
      stats);

  if (hit.hit) {
    write_hit_device(hit, hit_mask, leaf_ids, t, positions, depths, index);
  } else {
    write_miss_device(hit_mask, leaf_ids, t, positions, depths, index);
  }
}

__global__ void raycast_wide_kernel(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    std::uint8_t* hit_mask,
    std::int32_t* leaf_ids,
    float* t,
    glm::vec3* positions,
    std::int32_t* depths,
    std::size_t count,
    bool return_payload_indices,
    TraversalStats* stats) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (index >= count) {
    return;
  }

  const HitCandidateDevice hit = raycast_single_wide_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[index],
      directions[index],
      return_payload_indices,
      stats);

  if (hit.hit) {
    write_hit_device(hit, hit_mask, leaf_ids, t, positions, depths, index);
  } else {
    write_miss_device(hit_mask, leaf_ids, t, positions, depths, index);
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

}  // namespace

void raycast_cuda(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* origins,
    const glm::vec3* directions,
    std::uint8_t* hit_mask,
    std::int32_t* leaf_ids,
    float* t,
    glm::vec3* positions,
    std::int32_t* depths,
    std::size_t count,
    const RaycastOptions& options,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > kMaxDepth) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }
  check_not_null(nodes, num_nodes, "nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");
  check_not_null(hit_mask, count, "hit_mask");
  check_not_null(leaf_ids, count, "leaf_ids");
  check_not_null(t, count, "t");
  check_not_null(positions, count, "positions");
  check_not_null(depths, count, "depths");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  raycast_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      origins,
      directions,
      hit_mask,
      leaf_ids,
      t,
      positions,
      depths,
      count,
      options.return_payload_indices,
      options.stats);

  check_cuda_launch(cudaGetLastError(), "raycast_kernel launch");
}

void raycast_wide_cuda(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* origins,
    const glm::vec3* directions,
    std::uint8_t* hit_mask,
    std::int32_t* leaf_ids,
    float* t,
    glm::vec3* positions,
    std::int32_t* depths,
    std::size_t count,
    const RaycastOptions& options,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > kMaxDepth || (max_depth % 2) != 0) {
    throw ValidationError("wide4 max_depth must be even and in the range [0, 30]");
  }
  check_not_null(nodes, num_nodes, "wide_nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");
  check_not_null(hit_mask, count, "hit_mask");
  check_not_null(leaf_ids, count, "leaf_ids");
  check_not_null(t, count, "t");
  check_not_null(positions, count, "positions");
  check_not_null(depths, count, "depths");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  raycast_wide_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      origins,
      directions,
      hit_mask,
      leaf_ids,
      t,
      positions,
      depths,
      count,
      options.return_payload_indices,
      options.stats);

  check_cuda_launch(cudaGetLastError(), "raycast_wide_kernel launch");
}

}  // namespace svo
