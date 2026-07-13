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

struct WideDdaCellDevice {
  int child_index = 0;
  AabbDevice bounds{};
  float t_near = 0.0f;
  float t_far = 0.0f;
};

struct WideDdaStateDevice {
  bool valid = false;
  bool reverse = false;
  AabbDevice bounds{};
  glm::vec3 cell_size{0.0f, 0.0f, 0.0f};
  float base_t = 0.0f;
  float max_s = 0.0f;
  float current_s = 0.0f;
  int cell_x = 0;
  int cell_y = 0;
  int cell_z = 0;
  int step_x = 0;
  int step_y = 0;
  int step_z = 0;
  float next_x = 0.0f;
  float next_y = 0.0f;
  float next_z = 0.0f;
  float delta_x = 0.0f;
  float delta_y = 0.0f;
  float delta_z = 0.0f;
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

__device__ int clamp_cell_device(int cell) noexcept {
  return max(0, min(3, cell));
}

__device__ int local_cell_for_position_device(
    float position,
    float min_bound,
    float cell_size,
    float walk_direction) noexcept {
  const float local = (position - min_bound) / cell_size;
  const float rounded = roundf(local);
  if (fabsf(local - rounded) <= 8.0f * kEpsilon) {
    const int boundary = static_cast<int>(rounded);
    if (boundary <= 0) {
      return 0;
    }
    if (boundary >= 4) {
      return 3;
    }
    return walk_direction < -kEpsilon ? boundary - 1 : boundary;
  }
  return clamp_cell_device(static_cast<int>(floorf(local)));
}

__device__ void setup_wide_dda_axis_device(
    float origin_axis,
    float direction_axis,
    float min_bound,
    float cell_size,
    int cell,
    int& step,
    float& next_s,
    float& delta_s) noexcept {
  if (fabsf(direction_axis) <= kEpsilon) {
    step = 0;
    next_s = CUDART_INF_F;
    delta_s = CUDART_INF_F;
    return;
  }
  step = direction_axis > 0.0f ? 1 : -1;
  const int boundary_index = step > 0 ? cell + 1 : cell;
  const float boundary = min_bound + cell_size * static_cast<float>(boundary_index);
  next_s = (boundary - origin_axis) / direction_axis;
  if (next_s < 0.0f && next_s > -8.0f * kEpsilon) {
    next_s = 0.0f;
  }
  delta_s = cell_size / fabsf(direction_axis);
}

__device__ WideDdaStateDevice make_wide_dda_state_device(
    const AabbDevice& bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float t_near,
    float t_far,
    bool reverse) noexcept {
  WideDdaStateDevice state;
  const float clipped_near = fmaxf(t_near, 0.0f);
  if (t_far < clipped_near) {
    return state;
  }

  state.valid = true;
  state.reverse = reverse;
  state.bounds = bounds;
  state.base_t = reverse ? t_far : clipped_near;
  state.max_s = t_far - clipped_near;
  state.cell_size = (bounds.max - bounds.min) * 0.25f;
  const glm::vec3 walk_origin = origin + direction * state.base_t;
  const glm::vec3 walk_direction = reverse ? -direction : direction;

  state.cell_x = local_cell_for_position_device(walk_origin.x, bounds.min.x, state.cell_size.x, walk_direction.x);
  state.cell_y = local_cell_for_position_device(walk_origin.y, bounds.min.y, state.cell_size.y, walk_direction.y);
  state.cell_z = local_cell_for_position_device(walk_origin.z, bounds.min.z, state.cell_size.z, walk_direction.z);

  setup_wide_dda_axis_device(
      walk_origin.x,
      walk_direction.x,
      bounds.min.x,
      state.cell_size.x,
      state.cell_x,
      state.step_x,
      state.next_x,
      state.delta_x);
  setup_wide_dda_axis_device(
      walk_origin.y,
      walk_direction.y,
      bounds.min.y,
      state.cell_size.y,
      state.cell_y,
      state.step_y,
      state.next_y,
      state.delta_y);
  setup_wide_dda_axis_device(
      walk_origin.z,
      walk_direction.z,
      bounds.min.z,
      state.cell_size.z,
      state.cell_z,
      state.step_z,
      state.next_z,
      state.delta_z);
  return state;
}

__device__ bool next_wide_dda_cell_device(WideDdaStateDevice& state, WideDdaCellDevice& cell) noexcept {
  if (!state.valid || state.current_s > state.max_s + kEpsilon || state.cell_x < 0 || state.cell_x >= 4 ||
      state.cell_y < 0 || state.cell_y >= 4 || state.cell_z < 0 || state.cell_z >= 4) {
    return false;
  }

  const float next_s = fminf(fminf(state.next_x, state.next_y), fminf(state.next_z, state.max_s));
  cell.child_index = state.cell_x | (state.cell_y << 2) | (state.cell_z << 4);
  const glm::vec3 child_min = state.bounds.min +
      state.cell_size *
          glm::vec3{static_cast<float>(state.cell_x), static_cast<float>(state.cell_y), static_cast<float>(state.cell_z)};
  cell.bounds = {child_min, child_min + state.cell_size};
  if (state.reverse) {
    cell.t_near = state.base_t - next_s;
    cell.t_far = state.base_t - state.current_s;
  } else {
    cell.t_near = state.base_t + state.current_s;
    cell.t_far = state.base_t + next_s;
  }

  if (next_s >= state.max_s - kEpsilon) {
    state.valid = false;
    return true;
  }

  state.current_s = next_s;
  if (state.next_x <= state.next_y + kEpsilon && state.next_x <= state.next_z + kEpsilon) {
    state.cell_x += state.step_x;
    state.next_x += state.delta_x;
  } else if (state.next_y <= state.next_z + kEpsilon) {
    state.cell_y += state.step_y;
    state.next_y += state.delta_y;
  } else {
    state.cell_z += state.step_z;
    state.next_z += state.delta_z;
  }
  return true;
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

    WideDdaStateDevice dda;
    float entry_t_near = 0.0f;
    float entry_t_far = 0.0f;
    if (intersect_aabb_device(entry.bounds, origin, normalized_direction, entry_t_near, entry_t_far)) {
      dda = make_wide_dda_state_device(entry.bounds, origin, normalized_direction, entry_t_near, entry_t_far, false);
    } else {
      dda.valid = false;
    }
    WideDdaCellDevice cell;
    while (next_wide_dda_cell_device(dda, cell)) {
      const int child_index = cell.child_index;
      const std::uint64_t child_bit = 1ull << child_index;
      add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);
      if ((child_mask & child_bit) == 0u) {
        continue;
      }

      if (best.hit && fmaxf(cell.t_near, 0.0f) > best.t + kEpsilon) {
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
            cell.bounds,
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
        stack[stack_size++] = StackEntry{child_node_index, entry.depth_remaining - 2, cell.bounds};
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
