#include <svo/Renderer.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <math_constants.h>

#include <svo/Error.hpp>

namespace svo {
namespace {

constexpr float kEpsilon = 1.0e-6f;
constexpr int kMaxDepth = 30;
constexpr int kMaxStackEntries = 8 * (kMaxDepth + 1);
constexpr int kMaxWideStackEntries = 64 * ((kMaxDepth / 2) + 1);
constexpr int kMaxRenderSegments = 512;

struct AabbDevice {
  glm::vec3 min;
  glm::vec3 max;
};

struct StackEntryDevice {
  bool leaf = false;
  std::size_t node_index = 0;
  std::int32_t leaf_id = -1;
  int depth_remaining = 0;
  AabbDevice bounds{};
  float t_near = 0.0f;
  float t_far = 0.0f;
};

struct AccumulatorDevice {
  glm::vec3 rgb{0.0f, 0.0f, 0.0f};
  float depth = 0.0f;
  float opacity = 0.0f;
  float transmittance = 1.0f;
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

__device__ float segment_sort_key_device(const StackEntryDevice& candidate, float near_plane) noexcept {
  return fmaxf(fmaxf(candidate.t_near, near_plane), 0.0f);
}

__device__ bool segment_overlaps_device(float t_near, float t_far, float near_plane, float far_plane) noexcept {
  return fminf(t_far, far_plane) > fmaxf(fmaxf(t_near, near_plane), 0.0f);
}

__device__ void sort_candidates_device(StackEntryDevice* candidates, int count, float near_plane) noexcept {
  for (int outer = 1; outer < count; ++outer) {
    StackEntryDevice value = candidates[outer];
    float value_key = segment_sort_key_device(value, near_plane);
    int inner = outer - 1;
    while (inner >= 0) {
      const float inner_key = segment_sort_key_device(candidates[inner], near_plane);
      if (inner_key < value_key || (fabsf(inner_key - value_key) <= kEpsilon && candidates[inner].leaf_id <= value.leaf_id)) {
        break;
      }
      candidates[inner + 1] = candidates[inner];
      --inner;
    }
    candidates[inner + 1] = value;
  }
}

__device__ void composite_leaf_device(
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    const StackEntryDevice& segment,
    float near_plane,
    float far_plane,
    float early_stop_transmittance,
    TraversalStats* stats,
    AccumulatorDevice& accum) noexcept {
  (void)early_stop_transmittance;
  if (segment.leaf_id < 0 || static_cast<std::size_t>(segment.leaf_id) >= num_leaves) {
    return;
  }

  const float t0 = fmaxf(fmaxf(segment.t_near, near_plane), 0.0f);
  const float t1 = fminf(segment.t_far, far_plane);
  if (t1 <= t0) {
    return;
  }
  add_stat_device(stats != nullptr ? &stats->leaf_segments : nullptr);

  const std::uint32_t payload_index = leaf_payload_indices[static_cast<std::size_t>(segment.leaf_id)];
  if (static_cast<std::size_t>(payload_index) >= payload_rows) {
    return;
  }

  const float density = fmaxf(sigma[static_cast<std::size_t>(payload_index)], 0.0f);
  if (density <= 0.0f) {
    return;
  }

  const float alpha = 1.0f - expf(-density * (t1 - t0));
  const float weight = accum.transmittance * alpha;
  const std::size_t color_index = static_cast<std::size_t>(payload_index) * 3u;
  const glm::vec3 leaf_color{color[color_index], color[color_index + 1u], color[color_index + 2u]};
  const float midpoint = 0.5f * (t0 + t1);

  accum.rgb += weight * leaf_color;
  accum.depth += weight * midpoint;
  accum.opacity += weight;
  accum.transmittance *= 1.0f - alpha;
}

__device__ void render_single_device(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const AabbDevice& root_bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    float near_plane,
    float far_plane,
    glm::vec3 background_color,
    float early_stop_transmittance,
    TraversalStats* stats,
    glm::vec3& rgb,
    float& depth,
    float& opacity) noexcept {
  AccumulatorDevice accum;
  if (num_nodes > 0 && finite_vec3_device(origin)) {
    bool valid_direction = false;
    const glm::vec3 normalized_direction = normalize_or_zero_device(direction, valid_direction);
    if (valid_direction) {
      float root_t_near = 0.0f;
      float root_t_far = 0.0f;
      if (intersect_aabb_device(root_bounds, origin, normalized_direction, root_t_near, root_t_far) &&
          segment_overlaps_device(root_t_near, root_t_far, near_plane, far_plane)) {
        if (max_depth == 0) {
          StackEntryDevice root_leaf;
          root_leaf.leaf = true;
          root_leaf.leaf_id = 0;
          root_leaf.bounds = root_bounds;
          root_leaf.t_near = root_t_near;
          root_leaf.t_far = root_t_far;
          composite_leaf_device(
              leaf_payload_indices,
              num_leaves,
              sigma,
              color,
              payload_rows,
              root_leaf,
              near_plane,
              far_plane,
              early_stop_transmittance,
              stats,
              accum);
        } else {
          StackEntryDevice stack[kMaxStackEntries];
          int stack_size = 0;
          stack[stack_size++] = StackEntryDevice{false, 0u, -1, max_depth, root_bounds, root_t_near, root_t_far};
          add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
          max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));

          while (stack_size > 0 && accum.transmittance > early_stop_transmittance) {
            const StackEntryDevice entry = stack[--stack_size];
            if (entry.leaf) {
              composite_leaf_device(
                  leaf_payload_indices,
                  num_leaves,
                  sigma,
                  color,
                  payload_rows,
                  entry,
                  near_plane,
                  far_plane,
                  early_stop_transmittance,
                  stats,
                  accum);
              continue;
            }
            if (entry.node_index >= num_nodes || entry.depth_remaining <= 0) {
              continue;
            }
            add_stat_device(stats != nullptr ? &stats->nodes_visited : nullptr);
            add_stat_device(stats != nullptr ? &stats->stack_pops : nullptr);

            const NodeDescriptor descriptor = nodes[entry.node_index];
            const std::uint8_t child_mask = descriptor.child_mask();
            const std::uint8_t leaf_mask = descriptor.leaf_mask();
            const std::uint8_t internal_mask = descriptor.internal_child_mask();
            StackEntryDevice candidates[8];
            int candidate_count = 0;

            for (int child_index = 0; child_index < 8; ++child_index) {
              const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);
              if ((child_mask & child_bit) == 0u) {
                continue;
              }
              add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);

              const AabbDevice child = child_bounds_device(entry.bounds, child_index);
              float t_near = 0.0f;
              float t_far = 0.0f;
              if (!intersect_aabb_device(child, origin, normalized_direction, t_near, t_far) ||
                  !segment_overlaps_device(t_near, t_far, near_plane, far_plane)) {
                continue;
              }

              StackEntryDevice candidate;
              candidate.leaf = (leaf_mask & child_bit) != 0u;
              candidate.bounds = child;
              candidate.t_near = t_near;
              candidate.t_far = t_far;
              if (candidate.leaf) {
                candidate.leaf_id = static_cast<std::int32_t>(
                    static_cast<std::size_t>(descriptor.payload_base()) +
                    static_cast<std::size_t>(prefix_rank_device(leaf_mask, child_index)));
              } else {
                candidate.node_index = static_cast<std::size_t>(descriptor.child_base()) +
                    static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
                candidate.depth_remaining = entry.depth_remaining - 1;
              }
              candidates[candidate_count++] = candidate;
            }

            sort_candidates_device(candidates, candidate_count, near_plane);
            for (int index = candidate_count - 1; index >= 0; --index) {
              if (stack_size >= kMaxStackEntries) {
                stack_size = 0;
                break;
              }
              stack[stack_size++] = candidates[index];
              add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
              max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));
            }
          }
          if (accum.transmittance <= early_stop_transmittance) {
            add_stat_device(stats != nullptr ? &stats->early_terminations : nullptr);
          }
        }
      }
    }
  }

  rgb = accum.rgb + accum.transmittance * background_color;
  opacity = accum.opacity;
  depth = accum.opacity > 0.0f ? accum.depth / accum.opacity : CUDART_INF_F;
}

__device__ void render_single_wide_device(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const AabbDevice& root_bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    float near_plane,
    float far_plane,
    glm::vec3 background_color,
    float early_stop_transmittance,
    TraversalStats* stats,
    glm::vec3& rgb,
    float& depth,
    float& opacity) noexcept {
  AccumulatorDevice accum;
  if (num_nodes > 0 && finite_vec3_device(origin)) {
    bool valid_direction = false;
    const glm::vec3 normalized_direction = normalize_or_zero_device(direction, valid_direction);
    if (valid_direction) {
      float root_t_near = 0.0f;
      float root_t_far = 0.0f;
      if (intersect_aabb_device(root_bounds, origin, normalized_direction, root_t_near, root_t_far) &&
          segment_overlaps_device(root_t_near, root_t_far, near_plane, far_plane)) {
        if (max_depth == 0) {
          StackEntryDevice root_leaf;
          root_leaf.leaf = true;
          root_leaf.leaf_id = 0;
          root_leaf.bounds = root_bounds;
          root_leaf.t_near = root_t_near;
          root_leaf.t_far = root_t_far;
          composite_leaf_device(
              leaf_payload_indices,
              num_leaves,
              sigma,
              color,
              payload_rows,
              root_leaf,
              near_plane,
              far_plane,
              early_stop_transmittance,
              stats,
              accum);
        } else {
          StackEntryDevice stack[kMaxWideStackEntries];
          int stack_size = 0;
          stack[stack_size++] = StackEntryDevice{false, 0u, -1, max_depth, root_bounds, root_t_near, root_t_far};
          add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
          max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));

          while (stack_size > 0 && accum.transmittance > early_stop_transmittance) {
            const StackEntryDevice entry = stack[--stack_size];
            if (entry.leaf) {
              composite_leaf_device(
                  leaf_payload_indices,
                  num_leaves,
                  sigma,
                  color,
                  payload_rows,
                  entry,
                  near_plane,
                  far_plane,
                  early_stop_transmittance,
                  stats,
                  accum);
              continue;
            }
            if (entry.node_index >= num_nodes || entry.depth_remaining <= 0) {
              continue;
            }
            add_stat_device(stats != nullptr ? &stats->nodes_visited : nullptr);
            add_stat_device(stats != nullptr ? &stats->stack_pops : nullptr);

            const WideNodeDescriptor descriptor = nodes[entry.node_index];
            const std::uint64_t child_mask = descriptor.child_mask();
            const std::uint64_t leaf_mask = descriptor.leaf_mask();
            const std::uint64_t internal_mask = descriptor.internal_child_mask();

            WideDdaStateDevice dda = make_wide_dda_state_device(
                entry.bounds,
                origin,
                normalized_direction,
                entry.t_near,
                entry.t_far,
                true);
            WideDdaCellDevice cell;
            while (next_wide_dda_cell_device(dda, cell)) {
              const int child_index = cell.child_index;
              const std::uint64_t child_bit = 1ull << child_index;
              add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);
              if ((child_mask & child_bit) == 0u) {
                continue;
              }

              if (!segment_overlaps_device(cell.t_near, cell.t_far, near_plane, far_plane)) {
                continue;
              }

              StackEntryDevice candidate;
              candidate.leaf = (leaf_mask & child_bit) != 0u;
              candidate.bounds = cell.bounds;
              candidate.t_near = cell.t_near;
              candidate.t_far = cell.t_far;
              if (candidate.leaf) {
                candidate.leaf_id = static_cast<std::int32_t>(
                    static_cast<std::size_t>(descriptor.payload_base()) +
                    static_cast<std::size_t>(prefix_rank_device(leaf_mask, child_index)));
              } else {
                candidate.node_index = static_cast<std::size_t>(descriptor.child_base()) +
                    static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
                candidate.depth_remaining = entry.depth_remaining - 2;
              }
              if (stack_size >= kMaxWideStackEntries) {
                stack_size = 0;
                break;
              }
              stack[stack_size++] = candidate;
              add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
              max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));
            }
          }
          if (accum.transmittance <= early_stop_transmittance) {
            add_stat_device(stats != nullptr ? &stats->early_terminations : nullptr);
          }
        }
      }
    }
  }

  rgb = accum.rgb + accum.transmittance * background_color;
  opacity = accum.opacity;
  depth = accum.opacity > 0.0f ? accum.depth / accum.opacity : CUDART_INF_F;
}

struct RenderSegmentDevice {
  std::uint32_t payload_index = 0u;
  float raw_sigma = 0.0f;
  float delta = 0.0f;
  float alpha = 0.0f;
  float transmittance = 1.0f;
  glm::vec3 color{0.0f, 0.0f, 0.0f};
};

struct CountIntervalSinkDevice {
  std::uint32_t count = 0u;
  std::uint32_t ray_index = 0u;
  std::uint32_t* ray_indices = nullptr;
  float* t_start = nullptr;
  float* t_end = nullptr;
  std::int32_t* leaf_ids = nullptr;
  std::uint32_t* payload_indices = nullptr;
};

struct EmitIntervalSinkDevice {
  std::uint32_t cursor = 0u;
  std::uint32_t ray_index = 0u;
  std::uint32_t* ray_indices = nullptr;
  float* t_start = nullptr;
  float* t_end = nullptr;
  std::int32_t* leaf_ids = nullptr;
  std::uint32_t* payload_indices = nullptr;
};

__device__ float dot_vec3_device(const glm::vec3& lhs, const glm::vec3& rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

template <typename Sink>
__device__ bool append_interval_device(
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    const StackEntryDevice& segment,
    float near_plane,
    float far_plane,
    Sink& sink,
    TraversalStats* stats) noexcept {
  if (segment.leaf_id < 0 || static_cast<std::size_t>(segment.leaf_id) >= num_leaves) {
    return true;
  }

  const float t0 = fmaxf(fmaxf(segment.t_near, near_plane), 0.0f);
  const float t1 = fminf(segment.t_far, far_plane);
  if (t1 <= t0) {
    return true;
  }

  add_stat_device(stats != nullptr ? &stats->leaf_segments : nullptr);
  const std::uint32_t payload_index = leaf_payload_indices[static_cast<std::size_t>(segment.leaf_id)];
  if constexpr (std::is_same_v<Sink, CountIntervalSinkDevice>) {
    ++sink.count;
  } else {
    const std::uint32_t index = sink.cursor++;
    sink.ray_indices[index] = sink.ray_index;
    sink.t_start[index] = t0;
    sink.t_end[index] = t1;
    sink.leaf_ids[index] = segment.leaf_id;
    sink.payload_indices[index] = payload_index;
  }
  return true;
}

template <typename Sink>
__device__ bool collect_render_intervals_device(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const AabbDevice& root_bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float near_plane,
    float far_plane,
    Sink& sink,
    TraversalStats* stats) noexcept {
  if (num_nodes == 0 || !finite_vec3_device(origin)) {
    return true;
  }

  bool valid_direction = false;
  const glm::vec3 normalized_direction = normalize_or_zero_device(direction, valid_direction);
  if (!valid_direction) {
    return true;
  }

  float root_t_near = 0.0f;
  float root_t_far = 0.0f;
  if (!intersect_aabb_device(root_bounds, origin, normalized_direction, root_t_near, root_t_far) ||
      !segment_overlaps_device(root_t_near, root_t_far, near_plane, far_plane)) {
    return true;
  }

  if (max_depth == 0) {
    StackEntryDevice root_leaf;
    root_leaf.leaf = true;
    root_leaf.leaf_id = 0;
    root_leaf.bounds = root_bounds;
    root_leaf.t_near = root_t_near;
    root_leaf.t_far = root_t_far;
    return append_interval_device(leaf_payload_indices, num_leaves, root_leaf, near_plane, far_plane, sink, stats);
  }

  StackEntryDevice stack[kMaxStackEntries];
  int stack_size = 0;
  stack[stack_size++] = StackEntryDevice{false, 0u, -1, max_depth, root_bounds, root_t_near, root_t_far};
  add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
  max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));

  while (stack_size > 0) {
    const StackEntryDevice entry = stack[--stack_size];
    if (entry.leaf) {
      if (!append_interval_device(leaf_payload_indices, num_leaves, entry, near_plane, far_plane, sink, stats)) {
        return false;
      }
      continue;
    }
    if (entry.node_index >= num_nodes || entry.depth_remaining <= 0) {
      continue;
    }
    add_stat_device(stats != nullptr ? &stats->nodes_visited : nullptr);
    add_stat_device(stats != nullptr ? &stats->stack_pops : nullptr);

    const NodeDescriptor descriptor = nodes[entry.node_index];
    const std::uint8_t child_mask = descriptor.child_mask();
    const std::uint8_t leaf_mask = descriptor.leaf_mask();
    const std::uint8_t internal_mask = descriptor.internal_child_mask();
    StackEntryDevice candidates[8];
    int candidate_count = 0;

    for (int child_index = 0; child_index < 8; ++child_index) {
      const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);
      if ((child_mask & child_bit) == 0u) {
        continue;
      }
      add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);

      const AabbDevice child = child_bounds_device(entry.bounds, child_index);
      float t_near = 0.0f;
      float t_far = 0.0f;
      if (!intersect_aabb_device(child, origin, normalized_direction, t_near, t_far) ||
          !segment_overlaps_device(t_near, t_far, near_plane, far_plane)) {
        continue;
      }

      StackEntryDevice candidate;
      candidate.leaf = (leaf_mask & child_bit) != 0u;
      candidate.bounds = child;
      candidate.t_near = t_near;
      candidate.t_far = t_far;
      if (candidate.leaf) {
        candidate.leaf_id = static_cast<std::int32_t>(
            static_cast<std::size_t>(descriptor.payload_base()) +
            static_cast<std::size_t>(prefix_rank_device(leaf_mask, child_index)));
      } else {
        candidate.node_index = static_cast<std::size_t>(descriptor.child_base()) +
            static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
        candidate.depth_remaining = entry.depth_remaining - 1;
      }
      candidates[candidate_count++] = candidate;
    }

    sort_candidates_device(candidates, candidate_count, near_plane);
    for (int index = candidate_count - 1; index >= 0; --index) {
      if (stack_size >= kMaxStackEntries) {
        return false;
      }
      stack[stack_size++] = candidates[index];
      add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
      max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));
    }
  }

  return true;
}

template <typename Sink>
__device__ bool collect_render_intervals_wide_device(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const AabbDevice& root_bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float near_plane,
    float far_plane,
    Sink& sink,
    TraversalStats* stats) noexcept {
  if (num_nodes == 0 || !finite_vec3_device(origin)) {
    return true;
  }

  bool valid_direction = false;
  const glm::vec3 normalized_direction = normalize_or_zero_device(direction, valid_direction);
  if (!valid_direction) {
    return true;
  }

  float root_t_near = 0.0f;
  float root_t_far = 0.0f;
  if (!intersect_aabb_device(root_bounds, origin, normalized_direction, root_t_near, root_t_far) ||
      !segment_overlaps_device(root_t_near, root_t_far, near_plane, far_plane)) {
    return true;
  }

  if (max_depth == 0) {
    StackEntryDevice root_leaf;
    root_leaf.leaf = true;
    root_leaf.leaf_id = 0;
    root_leaf.bounds = root_bounds;
    root_leaf.t_near = root_t_near;
    root_leaf.t_far = root_t_far;
    return append_interval_device(leaf_payload_indices, num_leaves, root_leaf, near_plane, far_plane, sink, stats);
  }

  StackEntryDevice stack[kMaxWideStackEntries];
  int stack_size = 0;
  stack[stack_size++] = StackEntryDevice{false, 0u, -1, max_depth, root_bounds, root_t_near, root_t_far};
  add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
  max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));

  while (stack_size > 0) {
    const StackEntryDevice entry = stack[--stack_size];
    if (entry.leaf) {
      if (!append_interval_device(leaf_payload_indices, num_leaves, entry, near_plane, far_plane, sink, stats)) {
        return false;
      }
      continue;
    }
    if (entry.node_index >= num_nodes || entry.depth_remaining <= 0) {
      continue;
    }
    add_stat_device(stats != nullptr ? &stats->nodes_visited : nullptr);
    add_stat_device(stats != nullptr ? &stats->stack_pops : nullptr);

    const WideNodeDescriptor descriptor = nodes[entry.node_index];
    const std::uint64_t child_mask = descriptor.child_mask();
    const std::uint64_t leaf_mask = descriptor.leaf_mask();
    const std::uint64_t internal_mask = descriptor.internal_child_mask();

    WideDdaStateDevice dda = make_wide_dda_state_device(
        entry.bounds,
        origin,
        normalized_direction,
        entry.t_near,
        entry.t_far,
        true);
    WideDdaCellDevice cell;
    while (next_wide_dda_cell_device(dda, cell)) {
      const int child_index = cell.child_index;
      const std::uint64_t child_bit = 1ull << child_index;
      add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);
      if ((child_mask & child_bit) == 0u) {
        continue;
      }
      if (!segment_overlaps_device(cell.t_near, cell.t_far, near_plane, far_plane)) {
        continue;
      }

      StackEntryDevice candidate;
      candidate.leaf = (leaf_mask & child_bit) != 0u;
      candidate.bounds = cell.bounds;
      candidate.t_near = cell.t_near;
      candidate.t_far = cell.t_far;
      if (candidate.leaf) {
        candidate.leaf_id = static_cast<std::int32_t>(
            static_cast<std::size_t>(descriptor.payload_base()) +
            static_cast<std::size_t>(prefix_rank_device(leaf_mask, child_index)));
      } else {
        candidate.node_index = static_cast<std::size_t>(descriptor.child_base()) +
            static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
        candidate.depth_remaining = entry.depth_remaining - 2;
      }
      if (stack_size >= kMaxWideStackEntries) {
        return false;
      }
      stack[stack_size++] = candidate;
      add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
      max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));
    }
  }

  return true;
}

__device__ bool append_render_segment_device(
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    const StackEntryDevice& segment,
    float near_plane,
    float far_plane,
    RenderSegmentDevice* segments,
    int& segment_count,
    float& transmittance,
    TraversalStats* stats) noexcept {
  if (segment.leaf_id < 0 || static_cast<std::size_t>(segment.leaf_id) >= num_leaves) {
    return true;
  }

  const float t0 = fmaxf(fmaxf(segment.t_near, near_plane), 0.0f);
  const float t1 = fminf(segment.t_far, far_plane);
  if (t1 <= t0) {
    return true;
  }
  add_stat_device(stats != nullptr ? &stats->leaf_segments : nullptr);

  const std::uint32_t payload_index = leaf_payload_indices[static_cast<std::size_t>(segment.leaf_id)];
  if (static_cast<std::size_t>(payload_index) >= payload_rows) {
    return true;
  }

  const float raw_sigma = sigma[static_cast<std::size_t>(payload_index)];
  const float density = fmaxf(raw_sigma, 0.0f);
  if (density <= 0.0f) {
    return true;
  }

  if (segment_count >= kMaxRenderSegments) {
    return false;
  }

  const float delta = t1 - t0;
  const float alpha = 1.0f - expf(-density * delta);
  const std::size_t color_index = static_cast<std::size_t>(payload_index) * 3u;
  RenderSegmentDevice output;
  output.payload_index = payload_index;
  output.raw_sigma = raw_sigma;
  output.delta = delta;
  output.alpha = alpha;
  output.transmittance = transmittance;
  output.color = {color[color_index], color[color_index + 1u], color[color_index + 2u]};
  segments[segment_count++] = output;
  transmittance *= 1.0f - alpha;
  return true;
}

__device__ bool collect_render_segments_device(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const AabbDevice& root_bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    float near_plane,
    float far_plane,
    float early_stop_transmittance,
    RenderSegmentDevice* segments,
    int& segment_count,
    TraversalStats* stats) noexcept {
  segment_count = 0;
  if (num_nodes == 0 || !finite_vec3_device(origin)) {
    return true;
  }

  bool valid_direction = false;
  const glm::vec3 normalized_direction = normalize_or_zero_device(direction, valid_direction);
  if (!valid_direction) {
    return true;
  }

  float root_t_near = 0.0f;
  float root_t_far = 0.0f;
  if (!intersect_aabb_device(root_bounds, origin, normalized_direction, root_t_near, root_t_far) ||
      !segment_overlaps_device(root_t_near, root_t_far, near_plane, far_plane)) {
    return true;
  }

  float transmittance = 1.0f;
  if (max_depth == 0) {
    StackEntryDevice root_leaf;
    root_leaf.leaf = true;
    root_leaf.leaf_id = 0;
    root_leaf.bounds = root_bounds;
    root_leaf.t_near = root_t_near;
    root_leaf.t_far = root_t_far;
    return append_render_segment_device(
        leaf_payload_indices,
        num_leaves,
        sigma,
        color,
        payload_rows,
        root_leaf,
        near_plane,
        far_plane,
        segments,
        segment_count,
        transmittance,
        stats);
  }

  StackEntryDevice stack[kMaxStackEntries];
  int stack_size = 0;
  stack[stack_size++] = StackEntryDevice{false, 0u, -1, max_depth, root_bounds, root_t_near, root_t_far};
  add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
  max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));

  while (stack_size > 0 && transmittance > early_stop_transmittance) {
    const StackEntryDevice entry = stack[--stack_size];
    if (entry.leaf) {
      if (!append_render_segment_device(
              leaf_payload_indices,
              num_leaves,
              sigma,
              color,
              payload_rows,
              entry,
              near_plane,
              far_plane,
              segments,
              segment_count,
              transmittance,
              stats)) {
        return false;
      }
      continue;
    }
    if (entry.node_index >= num_nodes || entry.depth_remaining <= 0) {
      continue;
    }
    add_stat_device(stats != nullptr ? &stats->nodes_visited : nullptr);
    add_stat_device(stats != nullptr ? &stats->stack_pops : nullptr);

    const NodeDescriptor descriptor = nodes[entry.node_index];
    const std::uint8_t child_mask = descriptor.child_mask();
    const std::uint8_t leaf_mask = descriptor.leaf_mask();
    const std::uint8_t internal_mask = descriptor.internal_child_mask();
    StackEntryDevice candidates[8];
    int candidate_count = 0;

    for (int child_index = 0; child_index < 8; ++child_index) {
      const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);
      if ((child_mask & child_bit) == 0u) {
        continue;
      }
      add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);

      const AabbDevice child = child_bounds_device(entry.bounds, child_index);
      float t_near = 0.0f;
      float t_far = 0.0f;
      if (!intersect_aabb_device(child, origin, normalized_direction, t_near, t_far) ||
          !segment_overlaps_device(t_near, t_far, near_plane, far_plane)) {
        continue;
      }

      StackEntryDevice candidate;
      candidate.leaf = (leaf_mask & child_bit) != 0u;
      candidate.bounds = child;
      candidate.t_near = t_near;
      candidate.t_far = t_far;
      if (candidate.leaf) {
        candidate.leaf_id = static_cast<std::int32_t>(
            static_cast<std::size_t>(descriptor.payload_base()) +
            static_cast<std::size_t>(prefix_rank_device(leaf_mask, child_index)));
      } else {
        candidate.node_index = static_cast<std::size_t>(descriptor.child_base()) +
            static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
        candidate.depth_remaining = entry.depth_remaining - 1;
      }
      candidates[candidate_count++] = candidate;
    }

    sort_candidates_device(candidates, candidate_count, near_plane);
    for (int index = candidate_count - 1; index >= 0; --index) {
      if (stack_size >= kMaxStackEntries) {
        return false;
      }
      stack[stack_size++] = candidates[index];
      add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
      max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));
    }
  }
  if (transmittance <= early_stop_transmittance) {
    add_stat_device(stats != nullptr ? &stats->early_terminations : nullptr);
  }

  return true;
}

__device__ bool collect_render_segments_wide_device(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const AabbDevice& root_bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    float near_plane,
    float far_plane,
    float early_stop_transmittance,
    RenderSegmentDevice* segments,
    int& segment_count,
    TraversalStats* stats) noexcept {
  segment_count = 0;
  if (num_nodes == 0 || !finite_vec3_device(origin)) {
    return true;
  }

  bool valid_direction = false;
  const glm::vec3 normalized_direction = normalize_or_zero_device(direction, valid_direction);
  if (!valid_direction) {
    return true;
  }

  float root_t_near = 0.0f;
  float root_t_far = 0.0f;
  if (!intersect_aabb_device(root_bounds, origin, normalized_direction, root_t_near, root_t_far) ||
      !segment_overlaps_device(root_t_near, root_t_far, near_plane, far_plane)) {
    return true;
  }

  float transmittance = 1.0f;
  if (max_depth == 0) {
    StackEntryDevice root_leaf;
    root_leaf.leaf = true;
    root_leaf.leaf_id = 0;
    root_leaf.bounds = root_bounds;
    root_leaf.t_near = root_t_near;
    root_leaf.t_far = root_t_far;
    return append_render_segment_device(
        leaf_payload_indices,
        num_leaves,
        sigma,
        color,
        payload_rows,
        root_leaf,
        near_plane,
        far_plane,
        segments,
        segment_count,
        transmittance,
        stats);
  }

  StackEntryDevice stack[kMaxWideStackEntries];
  int stack_size = 0;
  stack[stack_size++] = StackEntryDevice{false, 0u, -1, max_depth, root_bounds, root_t_near, root_t_far};
  add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
  max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));

  while (stack_size > 0 && transmittance > early_stop_transmittance) {
    const StackEntryDevice entry = stack[--stack_size];
    if (entry.leaf) {
      if (!append_render_segment_device(
              leaf_payload_indices,
              num_leaves,
              sigma,
              color,
              payload_rows,
              entry,
              near_plane,
              far_plane,
              segments,
              segment_count,
              transmittance,
              stats)) {
        return false;
      }
      continue;
    }
    if (entry.node_index >= num_nodes || entry.depth_remaining <= 0) {
      continue;
    }
    add_stat_device(stats != nullptr ? &stats->nodes_visited : nullptr);
    add_stat_device(stats != nullptr ? &stats->stack_pops : nullptr);

    const WideNodeDescriptor descriptor = nodes[entry.node_index];
    const std::uint64_t child_mask = descriptor.child_mask();
    const std::uint64_t leaf_mask = descriptor.leaf_mask();
    const std::uint64_t internal_mask = descriptor.internal_child_mask();

    WideDdaStateDevice dda = make_wide_dda_state_device(
        entry.bounds,
        origin,
        normalized_direction,
        entry.t_near,
        entry.t_far,
        true);
    WideDdaCellDevice cell;
    while (next_wide_dda_cell_device(dda, cell)) {
      const int child_index = cell.child_index;
      const std::uint64_t child_bit = 1ull << child_index;
      add_stat_device(stats != nullptr ? &stats->child_candidates_tested : nullptr);
      if ((child_mask & child_bit) == 0u) {
        continue;
      }

      if (!segment_overlaps_device(cell.t_near, cell.t_far, near_plane, far_plane)) {
        continue;
      }

      StackEntryDevice candidate;
      candidate.leaf = (leaf_mask & child_bit) != 0u;
      candidate.bounds = cell.bounds;
      candidate.t_near = cell.t_near;
      candidate.t_far = cell.t_far;
      if (candidate.leaf) {
        candidate.leaf_id = static_cast<std::int32_t>(
            static_cast<std::size_t>(descriptor.payload_base()) +
            static_cast<std::size_t>(prefix_rank_device(leaf_mask, child_index)));
      } else {
        candidate.node_index = static_cast<std::size_t>(descriptor.child_base()) +
            static_cast<std::size_t>(prefix_rank_device(internal_mask, child_index));
        candidate.depth_remaining = entry.depth_remaining - 2;
      }
      if (stack_size >= kMaxWideStackEntries) {
        return false;
      }
      stack[stack_size++] = candidate;
      add_stat_device(stats != nullptr ? &stats->stack_pushes : nullptr);
      max_stat_device(stats != nullptr ? &stats->max_stack_depth : nullptr, static_cast<std::uint64_t>(stack_size));
    }
  }
  if (transmittance <= early_stop_transmittance) {
    add_stat_device(stats != nullptr ? &stats->early_terminations : nullptr);
  }

  return true;
}

__global__ void render_volume_backward_kernel(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const float* sigma,
    const float* color,
    const glm::vec3* grad_rgb,
    const float* grad_opacity,
    float* grad_sigma,
    float* grad_color,
    std::size_t count,
    std::size_t payload_rows,
    float near_plane,
    float far_plane,
    glm::vec3 background_color,
    float early_stop_transmittance,
    TraversalStats* stats,
    int* overflow_count) {
  const std::size_t ray_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (ray_index >= count) {
    return;
  }

  RenderSegmentDevice segments[kMaxRenderSegments];
  int segment_count = 0;
  const bool collected = collect_render_segments_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[ray_index],
      directions[ray_index],
      sigma,
      color,
      payload_rows,
      near_plane,
      far_plane,
      early_stop_transmittance,
      segments,
      segment_count,
      stats);
  if (!collected) {
    atomicAdd(overflow_count, 1);
    return;
  }

  const glm::vec3 upstream_rgb = grad_rgb[ray_index];
  const float upstream_opacity = grad_opacity[ray_index];
  glm::vec3 suffix_rgb = background_color;
  float suffix_transmittance = 1.0f;

  for (int index = segment_count - 1; index >= 0; --index) {
    const RenderSegmentDevice segment = segments[index];
    const float one_minus_alpha = 1.0f - segment.alpha;
    const float weight = segment.transmittance * segment.alpha;
    const std::size_t color_index = static_cast<std::size_t>(segment.payload_index) * 3u;

    atomicAdd(&grad_color[color_index], upstream_rgb.x * weight);
    atomicAdd(&grad_color[color_index + 1u], upstream_rgb.y * weight);
    atomicAdd(&grad_color[color_index + 2u], upstream_rgb.z * weight);

    const glm::vec3 color_delta = segment.color - suffix_rgb;
    const float grad_alpha =
        segment.transmittance * dot_vec3_device(upstream_rgb, color_delta) +
        upstream_opacity * segment.transmittance * suffix_transmittance;
    if (segment.raw_sigma > 0.0f) {
      atomicAdd(
          &grad_sigma[static_cast<std::size_t>(segment.payload_index)],
          grad_alpha * segment.delta * one_minus_alpha);
    }

    suffix_rgb = segment.alpha * segment.color + one_minus_alpha * suffix_rgb;
    suffix_transmittance = one_minus_alpha * suffix_transmittance;
  }
}

__global__ void render_volume_kernel(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const float* sigma,
    const float* color,
    glm::vec3* rgb,
    float* depth,
    float* opacity,
    std::size_t count,
    std::size_t payload_rows,
    float near_plane,
    float far_plane,
    glm::vec3 background_color,
    float early_stop_transmittance,
    TraversalStats* stats) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (index >= count) {
    return;
  }

  render_single_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[index],
      directions[index],
      sigma,
      color,
      payload_rows,
      near_plane,
      far_plane,
      background_color,
      early_stop_transmittance,
      stats,
      rgb[index],
      depth[index],
      opacity[index]);
}

__global__ void render_volume_wide_kernel(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const float* sigma,
    const float* color,
    glm::vec3* rgb,
    float* depth,
    float* opacity,
    std::size_t count,
    std::size_t payload_rows,
    float near_plane,
    float far_plane,
    glm::vec3 background_color,
    float early_stop_transmittance,
    TraversalStats* stats) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (index >= count) {
    return;
  }

  render_single_wide_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[index],
      directions[index],
      sigma,
      color,
      payload_rows,
      near_plane,
      far_plane,
      background_color,
      early_stop_transmittance,
      stats,
      rgb[index],
      depth[index],
      opacity[index]);
}

__global__ void render_volume_backward_wide_kernel(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const float* sigma,
    const float* color,
    const glm::vec3* grad_rgb,
    const float* grad_opacity,
    float* grad_sigma,
    float* grad_color,
    std::size_t count,
    std::size_t payload_rows,
    float near_plane,
    float far_plane,
    glm::vec3 background_color,
    float early_stop_transmittance,
    TraversalStats* stats,
    int* overflow_count) {
  const std::size_t ray_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (ray_index >= count) {
    return;
  }

  RenderSegmentDevice segments[kMaxRenderSegments];
  int segment_count = 0;
  const bool collected = collect_render_segments_wide_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[ray_index],
      directions[ray_index],
      sigma,
      color,
      payload_rows,
      near_plane,
      far_plane,
      early_stop_transmittance,
      segments,
      segment_count,
      stats);
  if (!collected) {
    atomicAdd(overflow_count, 1);
    return;
  }

  const glm::vec3 upstream_rgb = grad_rgb[ray_index];
  const float upstream_opacity = grad_opacity[ray_index];
  glm::vec3 suffix_rgb = background_color;
  float suffix_transmittance = 1.0f;

  for (int index = segment_count - 1; index >= 0; --index) {
    const RenderSegmentDevice segment = segments[index];
    const float one_minus_alpha = 1.0f - segment.alpha;
    const float weight = segment.transmittance * segment.alpha;
    const std::size_t color_index = static_cast<std::size_t>(segment.payload_index) * 3u;

    atomicAdd(&grad_color[color_index], upstream_rgb.x * weight);
    atomicAdd(&grad_color[color_index + 1u], upstream_rgb.y * weight);
    atomicAdd(&grad_color[color_index + 2u], upstream_rgb.z * weight);

    const glm::vec3 color_delta = segment.color - suffix_rgb;
    const float grad_alpha =
        segment.transmittance * dot_vec3_device(upstream_rgb, color_delta) +
        upstream_opacity * segment.transmittance * suffix_transmittance;
    if (segment.raw_sigma > 0.0f) {
      atomicAdd(
          &grad_sigma[static_cast<std::size_t>(segment.payload_index)],
          grad_alpha * segment.delta * one_minus_alpha);
    }

    suffix_rgb = segment.alpha * segment.color + one_minus_alpha * suffix_rgb;
    suffix_transmittance = one_minus_alpha * suffix_transmittance;
  }
}

__global__ void count_render_intervals_kernel(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    std::uint32_t* counts,
    std::size_t count,
    float near_plane,
    float far_plane,
    TraversalStats* stats,
    int* overflow_count) {
  const std::size_t ray_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (ray_index >= count) {
    return;
  }

  CountIntervalSinkDevice sink;
  const bool collected = collect_render_intervals_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[ray_index],
      directions[ray_index],
      near_plane,
      far_plane,
      sink,
      stats);
  if (!collected) {
    atomicAdd(overflow_count, 1);
    counts[ray_index] = 0u;
    return;
  }
  counts[ray_index] = sink.count;
}

__global__ void count_render_intervals_wide_kernel(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    std::uint32_t* counts,
    std::size_t count,
    float near_plane,
    float far_plane,
    TraversalStats* stats,
    int* overflow_count) {
  const std::size_t ray_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (ray_index >= count) {
    return;
  }

  CountIntervalSinkDevice sink;
  const bool collected = collect_render_intervals_wide_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[ray_index],
      directions[ray_index],
      near_plane,
      far_plane,
      sink,
      stats);
  if (!collected) {
    atomicAdd(overflow_count, 1);
    counts[ray_index] = 0u;
    return;
  }
  counts[ray_index] = sink.count;
}

__global__ void emit_render_intervals_kernel(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const std::uint32_t* offsets,
    std::uint32_t* ray_indices,
    float* t_start,
    float* t_end,
    std::int32_t* leaf_ids,
    std::uint32_t* payload_indices,
    std::size_t count,
    float near_plane,
    float far_plane,
    TraversalStats* stats,
    int* overflow_count) {
  const std::size_t ray_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (ray_index >= count) {
    return;
  }

  EmitIntervalSinkDevice sink;
  sink.cursor = offsets[ray_index];
  sink.ray_index = static_cast<std::uint32_t>(ray_index);
  sink.ray_indices = ray_indices;
  sink.t_start = t_start;
  sink.t_end = t_end;
  sink.leaf_ids = leaf_ids;
  sink.payload_indices = payload_indices;
  const bool collected = collect_render_intervals_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[ray_index],
      directions[ray_index],
      near_plane,
      far_plane,
      sink,
      stats);
  if (!collected) {
    atomicAdd(overflow_count, 1);
  }
}

__global__ void emit_render_intervals_wide_kernel(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    glm::vec3 min_bound,
    glm::vec3 max_bound,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const std::uint32_t* offsets,
    std::uint32_t* ray_indices,
    float* t_start,
    float* t_end,
    std::int32_t* leaf_ids,
    std::uint32_t* payload_indices,
    std::size_t count,
    float near_plane,
    float far_plane,
    TraversalStats* stats,
    int* overflow_count) {
  const std::size_t ray_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (ray_index >= count) {
    return;
  }

  EmitIntervalSinkDevice sink;
  sink.cursor = offsets[ray_index];
  sink.ray_index = static_cast<std::uint32_t>(ray_index);
  sink.ray_indices = ray_indices;
  sink.t_start = t_start;
  sink.t_end = t_end;
  sink.leaf_ids = leaf_ids;
  sink.payload_indices = payload_indices;
  const bool collected = collect_render_intervals_wide_device(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      AabbDevice{min_bound, max_bound},
      origins[ray_index],
      directions[ray_index],
      near_plane,
      far_plane,
      sink,
      stats);
  if (!collected) {
    atomicAdd(overflow_count, 1);
  }
}

__global__ void check_interval_offsets_kernel(const std::uint32_t* offsets, std::size_t count, int* overflow_count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  if (offsets[index + 1u] < offsets[index]) {
    atomicExch(overflow_count, 1);
  }
}

__global__ void render_volume_from_intervals_kernel(
    const std::uint32_t* offsets,
    const float* t_start,
    const float* t_end,
    const std::uint32_t* payload_indices,
    float* interval_alpha,
    float* interval_transmittance,
    const float* sigma,
    const float* color,
    glm::vec3* rgb,
    float* depth,
    float* opacity,
    std::size_t count,
    std::size_t payload_rows,
    glm::vec3 background_color,
    float early_stop_transmittance) {
  const std::size_t ray_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (ray_index >= count) {
    return;
  }

  AccumulatorDevice accum;
  const std::uint32_t begin = offsets[ray_index];
  const std::uint32_t end = offsets[ray_index + 1u];
  for (std::uint32_t interval = begin; interval < end; ++interval) {
    interval_alpha[interval] = 0.0f;
    if (accum.transmittance <= early_stop_transmittance) {
      interval_transmittance[interval] = 0.0f;
      continue;
    }
    interval_transmittance[interval] = accum.transmittance;

    const std::uint32_t payload_index = payload_indices[interval];
    if (static_cast<std::size_t>(payload_index) >= payload_rows) {
      continue;
    }

    const float delta = t_end[interval] - t_start[interval];
    if (delta <= 0.0f) {
      continue;
    }

    const float density = fmaxf(sigma[static_cast<std::size_t>(payload_index)], 0.0f);
    if (density <= 0.0f) {
      continue;
    }

    const float alpha = 1.0f - expf(-density * delta);
    const float weight = accum.transmittance * alpha;
    const std::size_t color_index = static_cast<std::size_t>(payload_index) * 3u;
    const glm::vec3 leaf_color{color[color_index], color[color_index + 1u], color[color_index + 2u]};
    interval_alpha[interval] = alpha;
    accum.rgb += weight * leaf_color;
    accum.depth += weight * (0.5f * (t_start[interval] + t_end[interval]));
    accum.opacity += weight;
    accum.transmittance *= 1.0f - alpha;
  }

  if (accum.transmittance <= early_stop_transmittance) {
    accum.transmittance = fmaxf(accum.transmittance, 0.0f);
  }
  rgb[ray_index] = accum.rgb + accum.transmittance * background_color;
  opacity[ray_index] = accum.opacity;
  depth[ray_index] = accum.opacity > 0.0f ? accum.depth / accum.opacity : CUDART_INF_F;
}

__global__ void render_volume_backward_from_intervals_kernel(
    const std::uint32_t* offsets,
    const float* t_start,
    const float* t_end,
    const std::uint32_t* payload_indices,
    const float* interval_alpha,
    const float* interval_transmittance,
    const float* sigma,
    const float* color,
    const glm::vec3* grad_rgb,
    const float* grad_opacity,
    float* grad_sigma,
    float* grad_color,
    std::size_t count,
    std::size_t payload_rows,
    glm::vec3 background_color) {
  const std::size_t ray_index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
      static_cast<std::size_t>(threadIdx.x);
  if (ray_index >= count) {
    return;
  }

  const glm::vec3 upstream_rgb = grad_rgb[ray_index];
  const float upstream_opacity = grad_opacity[ray_index];
  glm::vec3 suffix_rgb = background_color;
  float suffix_transmittance = 1.0f;
  const std::uint32_t begin = offsets[ray_index];
  const std::uint32_t end = offsets[ray_index + 1u];

  for (std::uint32_t reverse = end; reverse > begin; --reverse) {
    const std::uint32_t interval = reverse - 1u;
    const std::uint32_t payload_index = payload_indices[interval];
    const float alpha = interval_alpha[interval];
    const float one_minus_alpha = 1.0f - alpha;
    if (static_cast<std::size_t>(payload_index) < payload_rows) {
      const float transmittance = interval_transmittance[interval];
      const float weight = transmittance * alpha;
      const std::size_t color_index = static_cast<std::size_t>(payload_index) * 3u;
      const glm::vec3 leaf_color{color[color_index], color[color_index + 1u], color[color_index + 2u]};

      atomicAdd(&grad_color[color_index], upstream_rgb.x * weight);
      atomicAdd(&grad_color[color_index + 1u], upstream_rgb.y * weight);
      atomicAdd(&grad_color[color_index + 2u], upstream_rgb.z * weight);

      const glm::vec3 color_delta = leaf_color - suffix_rgb;
      const float grad_alpha =
          transmittance * dot_vec3_device(upstream_rgb, color_delta) +
          upstream_opacity * transmittance * suffix_transmittance;
      const float raw_sigma = sigma[static_cast<std::size_t>(payload_index)];
      const float delta = t_end[interval] - t_start[interval];
      if (raw_sigma > 0.0f && delta > 0.0f) {
        atomicAdd(
            &grad_sigma[static_cast<std::size_t>(payload_index)],
            grad_alpha * delta * one_minus_alpha);
      }

      suffix_rgb = alpha * leaf_color + one_minus_alpha * suffix_rgb;
    }
    suffix_transmittance = one_minus_alpha * suffix_transmittance;
  }
}

void check_not_null(const void* pointer, std::size_t count, const char* name) {
  if (count != 0 && pointer == nullptr) {
    throw ValidationError(std::string(name) + " cannot be null when count is non-zero");
  }
}

bool is_finite(float value) noexcept {
  return std::isfinite(value);
}

bool is_finite_vec3(const glm::vec3& value) noexcept {
  return is_finite(value.x) && is_finite(value.y) && is_finite(value.z);
}

void validate_options(const RenderOptions& options) {
  if (!is_finite(options.near_plane) || options.near_plane < 0.0f) {
    throw ValidationError("near_plane must be finite and non-negative");
  }
  if (std::isnan(options.far_plane) || options.far_plane <= options.near_plane) {
    throw ValidationError("far_plane must be greater than near_plane");
  }
  if (!is_finite_vec3(options.background_color)) {
    throw ValidationError("background_color must be finite");
  }
  if (!is_finite(options.early_stop_transmittance) || options.early_stop_transmittance < 0.0f ||
      options.early_stop_transmittance >= 1.0f) {
    throw ValidationError("early_stop_transmittance must be in [0, 1)");
  }
}

void check_cuda_launch(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw Error(std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

}  // namespace

void render_volume_cuda(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const float* sigma,
    const float* color,
    glm::vec3* rgb,
    float* depth,
    float* opacity,
    std::size_t count,
    std::size_t payload_rows,
    const RenderOptions& options,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > kMaxDepth) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }
  validate_options(options);
  check_not_null(nodes, num_nodes, "nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");
  check_not_null(rgb, count, "rgb");
  check_not_null(depth, count, "depth");
  check_not_null(opacity, count, "opacity");
  check_not_null(sigma, payload_rows, "sigma");
  if (payload_rows != 0 && color == nullptr) {
    throw ValidationError("color cannot be null when payload_rows is non-zero");
  }

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  render_volume_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      origins,
      directions,
      sigma,
      color,
      rgb,
      depth,
      opacity,
      count,
      payload_rows,
      options.near_plane,
      options.far_plane,
      options.background_color,
      options.early_stop_transmittance,
      options.stats);

  check_cuda_launch(cudaGetLastError(), "render_volume_kernel launch");
}

void render_volume_wide_cuda(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const float* sigma,
    const float* color,
    glm::vec3* rgb,
    float* depth,
    float* opacity,
    std::size_t count,
    std::size_t payload_rows,
    const RenderOptions& options,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > kMaxDepth || (max_depth % 2) != 0) {
    throw ValidationError("wide4 max_depth must be even and in the range [0, 30]");
  }
  validate_options(options);
  check_not_null(nodes, num_nodes, "wide_nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");
  check_not_null(rgb, count, "rgb");
  check_not_null(depth, count, "depth");
  check_not_null(opacity, count, "opacity");
  check_not_null(sigma, payload_rows, "sigma");
  if (payload_rows != 0 && color == nullptr) {
    throw ValidationError("color cannot be null when payload_rows is non-zero");
  }

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  render_volume_wide_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      origins,
      directions,
      sigma,
      color,
      rgb,
      depth,
      opacity,
      count,
      payload_rows,
      options.near_plane,
      options.far_plane,
      options.background_color,
      options.early_stop_transmittance,
      options.stats);

  check_cuda_launch(cudaGetLastError(), "render_volume_wide_kernel launch");
}

void render_volume_backward_cuda(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const float* sigma,
    const float* color,
    const glm::vec3* grad_rgb,
    const float* grad_opacity,
    float* grad_sigma,
    float* grad_color,
    std::size_t count,
    std::size_t payload_rows,
    const RenderOptions& options,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > kMaxDepth) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }
  validate_options(options);
  check_not_null(nodes, num_nodes, "nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");
  check_not_null(grad_rgb, count, "grad_rgb");
  check_not_null(grad_opacity, count, "grad_opacity");
  check_not_null(sigma, payload_rows, "sigma");
  if (payload_rows != 0 && color == nullptr) {
    throw ValidationError("color cannot be null when payload_rows is non-zero");
  }
  check_not_null(grad_sigma, payload_rows, "grad_sigma");
  check_not_null(grad_color, payload_rows * 3u, "grad_color");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  DeviceBuffer<int> device_overflow_count(1, Device::CUDA);
  check_cuda_launch(
      cudaMemsetAsync(device_overflow_count.data(), 0, sizeof(int), cuda_stream),
      "cudaMemsetAsync renderer backward overflow counter");

  render_volume_backward_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      origins,
      directions,
      sigma,
      color,
      grad_rgb,
      grad_opacity,
      grad_sigma,
      grad_color,
      count,
      payload_rows,
      options.near_plane,
      options.far_plane,
      options.background_color,
      options.early_stop_transmittance,
      options.stats,
      device_overflow_count.data());

  check_cuda_launch(cudaGetLastError(), "render_volume_backward_kernel launch");
  int host_overflow_count = 0;
  check_cuda_launch(
      cudaMemcpyAsync(
          &host_overflow_count,
          device_overflow_count.data(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync renderer backward overflow counter");
  check_cuda_launch(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize renderer backward overflow counter");
  if (host_overflow_count != 0) {
    throw Error("render_volume_backward_cuda exceeded the per-ray segment cache; reduce scene depth or split rays");
  }
}

void render_volume_backward_wide_cuda(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* origins,
    const glm::vec3* directions,
    const float* sigma,
    const float* color,
    const glm::vec3* grad_rgb,
    const float* grad_opacity,
    float* grad_sigma,
    float* grad_color,
    std::size_t count,
    std::size_t payload_rows,
    const RenderOptions& options,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > kMaxDepth || (max_depth % 2) != 0) {
    throw ValidationError("wide4 max_depth must be even and in the range [0, 30]");
  }
  validate_options(options);
  check_not_null(nodes, num_nodes, "wide_nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");
  check_not_null(grad_rgb, count, "grad_rgb");
  check_not_null(grad_opacity, count, "grad_opacity");
  check_not_null(sigma, payload_rows, "sigma");
  if (payload_rows != 0 && color == nullptr) {
    throw ValidationError("color cannot be null when payload_rows is non-zero");
  }
  check_not_null(grad_sigma, payload_rows, "grad_sigma");
  check_not_null(grad_color, payload_rows * 3u, "grad_color");

  if (count == 0) {
    return;
  }

  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  DeviceBuffer<int> device_overflow_count(1, Device::CUDA);
  check_cuda_launch(
      cudaMemsetAsync(device_overflow_count.data(), 0, sizeof(int), cuda_stream),
      "cudaMemsetAsync renderer backward overflow counter");

  render_volume_backward_wide_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      origins,
      directions,
      sigma,
      color,
      grad_rgb,
      grad_opacity,
      grad_sigma,
      grad_color,
      count,
      payload_rows,
      options.near_plane,
      options.far_plane,
      options.background_color,
      options.early_stop_transmittance,
      options.stats,
      device_overflow_count.data());

  check_cuda_launch(cudaGetLastError(), "render_volume_backward_wide_kernel launch");
  int host_overflow_count = 0;
  check_cuda_launch(
      cudaMemcpyAsync(
          &host_overflow_count,
          device_overflow_count.data(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync renderer backward overflow counter");
  check_cuda_launch(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize renderer backward overflow counter");
  if (host_overflow_count != 0) {
    throw Error("render_volume_backward_wide_cuda exceeded the per-ray segment cache; reduce scene depth or split rays");
  }
}

void build_render_intervals_cuda(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* origins,
    const glm::vec3* directions,
    std::size_t count,
    const RenderOptions& options,
    RenderIntervalBuffer& intervals,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > kMaxDepth) {
    throw ValidationError("max_depth must be in the range [0, 30]");
  }
  if (count >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw ValidationError("interval rendering supports fewer than INT_MAX rays per launch");
  }
  validate_options(options);
  check_not_null(nodes, num_nodes, "nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");

  intervals.ray_count = count;
  intervals.interval_count = 0;
  intervals.forward_aux_valid = false;
  if (count == 0) {
    intervals.counts.resize_discard(0, Device::CUDA);
    intervals.offsets.resize_discard(0, Device::CUDA);
    intervals.ray_indices.resize_discard(0, Device::CUDA);
    intervals.t_start.resize_discard(0, Device::CUDA);
    intervals.t_end.resize_discard(0, Device::CUDA);
    intervals.leaf_ids.resize_discard(0, Device::CUDA);
    intervals.payload_indices.resize_discard(0, Device::CUDA);
    intervals.alpha.resize_discard(0, Device::CUDA);
    intervals.transmittance.resize_discard(0, Device::CUDA);
    return;
  }

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  intervals.counts.resize_discard(count + 1u, Device::CUDA);
  intervals.offsets.resize_discard(count + 1u, Device::CUDA);

  DeviceBuffer<int> device_overflow_count(1, Device::CUDA);
  check_cuda_launch(
      cudaMemsetAsync(device_overflow_count.data(), 0, sizeof(int), cuda_stream),
      "cudaMemsetAsync renderer interval overflow counter");

  count_render_intervals_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      origins,
      directions,
      intervals.counts.data(),
      count,
      options.near_plane,
      options.far_plane,
      options.stats,
      device_overflow_count.data());
  check_cuda_launch(cudaGetLastError(), "count_render_intervals_kernel launch");
  check_cuda_launch(
      cudaMemsetAsync(intervals.counts.data() + count, 0, sizeof(std::uint32_t), cuda_stream),
      "cudaMemsetAsync renderer interval sentinel count");

  std::size_t temp_storage_bytes = 0;
  check_cuda_launch(
      cub::DeviceScan::ExclusiveSum(
          nullptr,
          temp_storage_bytes,
          intervals.counts.data(),
          intervals.offsets.data(),
          static_cast<int>(count + 1u),
          cuda_stream),
      "cub::DeviceScan::ExclusiveSum interval sizing");
  intervals.scan_temp_storage.resize_discard(temp_storage_bytes, Device::CUDA);
  check_cuda_launch(
      cub::DeviceScan::ExclusiveSum(
          intervals.scan_temp_storage.data(),
          temp_storage_bytes,
          intervals.counts.data(),
          intervals.offsets.data(),
          static_cast<int>(count + 1u),
          cuda_stream),
      "cub::DeviceScan::ExclusiveSum intervals");

  check_interval_offsets_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      intervals.offsets.data(), count, device_overflow_count.data());
  check_cuda_launch(cudaGetLastError(), "check_interval_offsets_kernel launch");

  std::uint32_t host_interval_count = 0u;
  int host_overflow_count = 0;
  check_cuda_launch(
      cudaMemcpyAsync(
          &host_interval_count,
          intervals.offsets.data() + count,
          sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync renderer interval count");
  check_cuda_launch(
      cudaMemcpyAsync(
          &host_overflow_count,
          device_overflow_count.data(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync renderer interval overflow counter");
  check_cuda_launch(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize renderer interval count");
  if (host_overflow_count != 0) {
    throw Error(
        "build_render_intervals_cuda exceeded the traversal stack or interval count uint32 capacity; "
        "reduce scene depth, split rays, or use direct rendering");
  }

  intervals.interval_count = static_cast<std::size_t>(host_interval_count);
  intervals.ray_indices.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.t_start.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.t_end.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.leaf_ids.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.payload_indices.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.alpha.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.transmittance.resize_discard(intervals.interval_count, Device::CUDA);

  check_cuda_launch(
      cudaMemsetAsync(device_overflow_count.data(), 0, sizeof(int), cuda_stream),
      "cudaMemsetAsync renderer interval emit overflow counter");
  if (intervals.interval_count != 0) {
    emit_render_intervals_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
        nodes,
        num_nodes,
        leaf_payload_indices,
        num_leaves,
        max_depth,
        root_bounds[0],
        root_bounds[1],
        origins,
        directions,
        intervals.offsets.data(),
        intervals.ray_indices.data(),
        intervals.t_start.data(),
        intervals.t_end.data(),
        intervals.leaf_ids.data(),
        intervals.payload_indices.data(),
        count,
        options.near_plane,
        options.far_plane,
        options.stats,
        device_overflow_count.data());
    check_cuda_launch(cudaGetLastError(), "emit_render_intervals_kernel launch");
  }

  host_overflow_count = 0;
  check_cuda_launch(
      cudaMemcpyAsync(
          &host_overflow_count,
          device_overflow_count.data(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync renderer interval emit overflow counter");
  check_cuda_launch(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize renderer interval emit");
  if (host_overflow_count != 0) {
    throw Error("build_render_intervals_cuda exceeded the traversal stack during emit; reduce scene depth or split rays");
  }
}

void build_render_intervals_wide_cuda(
    const WideNodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* origins,
    const glm::vec3* directions,
    std::size_t count,
    const RenderOptions& options,
    RenderIntervalBuffer& intervals,
    CudaStreamHandle stream) {
  if (max_depth < 0 || max_depth > kMaxDepth || (max_depth % 2) != 0) {
    throw ValidationError("wide4 max_depth must be even and in the range [0, 30]");
  }
  if (count >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw ValidationError("interval rendering supports fewer than INT_MAX rays per launch");
  }
  validate_options(options);
  check_not_null(nodes, num_nodes, "wide_nodes");
  check_not_null(leaf_payload_indices, num_leaves, "leaf_payload_indices");
  check_not_null(origins, count, "origins");
  check_not_null(directions, count, "directions");

  intervals.ray_count = count;
  intervals.interval_count = 0;
  intervals.forward_aux_valid = false;
  if (count == 0) {
    intervals.counts.resize_discard(0, Device::CUDA);
    intervals.offsets.resize_discard(0, Device::CUDA);
    intervals.ray_indices.resize_discard(0, Device::CUDA);
    intervals.t_start.resize_discard(0, Device::CUDA);
    intervals.t_end.resize_discard(0, Device::CUDA);
    intervals.leaf_ids.resize_discard(0, Device::CUDA);
    intervals.payload_indices.resize_discard(0, Device::CUDA);
    intervals.alpha.resize_discard(0, Device::CUDA);
    intervals.transmittance.resize_discard(0, Device::CUDA);
    return;
  }

  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  intervals.counts.resize_discard(count + 1u, Device::CUDA);
  intervals.offsets.resize_discard(count + 1u, Device::CUDA);

  DeviceBuffer<int> device_overflow_count(1, Device::CUDA);
  check_cuda_launch(
      cudaMemsetAsync(device_overflow_count.data(), 0, sizeof(int), cuda_stream),
      "cudaMemsetAsync renderer interval overflow counter");

  count_render_intervals_wide_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      nodes,
      num_nodes,
      leaf_payload_indices,
      num_leaves,
      max_depth,
      root_bounds[0],
      root_bounds[1],
      origins,
      directions,
      intervals.counts.data(),
      count,
      options.near_plane,
      options.far_plane,
      options.stats,
      device_overflow_count.data());
  check_cuda_launch(cudaGetLastError(), "count_render_intervals_wide_kernel launch");
  check_cuda_launch(
      cudaMemsetAsync(intervals.counts.data() + count, 0, sizeof(std::uint32_t), cuda_stream),
      "cudaMemsetAsync renderer interval sentinel count");

  std::size_t temp_storage_bytes = 0;
  check_cuda_launch(
      cub::DeviceScan::ExclusiveSum(
          nullptr,
          temp_storage_bytes,
          intervals.counts.data(),
          intervals.offsets.data(),
          static_cast<int>(count + 1u),
          cuda_stream),
      "cub::DeviceScan::ExclusiveSum interval sizing");
  intervals.scan_temp_storage.resize_discard(temp_storage_bytes, Device::CUDA);
  check_cuda_launch(
      cub::DeviceScan::ExclusiveSum(
          intervals.scan_temp_storage.data(),
          temp_storage_bytes,
          intervals.counts.data(),
          intervals.offsets.data(),
          static_cast<int>(count + 1u),
          cuda_stream),
      "cub::DeviceScan::ExclusiveSum intervals");

  check_interval_offsets_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      intervals.offsets.data(), count, device_overflow_count.data());
  check_cuda_launch(cudaGetLastError(), "check_interval_offsets_kernel launch");

  std::uint32_t host_interval_count = 0u;
  int host_overflow_count = 0;
  check_cuda_launch(
      cudaMemcpyAsync(
          &host_interval_count,
          intervals.offsets.data() + count,
          sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync renderer interval count");
  check_cuda_launch(
      cudaMemcpyAsync(
          &host_overflow_count,
          device_overflow_count.data(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync renderer interval overflow counter");
  check_cuda_launch(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize renderer interval count");
  if (host_overflow_count != 0) {
    throw Error(
        "build_render_intervals_wide_cuda exceeded the traversal stack or interval count uint32 capacity; "
        "reduce scene depth, split rays, or use direct rendering");
  }

  intervals.interval_count = static_cast<std::size_t>(host_interval_count);
  intervals.ray_indices.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.t_start.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.t_end.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.leaf_ids.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.payload_indices.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.alpha.resize_discard(intervals.interval_count, Device::CUDA);
  intervals.transmittance.resize_discard(intervals.interval_count, Device::CUDA);

  check_cuda_launch(
      cudaMemsetAsync(device_overflow_count.data(), 0, sizeof(int), cuda_stream),
      "cudaMemsetAsync renderer interval emit overflow counter");
  if (intervals.interval_count != 0) {
    emit_render_intervals_wide_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
        nodes,
        num_nodes,
        leaf_payload_indices,
        num_leaves,
        max_depth,
        root_bounds[0],
        root_bounds[1],
        origins,
        directions,
        intervals.offsets.data(),
        intervals.ray_indices.data(),
        intervals.t_start.data(),
        intervals.t_end.data(),
        intervals.leaf_ids.data(),
        intervals.payload_indices.data(),
        count,
        options.near_plane,
        options.far_plane,
        options.stats,
        device_overflow_count.data());
    check_cuda_launch(cudaGetLastError(), "emit_render_intervals_wide_kernel launch");
  }

  host_overflow_count = 0;
  check_cuda_launch(
      cudaMemcpyAsync(
          &host_overflow_count,
          device_overflow_count.data(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          cuda_stream),
      "cudaMemcpyAsync renderer interval emit overflow counter");
  check_cuda_launch(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize renderer interval emit");
  if (host_overflow_count != 0) {
    throw Error("build_render_intervals_wide_cuda exceeded the traversal stack during emit; reduce scene depth or split rays");
  }
}

void render_volume_from_intervals_cuda(
    RenderIntervalBuffer& intervals,
    const float* sigma,
    const float* color,
    glm::vec3* rgb,
    float* depth,
    float* opacity,
    std::size_t count,
    std::size_t payload_rows,
    const RenderOptions& options,
    CudaStreamHandle stream) {
  validate_options(options);
  if (intervals.ray_count != count) {
    throw ValidationError("interval buffer ray count does not match render ray count");
  }
  check_not_null(rgb, count, "rgb");
  check_not_null(depth, count, "depth");
  check_not_null(opacity, count, "opacity");
  check_not_null(sigma, payload_rows, "sigma");
  if (payload_rows != 0 && color == nullptr) {
    throw ValidationError("color cannot be null when payload_rows is non-zero");
  }
  if (count == 0) {
    intervals.forward_aux_valid = true;
    return;
  }
  check_not_null(intervals.offsets.data(), count + 1u, "interval offsets");
  check_not_null(intervals.t_start.data(), intervals.interval_count, "interval t_start");
  check_not_null(intervals.t_end.data(), intervals.interval_count, "interval t_end");
  check_not_null(intervals.payload_indices.data(), intervals.interval_count, "interval payload_indices");
  check_not_null(intervals.alpha.data(), intervals.interval_count, "interval alpha");
  check_not_null(intervals.transmittance.data(), intervals.interval_count, "interval transmittance");

  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  render_volume_from_intervals_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      intervals.offsets.data(),
      intervals.t_start.data(),
      intervals.t_end.data(),
      intervals.payload_indices.data(),
      intervals.alpha.data(),
      intervals.transmittance.data(),
      sigma,
      color,
      rgb,
      depth,
      opacity,
      count,
      payload_rows,
      options.background_color,
      options.early_stop_transmittance);
  check_cuda_launch(cudaGetLastError(), "render_volume_from_intervals_kernel launch");
  intervals.forward_aux_valid = true;
}

void render_volume_backward_from_intervals_cuda(
    const RenderIntervalBuffer& intervals,
    const float* sigma,
    const float* color,
    const glm::vec3* grad_rgb,
    const float* grad_opacity,
    float* grad_sigma,
    float* grad_color,
    std::size_t count,
    std::size_t payload_rows,
    const RenderOptions& options,
    CudaStreamHandle stream) {
  validate_options(options);
  if (intervals.ray_count != count) {
    throw ValidationError("interval buffer ray count does not match render ray count");
  }
  if (!intervals.forward_aux_valid) {
    throw ValidationError("interval buffer forward auxiliary data is not valid; run interval forward before backward");
  }
  check_not_null(grad_rgb, count, "grad_rgb");
  check_not_null(grad_opacity, count, "grad_opacity");
  check_not_null(sigma, payload_rows, "sigma");
  if (payload_rows != 0 && color == nullptr) {
    throw ValidationError("color cannot be null when payload_rows is non-zero");
  }
  check_not_null(grad_sigma, payload_rows, "grad_sigma");
  check_not_null(grad_color, payload_rows * 3u, "grad_color");
  if (count == 0) {
    return;
  }
  check_not_null(intervals.offsets.data(), count + 1u, "interval offsets");
  check_not_null(intervals.t_start.data(), intervals.interval_count, "interval t_start");
  check_not_null(intervals.t_end.data(), intervals.interval_count, "interval t_end");
  check_not_null(intervals.payload_indices.data(), intervals.interval_count, "interval payload_indices");
  check_not_null(intervals.alpha.data(), intervals.interval_count, "interval alpha");
  check_not_null(intervals.transmittance.data(), intervals.interval_count, "interval transmittance");

  constexpr int kBlockSize = 128;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  render_volume_backward_from_intervals_kernel<<<grid_size, kBlockSize, 0, cuda_stream>>>(
      intervals.offsets.data(),
      intervals.t_start.data(),
      intervals.t_end.data(),
      intervals.payload_indices.data(),
      intervals.alpha.data(),
      intervals.transmittance.data(),
      sigma,
      color,
      grad_rgb,
      grad_opacity,
      grad_sigma,
      grad_color,
      count,
      payload_rows,
      options.background_color);
  check_cuda_launch(cudaGetLastError(), "render_volume_backward_from_intervals_kernel launch");
}


}  // namespace svo
