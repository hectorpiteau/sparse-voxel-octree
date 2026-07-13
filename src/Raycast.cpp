#include <svo/Raycast.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <glm/geometric.hpp>

#include <svo/Error.hpp>

namespace svo {
namespace {

constexpr float kEpsilon = 1.0e-6f;

struct Aabb {
  glm::vec3 min;
  glm::vec3 max;
};

struct HitCandidate {
  bool hit = false;
  std::int32_t leaf_id = -1;
  float t = std::numeric_limits<float>::infinity();
  glm::vec3 position{
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  std::int32_t depth = -1;
};

struct WideDdaCell {
  int child_index = 0;
  Aabb bounds{};
  float t_near = 0.0f;
  float t_far = 0.0f;
};

bool is_finite(float value) noexcept {
  return std::isfinite(value);
}

bool is_finite_vec3(const glm::vec3& value) noexcept {
  return is_finite(value.x) && is_finite(value.y) && is_finite(value.z);
}

int count_bits(std::uint8_t value) noexcept {
  return std::popcount(value);
}

int count_bits(std::uint64_t value) noexcept {
  return std::popcount(value);
}

int prefix_rank(std::uint8_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint8_t lower_mask = static_cast<std::uint8_t>((1u << child_index) - 1u);
  return count_bits(static_cast<std::uint8_t>(mask & lower_mask));
}

int prefix_rank(std::uint64_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint64_t lower_mask = (1ull << child_index) - 1ull;
  return count_bits(mask & lower_mask);
}

Aabb child_bounds(const Aabb& bounds, int child_index) noexcept {
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

bool intersect_axis(float origin, float direction, float min_bound, float max_bound, float& t_min, float& t_max) {
  if (std::fabs(direction) <= kEpsilon) {
    return origin >= min_bound && origin <= max_bound;
  }

  float t0 = (min_bound - origin) / direction;
  float t1 = (max_bound - origin) / direction;
  if (t0 > t1) {
    std::swap(t0, t1);
  }

  t_min = std::max(t_min, t0);
  t_max = std::min(t_max, t1);
  return t_min <= t_max;
}

bool intersect_aabb(
    const Aabb& bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float& t_near,
    float& t_far) {
  t_near = -std::numeric_limits<float>::infinity();
  t_far = std::numeric_limits<float>::infinity();

  return intersect_axis(origin.x, direction.x, bounds.min.x, bounds.max.x, t_near, t_far) &&
      intersect_axis(origin.y, direction.y, bounds.min.y, bounds.max.y, t_near, t_far) &&
      intersect_axis(origin.z, direction.z, bounds.min.z, bounds.max.z, t_near, t_far) && t_far >= 0.0f;
}

int local_cell_for_position(float position, float min_bound, float cell_size, float walk_direction) noexcept {
  const float local = (position - min_bound) / cell_size;
  const float rounded = std::round(local);
  if (std::fabs(local - rounded) <= 8.0f * kEpsilon) {
    const int boundary = static_cast<int>(rounded);
    if (boundary <= 0) {
      return 0;
    }
    if (boundary >= 4) {
      return 3;
    }
    return walk_direction < -kEpsilon ? boundary - 1 : boundary;
  }
  return std::clamp(static_cast<int>(std::floor(local)), 0, 3);
}

void advance_axis(int& cell, int step, float& next_t, float delta_t) noexcept {
  cell += step;
  next_t += delta_t;
}

template <typename Visitor>
bool visit_wide_cells_dda(
    const Aabb& bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float t_near,
    float t_far,
    bool reverse,
    Visitor&& visitor) {
  const float clipped_near = std::max(t_near, 0.0f);
  if (t_far < clipped_near) {
    return true;
  }

  const float base_t = reverse ? t_far : clipped_near;
  const float max_s = reverse ? t_far - clipped_near : t_far - clipped_near;
  const glm::vec3 walk_origin = origin + direction * base_t;
  const glm::vec3 walk_direction = reverse ? -direction : direction;
  const glm::vec3 cell_size = (bounds.max - bounds.min) * 0.25f;

  int cell_x = local_cell_for_position(walk_origin.x, bounds.min.x, cell_size.x, walk_direction.x);
  int cell_y = local_cell_for_position(walk_origin.y, bounds.min.y, cell_size.y, walk_direction.y);
  int cell_z = local_cell_for_position(walk_origin.z, bounds.min.z, cell_size.z, walk_direction.z);

  const auto axis_setup = [](float origin_axis,
                             float direction_axis,
                             float min_bound,
                             float cell_size_axis,
                             int cell,
                             int& step,
                             float& next_s,
                             float& delta_s) {
    if (std::fabs(direction_axis) <= kEpsilon) {
      step = 0;
      next_s = std::numeric_limits<float>::infinity();
      delta_s = std::numeric_limits<float>::infinity();
      return;
    }
    step = direction_axis > 0.0f ? 1 : -1;
    const int boundary_index = step > 0 ? cell + 1 : cell;
    const float boundary = min_bound + cell_size_axis * static_cast<float>(boundary_index);
    next_s = (boundary - origin_axis) / direction_axis;
    if (next_s < 0.0f && next_s > -8.0f * kEpsilon) {
      next_s = 0.0f;
    }
    delta_s = cell_size_axis / std::fabs(direction_axis);
  };

  int step_x = 0;
  int step_y = 0;
  int step_z = 0;
  float next_x = 0.0f;
  float next_y = 0.0f;
  float next_z = 0.0f;
  float delta_x = 0.0f;
  float delta_y = 0.0f;
  float delta_z = 0.0f;
  axis_setup(walk_origin.x, walk_direction.x, bounds.min.x, cell_size.x, cell_x, step_x, next_x, delta_x);
  axis_setup(walk_origin.y, walk_direction.y, bounds.min.y, cell_size.y, cell_y, step_y, next_y, delta_y);
  axis_setup(walk_origin.z, walk_direction.z, bounds.min.z, cell_size.z, cell_z, step_z, next_z, delta_z);

  float current_s = 0.0f;
  for (int iteration = 0; iteration < 96; ++iteration) {
    if (cell_x < 0 || cell_x >= 4 || cell_y < 0 || cell_y >= 4 || cell_z < 0 || cell_z >= 4 ||
        current_s > max_s + kEpsilon) {
      break;
    }

    const float next_s = std::min({next_x, next_y, next_z, max_s});
    const int child_index = cell_x | (cell_y << 2) | (cell_z << 4);
    const glm::vec3 child_min =
        bounds.min + cell_size * glm::vec3{static_cast<float>(cell_x), static_cast<float>(cell_y), static_cast<float>(cell_z)};
    WideDdaCell cell;
    cell.child_index = child_index;
    cell.bounds = {child_min, child_min + cell_size};
    if (reverse) {
      cell.t_near = base_t - next_s;
      cell.t_far = base_t - current_s;
    } else {
      cell.t_near = base_t + current_s;
      cell.t_far = base_t + next_s;
    }
    if (!visitor(cell)) {
      return false;
    }
    if (next_s >= max_s - kEpsilon) {
      break;
    }

    current_s = next_s;
    if (next_x <= next_y + kEpsilon && next_x <= next_z + kEpsilon) {
      advance_axis(cell_x, step_x, next_x, delta_x);
    } else if (next_y <= next_z + kEpsilon) {
      advance_axis(cell_y, step_y, next_y, delta_y);
    } else {
      advance_axis(cell_z, step_z, next_z, delta_z);
    }
  }

  return true;
}

bool is_better_hit(const HitCandidate& candidate, const HitCandidate& best) noexcept {
  if (!candidate.hit) {
    return false;
  }
  if (!best.hit) {
    return true;
  }
  if (candidate.t < best.t - kEpsilon) {
    return true;
  }
  return std::fabs(candidate.t - best.t) <= kEpsilon && candidate.leaf_id < best.leaf_id;
}

void add_stat(std::uint64_t& field, std::uint64_t value = 1) noexcept {
  field += value;
}

void update_max_stat(std::uint64_t& field, std::uint64_t value) noexcept {
  field = std::max(field, value);
}

void consider_leaf(
    const Octree& octree,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const Aabb& bounds,
    std::int32_t leaf_id,
    std::int32_t depth,
    const RaycastOptions& options,
    HitCandidate& best) {
  float t_near = 0.0f;
  float t_far = 0.0f;
  if (!intersect_aabb(bounds, origin, direction, t_near, t_far)) {
    return;
  }
  if (options.stats != nullptr) {
    add_stat(options.stats->leaf_segments);
  }

  HitCandidate candidate;
  candidate.hit = true;
  candidate.leaf_id = options.return_payload_indices
      ? static_cast<std::int32_t>(octree.leaf_payload_indices()[static_cast<std::size_t>(leaf_id)])
      : leaf_id;
  candidate.t = std::max(t_near, 0.0f);
  candidate.position = origin + direction * candidate.t;
  candidate.depth = depth;

  if (is_better_hit(candidate, best)) {
    best = candidate;
  }
}

void traverse_node(
    const Octree& octree,
    std::size_t node_index,
    int depth_remaining,
    const Aabb& bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const RaycastOptions& options,
    HitCandidate& best) {
  if (node_index >= octree.nodes().size() || depth_remaining <= 0) {
    return;
  }
  if (options.stats != nullptr) {
    add_stat(options.stats->nodes_visited);
    add_stat(options.stats->stack_pops);
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
    if (options.stats != nullptr) {
      add_stat(options.stats->child_candidates_tested);
    }

    const Aabb child = child_bounds(bounds, child_index);
    float t_near = 0.0f;
    float t_far = 0.0f;
    if (!intersect_aabb(child, origin, direction, t_near, t_far)) {
      continue;
    }
    if (best.hit && std::max(t_near, 0.0f) > best.t + kEpsilon) {
      continue;
    }

    if ((leaf_mask & child_bit) != 0u) {
      const std::int32_t leaf_id = static_cast<std::int32_t>(
          static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank(leaf_mask, child_index)));
      const std::int32_t depth = octree.max_depth() - depth_remaining + 1;
      consider_leaf(octree, origin, direction, child, leaf_id, depth, options, best);
      continue;
    }

    const std::size_t child_node_index = static_cast<std::size_t>(descriptor.child_base()) +
        static_cast<std::size_t>(prefix_rank(internal_mask, child_index));
    if (options.stats != nullptr) {
      add_stat(options.stats->stack_pushes);
      update_max_stat(
          options.stats->max_stack_depth,
          static_cast<std::uint64_t>(octree.max_depth() - depth_remaining + 2));
    }
    traverse_node(
        octree,
        child_node_index,
        depth_remaining - 1,
        child,
        origin,
        direction,
        options,
        best);
  }
}

void traverse_wide_node(
    const Octree& octree,
    std::size_t node_index,
    int depth_remaining,
    const Aabb& bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float node_t_near,
    float node_t_far,
    const RaycastOptions& options,
    HitCandidate& best) {
  if (node_index >= octree.wide_nodes().size() || depth_remaining <= 0) {
    return;
  }
  if (options.stats != nullptr) {
    add_stat(options.stats->nodes_visited);
    add_stat(options.stats->stack_pops);
  }

  const WideNodeDescriptor descriptor = octree.wide_nodes()[node_index];
  const std::uint64_t child_mask = descriptor.child_mask();
  const std::uint64_t leaf_mask = descriptor.leaf_mask();
  const std::uint64_t internal_mask = descriptor.internal_child_mask();

  visit_wide_cells_dda(bounds, origin, direction, node_t_near, node_t_far, false, [&](const WideDdaCell& cell) {
    const int child_index = cell.child_index;
    const std::uint64_t child_bit = 1ull << child_index;
    if (options.stats != nullptr) {
      add_stat(options.stats->child_candidates_tested);
    }
    if ((child_mask & child_bit) == 0u) {
      return true;
    }

    if (best.hit && std::max(cell.t_near, 0.0f) > best.t + kEpsilon) {
      return true;
    }

    if ((leaf_mask & child_bit) != 0u) {
      const std::int32_t leaf_id = static_cast<std::int32_t>(
          static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank(leaf_mask, child_index)));
      const std::int32_t depth = octree.max_depth() - depth_remaining + 2;
      consider_leaf(octree, origin, direction, cell.bounds, leaf_id, depth, options, best);
      return true;
    }

    const std::size_t child_node_index = static_cast<std::size_t>(descriptor.child_base()) +
        static_cast<std::size_t>(prefix_rank(internal_mask, child_index));
    if (options.stats != nullptr) {
      add_stat(options.stats->stack_pushes);
      update_max_stat(
          options.stats->max_stack_depth,
          static_cast<std::uint64_t>(octree.max_depth() - depth_remaining + 3));
    }
    traverse_wide_node(
        octree,
        child_node_index,
        depth_remaining - 2,
        cell.bounds,
        origin,
        direction,
        cell.t_near,
        cell.t_far,
        options,
        best);
    return true;
  });
}

glm::vec3 normalized_direction(const glm::vec3& direction) {
  if (!is_finite_vec3(direction)) {
    throw ValidationError("ray directions must be finite");
  }
  const float length = glm::length(direction);
  if (!is_finite(length) || length <= kEpsilon) {
    throw ValidationError("ray directions must be non-zero");
  }
  return direction / length;
}

HitCandidate raycast_one(
    const Octree& octree,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const RaycastOptions& options) {
  if (!is_finite_vec3(origin)) {
    throw ValidationError("ray origins must be finite");
  }
  const glm::vec3 normalized = normalized_direction(direction);

  HitCandidate best;
  if (octree.num_nodes() == 0) {
    return best;
  }

  const Aabb root{octree.root_bounds()[0], octree.root_bounds()[1]};
  float root_t_near = 0.0f;
  float root_t_far = 0.0f;
  if (!intersect_aabb(root, origin, normalized, root_t_near, root_t_far)) {
    return best;
  }

  if (octree.max_depth() == 0) {
    consider_leaf(octree, origin, normalized, root, 0, 0, options, best);
    return best;
  }

  if (options.stats != nullptr) {
    add_stat(options.stats->stack_pushes);
    update_max_stat(options.stats->max_stack_depth, 1);
  }
  if (octree.branching() == BranchingMode::Wide4) {
    traverse_wide_node(octree, 0, octree.max_depth(), root, origin, normalized, root_t_near, root_t_far, options, best);
  } else {
    traverse_node(octree, 0, octree.max_depth(), root, origin, normalized, options, best);
  }
  return best;
}

}  // namespace

RaycastBatch raycast_cpu(
    const Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const RaycastOptions& options) {
  if (octree.device() != Device::CPU) {
    throw ValidationError("raycast_cpu requires a CPU octree");
  }
  if (origins.size() != directions.size()) {
    throw ValidationError("ray origins and directions must have the same count");
  }

  RaycastBatch batch;
  batch.width = static_cast<int>(origins.size());
  batch.height = origins.empty() ? 0 : 1;
  batch.hit_mask.reserve(origins.size());
  batch.leaf_ids.reserve(origins.size());
  batch.t.reserve(origins.size());
  batch.positions.reserve(origins.size());
  batch.depths.reserve(origins.size());

  for (std::size_t index = 0; index < origins.size(); ++index) {
    const HitCandidate hit = raycast_one(octree, origins[index], directions[index], options);
    batch.hit_mask.push_back(hit.hit ? 1u : 0u);
    batch.leaf_ids.push_back(hit.leaf_id);
    batch.t.push_back(hit.t);
    batch.positions.push_back(hit.position);
    batch.depths.push_back(hit.depth);
  }

  return batch;
}

RaycastBatch raycast_cpu(
    const Octree& octree,
    const RayBatch& rays,
    const RaycastOptions& options) {
  if (rays.origins.size() != rays.directions.size()) {
    throw ValidationError("RayBatch origins and directions must have the same count");
  }
  const std::size_t expected_count =
      static_cast<std::size_t>(rays.width) * static_cast<std::size_t>(rays.height);
  if (expected_count != rays.origins.size()) {
    throw ValidationError("RayBatch dimensions must match origin and direction count");
  }

  RaycastBatch result = raycast_cpu(octree, rays.origins, rays.directions, options);
  result.width = rays.width;
  result.height = rays.height;
  return result;
}

}  // namespace svo
