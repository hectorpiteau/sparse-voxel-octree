#include <svo/Raycast.hpp>

#include <algorithm>
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

bool is_finite(float value) noexcept {
  return std::isfinite(value);
}

bool is_finite_vec3(const glm::vec3& value) noexcept {
  return is_finite(value.x) && is_finite(value.y) && is_finite(value.z);
}

int count_bits(std::uint8_t value) noexcept {
  int count = 0;
  while (value != 0u) {
    count += static_cast<int>(value & 1u);
    value = static_cast<std::uint8_t>(value >> 1u);
  }
  return count;
}

int prefix_rank(std::uint8_t mask, int child_index) noexcept {
  if (child_index <= 0) {
    return 0;
  }
  const std::uint8_t lower_mask = static_cast<std::uint8_t>((1u << child_index) - 1u);
  return count_bits(static_cast<std::uint8_t>(mask & lower_mask));
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

  const NodeDescriptor descriptor = octree.nodes()[node_index];
  const std::uint8_t child_mask = descriptor.child_mask();
  const std::uint8_t leaf_mask = descriptor.leaf_mask();
  const std::uint8_t internal_mask = descriptor.internal_child_mask();

  for (int child_index = 0; child_index < 8; ++child_index) {
    const std::uint8_t child_bit = static_cast<std::uint8_t>(1u << child_index);
    if ((child_mask & child_bit) == 0u) {
      continue;
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
  if (octree.nodes().empty()) {
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

  traverse_node(octree, 0, octree.max_depth(), root, origin, normalized, options, best);
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
