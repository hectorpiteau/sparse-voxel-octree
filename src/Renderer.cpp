#include <svo/Renderer.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <glm/geometric.hpp>

#include <svo/Error.hpp>

namespace svo {
namespace {

constexpr float kEpsilon = 1.0e-6f;

struct Aabb {
  glm::vec3 min;
  glm::vec3 max;
};

struct SegmentCandidate {
  bool leaf = false;
  std::size_t node_index = 0;
  std::int32_t leaf_id = -1;
  std::int32_t depth = -1;
  int depth_remaining = 0;
  Aabb bounds{};
  float t_near = 0.0f;
  float t_far = 0.0f;
};

struct Accumulator {
  glm::vec3 rgb{0.0f, 0.0f, 0.0f};
  float depth = 0.0f;
  float opacity = 0.0f;
  float transmittance = 1.0f;
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

float segment_sort_key(const SegmentCandidate& candidate, const RenderOptions& options) noexcept {
  return std::max({candidate.t_near, options.near_plane, 0.0f});
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

void validate_render_options(const RenderOptions& options) {
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

void validate_payload(const Octree& octree, const float* sigma, const float* color, std::size_t payload_rows) {
  if (payload_rows > 0 && sigma == nullptr) {
    throw ValidationError("sigma cannot be null when payload_rows is non-zero");
  }
  if (payload_rows > 0 && color == nullptr) {
    throw ValidationError("color cannot be null when payload_rows is non-zero");
  }
  for (std::uint32_t payload_index : octree.leaf_payload_indices()) {
    if (payload_index >= payload_rows) {
      throw ValidationError("leaf payload index is outside the payload row range");
    }
  }
}

void composite_leaf(
    const Octree& octree,
    const float* sigma,
    const float* color,
    const SegmentCandidate& segment,
    const RenderOptions& options,
    Accumulator& accum) {
  if (segment.leaf_id < 0 || static_cast<std::size_t>(segment.leaf_id) >= octree.leaf_payload_indices().size()) {
    return;
  }

  const float t0 = std::max({segment.t_near, options.near_plane, 0.0f});
  const float t1 = std::min(segment.t_far, options.far_plane);
  if (t1 <= t0) {
    return;
  }

  const std::uint32_t payload_index = octree.leaf_payload_indices()[static_cast<std::size_t>(segment.leaf_id)];
  const float density = std::max(sigma[static_cast<std::size_t>(payload_index)], 0.0f);
  if (density <= 0.0f) {
    return;
  }

  const float alpha = 1.0f - std::exp(-density * (t1 - t0));
  const float weight = accum.transmittance * alpha;
  const auto color_index = static_cast<std::size_t>(payload_index) * 3u;
  const glm::vec3 leaf_color{color[color_index], color[color_index + 1u], color[color_index + 2u]};
  const float midpoint = 0.5f * (t0 + t1);

  accum.rgb += weight * leaf_color;
  accum.depth += weight * midpoint;
  accum.opacity += weight;
  accum.transmittance *= 1.0f - alpha;
}

void traverse_node(
    const Octree& octree,
    std::size_t node_index,
    int depth_remaining,
    const Aabb& bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const float* sigma,
    const float* color,
    const RenderOptions& options,
    Accumulator& accum) {
  if (node_index >= octree.nodes().size() || depth_remaining <= 0 ||
      accum.transmittance <= options.early_stop_transmittance) {
    return;
  }

  const NodeDescriptor descriptor = octree.nodes()[node_index];
  const std::uint8_t child_mask = descriptor.child_mask();
  const std::uint8_t leaf_mask = descriptor.leaf_mask();
  const std::uint8_t internal_mask = descriptor.internal_child_mask();
  std::vector<SegmentCandidate> candidates;
  candidates.reserve(8);

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
    if (std::min(t_far, options.far_plane) <= std::max({t_near, options.near_plane, 0.0f})) {
      continue;
    }

    SegmentCandidate candidate;
    candidate.leaf = (leaf_mask & child_bit) != 0u;
    candidate.bounds = child;
    candidate.t_near = t_near;
    candidate.t_far = t_far;
    if (candidate.leaf) {
      candidate.leaf_id = static_cast<std::int32_t>(
          static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank(leaf_mask, child_index)));
      candidate.depth = octree.max_depth() - depth_remaining + 1;
    } else {
      candidate.node_index = static_cast<std::size_t>(descriptor.child_base()) +
          static_cast<std::size_t>(prefix_rank(internal_mask, child_index));
      candidate.depth_remaining = depth_remaining - 1;
    }
    candidates.push_back(candidate);
  }

  std::sort(candidates.begin(), candidates.end(), [&](const SegmentCandidate& lhs, const SegmentCandidate& rhs) {
    const float lhs_key = segment_sort_key(lhs, options);
    const float rhs_key = segment_sort_key(rhs, options);
    if (std::fabs(lhs_key - rhs_key) > kEpsilon) {
      return lhs_key < rhs_key;
    }
    return lhs.leaf_id < rhs.leaf_id;
  });

  for (const SegmentCandidate& candidate : candidates) {
    if (accum.transmittance <= options.early_stop_transmittance) {
      break;
    }
    if (candidate.leaf) {
      composite_leaf(octree, sigma, color, candidate, options, accum);
    } else {
      traverse_node(
          octree,
          candidate.node_index,
          candidate.depth_remaining,
          candidate.bounds,
          origin,
          direction,
          sigma,
          color,
          options,
          accum);
    }
  }
}

void render_one(
    const Octree& octree,
    const glm::vec3& origin,
    const glm::vec3& direction,
    const float* sigma,
    const float* color,
    const RenderOptions& options,
    glm::vec3& rgb,
    float& depth,
    float& opacity) {
  if (!is_finite_vec3(origin)) {
    throw ValidationError("ray origins must be finite");
  }
  const glm::vec3 normalized = normalized_direction(direction);

  Accumulator accum;
  if (!octree.nodes().empty()) {
    const Aabb root{octree.root_bounds()[0], octree.root_bounds()[1]};
    float root_t_near = 0.0f;
    float root_t_far = 0.0f;
    if (intersect_aabb(root, origin, normalized, root_t_near, root_t_far) &&
        std::min(root_t_far, options.far_plane) > std::max({root_t_near, options.near_plane, 0.0f})) {
      if (octree.max_depth() == 0) {
        SegmentCandidate root_leaf;
        root_leaf.leaf = true;
        root_leaf.leaf_id = 0;
        root_leaf.depth = 0;
        root_leaf.bounds = root;
        root_leaf.t_near = root_t_near;
        root_leaf.t_far = root_t_far;
        composite_leaf(octree, sigma, color, root_leaf, options, accum);
      } else {
        traverse_node(
            octree,
            0,
            octree.max_depth(),
            root,
            origin,
            normalized,
            sigma,
            color,
            options,
            accum);
      }
    }
  }

  rgb = accum.rgb + accum.transmittance * options.background_color;
  opacity = accum.opacity;
  depth = accum.opacity > 0.0f ? accum.depth / accum.opacity : std::numeric_limits<float>::infinity();
}

}  // namespace

RenderBatch render_volume_cpu(
    const Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    const RenderOptions& options) {
  if (octree.device() != Device::CPU) {
    throw ValidationError("render_volume_cpu requires a CPU octree");
  }
  if (origins.size() != directions.size()) {
    throw ValidationError("ray origins and directions must have the same count");
  }
  validate_render_options(options);
  validate_payload(octree, sigma, color, payload_rows);

  RenderBatch batch;
  batch.width = static_cast<int>(origins.size());
  batch.height = origins.empty() ? 0 : 1;
  batch.rgb.resize(origins.size());
  batch.depth.resize(origins.size());
  batch.opacity.resize(origins.size());

  for (std::size_t index = 0; index < origins.size(); ++index) {
    render_one(
        octree,
        origins[index],
        directions[index],
        sigma,
        color,
        options,
        batch.rgb[index],
        batch.depth[index],
        batch.opacity[index]);
  }

  return batch;
}

RenderBatch render_volume_cpu(
    const Octree& octree,
    const RayBatch& rays,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    const RenderOptions& options) {
  if (rays.origins.size() != rays.directions.size()) {
    throw ValidationError("RayBatch origins and directions must have the same count");
  }
  if (rays.width < 0 || rays.height < 0) {
    throw ValidationError("RayBatch dimensions must be non-negative");
  }
  const std::size_t expected_count =
      static_cast<std::size_t>(rays.width) * static_cast<std::size_t>(rays.height);
  if (expected_count != rays.origins.size()) {
    throw ValidationError("RayBatch dimensions must match origin and direction count");
  }

  RenderBatch result = render_volume_cpu(octree, rays.origins, rays.directions, sigma, color, payload_rows, options);
  result.width = rays.width;
  result.height = rays.height;
  return result;
}

}  // namespace svo
