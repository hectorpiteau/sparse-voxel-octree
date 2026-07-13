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

struct WideDdaCell {
  int child_index = 0;
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
  const float max_s = t_far - clipped_near;
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

void add_stat(std::uint64_t& field, std::uint64_t value = 1) noexcept {
  field += value;
}

void update_max_stat(std::uint64_t& field, std::uint64_t value) noexcept {
  field = std::max(field, value);
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
  if (options.stats != nullptr) {
    add_stat(options.stats->leaf_segments);
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
    if (options.stats != nullptr && accum.transmittance <= options.early_stop_transmittance) {
      add_stat(options.stats->early_terminations);
    }
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
  std::vector<SegmentCandidate> candidates;
  candidates.reserve(8);

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
      if (options.stats != nullptr) {
        add_stat(options.stats->early_terminations);
      }
      break;
    }
    if (candidate.leaf) {
      composite_leaf(octree, sigma, color, candidate, options, accum);
    } else {
      if (options.stats != nullptr) {
        add_stat(options.stats->stack_pushes);
        update_max_stat(
            options.stats->max_stack_depth,
            static_cast<std::uint64_t>(octree.max_depth() - candidate.depth_remaining + 1));
      }
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

void traverse_wide_node(
    const Octree& octree,
    std::size_t node_index,
    int depth_remaining,
    const Aabb& bounds,
    const glm::vec3& origin,
    const glm::vec3& direction,
    float node_t_near,
    float node_t_far,
    const float* sigma,
    const float* color,
    const RenderOptions& options,
    Accumulator& accum) {
  if (node_index >= octree.wide_nodes().size() || depth_remaining <= 0 ||
      accum.transmittance <= options.early_stop_transmittance) {
    if (options.stats != nullptr && accum.transmittance <= options.early_stop_transmittance) {
      add_stat(options.stats->early_terminations);
    }
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
    if (accum.transmittance <= options.early_stop_transmittance) {
      if (options.stats != nullptr) {
        add_stat(options.stats->early_terminations);
      }
      return false;
    }

    const int child_index = cell.child_index;
    const std::uint64_t child_bit = 1ull << child_index;
    if (options.stats != nullptr) {
      add_stat(options.stats->child_candidates_tested);
    }
    if ((child_mask & child_bit) == 0u) {
      return true;
    }

    if (std::min(cell.t_far, options.far_plane) <= std::max({cell.t_near, options.near_plane, 0.0f})) {
      return true;
    }

    SegmentCandidate candidate;
    candidate.leaf = (leaf_mask & child_bit) != 0u;
    candidate.bounds = cell.bounds;
    candidate.t_near = cell.t_near;
    candidate.t_far = cell.t_far;
    if (candidate.leaf) {
      candidate.leaf_id = static_cast<std::int32_t>(
          static_cast<std::size_t>(descriptor.payload_base()) +
          static_cast<std::size_t>(prefix_rank(leaf_mask, child_index)));
      candidate.depth = octree.max_depth() - depth_remaining + 2;
      composite_leaf(octree, sigma, color, candidate, options, accum);
      return true;
    } else {
      candidate.node_index = static_cast<std::size_t>(descriptor.child_base()) +
          static_cast<std::size_t>(prefix_rank(internal_mask, child_index));
      candidate.depth_remaining = depth_remaining - 2;
      if (options.stats != nullptr) {
        add_stat(options.stats->stack_pushes);
        update_max_stat(
            options.stats->max_stack_depth,
            static_cast<std::uint64_t>(octree.max_depth() - candidate.depth_remaining + 2));
      }
      traverse_wide_node(
          octree,
          candidate.node_index,
          candidate.depth_remaining,
          candidate.bounds,
          origin,
          direction,
          candidate.t_near,
          candidate.t_far,
          sigma,
          color,
          options,
          accum);
    }
    return true;
  });
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
  if (octree.num_nodes() != 0) {
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
        if (options.stats != nullptr) {
          add_stat(options.stats->stack_pushes);
          update_max_stat(options.stats->max_stack_depth, 1);
        }
        if (octree.branching() == BranchingMode::Wide4) {
          traverse_wide_node(
              octree,
              0,
              octree.max_depth(),
              root,
              origin,
              normalized,
              root_t_near,
              root_t_far,
              sigma,
              color,
              options,
              accum);
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
