#pragma once

#include <cstdint>
#include <vector>

#include <svo/Camera.hpp>
#include <svo/Octree.hpp>

namespace svo {

struct RaycastOptions {
  bool return_payload_indices = false;
};

struct RaycastBatch {
  std::vector<std::uint8_t> hit_mask;
  std::vector<std::int32_t> leaf_ids;
  std::vector<float> t;
  std::vector<glm::vec3> positions;
  std::vector<std::int32_t> depths;
  int width = 0;
  int height = 0;
};

RaycastBatch raycast_cpu(
    const Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const RaycastOptions& options = {});

RaycastBatch raycast_cpu(
    const Octree& octree,
    const RayBatch& rays,
    const RaycastOptions& options = {});

}  // namespace svo
