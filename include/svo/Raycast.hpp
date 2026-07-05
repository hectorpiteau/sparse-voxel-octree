#pragma once

#include <cstddef>
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

#if SVO_ENABLE_CUDA
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
    const RaycastOptions& options = {},
    CudaStreamHandle stream = nullptr);
#endif

}  // namespace svo
