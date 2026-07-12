#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <svo/Camera.hpp>
#include <svo/Octree.hpp>

namespace svo {

struct RenderBatch {
  std::vector<glm::vec3> rgb;
  std::vector<float> depth;
  std::vector<float> opacity;
  int width = 0;
  int height = 0;
};

RenderBatch render_volume_cpu(
    const Octree& octree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    const RenderOptions& options = {});

RenderBatch render_volume_cpu(
    const Octree& octree,
    const RayBatch& rays,
    const float* sigma,
    const float* color,
    std::size_t payload_rows,
    const RenderOptions& options = {});

#if SVO_ENABLE_CUDA
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
    const RenderOptions& options = {},
    CudaStreamHandle stream = nullptr);

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
    const RenderOptions& options = {},
    CudaStreamHandle stream = nullptr);
#endif

}  // namespace svo
