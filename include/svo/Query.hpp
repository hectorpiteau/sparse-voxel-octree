#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <svo/Octree.hpp>

namespace svo {

std::vector<std::int32_t> query_points(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const QueryOptions& options = {});

#if SVO_ENABLE_CUDA
void query_points_cuda(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    std::int32_t* results,
    std::size_t count,
    const QueryOptions& options = {},
    CudaStreamHandle stream = nullptr);
#endif

}  // namespace svo
