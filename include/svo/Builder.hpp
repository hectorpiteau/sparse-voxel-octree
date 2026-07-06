#pragma once

#include <vector>

#include <svo/Octree.hpp>

namespace svo {

Octree build_octree_cpu(const std::vector<glm::ivec3>& coordinates, const BuildOptions& options);

Octree build_octree_cpu(
    const std::vector<glm::ivec3>& coordinates,
    const std::vector<std::uint32_t>& payload_indices,
    const BuildOptions& options);

}  // namespace svo
