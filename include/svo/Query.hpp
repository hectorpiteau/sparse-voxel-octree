#pragma once

#include <cstdint>
#include <vector>

#include <svo/Octree.hpp>

namespace svo {

std::vector<std::int32_t> query_points(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const QueryOptions& options = {});

}  // namespace svo
