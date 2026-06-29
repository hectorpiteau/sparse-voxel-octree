#pragma once

#include <array>

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_int3.hpp>

namespace svo {

using RootBounds = std::array<glm::vec3, 2>;

inline RootBounds default_root_bounds() noexcept {
  return {glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{1.0f, 1.0f, 1.0f}};
}

}  // namespace svo
