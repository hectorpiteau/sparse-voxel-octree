#pragma once

#include <string_view>

namespace svo {

std::string_view version() noexcept;
bool cuda_enabled() noexcept;

}  // namespace svo
