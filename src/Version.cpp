#include <svo/Version.hpp>

namespace svo {

std::string_view version() noexcept {
  return "0.1.0";
}

bool cuda_enabled() noexcept {
#if SVO_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

}  // namespace svo
