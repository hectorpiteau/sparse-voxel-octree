#include <svo/Error.hpp>
#include <svo/Octree.hpp>
#include <svo/Version.hpp>

#include <iostream>
#include <string>
#include <string_view>

int main() {
  if (svo::version() != std::string_view{"0.1.0"}) {
    std::cerr << "unexpected version\n";
    return 1;
  }

  if (sizeof(svo::NodeDescriptor) != sizeof(std::uint64_t)) {
    std::cerr << "NodeDescriptor must remain a single 64-bit word\n";
    return 1;
  }

  constexpr svo::NodeDescriptor descriptor =
      svo::NodeDescriptor::pack(0b00000111u, 0b00000011u, 12u, 34u);
  if (descriptor.child_mask() != 0b00000111u || descriptor.leaf_mask() != 0b00000011u ||
      descriptor.child_base() != 12u || descriptor.payload_base() != 34u) {
    std::cerr << "descriptor bit packing is incorrect\n";
    return 1;
  }

  svo::Octree empty_tree;
  empty_tree.validate();

  std::string error_message;
  try {
    svo::Octree invalid_tree{
        1,
        svo::Device::CPU,
        svo::default_root_bounds(),
        {svo::NodeDescriptor::pack(0b00000001u, 0b00000010u, 0u, 0u)},
        {}};
    invalid_tree.validate();
    std::cerr << "invalid tree unexpectedly validated\n";
    return 1;
  } catch (const svo::ValidationError& error) {
    error_message = error.what();
  }

  if (error_message.find("subset of child mask") == std::string::npos) {
    std::cerr << "unexpected validation error: " << error_message << '\n';
    return 1;
  }

  return 0;
}
