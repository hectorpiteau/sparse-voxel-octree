#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <svo/Octree.hpp>

namespace svo {

enum class SerializedArrayDType : std::uint8_t {
  Float32 = 1,
};

struct SerializedArray {
  std::string name;
  SerializedArrayDType dtype = SerializedArrayDType::Float32;
  std::vector<std::uint64_t> shape;
  std::vector<std::byte> data;
};

struct SerializedScene {
  Octree tree;
  std::vector<SerializedArray> arrays;
};

void save_svo(const std::string& path, const SerializedScene& scene);
SerializedScene load_svo(const std::string& path);

}  // namespace svo
