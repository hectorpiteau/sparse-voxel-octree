#include <svo/Builder.hpp>
#include <svo/Error.hpp>
#include <svo/Serialization.hpp>

#include <glm/ext/vector_int3.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void require_same_tree(const svo::Octree& expected, const svo::Octree& actual) {
  require(expected.max_depth() == actual.max_depth(), "max_depth should roundtrip");
  require(expected.branching() == actual.branching(), "branching should roundtrip");
  require(expected.num_nodes() == actual.num_nodes(), "node count should roundtrip");
  require(expected.leaf_payload_indices() == actual.leaf_payload_indices(), "leaf payload indices should roundtrip");
  require(expected.leaf_specs().size() == actual.leaf_specs().size(), "leaf spec count should roundtrip");
  for (int corner = 0; corner < 2; ++corner) {
    for (int axis = 0; axis < 3; ++axis) {
      require(expected.root_bounds()[corner][axis] == actual.root_bounds()[corner][axis], "root bounds should roundtrip");
    }
  }
  require(expected.nodes().size() == actual.nodes().size(), "octree8 descriptor count should roundtrip");
  for (std::size_t index = 0; index < expected.nodes().size(); ++index) {
    require(expected.nodes()[index].bits() == actual.nodes()[index].bits(), "octree8 descriptor bits should roundtrip");
  }
  require(expected.wide_nodes().size() == actual.wide_nodes().size(), "wide descriptor count should roundtrip");
  for (std::size_t index = 0; index < expected.wide_nodes().size(); ++index) {
    require(expected.wide_nodes()[index].child_mask() == actual.wide_nodes()[index].child_mask(), "wide child mask should roundtrip");
    require(expected.wide_nodes()[index].leaf_mask() == actual.wide_nodes()[index].leaf_mask(), "wide leaf mask should roundtrip");
    require(expected.wide_nodes()[index].child_base() == actual.wide_nodes()[index].child_base(), "wide child base should roundtrip");
    require(expected.wide_nodes()[index].payload_base() == actual.wide_nodes()[index].payload_base(), "wide payload base should roundtrip");
  }
  for (std::size_t index = 0; index < expected.leaf_specs().size(); ++index) {
    const svo::LeafSpec& a = expected.leaf_specs()[index];
    const svo::LeafSpec& b = actual.leaf_specs()[index];
    require(a.coord_min == b.coord_min, "leaf spec coordinate should roundtrip");
    require(a.depth == b.depth, "leaf spec depth should roundtrip");
    require(a.payload_index == b.payload_index, "leaf spec payload index should roundtrip");
  }
}

svo::SerializedArray float_array(std::string name, std::vector<std::uint64_t> shape, const std::vector<float>& values) {
  svo::SerializedArray array;
  array.name = std::move(name);
  array.dtype = svo::SerializedArrayDType::Float32;
  array.shape = std::move(shape);
  array.data.resize(values.size() * sizeof(float));
  std::memcpy(array.data.data(), values.data(), array.data.size());
  return array;
}

std::filesystem::path temp_path(const std::string& name) {
  return std::filesystem::temp_directory_path() / name;
}

template <typename T>
void write_value(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

}  // namespace

int main() {
  {
    svo::BuildOptions options;
    options.max_depth = 3;
    options.root_bounds = {glm::vec3{-1.5f, -1.5f, -1.5f}, glm::vec3{1.5f, 1.5f, 1.5f}};
    const std::vector<glm::ivec3> coords{{0, 0, 0}, {7, 7, 7}, {3, 4, 5}};
    const std::vector<std::uint32_t> payload_indices{4u, 2u, 7u};
    const svo::Octree tree = svo::Octree::from_voxels_cpu(coords, payload_indices, options);
    const std::vector<float> sigma{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    const std::vector<float> color(sigma.size() * 3u, 0.25f);
    svo::SerializedScene scene{
        tree,
        {
            float_array("sigma", {sigma.size()}, sigma),
            float_array("color", {sigma.size(), 3u}, color),
        }};
    const std::filesystem::path path = temp_path("svo_serialization_octree8.svo");
    svo::save_svo(path.string(), scene);
    const svo::SerializedScene loaded = svo::load_svo(path.string());
    require_same_tree(tree, loaded.tree);
    require(loaded.arrays.size() == 2, "payload array count should roundtrip");
    require(loaded.arrays[0].name == "sigma", "sigma payload name should roundtrip");
    require(loaded.arrays[0].data == scene.arrays[0].data, "sigma payload bytes should roundtrip");
    std::filesystem::remove(path);
  }

  {
    svo::BuildOptions options;
    options.max_depth = 4;
    options.branching = svo::BranchingMode::Wide4;
    const svo::Octree tree =
        svo::Octree::from_voxels_cpu({glm::ivec3{0, 0, 0}, glm::ivec3{15, 15, 15}}, options);
    const std::filesystem::path path = temp_path("svo_serialization_wide4.svo");
    svo::save_svo(path.string(), svo::SerializedScene{tree, {}});
    const svo::SerializedScene loaded = svo::load_svo(path.string());
    require_same_tree(tree, loaded.tree);
    std::filesystem::remove(path);
  }

  {
    bool failed = false;
    try {
      (void)svo::load_svo(temp_path("missing_svo_serialization_file.svo").string());
    } catch (const svo::ValidationError&) {
      failed = true;
    }
    require(failed, "loading a missing file should fail clearly");
  }

  {
    const std::filesystem::path path = temp_path("svo_serialization_bad_count.svo");
    std::ofstream out(path, std::ios::binary);
    const char magic[8] = {'S', 'V', 'O', 'B', 'I', 'N', '1', '\0'};
    out.write(magic, sizeof(magic));
    write_value(out, std::uint32_t{1});
    write_value(out, std::uint32_t{0x01020304u});
    write_value(out, std::uint32_t{0});
    write_value(out, std::int32_t{1});
    for (int index = 0; index < 6; ++index) {
      write_value(out, index < 3 ? 0.0f : 1.0f);
    }
    write_value(out, std::uint64_t{1000000});
    write_value(out, std::uint64_t{0});
    write_value(out, std::uint64_t{0});
    write_value(out, std::uint64_t{0});
    write_value(out, std::uint64_t{0});
    out.close();

    bool failed = false;
    try {
      (void)svo::load_svo(path.string());
    } catch (const svo::ValidationError&) {
      failed = true;
    }
    require(failed, "corrupt oversized section counts should fail before allocation");
    std::filesystem::remove(path);
  }

  return 0;
}
