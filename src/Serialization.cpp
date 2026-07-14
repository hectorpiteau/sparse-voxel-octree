#include <svo/Serialization.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <svo/Error.hpp>

namespace svo {
namespace {

constexpr std::array<char, 8> kMagic{'S', 'V', 'O', 'B', 'I', 'N', '1', '\0'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304u;
constexpr std::uint64_t kMaxArrayNameBytes = 4096u;
constexpr std::uint64_t kMaxArrayDims = 8u;

template <typename T>
void write_value(std::ostream& out, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out) {
    throw ValidationError("failed to write SVO file");
  }
}

template <typename T>
T read_value(std::istream& in, const char* name) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    throw ValidationError(std::string("truncated SVO file while reading ") + name);
  }
  return value;
}

void write_bytes(std::ostream& out, const void* data, std::uint64_t byte_count) {
  if (byte_count == 0) {
    return;
  }
  if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    throw ValidationError("SVO section is too large to write");
  }
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(byte_count));
  if (!out) {
    throw ValidationError("failed to write SVO file");
  }
}

void read_bytes(std::istream& in, void* data, std::uint64_t byte_count, const char* name) {
  if (byte_count == 0) {
    return;
  }
  if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    throw ValidationError(std::string("SVO section is too large to read: ") + name);
  }
  in.read(static_cast<char*>(data), static_cast<std::streamsize>(byte_count));
  if (!in) {
    throw ValidationError(std::string("truncated SVO file while reading ") + name);
  }
}

std::uint64_t checked_bytes(std::uint64_t count, std::uint64_t element_size, const char* name) {
  if (element_size != 0 && count > std::numeric_limits<std::uint64_t>::max() / element_size) {
    throw ValidationError(std::string("SVO section byte count overflow: ") + name);
  }
  return count * element_size;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* name) {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    throw ValidationError(std::string("SVO section byte count overflow: ") + name);
  }
  return a + b;
}

std::uint64_t stream_size(std::istream& in) {
  const std::istream::pos_type current = in.tellg();
  if (current < 0) {
    throw ValidationError("failed to inspect SVO file position");
  }
  in.seekg(0, std::ios::end);
  const std::istream::pos_type end = in.tellg();
  if (end < 0) {
    throw ValidationError("failed to inspect SVO file size");
  }
  in.seekg(current);
  return static_cast<std::uint64_t>(end);
}

std::uint64_t bytes_read(std::istream& in) {
  const std::istream::pos_type position = in.tellg();
  if (position < 0) {
    throw ValidationError("failed to inspect SVO file position");
  }
  return static_cast<std::uint64_t>(position);
}

std::uint64_t remaining_bytes(std::istream& in, std::uint64_t file_size) {
  const std::uint64_t position = bytes_read(in);
  if (position > file_size) {
    throw ValidationError("SVO file read position exceeds file size");
  }
  return file_size - position;
}

void require_remaining_bytes(std::istream& in, std::uint64_t file_size, std::uint64_t byte_count, const char* name) {
  if (byte_count > remaining_bytes(in, file_size)) {
    throw ValidationError(std::string("SVO file section exceeds remaining file size: ") + name);
  }
}

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
  write_bytes(out, values.data(), checked_bytes(values.size(), sizeof(T), "vector"));
}

std::uint64_t array_element_size(SerializedArrayDType dtype) {
  switch (dtype) {
    case SerializedArrayDType::Float32:
      return sizeof(float);
  }
  throw ValidationError("unsupported serialized array dtype");
}

std::uint64_t array_element_count(const std::vector<std::uint64_t>& shape) {
  std::uint64_t count = 1;
  for (const std::uint64_t dim : shape) {
    if (dim == 0) {
      return 0;
    }
    if (count > std::numeric_limits<std::uint64_t>::max() / dim) {
      throw ValidationError("serialized array shape overflows element count");
    }
    count *= dim;
  }
  return count;
}

void validate_array(const SerializedArray& array) {
  if (array.name.empty()) {
    throw ValidationError("serialized array name cannot be empty");
  }
  if (array.name.size() > kMaxArrayNameBytes) {
    throw ValidationError("serialized array name is too long");
  }
  if (array.shape.size() > kMaxArrayDims) {
    throw ValidationError("serialized array has too many dimensions");
  }
  const std::uint64_t expected_bytes = checked_bytes(
      array_element_count(array.shape),
      array_element_size(array.dtype),
      "serialized array");
  if (array.data.size() != expected_bytes) {
    throw ValidationError("serialized array byte count does not match dtype and shape");
  }
}

std::vector<LeafSpec> read_leaf_specs(std::istream& in, std::uint64_t count, std::uint64_t file_size) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw ValidationError("leaf spec count exceeds host size_t range");
  }
  require_remaining_bytes(in, file_size, checked_bytes(count, 5u * sizeof(std::uint32_t), "leaf specs"), "leaf specs");
  std::vector<LeafSpec> specs;
  specs.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    LeafSpec spec;
    spec.coord_min.x = read_value<std::int32_t>(in, "leaf spec x");
    spec.coord_min.y = read_value<std::int32_t>(in, "leaf spec y");
    spec.coord_min.z = read_value<std::int32_t>(in, "leaf spec z");
    spec.depth = read_value<std::int32_t>(in, "leaf spec depth");
    spec.payload_index = read_value<std::uint32_t>(in, "leaf spec payload index");
    specs.push_back(spec);
  }
  return specs;
}

void write_leaf_specs(std::ostream& out, const std::vector<LeafSpec>& specs) {
  for (const LeafSpec& spec : specs) {
    write_value(out, static_cast<std::int32_t>(spec.coord_min.x));
    write_value(out, static_cast<std::int32_t>(spec.coord_min.y));
    write_value(out, static_cast<std::int32_t>(spec.coord_min.z));
    write_value(out, static_cast<std::int32_t>(spec.depth));
    write_value(out, static_cast<std::uint32_t>(spec.payload_index));
  }
}

}  // namespace

void save_svo(const std::string& path, const SerializedScene& scene) {
  scene.tree.validate();
  for (const SerializedArray& array : scene.arrays) {
    validate_array(array);
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw ValidationError("failed to open SVO file for writing: " + path);
  }

  out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  write_value(out, kFormatVersion);
  write_value(out, kEndianMarker);
  write_value(out, static_cast<std::uint32_t>(scene.tree.branching()));
  write_value(out, static_cast<std::int32_t>(scene.tree.max_depth()));
  for (int corner = 0; corner < 2; ++corner) {
    for (int axis = 0; axis < 3; ++axis) {
      write_value(out, scene.tree.root_bounds()[corner][axis]);
    }
  }

  write_value(out, static_cast<std::uint64_t>(scene.tree.nodes().size()));
  write_value(out, static_cast<std::uint64_t>(scene.tree.wide_nodes().size()));
  write_value(out, static_cast<std::uint64_t>(scene.tree.leaf_payload_indices().size()));
  write_value(out, static_cast<std::uint64_t>(scene.tree.leaf_specs().size()));
  write_value(out, static_cast<std::uint64_t>(scene.arrays.size()));

  for (const NodeDescriptor& node : scene.tree.nodes()) {
    write_value(out, node.bits());
  }
  for (const WideNodeDescriptor& node : scene.tree.wide_nodes()) {
    write_value(out, node.child_mask());
    write_value(out, node.leaf_mask());
    write_value(out, node.child_base());
    write_value(out, node.payload_base());
  }
  write_vector(out, scene.tree.leaf_payload_indices());
  write_leaf_specs(out, scene.tree.leaf_specs());

  for (const SerializedArray& array : scene.arrays) {
    write_value(out, static_cast<std::uint64_t>(array.name.size()));
    write_bytes(out, array.name.data(), array.name.size());
    write_value(out, static_cast<std::uint32_t>(array.dtype));
    write_value(out, static_cast<std::uint64_t>(array.shape.size()));
    for (const std::uint64_t dim : array.shape) {
      write_value(out, dim);
    }
    write_value(out, static_cast<std::uint64_t>(array.data.size()));
    write_bytes(out, array.data.data(), array.data.size());
  }
}

SerializedScene load_svo(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw ValidationError("failed to open SVO file for reading: " + path);
  }
  const std::uint64_t file_size = stream_size(in);

  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!in || magic != kMagic) {
    throw ValidationError("invalid SVO file magic");
  }

  const std::uint32_t version = read_value<std::uint32_t>(in, "version");
  if (version != kFormatVersion) {
    throw ValidationError("unsupported SVO file version");
  }
  const std::uint32_t endian = read_value<std::uint32_t>(in, "endian marker");
  if (endian != kEndianMarker) {
    throw ValidationError("SVO file endian marker is not supported on this host");
  }

  const auto branching_value = read_value<std::uint32_t>(in, "branching mode");
  BranchingMode branching;
  if (branching_value == static_cast<std::uint32_t>(BranchingMode::Octree8)) {
    branching = BranchingMode::Octree8;
  } else if (branching_value == static_cast<std::uint32_t>(BranchingMode::Wide4)) {
    branching = BranchingMode::Wide4;
  } else {
    throw ValidationError("SVO file contains invalid branching mode");
  }

  const int max_depth = read_value<std::int32_t>(in, "max depth");
  RootBounds root_bounds{};
  for (int corner = 0; corner < 2; ++corner) {
    for (int axis = 0; axis < 3; ++axis) {
      root_bounds[corner][axis] = read_value<float>(in, "root bounds");
    }
  }

  const std::uint64_t node_count = read_value<std::uint64_t>(in, "node count");
  const std::uint64_t wide_node_count = read_value<std::uint64_t>(in, "wide node count");
  const std::uint64_t leaf_count = read_value<std::uint64_t>(in, "leaf count");
  const std::uint64_t leaf_spec_count = read_value<std::uint64_t>(in, "leaf spec count");
  const std::uint64_t array_count = read_value<std::uint64_t>(in, "array count");

  if (node_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      wide_node_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      leaf_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      leaf_spec_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      array_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw ValidationError("SVO file count exceeds host size_t range");
  }

  std::uint64_t required_topology_bytes = checked_bytes(node_count, sizeof(std::uint64_t), "node descriptors");
  required_topology_bytes = checked_add(
      required_topology_bytes,
      checked_bytes(wide_node_count, 2u * sizeof(std::uint64_t) + 2u * sizeof(std::uint32_t), "wide node descriptors"),
      "topology sections");
  required_topology_bytes = checked_add(
      required_topology_bytes,
      checked_bytes(leaf_count, sizeof(std::uint32_t), "leaf payload indices"),
      "topology sections");
  required_topology_bytes = checked_add(
      required_topology_bytes,
      checked_bytes(leaf_spec_count, 5u * sizeof(std::uint32_t), "leaf specs"),
      "topology sections");
  required_topology_bytes = checked_add(required_topology_bytes, array_count * sizeof(std::uint64_t), "array headers");
  require_remaining_bytes(in, file_size, required_topology_bytes, "topology and array headers");

  std::vector<NodeDescriptor> nodes;
  nodes.reserve(static_cast<std::size_t>(node_count));
  for (std::uint64_t index = 0; index < node_count; ++index) {
    nodes.emplace_back(read_value<std::uint64_t>(in, "node descriptor"));
  }

  std::vector<WideNodeDescriptor> wide_nodes;
  wide_nodes.reserve(static_cast<std::size_t>(wide_node_count));
  for (std::uint64_t index = 0; index < wide_node_count; ++index) {
    const std::uint64_t child_mask = read_value<std::uint64_t>(in, "wide child mask");
    const std::uint64_t leaf_mask = read_value<std::uint64_t>(in, "wide leaf mask");
    const std::uint32_t child_base = read_value<std::uint32_t>(in, "wide child base");
    const std::uint32_t payload_base = read_value<std::uint32_t>(in, "wide payload base");
    wide_nodes.push_back(WideNodeDescriptor::pack(child_mask, leaf_mask, child_base, payload_base));
  }

  std::vector<std::uint32_t> leaf_payload_indices(static_cast<std::size_t>(leaf_count));
  read_bytes(
      in,
      leaf_payload_indices.data(),
      checked_bytes(leaf_count, sizeof(std::uint32_t), "leaf payload indices"),
      "leaf payload indices");
  std::vector<LeafSpec> leaf_specs = read_leaf_specs(in, leaf_spec_count, file_size);

  std::vector<SerializedArray> arrays;
  arrays.reserve(static_cast<std::size_t>(array_count));
  for (std::uint64_t array_index = 0; array_index < array_count; ++array_index) {
    const std::uint64_t name_size = read_value<std::uint64_t>(in, "array name size");
    if (name_size == 0 || name_size > kMaxArrayNameBytes) {
      throw ValidationError("SVO file contains invalid array name size");
    }
    require_remaining_bytes(
        in,
        file_size,
        checked_add(name_size, sizeof(std::uint32_t) + sizeof(std::uint64_t), "array header"),
        "array header");
    SerializedArray array;
    array.name.resize(static_cast<std::size_t>(name_size));
    read_bytes(in, array.name.data(), name_size, "array name");
    const auto dtype_value = read_value<std::uint32_t>(in, "array dtype");
    if (dtype_value != static_cast<std::uint32_t>(SerializedArrayDType::Float32)) {
      throw ValidationError("SVO file contains unsupported array dtype");
    }
    array.dtype = SerializedArrayDType::Float32;
    const std::uint64_t dim_count = read_value<std::uint64_t>(in, "array dimension count");
    if (dim_count > kMaxArrayDims) {
      throw ValidationError("SVO file contains too many array dimensions");
    }
    require_remaining_bytes(
        in,
        file_size,
        checked_add(checked_bytes(dim_count, sizeof(std::uint64_t), "array dimensions"), sizeof(std::uint64_t), "array dimensions"),
        "array dimensions");
    array.shape.resize(static_cast<std::size_t>(dim_count));
    for (std::uint64_t dim_index = 0; dim_index < dim_count; ++dim_index) {
      array.shape[static_cast<std::size_t>(dim_index)] = read_value<std::uint64_t>(in, "array dimension");
    }
    const std::uint64_t byte_count = read_value<std::uint64_t>(in, "array byte count");
    const std::uint64_t expected_bytes = checked_bytes(
        array_element_count(array.shape),
        array_element_size(array.dtype),
        "serialized array");
    if (byte_count != expected_bytes) {
      throw ValidationError("SVO file array byte count does not match dtype and shape");
    }
    if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw ValidationError("SVO file array exceeds host size_t range");
    }
    require_remaining_bytes(in, file_size, byte_count, "array data");
    array.data.resize(static_cast<std::size_t>(byte_count));
    read_bytes(in, array.data.data(), byte_count, "array data");
    arrays.push_back(std::move(array));
  }

  if (bytes_read(in) != file_size) {
    throw ValidationError("SVO file contains trailing bytes");
  }

  Octree tree{
      max_depth,
      Device::CPU,
      root_bounds,
      branching,
      std::move(nodes),
      std::move(wide_nodes),
      std::move(leaf_payload_indices),
      std::move(leaf_specs)};
  tree.validate();

  return SerializedScene{std::move(tree), std::move(arrays)};
}

}  // namespace svo
