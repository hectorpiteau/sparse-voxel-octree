#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if SVO_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <svo/Camera.hpp>
#include <svo/CoarseOccupancy.hpp>
#include <svo/DeviceBuffer.hpp>
#include <svo/Error.hpp>
#include <svo/Interpolation.hpp>
#include <svo/Octree.hpp>
#include <svo/Query.hpp>
#include <svo/Raycast.hpp>
#include <svo/Renderer.hpp>
#include <svo/Serialization.hpp>
#include <svo/Version.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

static_assert(sizeof(glm::vec3) == 3 * sizeof(float), "Torch CUDA interop requires tightly packed glm::vec3");

bool is_c_contiguous(const py::array& array) {
  return (array.flags() & py::array::c_style) == py::array::c_style;
}

svo::RootBounds root_bounds_from_python(py::object root_bounds_obj) {
  if (root_bounds_obj.is_none()) {
    return svo::default_root_bounds();
  }

  py::array array = py::array::ensure(root_bounds_obj);
  if (!array) {
    throw py::type_error("root_bounds must be convertible to a NumPy array");
  }
  if (array.ndim() != 2 || array.shape(0) != 2 || array.shape(1) != 3) {
    throw py::value_error("root_bounds must have shape (2, 3)");
  }

  py::array_t<float, py::array::c_style> cast(array);
  auto view = cast.unchecked<2>();
  return {
      glm::vec3{view(0, 0), view(0, 1), view(0, 2)},
      glm::vec3{view(1, 0), view(1, 1), view(1, 2)},
  };
}

svo::BranchingMode branching_from_python(const std::string& branching) {
  if (branching == "octree8") {
    return svo::BranchingMode::Octree8;
  }
  if (branching == "wide4") {
    return svo::BranchingMode::Wide4;
  }
  throw py::value_error("branching must be 'octree8' or 'wide4'");
}

std::vector<glm::ivec3> coordinates_from_numpy(const py::array& array) {
  if (array.ndim() != 2 || array.shape(1) != 3) {
    throw py::value_error("coords must have shape (N, 3)");
  }
  if (!(py::isinstance<py::array_t<std::int32_t>>(array) || py::isinstance<py::array_t<std::int64_t>>(array))) {
    throw py::type_error("coords must have dtype int32 or int64");
  }
  if (!is_c_contiguous(array)) {
    throw py::value_error("coords must be C-contiguous");
  }

  std::vector<glm::ivec3> coordinates;
  coordinates.reserve(static_cast<std::size_t>(array.shape(0)));

  if (py::isinstance<py::array_t<std::int32_t>>(array)) {
    py::array_t<std::int32_t, py::array::c_style> cast(array);
    auto view = cast.unchecked<2>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      coordinates.emplace_back(view(index, 0), view(index, 1), view(index, 2));
    }
  } else {
    py::array_t<std::int64_t, py::array::c_style> cast(array);
    auto view = cast.unchecked<2>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      coordinates.emplace_back(
          static_cast<int>(view(index, 0)),
          static_cast<int>(view(index, 1)),
          static_cast<int>(view(index, 2)));
    }
  }

  return coordinates;
}

std::vector<std::uint32_t> payload_indices_from_numpy(const py::array& array, py::ssize_t expected_count) {
  if (array.ndim() != 1) {
    throw py::value_error("payload_indices must have shape (N,)");
  }
  if (array.shape(0) != expected_count) {
    throw py::value_error("payload_indices must have the same length as coords");
  }
  if (!(py::isinstance<py::array_t<std::int32_t>>(array) || py::isinstance<py::array_t<std::int64_t>>(array))) {
    throw py::type_error("payload_indices must have dtype int32 or int64");
  }
  if (!is_c_contiguous(array)) {
    throw py::value_error("payload_indices must be C-contiguous");
  }

  std::vector<std::uint32_t> payload_indices;
  payload_indices.reserve(static_cast<std::size_t>(array.shape(0)));

  if (py::isinstance<py::array_t<std::int32_t>>(array)) {
    py::array_t<std::int32_t, py::array::c_style> cast(array);
    auto view = cast.unchecked<1>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      if (view(index) < 0) {
        throw py::value_error("payload_indices must be non-negative");
      }
      payload_indices.push_back(static_cast<std::uint32_t>(view(index)));
    }
  } else {
    py::array_t<std::int64_t, py::array::c_style> cast(array);
    auto view = cast.unchecked<1>();
    constexpr std::int64_t max_index = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      if (view(index) < 0) {
        throw py::value_error("payload_indices must be non-negative");
      }
      if (view(index) > max_index) {
        throw py::value_error("payload_indices must be representable as int32");
      }
      payload_indices.push_back(static_cast<std::uint32_t>(view(index)));
    }
  }

  return payload_indices;
}

std::vector<int> depths_from_numpy(const py::array& array, py::ssize_t expected_count) {
  if (array.ndim() != 1) {
    throw py::value_error("depths must have shape (N,)");
  }
  if (array.shape(0) != expected_count) {
    throw py::value_error("depths must have the same length as coord_min");
  }
  if (!(py::isinstance<py::array_t<std::int32_t>>(array) || py::isinstance<py::array_t<std::int64_t>>(array))) {
    throw py::type_error("depths must have dtype int32 or int64");
  }
  if (!is_c_contiguous(array)) {
    throw py::value_error("depths must be C-contiguous");
  }

  std::vector<int> depths;
  depths.reserve(static_cast<std::size_t>(array.shape(0)));
  if (py::isinstance<py::array_t<std::int32_t>>(array)) {
    py::array_t<std::int32_t, py::array::c_style> cast(array);
    auto view = cast.unchecked<1>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      depths.push_back(static_cast<int>(view(index)));
    }
  } else {
    py::array_t<std::int64_t, py::array::c_style> cast(array);
    auto view = cast.unchecked<1>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      depths.push_back(static_cast<int>(view(index)));
    }
  }
  return depths;
}

std::vector<svo::LeafSpec> leaf_specs_from_numpy(
    const py::array& coord_min,
    const py::array& depths,
    const py::array& payload_indices) {
  const std::vector<glm::ivec3> coords = coordinates_from_numpy(coord_min);
  const std::vector<int> depth_values = depths_from_numpy(depths, coord_min.shape(0));
  const std::vector<std::uint32_t> payload_values = payload_indices_from_numpy(payload_indices, coord_min.shape(0));

  std::vector<svo::LeafSpec> specs;
  specs.reserve(coords.size());
  for (std::size_t index = 0; index < coords.size(); ++index) {
    specs.push_back(svo::LeafSpec{coords[index], depth_values[index], payload_values[index]});
  }
  return specs;
}

std::vector<glm::vec3> points_from_numpy(const py::array& array) {
  if (array.ndim() != 2 || array.shape(1) != 3) {
    throw py::value_error("points must have shape (N, 3)");
  }
  if (!(py::isinstance<py::array_t<float>>(array) || py::isinstance<py::array_t<double>>(array))) {
    throw py::type_error("points must have dtype float32 or float64");
  }
  if (!is_c_contiguous(array)) {
    throw py::value_error("points must be C-contiguous");
  }

  std::vector<glm::vec3> points;
  points.reserve(static_cast<std::size_t>(array.shape(0)));

  if (py::isinstance<py::array_t<float>>(array)) {
    py::array_t<float, py::array::c_style> cast(array);
    auto view = cast.unchecked<2>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      points.emplace_back(view(index, 0), view(index, 1), view(index, 2));
    }
  } else {
    py::array_t<double, py::array::c_style> cast(array);
    auto view = cast.unchecked<2>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      points.emplace_back(
          static_cast<float>(view(index, 0)),
          static_cast<float>(view(index, 1)),
          static_cast<float>(view(index, 2)));
    }
  }

  return points;
}

struct PythonPointBatch {
  std::vector<glm::vec3> points;
  std::vector<py::ssize_t> leading_shape;
};

PythonPointBatch point_batch_from_numpy(const py::array& array) {
  if (array.ndim() < 2 || array.shape(array.ndim() - 1) != 3) {
    throw py::value_error("points must have shape (..., 3)");
  }
  if (!(py::isinstance<py::array_t<float>>(array) || py::isinstance<py::array_t<double>>(array))) {
    throw py::type_error("points must have dtype float32 or float64");
  }
  if (!is_c_contiguous(array)) {
    throw py::value_error("points must be C-contiguous");
  }

  PythonPointBatch batch;
  py::ssize_t count = 1;
  for (py::ssize_t axis = 0; axis < array.ndim() - 1; ++axis) {
    batch.leading_shape.push_back(array.shape(axis));
    count *= array.shape(axis);
  }
  batch.points.reserve(static_cast<std::size_t>(count));

  if (py::isinstance<py::array_t<float>>(array)) {
    const auto* data = static_cast<const float*>(array.data());
    for (py::ssize_t index = 0; index < count; ++index) {
      batch.points.emplace_back(data[index * 3], data[index * 3 + 1], data[index * 3 + 2]);
    }
  } else {
    const auto* data = static_cast<const double*>(array.data());
    for (py::ssize_t index = 0; index < count; ++index) {
      batch.points.emplace_back(
          static_cast<float>(data[index * 3]),
          static_cast<float>(data[index * 3 + 1]),
          static_cast<float>(data[index * 3 + 2]));
    }
  }

  return batch;
}

struct PythonPayloadView {
  py::array array;
  py::ssize_t rows = 0;
  py::ssize_t channels = 1;
  bool scalar = true;
  bool double_precision = false;
};

PythonPayloadView payload_from_numpy(py::array array) {
  if (array.ndim() != 1 && array.ndim() != 2) {
    throw py::value_error("payload must have shape (P,) or (P, C)");
  }
  if (!(py::isinstance<py::array_t<float>>(array) || py::isinstance<py::array_t<double>>(array))) {
    throw py::type_error("payload must have dtype float32 or float64");
  }
  if (!is_c_contiguous(array)) {
    throw py::value_error("payload must be C-contiguous");
  }

  PythonPayloadView payload;
  payload.array = array;
  payload.rows = array.shape(0);
  payload.scalar = array.ndim() == 1;
  payload.channels = payload.scalar ? 1 : array.shape(1);
  payload.double_precision = py::isinstance<py::array_t<double>>(array);
  if (payload.channels <= 0) {
    throw py::value_error("payload must have at least one channel");
  }
  return payload;
}

std::vector<py::ssize_t> interpolation_output_shape(
    const std::vector<py::ssize_t>& leading_shape,
    const PythonPayloadView& payload) {
  std::vector<py::ssize_t> shape = leading_shape;
  if (!payload.scalar) {
    shape.push_back(payload.channels);
  }
  return shape;
}

py::object sample_trilinear_cpu_binding(
    const svo::Octree& octree,
    py::array points_array,
    py::array payload_array,
    double fill_value) {
  const PythonPointBatch points = point_batch_from_numpy(points_array);
  const PythonPayloadView payload = payload_from_numpy(payload_array);
  const std::vector<py::ssize_t> output_shape = interpolation_output_shape(points.leading_shape, payload);

  if (payload.double_precision) {
    const auto* payload_data = static_cast<const double*>(payload.array.data());
    std::vector<double> values;
    {
      py::gil_scoped_release release;
      values = svo::sample_trilinear_double(
          octree,
          points.points,
          payload_data,
          static_cast<std::size_t>(payload.rows),
          static_cast<std::size_t>(payload.channels),
          fill_value);
    }
    py::array_t<double> output(output_shape);
    std::copy(values.begin(), values.end(), static_cast<double*>(output.mutable_data()));
    return output;
  }

  const auto* payload_data = static_cast<const float*>(payload.array.data());
  std::vector<float> values;
  {
    py::gil_scoped_release release;
    values = svo::sample_trilinear_float(
        octree,
        points.points,
        payload_data,
        static_cast<std::size_t>(payload.rows),
        static_cast<std::size_t>(payload.channels),
        static_cast<float>(fill_value));
  }
  py::array_t<float> output(output_shape);
  std::copy(values.begin(), values.end(), static_cast<float*>(output.mutable_data()));
  return output;
}

py::array_t<std::int32_t> vector_to_numpy(const std::vector<std::int32_t>& values) {
  py::array_t<std::int32_t> output(values.size());
  auto view = output.mutable_unchecked<1>();
  for (py::ssize_t index = 0; index < view.shape(0); ++index) {
    view(index) = values[static_cast<std::size_t>(index)];
  }
  return output;
}

py::array_t<std::int32_t> payload_indices_to_numpy(const std::vector<std::uint32_t>& values) {
  py::array_t<std::int32_t> output(values.size());
  auto view = output.mutable_unchecked<1>();
  for (py::ssize_t index = 0; index < view.shape(0); ++index) {
    view(index) = static_cast<std::int32_t>(values[static_cast<std::size_t>(index)]);
  }
  return output;
}

py::tuple leaf_specs_to_numpy(const std::vector<svo::LeafSpec>& specs) {
  py::array_t<std::int32_t> coord_min({static_cast<py::ssize_t>(specs.size()), py::ssize_t{3}});
  py::array_t<std::int32_t> depths(specs.size());
  py::array_t<std::int32_t> payload_indices(specs.size());
  auto coord_view = coord_min.mutable_unchecked<2>();
  auto depth_view = depths.mutable_unchecked<1>();
  auto payload_view = payload_indices.mutable_unchecked<1>();
  for (py::ssize_t index = 0; index < depth_view.shape(0); ++index) {
    const svo::LeafSpec& spec = specs[static_cast<std::size_t>(index)];
    coord_view(index, 0) = spec.coord_min.x;
    coord_view(index, 1) = spec.coord_min.y;
    coord_view(index, 2) = spec.coord_min.z;
    depth_view(index) = static_cast<std::int32_t>(spec.depth);
    payload_view(index) = static_cast<std::int32_t>(spec.payload_index);
  }
  return py::make_tuple(coord_min, depths, payload_indices);
}

py::array_t<float> root_bounds_to_numpy(const svo::RootBounds& bounds) {
  py::array_t<float> output({2, 3});
  auto view = output.mutable_unchecked<2>();
  for (int axis = 0; axis < 3; ++axis) {
    view(0, axis) = bounds[0][axis];
    view(1, axis) = bounds[1][axis];
  }
  return output;
}

svo::SerializedArray serialized_array_from_numpy(const std::string& name, py::array array) {
  if (name.empty()) {
    throw py::value_error("payload array names cannot be empty");
  }
  if (!py::isinstance<py::array_t<float>>(array)) {
    throw py::type_error("serialized payload arrays must have dtype float32");
  }
  if (!is_c_contiguous(array)) {
    throw py::value_error("serialized payload arrays must be C-contiguous");
  }
  svo::SerializedArray result;
  result.name = name;
  result.dtype = svo::SerializedArrayDType::Float32;
  result.shape.reserve(static_cast<std::size_t>(array.ndim()));
  std::uint64_t element_count = 1;
  for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
    const auto dim = static_cast<std::uint64_t>(array.shape(axis));
    result.shape.push_back(dim);
    if (dim == 0) {
      element_count = 0;
    } else if (element_count != 0) {
      if (element_count > std::numeric_limits<std::uint64_t>::max() / dim) {
        throw py::value_error("serialized payload array shape overflows element count");
      }
      element_count *= dim;
    }
  }
  const std::uint64_t byte_count = element_count * static_cast<std::uint64_t>(sizeof(float));
  if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw py::value_error("serialized payload array is too large");
  }
  result.data.resize(static_cast<std::size_t>(byte_count));
  if (byte_count != 0) {
    std::memcpy(result.data.data(), array.data(), static_cast<std::size_t>(byte_count));
  }
  return result;
}

py::array serialized_array_to_numpy(const svo::SerializedArray& array) {
  if (array.dtype != svo::SerializedArrayDType::Float32) {
    throw py::value_error("unsupported serialized payload dtype");
  }
  std::vector<py::ssize_t> shape;
  shape.reserve(array.shape.size());
  for (const std::uint64_t dim : array.shape) {
    if (dim > static_cast<std::uint64_t>(std::numeric_limits<py::ssize_t>::max())) {
      throw py::value_error("serialized payload array dimension is too large for Python");
    }
    shape.push_back(static_cast<py::ssize_t>(dim));
  }
  py::array_t<float> output(shape);
  if (!array.data.empty()) {
    std::memcpy(output.mutable_data(), array.data.data(), array.data.size());
  }
  return output;
}

py::dict serialized_arrays_to_python(const std::vector<svo::SerializedArray>& arrays) {
  py::dict output;
  for (const svo::SerializedArray& array : arrays) {
    output[py::str(array.name)] = serialized_array_to_numpy(array);
  }
  return output;
}


glm::vec3 vec3_from_python(py::object value, const char* name) {
  py::array array = py::array::ensure(value);
  if (!array) {
    throw py::type_error(std::string(name) + " must be convertible to a NumPy array");
  }
  if (array.ndim() != 1 || array.shape(0) != 3) {
    throw py::value_error(std::string(name) + " must have shape (3,)");
  }

  py::array_t<float, py::array::c_style | py::array::forcecast> cast(array);
  auto view = cast.unchecked<1>();
  return {view(0), view(1), view(2)};
}

py::tuple ray_batch_to_numpy(const svo::RayBatch& rays) {
  py::array_t<float> origins({rays.height, rays.width, 3});
  py::array_t<float> directions({rays.height, rays.width, 3});
  auto origins_view = origins.mutable_unchecked<3>();
  auto directions_view = directions.mutable_unchecked<3>();

  for (int y = 0; y < rays.height; ++y) {
    for (int x = 0; x < rays.width; ++x) {
      const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(rays.width) +
          static_cast<std::size_t>(x);
      for (int axis = 0; axis < 3; ++axis) {
        origins_view(y, x, axis) = rays.origins[index][axis];
        directions_view(y, x, axis) = rays.directions[index][axis];
      }
    }
  }

  return py::make_tuple(origins, directions);
}


struct PythonRayBatch {
  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> directions;
  int width = 0;
  int height = 0;
  bool image_shaped = false;
};

void check_ray_array(const py::array& array, const char* name) {
  if (!(py::isinstance<py::array_t<float>>(array) || py::isinstance<py::array_t<double>>(array))) {
    throw py::type_error(std::string(name) + " must have dtype float32 or float64");
  }
  if (!is_c_contiguous(array)) {
    throw py::value_error(std::string(name) + " must be C-contiguous");
  }
  const bool flat = array.ndim() == 2 && array.shape(1) == 3;
  const bool image = array.ndim() == 3 && array.shape(2) == 3;
  if (!flat && !image) {
    throw py::value_error(std::string(name) + " must have shape (N, 3) or (H, W, 3)");
  }
}

std::vector<glm::vec3> ray_vectors_from_numpy(const py::array& array) {
  std::vector<glm::vec3> values;
  const py::ssize_t count = array.ndim() == 2 ? array.shape(0) : array.shape(0) * array.shape(1);
  values.reserve(static_cast<std::size_t>(count));

  if (py::isinstance<py::array_t<float>>(array)) {
    py::array_t<float, py::array::c_style> cast(array);
    if (array.ndim() == 2) {
      auto view = cast.unchecked<2>();
      for (py::ssize_t index = 0; index < view.shape(0); ++index) {
        values.emplace_back(view(index, 0), view(index, 1), view(index, 2));
      }
    } else {
      auto view = cast.unchecked<3>();
      for (py::ssize_t y = 0; y < view.shape(0); ++y) {
        for (py::ssize_t x = 0; x < view.shape(1); ++x) {
          values.emplace_back(view(y, x, 0), view(y, x, 1), view(y, x, 2));
        }
      }
    }
  } else {
    py::array_t<double, py::array::c_style> cast(array);
    if (array.ndim() == 2) {
      auto view = cast.unchecked<2>();
      for (py::ssize_t index = 0; index < view.shape(0); ++index) {
        values.emplace_back(
            static_cast<float>(view(index, 0)),
            static_cast<float>(view(index, 1)),
            static_cast<float>(view(index, 2)));
      }
    } else {
      auto view = cast.unchecked<3>();
      for (py::ssize_t y = 0; y < view.shape(0); ++y) {
        for (py::ssize_t x = 0; x < view.shape(1); ++x) {
          values.emplace_back(
              static_cast<float>(view(y, x, 0)),
              static_cast<float>(view(y, x, 1)),
              static_cast<float>(view(y, x, 2)));
        }
      }
    }
  }

  return values;
}

PythonRayBatch rays_from_numpy(py::array origins, py::array directions) {
  check_ray_array(origins, "origins");
  check_ray_array(directions, "directions");
  if (origins.ndim() != directions.ndim()) {
    throw py::value_error("origins and directions must have the same shape");
  }
  for (py::ssize_t axis = 0; axis < origins.ndim(); ++axis) {
    if (origins.shape(axis) != directions.shape(axis)) {
      throw py::value_error("origins and directions must have the same shape");
    }
  }

  PythonRayBatch rays;
  rays.image_shaped = origins.ndim() == 3;
  rays.height = rays.image_shaped ? static_cast<int>(origins.shape(0)) : 1;
  rays.width = rays.image_shaped ? static_cast<int>(origins.shape(1)) : static_cast<int>(origins.shape(0));
  rays.origins = ray_vectors_from_numpy(origins);
  rays.directions = ray_vectors_from_numpy(directions);
  return rays;
}

struct PythonRenderPayloadView {
  py::array sigma;
  py::array color;
  py::ssize_t rows = 0;
};

PythonRenderPayloadView render_payload_from_numpy(py::array sigma, py::array color) {
  if (sigma.ndim() != 1) {
    throw py::value_error("sigma must have shape (P,)");
  }
  if (color.ndim() != 2 || color.shape(1) != 3) {
    throw py::value_error("color must have shape (P, 3)");
  }
  if (!py::isinstance<py::array_t<float>>(sigma)) {
    throw py::type_error("sigma must have dtype float32");
  }
  if (!py::isinstance<py::array_t<float>>(color)) {
    throw py::type_error("color must have dtype float32");
  }
  if (!is_c_contiguous(sigma) || !is_c_contiguous(color)) {
    throw py::value_error("sigma and color must be C-contiguous");
  }
  if (sigma.shape(0) != color.shape(0)) {
    throw py::value_error("sigma and color must have the same row count");
  }

  PythonRenderPayloadView payload;
  payload.sigma = sigma;
  payload.color = color;
  payload.rows = sigma.shape(0);
  return payload;
}

svo::RenderOptions render_options_from_python(
    double near_plane,
    double far_plane,
    py::object background_color,
    double early_stop_transmittance,
    bool store_aux,
    bool enable_empty_space_skipping) {
  svo::RenderOptions options;
  options.near_plane = static_cast<float>(near_plane);
  options.far_plane = static_cast<float>(far_plane);
  options.background_color = vec3_from_python(background_color, "background_color");
  options.early_stop_transmittance = static_cast<float>(early_stop_transmittance);
  options.store_aux = store_aux;
  options.enable_empty_space_skipping = enable_empty_space_skipping;
  return options;
}

py::tuple render_batch_to_numpy(const svo::RenderBatch& results, bool image_shaped) {
  if (image_shaped) {
    py::array_t<float> rgb({results.height, results.width, 3});
    py::array_t<float> depth({results.height, results.width});
    py::array_t<float> opacity({results.height, results.width});
    auto rgb_view = rgb.mutable_unchecked<3>();
    auto depth_view = depth.mutable_unchecked<2>();
    auto opacity_view = opacity.mutable_unchecked<2>();
    for (int y = 0; y < results.height; ++y) {
      for (int x = 0; x < results.width; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(results.width) +
            static_cast<std::size_t>(x);
        for (int axis = 0; axis < 3; ++axis) {
          rgb_view(y, x, axis) = results.rgb[index][axis];
        }
        depth_view(y, x) = results.depth[index];
        opacity_view(y, x) = results.opacity[index];
      }
    }
    return py::make_tuple(rgb, depth, opacity);
  }

  const py::ssize_t count = static_cast<py::ssize_t>(results.rgb.size());
  py::array_t<float> rgb({count, static_cast<py::ssize_t>(3)});
  py::array_t<float> depth(count);
  py::array_t<float> opacity(count);
  auto rgb_view = rgb.mutable_unchecked<2>();
  auto depth_view = depth.mutable_unchecked<1>();
  auto opacity_view = opacity.mutable_unchecked<1>();
  for (py::ssize_t index = 0; index < count; ++index) {
    const std::size_t result_index = static_cast<std::size_t>(index);
    for (int axis = 0; axis < 3; ++axis) {
      rgb_view(index, axis) = results.rgb[result_index][axis];
    }
    depth_view(index) = results.depth[result_index];
    opacity_view(index) = results.opacity[result_index];
  }
  return py::make_tuple(rgb, depth, opacity);
}

py::tuple render_volume_cpu_binding(
    const svo::Octree& octree,
    py::array origins,
    py::array directions,
    py::array sigma,
    py::array color,
    double near_plane,
    double far_plane,
    py::object background_color,
    double early_stop_transmittance,
    bool store_aux,
    bool enable_empty_space_skipping) {
  if (!py::isinstance<py::array_t<float>>(origins) || !py::isinstance<py::array_t<float>>(directions)) {
    throw py::type_error("origins and directions must have dtype float32 for rendering");
  }
  const PythonRayBatch rays = rays_from_numpy(origins, directions);
  const PythonRenderPayloadView payload = render_payload_from_numpy(sigma, color);
  svo::RayBatch ray_batch;
  ray_batch.origins = rays.origins;
  ray_batch.directions = rays.directions;
  ray_batch.width = rays.width;
  ray_batch.height = rays.height;
  const svo::RenderOptions options = render_options_from_python(
      near_plane,
      far_plane,
      background_color,
      early_stop_transmittance,
      store_aux,
      enable_empty_space_skipping);
  const auto* sigma_data = static_cast<const float*>(payload.sigma.data());
  const auto* color_data = static_cast<const float*>(payload.color.data());
  svo::RenderBatch results;
  {
    py::gil_scoped_release release;
    results = svo::render_volume_cpu(
        octree,
        ray_batch,
        sigma_data,
        color_data,
        static_cast<std::size_t>(payload.rows),
        options);
  }
  return render_batch_to_numpy(results, rays.image_shaped);
}


py::tuple raycast_batch_to_numpy(const svo::RaycastBatch& results, bool image_shaped) {
  if (image_shaped) {
    py::array_t<bool> hit_mask({results.height, results.width});
    py::array_t<std::int32_t> leaf_ids({results.height, results.width});
    py::array_t<float> t({results.height, results.width});
    py::array_t<float> positions({results.height, results.width, 3});
    py::array_t<std::int32_t> depths({results.height, results.width});
    auto hit_view = hit_mask.mutable_unchecked<2>();
    auto leaf_view = leaf_ids.mutable_unchecked<2>();
    auto t_view = t.mutable_unchecked<2>();
    auto position_view = positions.mutable_unchecked<3>();
    auto depth_view = depths.mutable_unchecked<2>();

    for (int y = 0; y < results.height; ++y) {
      for (int x = 0; x < results.width; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(results.width) +
            static_cast<std::size_t>(x);
        hit_view(y, x) = results.hit_mask[index] != 0u;
        leaf_view(y, x) = results.leaf_ids[index];
        t_view(y, x) = results.t[index];
        depth_view(y, x) = results.depths[index];
        for (int axis = 0; axis < 3; ++axis) {
          position_view(y, x, axis) = results.positions[index][axis];
        }
      }
    }
    return py::make_tuple(hit_mask, leaf_ids, t, positions, depths);
  }

  const py::ssize_t count = static_cast<py::ssize_t>(results.hit_mask.size());
  py::array_t<bool> hit_mask(count);
  py::array_t<std::int32_t> leaf_ids(count);
  py::array_t<float> t(count);
  py::array_t<float> positions({count, static_cast<py::ssize_t>(3)});
  py::array_t<std::int32_t> depths(count);
  auto hit_view = hit_mask.mutable_unchecked<1>();
  auto leaf_view = leaf_ids.mutable_unchecked<1>();
  auto t_view = t.mutable_unchecked<1>();
  auto position_view = positions.mutable_unchecked<2>();
  auto depth_view = depths.mutable_unchecked<1>();

  for (py::ssize_t index = 0; index < count; ++index) {
    const std::size_t result_index = static_cast<std::size_t>(index);
    hit_view(index) = results.hit_mask[result_index] != 0u;
    leaf_view(index) = results.leaf_ids[result_index];
    t_view(index) = results.t[result_index];
    depth_view(index) = results.depths[result_index];
    for (int axis = 0; axis < 3; ++axis) {
      position_view(index, axis) = results.positions[result_index][axis];
    }
  }
  return py::make_tuple(hit_mask, leaf_ids, t, positions, depths);
}


#if SVO_ENABLE_CUDA
void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw svo::Error(std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

int current_cuda_device() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  return device;
}

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int device) : previous_device_(current_cuda_device()) {
    if (previous_device_ != device) {
      check_cuda(cudaSetDevice(device), "cudaSetDevice");
      changed_ = true;
    }
  }

  ~CudaDeviceGuard() {
    if (changed_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

 private:
  int previous_device_ = 0;
  bool changed_ = false;
};

bool object_type_is_from_torch(py::handle value) {
  const py::object type = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(Py_TYPE(value.ptr())));
  const std::string module = py::cast<std::string>(type.attr("__module__"));
  return module.rfind("torch", 0) == 0;
}

py::object import_torch() {
  try {
    return py::module_::import("torch");
  } catch (const py::error_already_set& error) {
    if (error.matches(PyExc_ModuleNotFoundError)) {
      throw py::type_error("Torch tensor input requires torch to be installed");
    }
    throw;
  }
}

bool py_equal(py::handle lhs, py::handle rhs) {
  const int result = PyObject_RichCompareBool(lhs.ptr(), rhs.ptr(), Py_EQ);
  if (result < 0) {
    throw py::error_already_set();
  }
  return result == 1;
}

class CudaCoarseOccupancyOwner {
 public:
  CudaCoarseOccupancyOwner(const svo::Octree& host_octree, int resolution, int device_index)
      : device_index_(device_index) {
    CudaDeviceGuard guard(device_index_);
    host_grid_ = svo::CoarseOccupancyGrid::from_octree(host_octree, resolution);
    device_grid_.upload(host_grid_);
    view_ = device_grid_.view();
  }

  ~CudaCoarseOccupancyOwner() {
    release();
  }

  CudaCoarseOccupancyOwner(const CudaCoarseOccupancyOwner&) = delete;
  CudaCoarseOccupancyOwner& operator=(const CudaCoarseOccupancyOwner&) = delete;
  CudaCoarseOccupancyOwner(CudaCoarseOccupancyOwner&&) noexcept = default;
  CudaCoarseOccupancyOwner& operator=(CudaCoarseOccupancyOwner&&) noexcept = default;

  void release() noexcept {
    device_grid_ = svo::DeviceCoarseOccupancyGrid{};
    view_ = {};
  }

  const svo::CoarseOccupancyDeviceView& view() const noexcept { return view_; }
  int resolution() const noexcept { return host_grid_.resolution(); }
  std::size_t size_bytes() const noexcept { return host_grid_.size_bytes(); }
  int device_index() const noexcept { return device_index_; }

 private:
  int device_index_ = 0;
  svo::CoarseOccupancyGrid host_grid_;
  svo::DeviceCoarseOccupancyGrid device_grid_;
  svo::CoarseOccupancyDeviceView view_;
};

bool is_torch_tensor(py::handle torch, py::handle value) {
  return py::cast<bool>(torch.attr("is_tensor")(value));
}

int torch_tensor_device_index(py::handle torch, py::handle tensor) {
  if (!py::cast<bool>(tensor.attr("is_cuda"))) {
    throw py::type_error("Torch tensor inputs to CudaOctree must be CUDA tensors");
  }

  py::object device = tensor.attr("device");
  py::object index = device.attr("index");
  if (index.is_none()) {
    return py::cast<int>(torch.attr("cuda").attr("current_device")());
  }
  return py::cast<int>(index);
}

void check_torch_float32_tensor(
    py::handle torch,
    py::handle tensor,
    const char* name,
    int expected_device_index) {
  if (!is_torch_tensor(torch, tensor)) {
    throw py::type_error(std::string(name) + " must be a Torch tensor");
  }
  if (!py::cast<bool>(tensor.attr("is_cuda"))) {
    throw py::type_error(std::string(name) + " must be a CUDA Torch tensor");
  }
  if (!py_equal(tensor.attr("dtype"), torch.attr("float32"))) {
    throw py::type_error(std::string(name) + " must have dtype torch.float32");
  }
  if (!py::cast<bool>(tensor.attr("is_contiguous")())) {
    throw py::value_error(std::string(name) + " must be contiguous");
  }

  const int tensor_device_index = torch_tensor_device_index(torch, tensor);
  if (tensor_device_index != expected_device_index) {
    std::ostringstream message;
    message << name << " is on cuda:" << tensor_device_index
            << " but this CudaOctree owns data on cuda:" << expected_device_index;
    throw py::value_error(message.str());
  }
}

std::vector<py::ssize_t> torch_shape(py::handle tensor) {
  std::vector<py::ssize_t> shape;
  py::tuple tuple = py::cast<py::tuple>(tensor.attr("shape"));
  shape.reserve(static_cast<std::size_t>(tuple.size()));
  for (py::handle value : tuple) {
    shape.push_back(py::cast<py::ssize_t>(value));
  }
  return shape;
}

std::size_t checked_count_from_shape(py::ssize_t leading, const char* name) {
  if (leading < 0) {
    throw py::value_error(std::string(name) + " shape contains a negative dimension");
  }
  return static_cast<std::size_t>(leading);
}

std::size_t checked_flat_ray_count(const std::vector<py::ssize_t>& shape, const char* name) {
  if (shape.size() == 2) {
    return checked_count_from_shape(shape[0], name);
  }
  const std::size_t height = checked_count_from_shape(shape[0], name);
  const std::size_t width = checked_count_from_shape(shape[1], name);
  if (height != 0 && width > std::numeric_limits<std::size_t>::max() / height) {
    throw py::value_error(std::string(name) + " shape is too large");
  }
  return height * width;
}

std::uintptr_t torch_data_ptr(py::handle tensor) {
  return py::cast<std::uintptr_t>(tensor.attr("data_ptr")());
}

svo::CudaStreamHandle torch_current_stream(py::handle torch, py::handle tensor) {
  py::object stream = torch.attr("cuda").attr("current_stream")(tensor.attr("device"));
  const std::uintptr_t stream_pointer = py::cast<std::uintptr_t>(stream.attr("cuda_stream"));
  return reinterpret_cast<svo::CudaStreamHandle>(stream_pointer);
}

py::object torch_empty_like_device(
    py::handle torch,
    const py::tuple& shape,
    py::object dtype,
    py::object device) {
  return torch.attr("empty")(shape, "dtype"_a = dtype, "device"_a = device);
}

py::tuple tuple_from_shape(const std::vector<py::ssize_t>& shape) {
  py::tuple tuple(shape.size());
  for (std::size_t index = 0; index < shape.size(); ++index) {
    tuple[index] = py::int_(shape[index]);
  }
  return tuple;
}

std::size_t checked_torch_point_count(const std::vector<py::ssize_t>& shape, const char* name) {
  if (shape.size() < 2 || shape.back() != 3) {
    throw py::value_error(std::string(name) + " must have shape (..., 3)");
  }
  std::size_t count = 1;
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
    if (shape[axis] < 0) {
      throw py::value_error(std::string(name) + " shape contains a negative dimension");
    }
    const std::size_t dim = static_cast<std::size_t>(shape[axis]);
    if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
      throw py::value_error(std::string(name) + " shape is too large");
    }
    count *= dim;
  }
  return count;
}

std::vector<py::ssize_t> torch_point_leading_shape(const std::vector<py::ssize_t>& shape) {
  return std::vector<py::ssize_t>(shape.begin(), shape.end() - 1);
}

bool torch_dtype_is_float(py::handle torch, py::handle dtype) {
  return py_equal(dtype, torch.attr("float32")) || py_equal(dtype, torch.attr("float64"));
}

void check_torch_floating_tensor(
    py::handle torch,
    py::handle tensor,
    const char* name,
    int expected_device_index) {
  if (!is_torch_tensor(torch, tensor)) {
    throw py::type_error(std::string(name) + " must be a Torch tensor");
  }
  if (!py::cast<bool>(tensor.attr("is_cuda"))) {
    throw py::type_error(std::string(name) + " must be a CUDA Torch tensor");
  }
  if (!torch_dtype_is_float(torch, tensor.attr("dtype"))) {
    throw py::type_error(std::string(name) + " must have dtype torch.float32 or torch.float64");
  }
  if (!py::cast<bool>(tensor.attr("is_contiguous")())) {
    throw py::value_error(std::string(name) + " must be contiguous");
  }

  const int tensor_device_index = torch_tensor_device_index(torch, tensor);
  if (tensor_device_index != expected_device_index) {
    std::ostringstream message;
    message << name << " is on cuda:" << tensor_device_index
            << " but this CudaOctree owns data on cuda:" << expected_device_index;
    throw py::value_error(message.str());
  }
}

void check_torch_same_dtype(py::handle lhs, py::handle rhs, const char* name) {
  if (!py_equal(lhs.attr("dtype"), rhs.attr("dtype"))) {
    throw py::type_error(std::string(name) + " must have the same dtype as payload");
  }
}

struct TorchPayloadShape {
  py::ssize_t rows = 0;
  py::ssize_t channels = 1;
  bool scalar = true;
  bool double_precision = false;
};

TorchPayloadShape torch_payload_shape(py::handle torch, py::handle payload) {
  const std::vector<py::ssize_t> shape = torch_shape(payload);
  if (shape.size() != 1 && shape.size() != 2) {
    throw py::value_error("payload must have shape (P,) or (P, C)");
  }
  TorchPayloadShape payload_shape;
  payload_shape.rows = shape[0];
  payload_shape.scalar = shape.size() == 1;
  payload_shape.channels = payload_shape.scalar ? 1 : shape[1];
  payload_shape.double_precision = py_equal(payload.attr("dtype"), torch.attr("float64"));
  if (payload_shape.rows < 0 || payload_shape.channels <= 0) {
    throw py::value_error("payload must have shape (P,) or (P, C) with C > 0");
  }
  return payload_shape;
}

std::vector<py::ssize_t> torch_interpolation_output_shape(
    const std::vector<py::ssize_t>& point_shape,
    const TorchPayloadShape& payload_shape) {
  std::vector<py::ssize_t> output_shape = torch_point_leading_shape(point_shape);
  if (!payload_shape.scalar) {
    output_shape.push_back(payload_shape.channels);
  }
  return output_shape;
}

void check_torch_shape_equals(
    py::handle tensor,
    const std::vector<py::ssize_t>& expected_shape,
    const char* name) {
  const std::vector<py::ssize_t> actual_shape = torch_shape(tensor);
  if (actual_shape != expected_shape) {
    throw py::value_error(std::string(name) + " has an unexpected shape");
  }
}

class CudaOctreeOwner {
 public:
  explicit CudaOctreeOwner(const svo::Octree& host_octree)
      : host_octree_(host_octree),
        device_index_(current_cuda_device()),
        device_nodes_(svo::DeviceBuffer<svo::NodeDescriptor>::from_host(host_octree.nodes(), svo::Device::CUDA)),
        device_wide_nodes_(svo::DeviceBuffer<svo::WideNodeDescriptor>::from_host(host_octree.wide_nodes(), svo::Device::CUDA)),
        device_leaf_payload_indices_(svo::DeviceBuffer<std::uint32_t>::from_host(
            host_octree.leaf_payload_indices(),
            svo::Device::CUDA)) {}

  ~CudaOctreeOwner() {
    release();
  }

  CudaOctreeOwner(const CudaOctreeOwner&) = delete;
  CudaOctreeOwner& operator=(const CudaOctreeOwner&) = delete;
  CudaOctreeOwner(CudaOctreeOwner&&) noexcept = default;
  CudaOctreeOwner& operator=(CudaOctreeOwner&&) noexcept = default;

  void release() noexcept {
    try {
      CudaDeviceGuard guard(device_index_);
      device_nodes_.release();
      device_wide_nodes_.release();
      device_leaf_payload_indices_.release();
    } catch (...) {
      device_nodes_.release();
      device_wide_nodes_.release();
      device_leaf_payload_indices_.release();
    }
  }

  const svo::Octree& host_octree() const noexcept { return host_octree_; }
  int max_depth() const noexcept { return host_octree_.max_depth(); }
  std::int64_t num_nodes() const noexcept { return host_octree_.num_nodes(); }
  std::int64_t num_leaves() const noexcept { return host_octree_.num_leaves(); }
  int device_index() const noexcept { return device_index_; }
  const svo::RootBounds& root_bounds() const noexcept { return host_octree_.root_bounds(); }

  CudaCoarseOccupancyOwner coarse_occupancy(int resolution) const {
    return CudaCoarseOccupancyOwner(host_octree_, resolution, device_index_);
  }

  py::object query(py::object points, bool return_payload_indices) const {
    if (object_type_is_from_torch(points)) {
      return query_torch(points, return_payload_indices);
    }

    py::array points_array = py::array::ensure(points);
    if (!points_array) {
      throw py::type_error("points must be a NumPy array or CUDA Torch tensor");
    }
    return query_numpy(points_array, return_payload_indices);
  }

  py::array_t<std::int32_t> query_numpy(py::array points_array, bool return_payload_indices) const {
    const std::vector<glm::vec3> points = points_from_numpy(points_array);
    CudaDeviceGuard guard(device_index_);
    svo::DeviceBuffer<glm::vec3> device_points =
        svo::DeviceBuffer<glm::vec3>::from_host(points, svo::Device::CUDA);
    svo::DeviceBuffer<std::int32_t> device_results(points.size(), svo::Device::CUDA);

    svo::QueryOptions options;
    options.return_payload_indices = return_payload_indices;

    {
      py::gil_scoped_release release;
      if (host_octree_.branching() == svo::BranchingMode::Wide4) {
        svo::query_points_wide_cuda(
            device_wide_nodes_.data(),
            device_wide_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            device_points.data(),
            device_results.data(),
            device_results.size(),
            options);
      } else {
        svo::query_points_cuda(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            device_points.data(),
            device_results.data(),
            device_results.size(),
            options);
      }
    }

    return vector_to_numpy(device_results.to_host());
  }

  py::object query_torch(py::object points, bool return_payload_indices) const {
    py::object torch = import_torch();
    check_torch_float32_tensor(torch, points, "points", device_index_);

    const std::vector<py::ssize_t> shape = torch_shape(points);
    if (shape.size() != 2 || shape[1] != 3) {
      throw py::value_error("points must have shape (N, 3)");
    }

    const std::size_t count = checked_count_from_shape(shape[0], "points");
    py::object results = torch_empty_like_device(
        torch,
        py::make_tuple(shape[0]),
        torch.attr("int32"),
        points.attr("device"));
    if (count == 0) {
      return results;
    }

    svo::QueryOptions options;
    options.return_payload_indices = return_payload_indices;
    svo::CudaStreamHandle stream = torch_current_stream(torch, points);
    const auto* point_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(points));
    auto* result_data = reinterpret_cast<std::int32_t*>(torch_data_ptr(results));
    CudaDeviceGuard guard(device_index_);
    {
      py::gil_scoped_release release;
      if (host_octree_.branching() == svo::BranchingMode::Wide4) {
        svo::query_points_wide_cuda(
            device_wide_nodes_.data(),
            device_wide_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            point_data,
            result_data,
            count,
            options,
            stream);
      } else {
        svo::query_points_cuda(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            point_data,
            result_data,
            count,
            options,
            stream);
      }
    }

    return results;
  }

  py::object raycast(py::object origins, py::object directions, bool return_payload_indices) const {
    if (object_type_is_from_torch(origins) || object_type_is_from_torch(directions)) {
      return raycast_torch(origins, directions, return_payload_indices);
    }

    py::array origins_array = py::array::ensure(origins);
    py::array directions_array = py::array::ensure(directions);
    if (!origins_array || !directions_array) {
      throw py::type_error("origins and directions must be NumPy arrays or CUDA Torch tensors");
    }
    return raycast_numpy(origins_array, directions_array, return_payload_indices);
  }

  py::tuple raycast_numpy(py::array origins_array, py::array directions_array, bool return_payload_indices) const {
    const PythonRayBatch rays = rays_from_numpy(origins_array, directions_array);
    CudaDeviceGuard guard(device_index_);
    svo::DeviceBuffer<glm::vec3> device_origins =
        svo::DeviceBuffer<glm::vec3>::from_host(rays.origins, svo::Device::CUDA);
    svo::DeviceBuffer<glm::vec3> device_directions =
        svo::DeviceBuffer<glm::vec3>::from_host(rays.directions, svo::Device::CUDA);
    svo::DeviceBuffer<std::uint8_t> device_hit_mask(rays.origins.size(), svo::Device::CUDA);
    svo::DeviceBuffer<std::int32_t> device_leaf_ids(rays.origins.size(), svo::Device::CUDA);
    svo::DeviceBuffer<float> device_t(rays.origins.size(), svo::Device::CUDA);
    svo::DeviceBuffer<glm::vec3> device_positions(rays.origins.size(), svo::Device::CUDA);
    svo::DeviceBuffer<std::int32_t> device_depths(rays.origins.size(), svo::Device::CUDA);

    svo::RaycastOptions options;
    options.return_payload_indices = return_payload_indices;

    {
      py::gil_scoped_release release;
      if (host_octree_.branching() == svo::BranchingMode::Wide4) {
        svo::raycast_wide_cuda(
            device_wide_nodes_.data(),
            device_wide_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            device_origins.data(),
            device_directions.data(),
            device_hit_mask.data(),
            device_leaf_ids.data(),
            device_t.data(),
            device_positions.data(),
            device_depths.data(),
            rays.origins.size(),
            options);
      } else {
        svo::raycast_cuda(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            device_origins.data(),
            device_directions.data(),
            device_hit_mask.data(),
            device_leaf_ids.data(),
            device_t.data(),
            device_positions.data(),
            device_depths.data(),
            rays.origins.size(),
            options);
      }
    }

    svo::RaycastBatch results;
    results.width = rays.width;
    results.height = rays.height;
    results.hit_mask = device_hit_mask.to_host();
    results.leaf_ids = device_leaf_ids.to_host();
    results.t = device_t.to_host();
    results.positions = device_positions.to_host();
    results.depths = device_depths.to_host();
    return raycast_batch_to_numpy(results, rays.image_shaped);
  }

  py::object raycast_torch(py::object origins, py::object directions, bool return_payload_indices) const {
    py::object torch = import_torch();
    check_torch_float32_tensor(torch, origins, "origins", device_index_);
    check_torch_float32_tensor(torch, directions, "directions", device_index_);

    const std::vector<py::ssize_t> shape = torch_shape(origins);
    const std::vector<py::ssize_t> direction_shape = torch_shape(directions);
    if (shape != direction_shape) {
      throw py::value_error("origins and directions must have the same shape");
    }
    const bool flat = shape.size() == 2 && shape[1] == 3;
    const bool image = shape.size() == 3 && shape[2] == 3;
    if (!flat && !image) {
      throw py::value_error("origins and directions must have shape (N, 3) or (H, W, 3)");
    }

    const std::size_t count = checked_flat_ray_count(shape, "origins");
    py::tuple output_shape;
    if (flat) {
      output_shape = py::make_tuple(shape[0]);
    } else {
      output_shape = py::make_tuple(shape[0], shape[1]);
    }
    py::object device = origins.attr("device");
    py::object hit_mask = torch_empty_like_device(torch, output_shape, torch.attr("bool"), device);
    py::object leaf_ids = torch_empty_like_device(torch, output_shape, torch.attr("int32"), device);
    py::object t = torch_empty_like_device(torch, output_shape, torch.attr("float32"), device);
    py::object positions = torch_empty_like_device(torch, tuple_from_shape(shape), torch.attr("float32"), device);
    py::object depths = torch_empty_like_device(torch, output_shape, torch.attr("int32"), device);
    if (count == 0) {
      return py::make_tuple(hit_mask, leaf_ids, t, positions, depths);
    }

    svo::RaycastOptions options;
    options.return_payload_indices = return_payload_indices;
    svo::CudaStreamHandle stream = torch_current_stream(torch, origins);
    const auto* origin_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(origins));
    const auto* direction_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(directions));
    auto* hit_mask_data = reinterpret_cast<std::uint8_t*>(torch_data_ptr(hit_mask));
    auto* leaf_id_data = reinterpret_cast<std::int32_t*>(torch_data_ptr(leaf_ids));
    auto* t_data = reinterpret_cast<float*>(torch_data_ptr(t));
    auto* position_data = reinterpret_cast<glm::vec3*>(torch_data_ptr(positions));
    auto* depth_data = reinterpret_cast<std::int32_t*>(torch_data_ptr(depths));
    CudaDeviceGuard guard(device_index_);
    {
      py::gil_scoped_release release;
      if (host_octree_.branching() == svo::BranchingMode::Wide4) {
        svo::raycast_wide_cuda(
            device_wide_nodes_.data(),
            device_wide_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            origin_data,
            direction_data,
            hit_mask_data,
            leaf_id_data,
            t_data,
            position_data,
            depth_data,
            count,
            options,
            stream);
      } else {
        svo::raycast_cuda(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            origin_data,
            direction_data,
            hit_mask_data,
            leaf_id_data,
            t_data,
            position_data,
            depth_data,
            count,
            options,
            stream);
      }
    }

    return py::make_tuple(hit_mask, leaf_ids, t, positions, depths);
  }


  py::object sample_trilinear_torch(py::object points, py::object payload, double fill_value) const {
    if (host_octree_.branching() == svo::BranchingMode::Wide4) {
      throw py::type_error("trilinear interpolation does not support wide4 trees yet");
    }
    py::object torch = import_torch();
    check_torch_float32_tensor(torch, points, "points", device_index_);
    check_torch_floating_tensor(torch, payload, "payload", device_index_);

    const std::vector<py::ssize_t> point_shape = torch_shape(points);
    const std::size_t count = checked_torch_point_count(point_shape, "points");
    const TorchPayloadShape payload_shape = torch_payload_shape(torch, payload);
    validate_payload_rows(static_cast<std::size_t>(payload_shape.rows));

    const std::vector<py::ssize_t> output_shape = torch_interpolation_output_shape(point_shape, payload_shape);
    py::object output = torch_empty_like_device(
        torch,
        tuple_from_shape(output_shape),
        payload.attr("dtype"),
        points.attr("device"));
    if (count == 0) {
      return output;
    }

    svo::CudaStreamHandle stream = torch_current_stream(torch, points);
    const auto* point_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(points));
    CudaDeviceGuard guard(device_index_);
    if (payload_shape.double_precision) {
      const auto* payload_data = reinterpret_cast<const double*>(torch_data_ptr(payload));
      auto* output_data = reinterpret_cast<double*>(torch_data_ptr(output));
      {
        py::gil_scoped_release release;
        svo::sample_trilinear_cuda_double(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            point_data,
            payload_data,
            output_data,
            count,
            static_cast<std::size_t>(payload_shape.rows),
            static_cast<std::size_t>(payload_shape.channels),
            fill_value,
            stream);
      }
    } else {
      const auto* payload_data = reinterpret_cast<const float*>(torch_data_ptr(payload));
      auto* output_data = reinterpret_cast<float*>(torch_data_ptr(output));
      {
        py::gil_scoped_release release;
        svo::sample_trilinear_cuda_float(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            point_data,
            payload_data,
            output_data,
            count,
            static_cast<std::size_t>(payload_shape.rows),
            static_cast<std::size_t>(payload_shape.channels),
            static_cast<float>(fill_value),
            stream);
      }
    }

    return output;
  }

  py::object sample_trilinear_backward_torch(
      py::object points,
      py::object payload,
      py::object grad_outputs,
      double fill_value) const {
    if (host_octree_.branching() == svo::BranchingMode::Wide4) {
      throw py::type_error("trilinear interpolation does not support wide4 trees yet");
    }
    py::object torch = import_torch();
    check_torch_float32_tensor(torch, points, "points", device_index_);
    check_torch_floating_tensor(torch, payload, "payload", device_index_);
    check_torch_floating_tensor(torch, grad_outputs, "grad_outputs", device_index_);
    check_torch_same_dtype(grad_outputs, payload, "grad_outputs");

    const std::vector<py::ssize_t> point_shape = torch_shape(points);
    const std::size_t count = checked_torch_point_count(point_shape, "points");
    const TorchPayloadShape payload_shape = torch_payload_shape(torch, payload);
    validate_payload_rows(static_cast<std::size_t>(payload_shape.rows));
    check_torch_shape_equals(
        grad_outputs,
        torch_interpolation_output_shape(point_shape, payload_shape),
        "grad_outputs");

    py::object grad_payload = torch.attr("zeros_like")(payload);
    if (count == 0) {
      return grad_payload;
    }

    svo::CudaStreamHandle stream = torch_current_stream(torch, points);
    const auto* point_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(points));
    CudaDeviceGuard guard(device_index_);
    if (payload_shape.double_precision) {
      const auto* grad_output_data = reinterpret_cast<const double*>(torch_data_ptr(grad_outputs));
      auto* grad_payload_data = reinterpret_cast<double*>(torch_data_ptr(grad_payload));
      {
        py::gil_scoped_release release;
        svo::sample_trilinear_backward_cuda_double(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            point_data,
            grad_output_data,
            grad_payload_data,
            count,
            static_cast<std::size_t>(payload_shape.rows),
            static_cast<std::size_t>(payload_shape.channels),
            fill_value,
            stream);
      }
    } else {
      const auto* grad_output_data = reinterpret_cast<const float*>(torch_data_ptr(grad_outputs));
      auto* grad_payload_data = reinterpret_cast<float*>(torch_data_ptr(grad_payload));
      {
        py::gil_scoped_release release;
        svo::sample_trilinear_backward_cuda_float(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            point_data,
            grad_output_data,
            grad_payload_data,
            count,
            static_cast<std::size_t>(payload_shape.rows),
            static_cast<std::size_t>(payload_shape.channels),
            static_cast<float>(fill_value),
            stream);
      }
    }

    return grad_payload;
  }

  py::object render_volume_backward_torch(
      py::object origins,
      py::object directions,
      py::object sigma,
      py::object color,
      py::object grad_rgb,
      py::object grad_opacity,
      double near_plane,
      double far_plane,
      py::object background_color,
      double early_stop_transmittance,
      bool store_aux,
      bool enable_empty_space_skipping) const {
    py::object torch = import_torch();
    check_torch_float32_tensor(torch, origins, "origins", device_index_);
    check_torch_float32_tensor(torch, directions, "directions", device_index_);
    check_torch_float32_tensor(torch, sigma, "sigma", device_index_);
    check_torch_float32_tensor(torch, color, "color", device_index_);
    check_torch_float32_tensor(torch, grad_rgb, "grad_rgb", device_index_);
    check_torch_float32_tensor(torch, grad_opacity, "grad_opacity", device_index_);

    const std::vector<py::ssize_t> ray_shape = torch_shape(origins);
    const std::vector<py::ssize_t> direction_shape = torch_shape(directions);
    if (ray_shape != direction_shape) {
      throw py::value_error("origins and directions must have the same shape");
    }
    const std::size_t count = checked_torch_point_count(ray_shape, "origins");

    const std::vector<py::ssize_t> sigma_shape = torch_shape(sigma);
    if (sigma_shape.size() != 1) {
      throw py::value_error("sigma must have shape (P,)");
    }
    const std::vector<py::ssize_t> color_shape = torch_shape(color);
    if (color_shape.size() != 2 || color_shape[1] != 3) {
      throw py::value_error("color must have shape (P, 3)");
    }
    if (sigma_shape[0] != color_shape[0]) {
      throw py::value_error("sigma and color must have the same row count");
    }
    if (sigma_shape[0] < 0) {
      throw py::value_error("sigma shape contains a negative dimension");
    }
    const std::size_t payload_rows = static_cast<std::size_t>(sigma_shape[0]);
    validate_payload_rows(payload_rows);

    std::vector<py::ssize_t> opacity_shape = torch_point_leading_shape(ray_shape);
    std::vector<py::ssize_t> rgb_shape = opacity_shape;
    rgb_shape.push_back(3);
    check_torch_shape_equals(grad_rgb, rgb_shape, "grad_rgb");
    check_torch_shape_equals(grad_opacity, opacity_shape, "grad_opacity");

    py::object grad_sigma = torch.attr("zeros_like")(sigma);
    py::object grad_color = torch.attr("zeros_like")(color);
    if (count == 0) {
      return py::make_tuple(grad_sigma, grad_color);
    }

    svo::RenderOptions options = render_options_from_python(
        near_plane,
        far_plane,
        background_color,
        early_stop_transmittance,
        store_aux,
        enable_empty_space_skipping);
    svo::CudaStreamHandle stream = torch_current_stream(torch, origins);
    const auto* origin_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(origins));
    const auto* direction_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(directions));
    const auto* sigma_data = reinterpret_cast<const float*>(torch_data_ptr(sigma));
    const auto* color_data = reinterpret_cast<const float*>(torch_data_ptr(color));
    const auto* grad_rgb_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(grad_rgb));
    const auto* grad_opacity_data = reinterpret_cast<const float*>(torch_data_ptr(grad_opacity));
    auto* grad_sigma_data = reinterpret_cast<float*>(torch_data_ptr(grad_sigma));
    auto* grad_color_data = reinterpret_cast<float*>(torch_data_ptr(grad_color));
    CudaDeviceGuard guard(device_index_);
    {
      py::gil_scoped_release release;
      if (host_octree_.branching() == svo::BranchingMode::Wide4) {
        svo::render_volume_backward_wide_cuda(
            device_wide_nodes_.data(),
            device_wide_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            origin_data,
            direction_data,
            sigma_data,
            color_data,
            grad_rgb_data,
            grad_opacity_data,
            grad_sigma_data,
            grad_color_data,
            count,
            payload_rows,
            options,
            stream);
      } else {
        svo::render_volume_backward_cuda(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            origin_data,
            direction_data,
            sigma_data,
            color_data,
            grad_rgb_data,
            grad_opacity_data,
            grad_sigma_data,
            grad_color_data,
            count,
            payload_rows,
            options,
            stream);
      }
    }

    return py::make_tuple(grad_sigma, grad_color);
  }


  py::object render_volume_torch(
      py::object origins,
      py::object directions,
      py::object sigma,
      py::object color,
      double near_plane,
      double far_plane,
      py::object background_color,
      double early_stop_transmittance,
      bool store_aux,
      bool enable_empty_space_skipping,
      py::object coarse_occupancy_obj = py::none()) const {
    py::object torch = import_torch();
    check_torch_float32_tensor(torch, origins, "origins", device_index_);
    check_torch_float32_tensor(torch, directions, "directions", device_index_);
    check_torch_float32_tensor(torch, sigma, "sigma", device_index_);
    check_torch_float32_tensor(torch, color, "color", device_index_);

    const std::vector<py::ssize_t> ray_shape = torch_shape(origins);
    const std::vector<py::ssize_t> direction_shape = torch_shape(directions);
    if (ray_shape != direction_shape) {
      throw py::value_error("origins and directions must have the same shape");
    }
    const std::size_t count = checked_torch_point_count(ray_shape, "origins");

    const std::vector<py::ssize_t> sigma_shape = torch_shape(sigma);
    if (sigma_shape.size() != 1) {
      throw py::value_error("sigma must have shape (P,)");
    }
    const std::vector<py::ssize_t> color_shape = torch_shape(color);
    if (color_shape.size() != 2 || color_shape[1] != 3) {
      throw py::value_error("color must have shape (P, 3)");
    }
    if (sigma_shape[0] != color_shape[0]) {
      throw py::value_error("sigma and color must have the same row count");
    }
    if (sigma_shape[0] < 0) {
      throw py::value_error("sigma shape contains a negative dimension");
    }
    const std::size_t payload_rows = static_cast<std::size_t>(sigma_shape[0]);
    validate_payload_rows(payload_rows);

    py::object device = origins.attr("device");
    const std::vector<py::ssize_t> depth_shape = torch_point_leading_shape(ray_shape);
    std::vector<py::ssize_t> rgb_shape = depth_shape;
    rgb_shape.push_back(3);
    py::object rgb = torch_empty_like_device(torch, tuple_from_shape(rgb_shape), torch.attr("float32"), device);
    py::object depth = torch_empty_like_device(torch, tuple_from_shape(depth_shape), torch.attr("float32"), device);
    py::object opacity = torch_empty_like_device(torch, tuple_from_shape(depth_shape), torch.attr("float32"), device);
    if (count == 0) {
      return py::make_tuple(rgb, depth, opacity);
    }

    svo::RenderOptions options = render_options_from_python(
        near_plane,
        far_plane,
        background_color,
        early_stop_transmittance,
        store_aux,
        enable_empty_space_skipping);
    svo::CudaStreamHandle stream = torch_current_stream(torch, origins);
    const auto* origin_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(origins));
    const auto* direction_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(directions));
    const auto* sigma_data = reinterpret_cast<const float*>(torch_data_ptr(sigma));
    const auto* color_data = reinterpret_cast<const float*>(torch_data_ptr(color));
    const CudaCoarseOccupancyOwner* coarse_occupancy = nullptr;
    if (!coarse_occupancy_obj.is_none()) {
      coarse_occupancy = py::cast<const CudaCoarseOccupancyOwner*>(coarse_occupancy_obj);
      if (coarse_occupancy == nullptr) {
        throw py::type_error("coarse_occupancy must be created by CudaOctree._coarse_occupancy()");
      }
      if (coarse_occupancy->device_index() != device_index_) {
        throw py::value_error("coarse_occupancy belongs to a different CUDA device");
      }
      options.coarse_occupancy = &coarse_occupancy->view();
    }
    auto* rgb_data = reinterpret_cast<glm::vec3*>(torch_data_ptr(rgb));
    auto* depth_data = reinterpret_cast<float*>(torch_data_ptr(depth));
    auto* opacity_data = reinterpret_cast<float*>(torch_data_ptr(opacity));
    CudaDeviceGuard guard(device_index_);
    {
      py::gil_scoped_release release;
      if (host_octree_.branching() == svo::BranchingMode::Wide4) {
        svo::render_volume_wide_cuda(
            device_wide_nodes_.data(),
            device_wide_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            origin_data,
            direction_data,
            sigma_data,
            color_data,
            rgb_data,
            depth_data,
            opacity_data,
            count,
            payload_rows,
            options,
            stream);
      } else {
        svo::render_volume_cuda(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            origin_data,
            direction_data,
            sigma_data,
            color_data,
            rgb_data,
            depth_data,
            opacity_data,
            count,
            payload_rows,
            options,
            stream);
      }
    }

    return py::make_tuple(rgb, depth, opacity);
  }

  py::object render_volume_intervals_torch(
      py::object origins,
      py::object directions,
      py::object sigma,
      py::object color,
      double near_plane,
      double far_plane,
      py::object background_color,
      double early_stop_transmittance,
      bool store_aux,
      bool enable_empty_space_skipping) const {
    py::object torch = import_torch();
    check_torch_float32_tensor(torch, origins, "origins", device_index_);
    check_torch_float32_tensor(torch, directions, "directions", device_index_);
    check_torch_float32_tensor(torch, sigma, "sigma", device_index_);
    check_torch_float32_tensor(torch, color, "color", device_index_);

    const std::vector<py::ssize_t> ray_shape = torch_shape(origins);
    const std::vector<py::ssize_t> direction_shape = torch_shape(directions);
    if (ray_shape != direction_shape) {
      throw py::value_error("origins and directions must have the same shape");
    }
    const std::size_t count = checked_torch_point_count(ray_shape, "origins");

    const std::vector<py::ssize_t> sigma_shape = torch_shape(sigma);
    if (sigma_shape.size() != 1) {
      throw py::value_error("sigma must have shape (P,)");
    }
    const std::vector<py::ssize_t> color_shape = torch_shape(color);
    if (color_shape.size() != 2 || color_shape[1] != 3) {
      throw py::value_error("color must have shape (P, 3)");
    }
    if (sigma_shape[0] != color_shape[0]) {
      throw py::value_error("sigma and color must have the same row count");
    }
    if (sigma_shape[0] < 0) {
      throw py::value_error("sigma shape contains a negative dimension");
    }
    const std::size_t payload_rows = static_cast<std::size_t>(sigma_shape[0]);
    validate_payload_rows(payload_rows);

    py::object device = origins.attr("device");
    const std::vector<py::ssize_t> depth_shape = torch_point_leading_shape(ray_shape);
    std::vector<py::ssize_t> rgb_shape = depth_shape;
    rgb_shape.push_back(3);
    py::object rgb = torch_empty_like_device(torch, tuple_from_shape(rgb_shape), torch.attr("float32"), device);
    py::object depth = torch_empty_like_device(torch, tuple_from_shape(depth_shape), torch.attr("float32"), device);
    py::object opacity = torch_empty_like_device(torch, tuple_from_shape(depth_shape), torch.attr("float32"), device);
    auto intervals = std::make_shared<svo::RenderIntervalBuffer>();
    if (count == 0) {
      return py::make_tuple(rgb, depth, opacity, intervals);
    }

    const svo::RenderOptions options = render_options_from_python(
        near_plane,
        far_plane,
        background_color,
        early_stop_transmittance,
        store_aux,
        enable_empty_space_skipping);
    svo::CudaStreamHandle stream = torch_current_stream(torch, origins);
    const auto* origin_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(origins));
    const auto* direction_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(directions));
    const auto* sigma_data = reinterpret_cast<const float*>(torch_data_ptr(sigma));
    const auto* color_data = reinterpret_cast<const float*>(torch_data_ptr(color));
    auto* rgb_data = reinterpret_cast<glm::vec3*>(torch_data_ptr(rgb));
    auto* depth_data = reinterpret_cast<float*>(torch_data_ptr(depth));
    auto* opacity_data = reinterpret_cast<float*>(torch_data_ptr(opacity));
    CudaDeviceGuard guard(device_index_);
    {
      py::gil_scoped_release release;
      if (host_octree_.branching() == svo::BranchingMode::Wide4) {
        svo::build_render_intervals_wide_cuda(
            device_wide_nodes_.data(),
            device_wide_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            origin_data,
            direction_data,
            count,
            options,
            *intervals,
            stream);
      } else {
        svo::build_render_intervals_cuda(
            device_nodes_.data(),
            device_nodes_.size(),
            device_leaf_payload_indices_.data(),
            device_leaf_payload_indices_.size(),
            host_octree_.max_depth(),
            host_octree_.root_bounds(),
            origin_data,
            direction_data,
            count,
            options,
            *intervals,
            stream);
      }
      svo::render_volume_from_intervals_cuda(
          *intervals,
          sigma_data,
          color_data,
          rgb_data,
          depth_data,
          opacity_data,
          count,
          payload_rows,
          options,
          stream);
    }

    return py::make_tuple(rgb, depth, opacity, intervals);
  }

  py::object render_volume_backward_intervals_torch(
      py::object origins,
      py::object directions,
      py::object sigma,
      py::object color,
      py::object grad_rgb,
      py::object grad_opacity,
      py::object interval_buffer,
      double near_plane,
      double far_plane,
      py::object background_color,
      double early_stop_transmittance,
      bool store_aux,
      bool enable_empty_space_skipping) const {
    py::object torch = import_torch();
    check_torch_float32_tensor(torch, origins, "origins", device_index_);
    check_torch_float32_tensor(torch, directions, "directions", device_index_);
    check_torch_float32_tensor(torch, sigma, "sigma", device_index_);
    check_torch_float32_tensor(torch, color, "color", device_index_);
    check_torch_float32_tensor(torch, grad_rgb, "grad_rgb", device_index_);
    check_torch_float32_tensor(torch, grad_opacity, "grad_opacity", device_index_);

    const std::vector<py::ssize_t> ray_shape = torch_shape(origins);
    const std::vector<py::ssize_t> direction_shape = torch_shape(directions);
    if (ray_shape != direction_shape) {
      throw py::value_error("origins and directions must have the same shape");
    }
    const std::size_t count = checked_torch_point_count(ray_shape, "origins");

    const std::vector<py::ssize_t> sigma_shape = torch_shape(sigma);
    if (sigma_shape.size() != 1) {
      throw py::value_error("sigma must have shape (P,)");
    }
    const std::vector<py::ssize_t> color_shape = torch_shape(color);
    if (color_shape.size() != 2 || color_shape[1] != 3) {
      throw py::value_error("color must have shape (P, 3)");
    }
    if (sigma_shape[0] != color_shape[0]) {
      throw py::value_error("sigma and color must have the same row count");
    }
    if (sigma_shape[0] < 0) {
      throw py::value_error("sigma shape contains a negative dimension");
    }
    const std::size_t payload_rows = static_cast<std::size_t>(sigma_shape[0]);
    validate_payload_rows(payload_rows);

    std::vector<py::ssize_t> opacity_shape = torch_point_leading_shape(ray_shape);
    std::vector<py::ssize_t> rgb_shape = opacity_shape;
    rgb_shape.push_back(3);
    check_torch_shape_equals(grad_rgb, rgb_shape, "grad_rgb");
    check_torch_shape_equals(grad_opacity, opacity_shape, "grad_opacity");

    auto intervals = interval_buffer.cast<std::shared_ptr<svo::RenderIntervalBuffer>>();
    if (!intervals) {
      throw py::value_error("interval_buffer must be a renderer interval buffer");
    }

    py::object grad_sigma = torch.attr("zeros_like")(sigma);
    py::object grad_color = torch.attr("zeros_like")(color);
    if (count == 0) {
      return py::make_tuple(grad_sigma, grad_color);
    }

    const svo::RenderOptions options = render_options_from_python(
        near_plane,
        far_plane,
        background_color,
        early_stop_transmittance,
        store_aux,
        enable_empty_space_skipping);
    svo::CudaStreamHandle stream = torch_current_stream(torch, origins);
    const auto* sigma_data = reinterpret_cast<const float*>(torch_data_ptr(sigma));
    const auto* color_data = reinterpret_cast<const float*>(torch_data_ptr(color));
    const auto* grad_rgb_data = reinterpret_cast<const glm::vec3*>(torch_data_ptr(grad_rgb));
    const auto* grad_opacity_data = reinterpret_cast<const float*>(torch_data_ptr(grad_opacity));
    auto* grad_sigma_data = reinterpret_cast<float*>(torch_data_ptr(grad_sigma));
    auto* grad_color_data = reinterpret_cast<float*>(torch_data_ptr(grad_color));
    CudaDeviceGuard guard(device_index_);
    {
      py::gil_scoped_release release;
      svo::render_volume_backward_from_intervals_cuda(
          *intervals,
          sigma_data,
          color_data,
          grad_rgb_data,
          grad_opacity_data,
          grad_sigma_data,
          grad_color_data,
          count,
          payload_rows,
          options,
          stream);
    }

    return py::make_tuple(grad_sigma, grad_color);
  }

 private:
  void validate_payload_rows(std::size_t payload_rows) const {
    for (std::uint32_t payload_index : host_octree_.leaf_payload_indices()) {
      if (payload_index >= payload_rows) {
        throw py::value_error("leaf payload index is outside the payload row range");
      }
    }
  }

  svo::Octree host_octree_;
  int device_index_ = 0;
  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes_;
  svo::DeviceBuffer<svo::WideNodeDescriptor> device_wide_nodes_;
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices_;
};

std::string cuda_octree_repr(const CudaOctreeOwner& octree) {
  std::ostringstream stream;
  stream << "CudaOctree(max_depth=" << octree.max_depth() << ", num_nodes=" << octree.num_nodes()
         << ", num_leaves=" << octree.num_leaves() << ", branching='"
         << svo::branching_mode_name(octree.host_octree().branching()) << "', device='cuda:"
         << octree.device_index() << "')";
  return stream.str();
}
#endif

std::string octree_repr(const svo::Octree& octree) {
  std::ostringstream stream;
  stream << "Octree(max_depth=" << octree.max_depth() << ", num_nodes=" << octree.num_nodes()
         << ", num_leaves=" << octree.num_leaves() << ", branching='"
         << svo::branching_mode_name(octree.branching()) << "', device='" << svo::device_name(octree.device())
         << "')";
  return stream.str();
}

}  // namespace

PYBIND11_MODULE(_svo, module) {
  module.doc() = "Sparse voxel octree bindings.";

  py::register_exception<svo::Error>(module, "SvoError");
  py::register_exception<svo::ValidationError>(module, "ValidationError");

  py::enum_<svo::CameraConvention>(module, "CameraConvention")
      .value("OpenGL", svo::CameraConvention::OpenGL)
      .value("ComputerVision", svo::CameraConvention::ComputerVision)
      .export_values();

  py::enum_<svo::BranchingMode>(module, "BranchingMode")
      .value("Octree8", svo::BranchingMode::Octree8)
      .value("Wide4", svo::BranchingMode::Wide4)
      .export_values();

  py::class_<svo::CameraIntrinsics>(module, "CameraIntrinsics")
      .def(
          py::init<int, int, float, float, float, float>(),
          py::arg("width"),
          py::arg("height"),
          py::arg("fx"),
          py::arg("fy"),
          py::arg("cx"),
          py::arg("cy"))
      .def_readwrite("width", &svo::CameraIntrinsics::width)
      .def_readwrite("height", &svo::CameraIntrinsics::height)
      .def_readwrite("fx", &svo::CameraIntrinsics::fx)
      .def_readwrite("fy", &svo::CameraIntrinsics::fy)
      .def_readwrite("cx", &svo::CameraIntrinsics::cx)
      .def_readwrite("cy", &svo::CameraIntrinsics::cy);

  py::class_<svo::Camera>(module, "Camera")
      .def_static(
          "look_at",
          [](py::object origin,
             py::object target,
             py::object up,
             int width,
             int height,
             float vertical_fov_y_degrees,
             svo::CameraConvention convention) {
            return svo::Camera::look_at(
                vec3_from_python(origin, "origin"),
                vec3_from_python(target, "target"),
                vec3_from_python(up, "up"),
                width,
                height,
                vertical_fov_y_degrees,
                convention);
          },
          py::arg("origin"),
          py::arg("target"),
          py::arg("up"),
          py::arg("width"),
          py::arg("height"),
          py::arg("vertical_fov_y_degrees"),
          py::arg("convention") = svo::CameraConvention::OpenGL)
      .def_static(
          "from_intrinsics",
          [](py::object origin,
             py::object target,
             py::object up,
             const svo::CameraIntrinsics& intrinsics,
             svo::CameraConvention convention) {
            return svo::Camera::from_intrinsics(
                vec3_from_python(origin, "origin"),
                vec3_from_python(target, "target"),
                vec3_from_python(up, "up"),
                intrinsics,
                convention);
          },
          py::arg("origin"),
          py::arg("target"),
          py::arg("up"),
          py::arg("intrinsics"),
          py::arg("convention") = svo::CameraConvention::OpenGL)
      .def(
          "generate_rays",
          [](const svo::Camera& camera) {
            svo::RayBatch rays;
            {
              py::gil_scoped_release release;
              rays = svo::generate_rays_cpu(camera);
            }
            return ray_batch_to_numpy(rays);
          })
      .def_property_readonly("width", &svo::Camera::width)
      .def_property_readonly("height", &svo::Camera::height)
      .def_property_readonly("intrinsics", &svo::Camera::intrinsics)
      .def_property_readonly("convention", &svo::Camera::convention);

  py::class_<svo::Octree>(module, "Octree", "Sparse voxel octree CPU wrapper.")
      .def_static(
          "from_voxels",
          [](py::array coords,
             int max_depth,
             const std::string& device,
             py::object root_bounds,
             py::object payload_indices,
             const std::string& branching) {
            if (device != "cpu") {
              throw py::value_error(
                  "Octree.from_voxels currently supports only device='cpu'; build on CPU and use "
                  "Octree.to('cuda') once Python CUDA ownership is implemented");
            }

            svo::BuildOptions options;
            options.max_depth = max_depth;
            options.device = svo::Device::CPU;
            options.root_bounds = root_bounds_from_python(root_bounds);
            options.branching = branching_from_python(branching);

            const std::vector<glm::ivec3> coordinates = coordinates_from_numpy(coords);
            if (payload_indices.is_none()) {
              svo::Octree octree;
              {
                py::gil_scoped_release release;
                octree = svo::Octree::from_voxels_cpu(coordinates, options);
              }
              return octree;
            }

            py::array payload_array = py::array::ensure(payload_indices);
            if (!payload_array) {
              throw py::type_error("payload_indices must be convertible to a NumPy array");
            }
            const std::vector<std::uint32_t> payload_values =
                payload_indices_from_numpy(payload_array, coords.shape(0));
            svo::Octree octree;
            {
              py::gil_scoped_release release;
              octree = svo::Octree::from_voxels_cpu(coordinates, payload_values, options);
            }
            return octree;
          },
          py::arg("coords"),
          py::arg("max_depth"),
          py::arg("device") = "cpu",
          py::arg("root_bounds") = py::none(),
          py::arg("payload_indices") = py::none(),
          py::arg("branching") = "octree8",
          R"pbdoc(
Build a CPU octree from occupied voxel coordinates.

Args:
    coords: NumPy array with shape (N, 3) and dtype int32 or int64.
    max_depth: Octree depth. Valid voxel coordinates lie in [0, 2^max_depth).
    device: Currently only "cpu" is supported.
    root_bounds: Optional array with shape (2, 3). Defaults to [0, 1]^3.
    payload_indices: Optional int array with shape (N,) mapping each input voxel to an external payload row.
    branching: "octree8" for the classic 2x2x2 tree or "wide4" for 4x4x4 nodes.
)pbdoc")
      .def_static(
          "from_leaf_specs",
          [](py::array coord_min,
             py::array depths,
             py::array payload_indices,
             int max_depth,
             const std::string& device,
             py::object root_bounds,
             const std::string& branching) {
            if (device != "cpu") {
              throw py::value_error("Octree.from_leaf_specs currently supports only device='cpu'; build on CPU and use to('cuda')");
            }
            svo::BuildOptions options;
            options.max_depth = max_depth;
            options.device = svo::Device::CPU;
            options.root_bounds = root_bounds_from_python(root_bounds);
            options.branching = branching_from_python(branching);
            const std::vector<svo::LeafSpec> specs = leaf_specs_from_numpy(coord_min, depths, payload_indices);
            svo::Octree octree;
            {
              py::gil_scoped_release release;
              octree = svo::Octree::from_leaf_specs_cpu(specs, options);
            }
            return octree;
          },
          py::arg("coord_min"),
          py::arg("depths"),
          py::arg("payload_indices"),
          py::arg("max_depth"),
          py::arg("device") = "cpu",
          py::arg("root_bounds") = py::none(),
          py::arg("branching") = "octree8",
          R"pbdoc(
Build a CPU octree from variable-depth leaf specs.
)pbdoc")
      .def_static(
          "full_grid",
          [](int max_depth, int leaf_depth, py::object root_bounds, const std::string& branching) {
            if (branching != "octree8") {
              throw py::value_error("Octree.full_grid currently supports only branching='octree8'");
            }
            if (max_depth < 0 || max_depth > 30) {
              throw py::value_error("max_depth must be in the range [0, 30]");
            }
            if (leaf_depth < 0 || leaf_depth > max_depth) {
              throw py::value_error("leaf_depth must satisfy 0 <= leaf_depth <= max_depth");
            }
            if (max_depth > 0 && leaf_depth == 0) {
              throw py::value_error("leaf_depth=0 is only supported when max_depth=0");
            }
            const std::int64_t count_per_axis = static_cast<std::int64_t>(1) << leaf_depth;
            const std::int64_t total_count = count_per_axis * count_per_axis * count_per_axis;
            if (total_count > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
              throw py::value_error("full_grid leaf count exceeds int32 payload index range");
            }
            std::vector<svo::LeafSpec> specs;
            specs.reserve(static_cast<std::size_t>(total_count));
            const int cell_size = 1 << (max_depth - leaf_depth);
            std::uint32_t payload_index = 0u;
            for (std::int64_t z = 0; z < count_per_axis; ++z) {
              for (std::int64_t y = 0; y < count_per_axis; ++y) {
                for (std::int64_t x = 0; x < count_per_axis; ++x) {
                  specs.push_back(svo::LeafSpec{
                      glm::ivec3{
                          static_cast<int>(x) * cell_size,
                          static_cast<int>(y) * cell_size,
                          static_cast<int>(z) * cell_size},
                      leaf_depth,
                      payload_index++});
                }
              }
            }
            svo::BuildOptions options;
            options.max_depth = max_depth;
            options.device = svo::Device::CPU;
            options.root_bounds = root_bounds_from_python(root_bounds);
            options.branching = svo::BranchingMode::Octree8;
            svo::Octree octree;
            {
              py::gil_scoped_release release;
              octree = svo::Octree::from_leaf_specs_cpu(specs, options);
              std::vector<svo::LeafSpec> ordered_specs = octree.leaf_specs();
              for (std::size_t index = 0; index < ordered_specs.size(); ++index) {
                ordered_specs[index].payload_index = static_cast<std::uint32_t>(index);
              }
              octree = svo::Octree::from_leaf_specs_cpu(ordered_specs, options);
            }
            return octree;
          },
          py::arg("max_depth"),
          py::arg("leaf_depth"),
          py::arg("root_bounds") = py::none(),
          py::arg("branching") = "octree8",
          R"pbdoc(
Build a dense coarse octree8 leaf grid while preserving a deeper max_depth budget.
)pbdoc")
      .def(
          "query",
          [](const svo::Octree& octree, py::array points, bool return_payload_indices) {
            svo::QueryOptions options;
            options.return_payload_indices = return_payload_indices;
            const std::vector<glm::vec3> point_values = points_from_numpy(points);
            std::vector<std::int32_t> results;
            {
              py::gil_scoped_release release;
              results = svo::query_points(octree, point_values, options);
            }
            return vector_to_numpy(results);
          },
          py::arg("points"),
          py::arg("return_payload_indices") = false,
          R"pbdoc(
Query points against the CPU octree.

Args:
    points: NumPy array with shape (N, 3) and dtype float32 or float64.
    return_payload_indices: Return payload indices instead of leaf IDs.

Returns:
    NumPy int32 array of shape (N,). Misses are encoded as -1.
)pbdoc")
      .def(
          "query_payload_indices",
          [](const svo::Octree& octree, py::array points) {
            const std::vector<glm::vec3> point_values = points_from_numpy(points);
            std::vector<std::int32_t> results;
            {
              py::gil_scoped_release release;
              results = svo::query_payload_indices(octree, point_values);
            }
            return vector_to_numpy(results);
          },
          py::arg("points"),
          R"pbdoc(
Query points and return payload indices directly.

Args:
    points: NumPy array with shape (N, 3) and dtype float32 or float64.

Returns:
    NumPy int32 array of shape (N,). Misses are encoded as -1.
)pbdoc")
      .def(
          "raycast",
          [](const svo::Octree& octree, py::array origins, py::array directions, bool return_payload_indices) {
            const PythonRayBatch rays = rays_from_numpy(origins, directions);
            svo::RaycastOptions options;
            options.return_payload_indices = return_payload_indices;
            svo::RayBatch ray_batch;
            ray_batch.origins = rays.origins;
            ray_batch.directions = rays.directions;
            ray_batch.width = rays.width;
            ray_batch.height = rays.height;
            svo::RaycastBatch results;
            {
              py::gil_scoped_release release;
              results = svo::raycast_cpu(octree, ray_batch, options);
            }
            return raycast_batch_to_numpy(results, rays.image_shaped);
          },
          py::arg("origins"),
          py::arg("directions"),
          py::arg("return_payload_indices") = false,
          R"pbdoc(
Raycast origin/direction batches against the CPU octree.

Args:
    origins: NumPy array with shape (N, 3) or (H, W, 3), dtype float32 or float64.
    directions: NumPy array with the same shape as origins.
    return_payload_indices: Return payload indices instead of leaf IDs.

Returns:
    Tuple (hit_mask, leaf_ids, t, positions, depths). Misses use leaf_id=-1,
    depth=-1, t=inf, and position=(nan, nan, nan).
)pbdoc")
      .def(
          "to",
          [](const svo::Octree& octree, const std::string& device) -> py::object {
            if (device == "cpu") {
              return py::cast(octree);
            }
            if (device == "cuda") {
#if SVO_ENABLE_CUDA
              return py::cast(CudaOctreeOwner(octree));
#else
              throw py::type_error(
                  "Octree.to('cuda') requires a Python extension built with SVO_ENABLE_CUDA=ON");
#endif
            }
            throw py::value_error("device must be 'cpu' or 'cuda'");
          },
          py::arg("device"),
          R"pbdoc(
Return an octree on the requested device.

With CUDA-enabled builds, device="cuda" returns a CUDA-owned octree whose topology
is resident on the GPU. CPU-only builds raise a clear error for device="cuda".
)pbdoc")
      .def(
          "query_cuda",
          [](const svo::Octree& octree, py::object points, bool return_payload_indices) {
#if SVO_ENABLE_CUDA
            return CudaOctreeOwner(octree).query(points, return_payload_indices);
#else
            (void)octree;
            (void)points;
            (void)return_payload_indices;
            throw py::type_error(
                "Octree.query_cuda requires a Python extension built with SVO_ENABLE_CUDA=ON");
#endif
          },
          py::arg("points"),
          py::arg("return_payload_indices") = false,
          R"pbdoc(
Query points through a temporary CUDA-owned octree.

This convenience path uploads octree topology for the call. CPU NumPy points are
copied to CUDA and return a CPU NumPy array; CUDA Torch tensor points stay on
GPU and return a CUDA torch.int32 tensor. For repeated queries, use
tree.to("cuda") to keep octree topology resident on the GPU.
)pbdoc")
      .def(
          "raycast_cuda",
          [](const svo::Octree& octree, py::object origins, py::object directions, bool return_payload_indices) {
#if SVO_ENABLE_CUDA
            return CudaOctreeOwner(octree).raycast(origins, directions, return_payload_indices);
#else
            (void)octree;
            (void)origins;
            (void)directions;
            (void)return_payload_indices;
            throw py::type_error(
                "Octree.raycast_cuda requires a Python extension built with SVO_ENABLE_CUDA=ON");
#endif
          },
          py::arg("origins"),
          py::arg("directions"),
          py::arg("return_payload_indices") = false,
          R"pbdoc(
Raycast through a temporary CUDA-owned octree.

This convenience path uploads octree topology for the call. CPU NumPy rays are
copied to CUDA and return CPU NumPy arrays; CUDA Torch tensor rays stay on GPU
and return CUDA Torch tensors. For repeated raycasts, use tree.to("cuda") to
keep octree topology resident on the GPU.
)pbdoc")
      .def_property_readonly("max_depth", &svo::Octree::max_depth)
      .def_property_readonly("num_nodes", &svo::Octree::num_nodes)
      .def_property_readonly("num_leaves", &svo::Octree::num_leaves)
      .def_property_readonly(
          "device",
          [](const svo::Octree& octree) { return std::string(svo::device_name(octree.device())); })
      .def_property_readonly(
          "branching",
          [](const svo::Octree& octree) { return std::string(svo::branching_mode_name(octree.branching())); })
      .def_property_readonly(
          "root_bounds",
          [](const svo::Octree& octree) { return root_bounds_to_numpy(octree.root_bounds()); })
      .def_property_readonly(
          "leaf_payload_indices",
          [](const svo::Octree& octree) { return payload_indices_to_numpy(octree.leaf_payload_indices()); })
      .def_property_readonly(
          "leaf_specs",
          [](const svo::Octree& octree) { return leaf_specs_to_numpy(octree.leaf_specs()); })
      .def("__repr__", &octree_repr);

#if SVO_ENABLE_CUDA
  py::class_<svo::RenderIntervalBuffer, std::shared_ptr<svo::RenderIntervalBuffer>>(
      module,
      "_RenderIntervalBuffer",
      "Internal CUDA render interval workspace.");

  py::class_<CudaOctreeOwner>(module, "CudaOctree", "CUDA-owned sparse voxel octree.")
      .def(
          "query",
          &CudaOctreeOwner::query,
          py::arg("points"),
          py::arg("return_payload_indices") = false,
          R"pbdoc(
Query points against CUDA-resident octree topology.

Args:
    points: CPU NumPy array with shape (N, 3), dtype float32 or float64,
        or contiguous CUDA Torch tensor with shape (N, 3), dtype torch.float32.
    return_payload_indices: Return payload indices instead of leaf IDs.

Returns:
    NumPy int32 array for NumPy input, or CUDA torch.int32 tensor for Torch
    CUDA input. Misses are encoded as -1.
)pbdoc")
      .def(
          "query_payload_indices",
          [](const CudaOctreeOwner& octree, py::object points) {
            return octree.query(points, true);
          },
          py::arg("points"),
          R"pbdoc(
Query points against CUDA-resident octree topology and return payload indices.

Returns:
    NumPy int32 array for NumPy input, or CUDA torch.int32 tensor for Torch
    CUDA input. Misses are encoded as -1.
)pbdoc")
      .def(
          "raycast",
          &CudaOctreeOwner::raycast,
          py::arg("origins"),
          py::arg("directions"),
          py::arg("return_payload_indices") = false,
          R"pbdoc(
Raycast against CUDA-resident octree topology.

Args:
    origins: CPU NumPy array with shape (N, 3) or (H, W, 3), dtype float32 or float64,
        or contiguous CUDA Torch tensor with the same shapes and dtype torch.float32.
    directions: Same shape, dtype, device, and backend as origins.
    return_payload_indices: Return payload indices instead of leaf IDs.

Returns:
    Tuple (hit_mask, leaf_ids, t, positions, depths), using NumPy arrays for
    NumPy input or CUDA Torch tensors for Torch CUDA input.
)pbdoc")
      .def(
          "_render_volume_backward_torch",
          &CudaOctreeOwner::render_volume_backward_torch,
          py::arg("origins"),
          py::arg("directions"),
          py::arg("sigma"),
          py::arg("color"),
          py::arg("grad_rgb"),
          py::arg("grad_opacity"),
          py::arg("near_plane") = 0.0,
          py::arg("far_plane") = std::numeric_limits<double>::infinity(),
          py::arg("background_color") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("early_stop_transmittance") = 1.0e-4,
          py::arg("store_aux") = false,
          py::arg("enable_empty_space_skipping") = true)
      .def(
          "_render_volume_torch",
          &CudaOctreeOwner::render_volume_torch,
          py::arg("origins"),
          py::arg("directions"),
          py::arg("sigma"),
          py::arg("color"),
          py::arg("near_plane") = 0.0,
          py::arg("far_plane") = std::numeric_limits<double>::infinity(),
          py::arg("background_color") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("early_stop_transmittance") = 1.0e-4,
          py::arg("store_aux") = false,
          py::arg("enable_empty_space_skipping") = true,
          py::arg("coarse_occupancy") = py::none())
      .def(
          "_render_volume_intervals_torch",
          &CudaOctreeOwner::render_volume_intervals_torch,
          py::arg("origins"),
          py::arg("directions"),
          py::arg("sigma"),
          py::arg("color"),
          py::arg("near_plane") = 0.0,
          py::arg("far_plane") = std::numeric_limits<double>::infinity(),
          py::arg("background_color") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("early_stop_transmittance") = 1.0e-4,
          py::arg("store_aux") = false,
          py::arg("enable_empty_space_skipping") = true)
      .def(
          "_render_volume_backward_intervals_torch",
          &CudaOctreeOwner::render_volume_backward_intervals_torch,
          py::arg("origins"),
          py::arg("directions"),
          py::arg("sigma"),
          py::arg("color"),
          py::arg("grad_rgb"),
          py::arg("grad_opacity"),
          py::arg("interval_buffer"),
          py::arg("near_plane") = 0.0,
          py::arg("far_plane") = std::numeric_limits<double>::infinity(),
          py::arg("background_color") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("early_stop_transmittance") = 1.0e-4,
          py::arg("store_aux") = false,
          py::arg("enable_empty_space_skipping") = true)
      .def(
          "_sample_trilinear_torch",
          &CudaOctreeOwner::sample_trilinear_torch,
          py::arg("points"),
          py::arg("payload"),
          py::arg("fill_value") = 0.0)
      .def(
          "_sample_trilinear_backward_torch",
          &CudaOctreeOwner::sample_trilinear_backward_torch,
          py::arg("points"),
          py::arg("payload"),
          py::arg("grad_outputs"),
          py::arg("fill_value") = 0.0)
      .def(
          "to",
          [](const CudaOctreeOwner& octree, const std::string& device) -> py::object {
            if (device == "cuda") {
              return py::cast(&octree, py::return_value_policy::reference);
            }
            if (device == "cpu") {
              return py::cast(octree.host_octree());
            }
            throw py::value_error("device must be 'cpu' or 'cuda'");
          },
          py::arg("device"))
      .def_property_readonly("max_depth", &CudaOctreeOwner::max_depth)
      .def_property_readonly("num_nodes", &CudaOctreeOwner::num_nodes)
      .def_property_readonly("num_leaves", &CudaOctreeOwner::num_leaves)
      .def_property_readonly("device", [](const CudaOctreeOwner&) { return std::string("cuda"); })
      .def_property_readonly("device_index", &CudaOctreeOwner::device_index)
      .def_property_readonly(
          "branching",
          [](const CudaOctreeOwner& octree) {
            return std::string(svo::branching_mode_name(octree.host_octree().branching()));
          })
      .def_property_readonly(
          "root_bounds",
          [](const CudaOctreeOwner& octree) { return root_bounds_to_numpy(octree.root_bounds()); })
      .def_property_readonly(
          "leaf_payload_indices",
          [](const CudaOctreeOwner& octree) { return payload_indices_to_numpy(octree.host_octree().leaf_payload_indices()); })
      .def_property_readonly(
          "leaf_specs",
          [](const CudaOctreeOwner& octree) { return leaf_specs_to_numpy(octree.host_octree().leaf_specs()); })
      .def(
          "_coarse_occupancy",
          &CudaOctreeOwner::coarse_occupancy,
          py::arg("resolution"),
          R"pbdoc(
Build a CUDA-resident coarse occupancy grid for debug/benchmark rendering.
)pbdoc")
      .def("_release", &CudaOctreeOwner::release)
      .def("__repr__", &cuda_octree_repr);

  py::class_<CudaCoarseOccupancyOwner>(module, "_CudaCoarseOccupancy", "CUDA coarse occupancy debug accelerator.")
      .def_property_readonly("resolution", &CudaCoarseOccupancyOwner::resolution)
      .def_property_readonly("size_bytes", &CudaCoarseOccupancyOwner::size_bytes)
      .def_property_readonly("device_index", &CudaCoarseOccupancyOwner::device_index)
      .def("_release", &CudaCoarseOccupancyOwner::release);
#endif

  module.def(
      "_save_svo",
      [](const std::string& path, const svo::Octree& tree, py::dict payloads) {
        svo::SerializedScene scene;
        scene.tree = tree;
        scene.arrays.reserve(static_cast<std::size_t>(payloads.size()));
        for (const auto& item : payloads) {
          const std::string name = py::cast<std::string>(item.first);
          py::array array = py::array::ensure(item.second);
          if (!array) {
            throw py::type_error("payload values must be convertible to NumPy arrays");
          }
          scene.arrays.push_back(serialized_array_from_numpy(name, array));
        }
        py::gil_scoped_release release;
        svo::save_svo(path, scene);
      },
      py::arg("path"),
      py::arg("tree"),
      py::arg("payloads") = py::dict(),
      R"pbdoc(
Save a CPU octree and optional float32 payload arrays to a versioned .svo file.

This private binding is wrapped by ``svo.save``.
)pbdoc");

  module.def(
      "_load_svo",
      [](const std::string& path) {
        svo::SerializedScene scene;
        {
          py::gil_scoped_release release;
          scene = svo::load_svo(path);
        }
        return py::make_tuple(std::move(scene.tree), serialized_arrays_to_python(scene.arrays));
      },
      py::arg("path"),
      R"pbdoc(
Load a versioned .svo file as (Octree, payload_dict).

This private binding is wrapped by ``svo.load``.
)pbdoc");

  module.def(
      "_sample_trilinear_cpu",
      &sample_trilinear_cpu_binding,
      py::arg("tree"),
      py::arg("points"),
      py::arg("payload"),
      py::arg("fill_value") = 0.0,
      R"pbdoc(
Sample leaf-centered payload values with trilinear interpolation on CPU.

This private binding is wrapped by ``svo.sample_trilinear``.
)pbdoc");

  module.def(
      "_render_volume_cpu",
      &render_volume_cpu_binding,
      py::arg("tree"),
      py::arg("origins"),
      py::arg("directions"),
      py::arg("sigma"),
      py::arg("color"),
      py::arg("near_plane") = 0.0,
      py::arg("far_plane") = std::numeric_limits<double>::infinity(),
      py::arg("background_color") = py::make_tuple(0.0, 0.0, 0.0),
      py::arg("early_stop_transmittance") = 1.0e-4,
      py::arg("store_aux") = false,
      py::arg("enable_empty_space_skipping") = true,
      R"pbdoc(
Render sigma/color leaf payloads along rays on CPU.

This private binding is wrapped by ``svo.render_volume``.
)pbdoc");

  module.def(
      "_core_version",
      []() { return std::string{svo::version()}; },
      "Return the compiled C++ core fallback version.");

  module.def(
      "cuda_enabled",
      []() {
#if SVO_ENABLE_CUDA
        return true;
#else
        return false;
#endif
      },
      "Return whether this extension was compiled with CUDA support.");
}
