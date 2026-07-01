#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <svo/Error.hpp>
#include <svo/Octree.hpp>
#include <svo/Query.hpp>

namespace py = pybind11;

namespace {

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

  py::array_t<float, py::array::c_style | py::array::forcecast> cast(array);
  auto view = cast.unchecked<2>();
  return {
      glm::vec3{view(0, 0), view(0, 1), view(0, 2)},
      glm::vec3{view(1, 0), view(1, 1), view(1, 2)},
  };
}

std::vector<glm::ivec3> coordinates_from_numpy(const py::array& array) {
  if (array.ndim() != 2 || array.shape(1) != 3) {
    throw py::value_error("coords must have shape (N, 3)");
  }
  if (!(py::isinstance<py::array_t<std::int32_t>>(array) || py::isinstance<py::array_t<std::int64_t>>(array))) {
    throw py::type_error("coords must have dtype int32 or int64");
  }

  std::vector<glm::ivec3> coordinates;
  coordinates.reserve(static_cast<std::size_t>(array.shape(0)));

  if (py::isinstance<py::array_t<std::int32_t>>(array)) {
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> cast(array);
    auto view = cast.unchecked<2>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      coordinates.emplace_back(view(index, 0), view(index, 1), view(index, 2));
    }
  } else {
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cast(array);
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

std::vector<glm::vec3> points_from_numpy(const py::array& array) {
  if (array.ndim() != 2 || array.shape(1) != 3) {
    throw py::value_error("points must have shape (N, 3)");
  }
  if (!(py::isinstance<py::array_t<float>>(array) || py::isinstance<py::array_t<double>>(array))) {
    throw py::type_error("points must have dtype float32 or float64");
  }

  std::vector<glm::vec3> points;
  points.reserve(static_cast<std::size_t>(array.shape(0)));

  if (py::isinstance<py::array_t<float>>(array)) {
    py::array_t<float, py::array::c_style | py::array::forcecast> cast(array);
    auto view = cast.unchecked<2>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
      points.emplace_back(view(index, 0), view(index, 1), view(index, 2));
    }
  } else {
    py::array_t<double, py::array::c_style | py::array::forcecast> cast(array);
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

py::array_t<std::int32_t> vector_to_numpy(const std::vector<std::int32_t>& values) {
  py::array_t<std::int32_t> output(values.size());
  auto view = output.mutable_unchecked<1>();
  for (py::ssize_t index = 0; index < view.shape(0); ++index) {
    view(index) = values[static_cast<std::size_t>(index)];
  }
  return output;
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

std::string octree_repr(const svo::Octree& octree) {
  std::ostringstream stream;
  stream << "Octree(max_depth=" << octree.max_depth() << ", num_nodes=" << octree.num_nodes()
         << ", num_leaves=" << octree.num_leaves() << ", device='" << svo::device_name(octree.device())
         << "')";
  return stream.str();
}

}  // namespace

PYBIND11_MODULE(_svo, module) {
  module.doc() = "Sparse voxel octree bindings.";

  py::register_exception<svo::Error>(module, "SvoError");
  py::register_exception<svo::ValidationError>(module, "ValidationError");

  py::class_<svo::Octree>(module, "Octree", "Sparse voxel octree CPU wrapper.")
      .def_static(
          "from_voxels",
          [](py::array coords, int max_depth, const std::string& device, py::object root_bounds) {
            if (device != "cpu") {
              throw py::value_error(
                  "Octree.from_voxels currently supports only device='cpu'; build on CPU and use "
                  "Octree.to('cuda') once Python CUDA ownership is implemented");
            }

            svo::BuildOptions options;
            options.max_depth = max_depth;
            options.device = svo::Device::CPU;
            options.root_bounds = root_bounds_from_python(root_bounds);

            return svo::Octree::from_voxels_cpu(coordinates_from_numpy(coords), options);
          },
          py::arg("coords"),
          py::arg("max_depth"),
          py::arg("device") = "cpu",
          py::arg("root_bounds") = py::none(),
          R"pbdoc(
Build a CPU octree from occupied voxel coordinates.

Args:
    coords: NumPy array with shape (N, 3) and dtype int32 or int64.
    max_depth: Octree depth. Valid voxel coordinates lie in [0, 2^max_depth).
    device: Currently only "cpu" is supported.
    root_bounds: Optional array with shape (2, 3). Defaults to [0, 1]^3.
)pbdoc")
      .def(
          "query",
          [](const svo::Octree& octree, py::array points, bool return_payload_indices) {
            svo::QueryOptions options;
            options.return_payload_indices = return_payload_indices;
            return vector_to_numpy(
                svo::query_points(octree, points_from_numpy(points), options));
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
          "to",
          [](const svo::Octree& octree, const std::string& device) {
            if (device == "cpu") {
              return octree;
            }
            if (device == "cuda") {
              throw py::type_error(
                  "Octree.to('cuda') is reserved for the upcoming Python CUDA octree owner; "
                  "the raw C++ query_points_cuda path exists, but Python CUDA ownership is not implemented yet");
            }
            throw py::value_error("device must be 'cpu' or 'cuda'");
          },
          py::arg("device"),
          R"pbdoc(
Return an octree on the requested device.

Currently only device="cpu" is implemented. device="cuda" is reserved for the
future Python CUDA owner that will avoid unnecessary host roundtrips.
)pbdoc")
      .def(
          "query_cuda",
          [](const svo::Octree&, py::object, bool) {
            throw py::type_error(
                "Octree.query_cuda is not implemented yet. Use the C++ query_points_cuda launcher "
                "for CUDA point lookup until Python CUDA tensor ownership is added.");
          },
          py::arg("points"),
          py::arg("return_payload_indices") = false,
          R"pbdoc(
Future CUDA point-query entry point.

This placeholder keeps the Python API shape explicit while CUDA tensor ownership,
stream handling, and dtype/contiguity validation are designed.
)pbdoc")
      .def_property_readonly("max_depth", &svo::Octree::max_depth)
      .def_property_readonly("num_nodes", &svo::Octree::num_nodes)
      .def_property_readonly("num_leaves", &svo::Octree::num_leaves)
      .def_property_readonly(
          "device",
          [](const svo::Octree& octree) { return std::string(svo::device_name(octree.device())); })
      .def_property_readonly(
          "root_bounds",
          [](const svo::Octree& octree) { return root_bounds_to_numpy(octree.root_bounds()); })
      .def("__repr__", &octree_repr);

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
