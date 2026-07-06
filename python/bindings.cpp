#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <svo/Camera.hpp>
#include <svo/DeviceBuffer.hpp>
#include <svo/Error.hpp>
#include <svo/Octree.hpp>
#include <svo/Query.hpp>
#include <svo/Raycast.hpp>

namespace py = pybind11;

namespace {

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

py::array_t<float> root_bounds_to_numpy(const svo::RootBounds& bounds) {
  py::array_t<float> output({2, 3});
  auto view = output.mutable_unchecked<2>();
  for (int axis = 0; axis < 3; ++axis) {
    view(0, axis) = bounds[0][axis];
    view(1, axis) = bounds[1][axis];
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
class CudaOctreeOwner {
 public:
  explicit CudaOctreeOwner(const svo::Octree& host_octree)
      : host_octree_(host_octree),
        device_nodes_(svo::DeviceBuffer<svo::NodeDescriptor>::from_host(host_octree.nodes(), svo::Device::CUDA)),
        device_leaf_payload_indices_(svo::DeviceBuffer<std::uint32_t>::from_host(
            host_octree.leaf_payload_indices(),
            svo::Device::CUDA)) {}

  const svo::Octree& host_octree() const noexcept { return host_octree_; }
  int max_depth() const noexcept { return host_octree_.max_depth(); }
  std::int64_t num_nodes() const noexcept { return host_octree_.num_nodes(); }
  std::int64_t num_leaves() const noexcept { return host_octree_.num_leaves(); }
  const svo::RootBounds& root_bounds() const noexcept { return host_octree_.root_bounds(); }

  py::array_t<std::int32_t> query(py::array points_array, bool return_payload_indices) const {
    const std::vector<glm::vec3> points = points_from_numpy(points_array);
    svo::DeviceBuffer<glm::vec3> device_points =
        svo::DeviceBuffer<glm::vec3>::from_host(points, svo::Device::CUDA);
    svo::DeviceBuffer<std::int32_t> device_results(points.size(), svo::Device::CUDA);

    svo::QueryOptions options;
    options.return_payload_indices = return_payload_indices;

    {
      py::gil_scoped_release release;
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

    return vector_to_numpy(device_results.to_host());
  }

  py::tuple raycast(py::array origins_array, py::array directions_array, bool return_payload_indices) const {
    const PythonRayBatch rays = rays_from_numpy(origins_array, directions_array);
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

 private:
  svo::Octree host_octree_;
  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes_;
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices_;
};

std::string cuda_octree_repr(const CudaOctreeOwner& octree) {
  std::ostringstream stream;
  stream << "CudaOctree(max_depth=" << octree.max_depth() << ", num_nodes=" << octree.num_nodes()
         << ", num_leaves=" << octree.num_leaves() << ", device='cuda')";
  return stream.str();
}
#endif

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

  py::enum_<svo::CameraConvention>(module, "CameraConvention")
      .value("OpenGL", svo::CameraConvention::OpenGL)
      .value("ComputerVision", svo::CameraConvention::ComputerVision)
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
            return ray_batch_to_numpy(svo::generate_rays_cpu(camera));
          })
      .def_property_readonly("width", &svo::Camera::width)
      .def_property_readonly("height", &svo::Camera::height)
      .def_property_readonly("intrinsics", &svo::Camera::intrinsics)
      .def_property_readonly("convention", &svo::Camera::convention);

  py::class_<svo::Octree>(module, "Octree", "Sparse voxel octree CPU wrapper.")
      .def_static(
          "from_voxels",
          [](py::array coords, int max_depth, const std::string& device, py::object root_bounds, py::object payload_indices) {
            if (device != "cpu") {
              throw py::value_error(
                  "Octree.from_voxels currently supports only device='cpu'; build on CPU and use "
                  "Octree.to('cuda') once Python CUDA ownership is implemented");
            }

            svo::BuildOptions options;
            options.max_depth = max_depth;
            options.device = svo::Device::CPU;
            options.root_bounds = root_bounds_from_python(root_bounds);

            const std::vector<glm::ivec3> coordinates = coordinates_from_numpy(coords);
            if (payload_indices.is_none()) {
              return svo::Octree::from_voxels_cpu(coordinates, options);
            }

            py::array payload_array = py::array::ensure(payload_indices);
            if (!payload_array) {
              throw py::type_error("payload_indices must be convertible to a NumPy array");
            }
            return svo::Octree::from_voxels_cpu(
                coordinates,
                payload_indices_from_numpy(payload_array, coords.shape(0)),
                options);
          },
          py::arg("coords"),
          py::arg("max_depth"),
          py::arg("device") = "cpu",
          py::arg("root_bounds") = py::none(),
          py::arg("payload_indices") = py::none(),
          R"pbdoc(
Build a CPU octree from occupied voxel coordinates.

Args:
    coords: NumPy array with shape (N, 3) and dtype int32 or int64.
    max_depth: Octree depth. Valid voxel coordinates lie in [0, 2^max_depth).
    device: Currently only "cpu" is supported.
    root_bounds: Optional array with shape (2, 3). Defaults to [0, 1]^3.
    payload_indices: Optional int array with shape (N,) mapping each input voxel to an external payload row.
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
          "query_payload_indices",
          [](const svo::Octree& octree, py::array points) {
            return vector_to_numpy(svo::query_payload_indices(octree, points_from_numpy(points)));
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
            return raycast_batch_to_numpy(
                svo::raycast_cpu(octree, ray_batch, options), rays.image_shaped);
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
          [](const svo::Octree& octree, py::array points, bool return_payload_indices) {
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

This convenience path copies CPU NumPy points to CUDA, runs the CUDA point-query
launcher, and returns a CPU NumPy array. For repeated queries, use tree.to("cuda")
to keep octree topology resident on the GPU.
)pbdoc")
      .def(
          "raycast_cuda",
          [](const svo::Octree& octree, py::array origins, py::array directions, bool return_payload_indices) {
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

This convenience path copies CPU NumPy rays to CUDA, runs the CUDA raycast
launcher, and returns CPU NumPy arrays matching Octree.raycast. For repeated
raycasts, use tree.to("cuda") to keep octree topology resident on the GPU.
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
      .def_property_readonly(
          "leaf_payload_indices",
          [](const svo::Octree& octree) { return payload_indices_to_numpy(octree.leaf_payload_indices()); })
      .def("__repr__", &octree_repr);

#if SVO_ENABLE_CUDA
  py::class_<CudaOctreeOwner>(module, "CudaOctree", "CUDA-owned sparse voxel octree.")
      .def(
          "query",
          &CudaOctreeOwner::query,
          py::arg("points"),
          py::arg("return_payload_indices") = false,
          R"pbdoc(
Query points against CUDA-resident octree topology.

Args:
    points: CPU NumPy array with shape (N, 3) and dtype float32 or float64.
    return_payload_indices: Return payload indices instead of leaf IDs.

Returns:
    NumPy int32 array of shape (N,). Misses are encoded as -1.
)pbdoc")
      .def(
          "query_payload_indices",
          [](const CudaOctreeOwner& octree, py::array points) {
            return octree.query(points, true);
          },
          py::arg("points"),
          R"pbdoc(
Query points against CUDA-resident octree topology and return payload indices.

Returns:
    NumPy int32 array of shape (N,). Misses are encoded as -1.
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
    origins: CPU NumPy array with shape (N, 3) or (H, W, 3), dtype float32 or float64.
    directions: CPU NumPy array with the same shape as origins.
    return_payload_indices: Return payload indices instead of leaf IDs.

Returns:
    Tuple (hit_mask, leaf_ids, t, positions, depths), matching Octree.raycast.
)pbdoc")
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
      .def_property_readonly(
          "root_bounds",
          [](const CudaOctreeOwner& octree) { return root_bounds_to_numpy(octree.root_bounds()); })
      .def_property_readonly(
          "leaf_payload_indices",
          [](const CudaOctreeOwner& octree) { return payload_indices_to_numpy(octree.host_octree().leaf_payload_indices()); })
      .def("__repr__", &cuda_octree_repr);
#endif

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
