#include "benchmark_common.hpp"

#include <svo/Renderer.hpp>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

struct RenderRun {
  svo_bench::TimingMetrics metrics;
  svo::TraversalStats stats;
};

const svo::SerializedArray* find_array(const svo::SerializedScene& scene, std::string_view name) {
  for (const svo::SerializedArray& array : scene.arrays) {
    if (array.name == name) {
      return &array;
    }
  }
  return nullptr;
}

std::vector<float> float_array_data(const svo::SerializedArray& array) {
  if (array.dtype != svo::SerializedArrayDType::Float32) {
    throw std::runtime_error("render benchmark scene payload arrays must be float32");
  }
  if ((array.data.size() % sizeof(float)) != 0u) {
    throw std::runtime_error("render benchmark scene payload byte count is not float-aligned");
  }
  std::vector<float> values(array.data.size() / sizeof(float));
  if (!array.data.empty()) {
    std::memcpy(values.data(), array.data.data(), array.data.size());
  }
  return values;
}

void payloads_from_scene(const svo::SerializedScene& scene, std::vector<float>& sigma, std::vector<float>& color) {
  const svo::SerializedArray* sigma_array = find_array(scene, "sigma");
  const svo::SerializedArray* color_array = find_array(scene, "color");
  if (sigma_array == nullptr || color_array == nullptr) {
    throw std::runtime_error("--scene-file for render benchmark must contain 'sigma' and 'color' arrays");
  }
  const std::uint64_t rows = scene.tree.leaf_payload_indices().empty()
      ? 0u
      : static_cast<std::uint64_t>(*std::max_element(
            scene.tree.leaf_payload_indices().begin(),
            scene.tree.leaf_payload_indices().end())) +
          1u;
  if (sigma_array->shape != std::vector<std::uint64_t>{rows}) {
    throw std::runtime_error("scene payload 'sigma' must have shape (P,)");
  }
  if (color_array->shape != std::vector<std::uint64_t>{rows, 3u}) {
    throw std::runtime_error("scene payload 'color' must have shape (P, 3)");
  }
  sigma = float_array_data(*sigma_array);
  color = float_array_data(*color_array);
}

bool run_forward(const svo_bench::BenchmarkConfig& config) {
  return config.operation == "default" || config.operation == "both" || config.operation == "forward";
}

bool run_backward(const svo_bench::BenchmarkConfig& config) {
  return config.operation == "default" || config.operation == "both" || config.operation == "backward";
}

void validate_operation(const svo_bench::BenchmarkConfig& config) {
  if (!run_forward(config) && !run_backward(config)) {
    throw std::runtime_error("--operation must be forward, backward, both, or omitted");
  }
}

template <typename Forward, typename Backward>
void time_render(
    cudaStream_t stream,
    const svo_bench::BenchmarkConfig& config,
    Forward&& forward,
    Backward&& backward,
    svo_bench::TimingMetrics& metrics) {
  if (run_forward(config)) {
    forward();
  }
  if (run_backward(config)) {
    backward();
  }
  svo_bench::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize render warmup");

  if (run_forward(config)) {
    metrics.kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
      for (int iteration = 0; iteration < config.iterations; ++iteration) {
        forward();
      }
    }) / static_cast<double>(config.iterations);
  }

  if (run_backward(config)) {
    metrics.backward_kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
      for (int iteration = 0; iteration < config.iterations; ++iteration) {
        backward();
      }
    }) / static_cast<double>(config.iterations);
  }
}

RenderRun run_octree8(
    const svo::Octree& tree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const std::vector<float>& sigma,
    const std::vector<float>& color,
    cudaStream_t stream,
    const svo_bench::BenchmarkConfig& config) {
  RenderRun run;
  svo::RenderOptions cpu_options;
  svo::TraversalStats cpu_stats;
  cpu_options.stats = config.profile ? &cpu_stats : nullptr;
  run.metrics.cpu_ms = svo_bench::time_wall_ms([&]() {
    (void)svo::render_volume_cpu(tree, origins, directions, sigma.data(), color.data(), sigma.size(), cpu_options);
  });

  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes(tree.nodes().size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_payload_indices(tree.leaf_payload_indices().size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_origins(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_directions(directions.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_sigma(sigma.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_color(color.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_depth(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_opacity(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_grad_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_grad_opacity(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_grad_sigma(sigma.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_grad_color(color.size(), svo::Device::CUDA);
  svo::DeviceBuffer<svo::TraversalStats> device_stats(1, svo::Device::CUDA);
  svo::DeviceCoarseOccupancyGrid device_coarse;
  svo::CoarseOccupancyDeviceView coarse_view;
  const std::vector<glm::vec3> grad_rgb(origins.size(), {1.0f, 1.0f, 1.0f});
  const std::vector<float> grad_opacity(origins.size(), 1.0f);
  svo::TraversalStats zero_stats{};
  const bool use_coarse = config.empty_space_accelerator == "coarse";
  if (use_coarse && config.render_strategy == "intervals") {
    throw std::runtime_error("--empty-space-accelerator coarse is not wired for --render-strategy intervals");
  }
  const svo::CoarseOccupancyGrid coarse_grid =
      use_coarse ? svo::CoarseOccupancyGrid::from_octree(tree, config.coarse_resolution) : svo::CoarseOccupancyGrid{};
  run.metrics.coarse_occupancy_bytes = use_coarse ? static_cast<std::uint64_t>(coarse_grid.size_bytes()) : 0u;

  run.metrics.h2d_ms = svo_bench::time_cuda_ms(stream, [&]() {
    device_nodes.copy_from_host(tree.nodes().data(), tree.nodes().size(), stream);
    device_payload_indices.copy_from_host(tree.leaf_payload_indices().data(), tree.leaf_payload_indices().size(), stream);
    device_origins.copy_from_host(origins.data(), origins.size(), stream);
    device_directions.copy_from_host(directions.data(), directions.size(), stream);
    device_sigma.copy_from_host(sigma.data(), sigma.size(), stream);
    device_color.copy_from_host(color.data(), color.size(), stream);
    device_grad_rgb.copy_from_host(grad_rgb.data(), grad_rgb.size(), stream);
    device_grad_opacity.copy_from_host(grad_opacity.data(), grad_opacity.size(), stream);
    device_stats.copy_from_host(&zero_stats, 1, stream);
    if (use_coarse) {
      device_coarse.upload(coarse_grid, stream);
      coarse_view = device_coarse.view();
    }
    svo_bench::check(cudaMemsetAsync(device_grad_sigma.data(), 0, sigma.size() * sizeof(float), stream), "cudaMemsetAsync grad_sigma");
    svo_bench::check(cudaMemsetAsync(device_grad_color.data(), 0, color.size() * sizeof(float), stream), "cudaMemsetAsync grad_color");
  });

  svo::RenderOptions options;
  options.stats = config.profile ? device_stats.data() : nullptr;
  options.coarse_occupancy = use_coarse ? &coarse_view : nullptr;
  svo::RenderIntervalBuffer intervals;
  if (config.render_strategy == "intervals") {
    auto build_intervals = [&]() {
      svo::build_render_intervals_cuda(
          device_nodes.data(),
          device_nodes.size(),
          device_payload_indices.data(),
          device_payload_indices.size(),
          tree.max_depth(),
          tree.root_bounds(),
          device_origins.data(),
          device_directions.data(),
          origins.size(),
          options,
          intervals,
          stream);
    };
    auto forward = [&]() {
      svo::render_volume_from_intervals_cuda(
          intervals,
          device_sigma.data(),
          device_color.data(),
          device_rgb.data(),
          device_depth.data(),
          device_opacity.data(),
          origins.size(),
          sigma.size(),
          options,
          stream);
    };
    auto backward = [&]() {
      svo::render_volume_backward_from_intervals_cuda(
          intervals,
          device_sigma.data(),
          device_color.data(),
          device_grad_rgb.data(),
          device_grad_opacity.data(),
          device_grad_sigma.data(),
          device_grad_color.data(),
          origins.size(),
          sigma.size(),
          options,
          stream);
    };

    build_intervals();
    forward();
    if (run_backward(config)) {
      backward();
    }
    svo_bench::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize interval render warmup");
    run.metrics.interval_build_ms = svo_bench::time_cuda_ms(stream, [&]() {
      for (int iteration = 0; iteration < config.iterations; ++iteration) {
        build_intervals();
      }
    }) / static_cast<double>(config.iterations);
    if (run_forward(config)) {
      run.metrics.kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
          forward();
        }
      }) / static_cast<double>(config.iterations);
    } else {
      forward();
      svo_bench::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize interval forward before backward");
    }
    if (run_backward(config)) {
      run.metrics.backward_kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
          backward();
        }
      }) / static_cast<double>(config.iterations);
    }
    run.metrics.interval_count = static_cast<std::uint64_t>(intervals.interval_count);
  } else {
    time_render(
        stream,
        config,
        [&]() {
          svo::render_volume_cuda(
              device_nodes.data(),
              device_nodes.size(),
              device_payload_indices.data(),
              device_payload_indices.size(),
              tree.max_depth(),
              tree.root_bounds(),
              device_origins.data(),
              device_directions.data(),
              device_sigma.data(),
              device_color.data(),
              device_rgb.data(),
              device_depth.data(),
              device_opacity.data(),
              origins.size(),
              sigma.size(),
              options,
              stream);
        },
        [&]() {
          svo::render_volume_backward_cuda(
              device_nodes.data(),
              device_nodes.size(),
              device_payload_indices.data(),
              device_payload_indices.size(),
              tree.max_depth(),
              tree.root_bounds(),
              device_origins.data(),
              device_directions.data(),
              device_sigma.data(),
              device_color.data(),
              device_grad_rgb.data(),
              device_grad_opacity.data(),
              device_grad_sigma.data(),
              device_grad_color.data(),
              origins.size(),
              sigma.size(),
              options,
              stream);
        },
        run.metrics);
  }

  run.metrics.d2h_ms = svo_bench::time_cuda_ms(stream, [&]() {
    if (config.profile) {
      device_stats.copy_to_host(&run.stats, 1, stream);
    }
    if (config.render_strategy == "intervals" && !intervals.counts.empty()) {
      const std::vector<std::uint32_t> interval_counts = intervals.counts.to_host(stream);
      for (std::uint32_t value : interval_counts) {
        run.metrics.max_intervals_per_ray = std::max(run.metrics.max_intervals_per_ray, static_cast<std::uint64_t>(value));
      }
    }
  });
  run.metrics.total_wall_ms = run.metrics.h2d_ms + run.metrics.interval_build_ms + run.metrics.kernel_ms +
      run.metrics.backward_kernel_ms + run.metrics.d2h_ms;
  return run;
}

RenderRun run_wide4(
    const svo::Octree& tree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    const std::vector<float>& sigma,
    const std::vector<float>& color,
    cudaStream_t stream,
    const svo_bench::BenchmarkConfig& config) {
  RenderRun run;
  svo::RenderOptions cpu_options;
  svo::TraversalStats cpu_stats;
  cpu_options.stats = config.profile ? &cpu_stats : nullptr;
  run.metrics.cpu_ms = svo_bench::time_wall_ms([&]() {
    (void)svo::render_volume_cpu(tree, origins, directions, sigma.data(), color.data(), sigma.size(), cpu_options);
  });

  svo::DeviceBuffer<svo::WideNodeDescriptor> device_nodes(tree.wide_nodes().size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_payload_indices(tree.leaf_payload_indices().size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_origins(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_directions(directions.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_sigma(sigma.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_color(color.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_depth(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_opacity(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_grad_rgb(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_grad_opacity(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_grad_sigma(sigma.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_grad_color(color.size(), svo::Device::CUDA);
  svo::DeviceBuffer<svo::TraversalStats> device_stats(1, svo::Device::CUDA);
  svo::DeviceCoarseOccupancyGrid device_coarse;
  svo::CoarseOccupancyDeviceView coarse_view;
  const std::vector<glm::vec3> grad_rgb(origins.size(), {1.0f, 1.0f, 1.0f});
  const std::vector<float> grad_opacity(origins.size(), 1.0f);
  svo::TraversalStats zero_stats{};
  const bool use_coarse = config.empty_space_accelerator == "coarse";
  if (use_coarse && config.render_strategy == "intervals") {
    throw std::runtime_error("--empty-space-accelerator coarse is not wired for --render-strategy intervals");
  }
  const svo::CoarseOccupancyGrid coarse_grid =
      use_coarse ? svo::CoarseOccupancyGrid::from_octree(tree, config.coarse_resolution) : svo::CoarseOccupancyGrid{};
  run.metrics.coarse_occupancy_bytes = use_coarse ? static_cast<std::uint64_t>(coarse_grid.size_bytes()) : 0u;

  run.metrics.h2d_ms = svo_bench::time_cuda_ms(stream, [&]() {
    device_nodes.copy_from_host(tree.wide_nodes().data(), tree.wide_nodes().size(), stream);
    device_payload_indices.copy_from_host(tree.leaf_payload_indices().data(), tree.leaf_payload_indices().size(), stream);
    device_origins.copy_from_host(origins.data(), origins.size(), stream);
    device_directions.copy_from_host(directions.data(), directions.size(), stream);
    device_sigma.copy_from_host(sigma.data(), sigma.size(), stream);
    device_color.copy_from_host(color.data(), color.size(), stream);
    device_grad_rgb.copy_from_host(grad_rgb.data(), grad_rgb.size(), stream);
    device_grad_opacity.copy_from_host(grad_opacity.data(), grad_opacity.size(), stream);
    device_stats.copy_from_host(&zero_stats, 1, stream);
    if (use_coarse) {
      device_coarse.upload(coarse_grid, stream);
      coarse_view = device_coarse.view();
    }
    svo_bench::check(cudaMemsetAsync(device_grad_sigma.data(), 0, sigma.size() * sizeof(float), stream), "cudaMemsetAsync grad_sigma");
    svo_bench::check(cudaMemsetAsync(device_grad_color.data(), 0, color.size() * sizeof(float), stream), "cudaMemsetAsync grad_color");
  });

  svo::RenderOptions options;
  options.stats = config.profile ? device_stats.data() : nullptr;
  options.coarse_occupancy = use_coarse ? &coarse_view : nullptr;
  svo::RenderIntervalBuffer intervals;
  if (config.render_strategy == "intervals") {
    auto build_intervals = [&]() {
      svo::build_render_intervals_wide_cuda(
          device_nodes.data(),
          device_nodes.size(),
          device_payload_indices.data(),
          device_payload_indices.size(),
          tree.max_depth(),
          tree.root_bounds(),
          device_origins.data(),
          device_directions.data(),
          origins.size(),
          options,
          intervals,
          stream);
    };
    auto forward = [&]() {
      svo::render_volume_from_intervals_cuda(
          intervals,
          device_sigma.data(),
          device_color.data(),
          device_rgb.data(),
          device_depth.data(),
          device_opacity.data(),
          origins.size(),
          sigma.size(),
          options,
          stream);
    };
    auto backward = [&]() {
      svo::render_volume_backward_from_intervals_cuda(
          intervals,
          device_sigma.data(),
          device_color.data(),
          device_grad_rgb.data(),
          device_grad_opacity.data(),
          device_grad_sigma.data(),
          device_grad_color.data(),
          origins.size(),
          sigma.size(),
          options,
          stream);
    };

    build_intervals();
    forward();
    if (run_backward(config)) {
      backward();
    }
    svo_bench::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize interval render warmup");
    run.metrics.interval_build_ms = svo_bench::time_cuda_ms(stream, [&]() {
      for (int iteration = 0; iteration < config.iterations; ++iteration) {
        build_intervals();
      }
    }) / static_cast<double>(config.iterations);
    if (run_forward(config)) {
      run.metrics.kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
          forward();
        }
      }) / static_cast<double>(config.iterations);
    } else {
      forward();
      svo_bench::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize interval forward before backward");
    }
    if (run_backward(config)) {
      run.metrics.backward_kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
          backward();
        }
      }) / static_cast<double>(config.iterations);
    }
    run.metrics.interval_count = static_cast<std::uint64_t>(intervals.interval_count);
  } else {
    time_render(
        stream,
        config,
        [&]() {
          svo::render_volume_wide_cuda(
              device_nodes.data(),
              device_nodes.size(),
              device_payload_indices.data(),
              device_payload_indices.size(),
              tree.max_depth(),
              tree.root_bounds(),
              device_origins.data(),
              device_directions.data(),
              device_sigma.data(),
              device_color.data(),
              device_rgb.data(),
              device_depth.data(),
              device_opacity.data(),
              origins.size(),
              sigma.size(),
              options,
              stream);
        },
        [&]() {
          svo::render_volume_backward_wide_cuda(
              device_nodes.data(),
              device_nodes.size(),
              device_payload_indices.data(),
              device_payload_indices.size(),
              tree.max_depth(),
              tree.root_bounds(),
              device_origins.data(),
              device_directions.data(),
              device_sigma.data(),
              device_color.data(),
              device_grad_rgb.data(),
              device_grad_opacity.data(),
              device_grad_sigma.data(),
              device_grad_color.data(),
              origins.size(),
              sigma.size(),
              options,
              stream);
        },
        run.metrics);
  }

  run.metrics.d2h_ms = svo_bench::time_cuda_ms(stream, [&]() {
    if (config.profile) {
      device_stats.copy_to_host(&run.stats, 1, stream);
    }
    if (config.render_strategy == "intervals" && !intervals.counts.empty()) {
      const std::vector<std::uint32_t> interval_counts = intervals.counts.to_host(stream);
      for (std::uint32_t value : interval_counts) {
        run.metrics.max_intervals_per_ray = std::max(run.metrics.max_intervals_per_ray, static_cast<std::uint64_t>(value));
      }
    }
  });
  run.metrics.total_wall_ms = run.metrics.h2d_ms + run.metrics.interval_build_ms + run.metrics.kernel_ms +
      run.metrics.backward_kernel_ms + run.metrics.d2h_ms;
  return run;
}

void print_result(
    const svo_bench::BenchmarkConfig& config,
    const svo::Octree& tree,
    const RenderRun& run,
    std::string_view branching) {
  const double active_kernel_ms = run_forward(config) && !run_backward(config)
      ? run.metrics.kernel_ms
      : run_backward(config) && !run_forward(config)
      ? run.metrics.backward_kernel_ms
      : run.metrics.kernel_ms + run.metrics.backward_kernel_ms;
  const double pixels_per_second = active_kernel_ms > 0.0 ? (static_cast<double>(config.count) / active_kernel_ms) * 1000.0 : 0.0;
  const double fps = active_kernel_ms > 0.0 ? 1000.0 / active_kernel_ms : 0.0;
  std::cout << "render " << branching << '\n';
  std::cout << "  operation: " << (config.operation == "default" ? "both" : config.operation) << '\n';
  std::cout << "  render_strategy: " << config.render_strategy << '\n';
  std::cout << "  max_depth: " << tree.max_depth() << '\n';
  std::cout << "  nodes: " << svo_bench::total_nodes(tree) << '\n';
  std::cout << "  leaves: " << tree.num_leaves() << '\n';
  std::cout << "  pixels: " << config.count << '\n';
  std::cout << "  coarse_occupancy_bytes: " << run.metrics.coarse_occupancy_bytes << '\n';
  std::cout << "  cpu_reference_wall_ms: " << run.metrics.cpu_ms << '\n';
  std::cout << "  h2d_ms: " << run.metrics.h2d_ms << '\n';
  std::cout << "  interval_build_ms: " << run.metrics.interval_build_ms << '\n';
  std::cout << "  forward_kernel_ms: " << run.metrics.kernel_ms << '\n';
  std::cout << "  backward_kernel_ms: " << run.metrics.backward_kernel_ms << '\n';
  std::cout << "  d2h_ms: " << run.metrics.d2h_ms << '\n';
  std::cout << "  total_wall_ms: " << run.metrics.total_wall_ms << '\n';
  if (config.render_strategy == "intervals") {
    const double avg_intervals = config.count > 0
        ? static_cast<double>(run.metrics.interval_count) / static_cast<double>(config.count)
        : 0.0;
    std::cout << "  interval_count: " << run.metrics.interval_count << '\n';
    std::cout << "  avg_intervals_per_ray: " << avg_intervals << '\n';
    std::cout << "  max_intervals_per_ray: " << run.metrics.max_intervals_per_ray << '\n';
  }
  std::cout << "  pixels_per_second: " << pixels_per_second << '\n';
  std::cout << "  fps: " << fps << '\n';
  if (config.profile) {
    svo_bench::print_stats(run.stats);
  }
  const std::string operation_name = std::string("render_") + (config.operation == "default" ? "both" : config.operation);
  svo_bench::append_jsonl(config, tree, operation_name, run.metrics, pixels_per_second, run.stats, "pixels_per_second");
}

}  // namespace

int main(int argc, char** argv) {
  const svo_bench::BenchmarkConfig config = svo_bench::parse_config(argc, argv, 1u << 18u);
  validate_operation(config);
  svo_bench::print_common_header(config, "render");
  std::cout << "  operation: " << (config.operation == "default" ? "both" : config.operation) << '\n';

  if (!config.scene_file.empty()) {
    const svo::SerializedScene scene = svo::load_svo(config.scene_file);
    std::vector<float> sigma;
    std::vector<float> color;
    payloads_from_scene(scene, sigma, color);
    std::vector<glm::vec3> origins;
    std::vector<glm::vec3> directions;
    svo_bench::make_rays_for_bounds(config.count, config.seed, scene.tree.root_bounds(), origins, directions);
    svo_bench::CudaStream stream;
    if (scene.tree.branching() == svo::BranchingMode::Wide4) {
      print_result(config, scene.tree, run_wide4(scene.tree, origins, directions, sigma, color, stream.get(), config), "wide4");
    } else {
      print_result(config, scene.tree, run_octree8(scene.tree, origins, directions, sigma, color, stream.get(), config), "octree8");
    }
    return 0;
  }

  const std::vector<glm::ivec3> coordinates = svo_bench::make_scene(config);
  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> directions;
  svo_bench::make_rays(config.count, config.seed, origins, directions);
  svo_bench::CudaStream stream;

  if (config.branching == "octree8" || config.branching == "both") {
    const svo::Octree tree = svo_bench::build_tree(coordinates, config, "octree8");
    std::vector<float> sigma;
    std::vector<float> color;
    svo_bench::make_payloads(tree, sigma, color);
    print_result(config, tree, run_octree8(tree, origins, directions, sigma, color, stream.get(), config), "octree8");
  }
  if (config.branching == "wide4" || config.branching == "both") {
    const svo::Octree tree = svo_bench::build_tree(coordinates, config, "wide4");
    std::vector<float> sigma;
    std::vector<float> color;
    svo_bench::make_payloads(tree, sigma, color);
    print_result(config, tree, run_wide4(tree, origins, directions, sigma, color, stream.get(), config), "wide4");
  }
  return 0;
}
