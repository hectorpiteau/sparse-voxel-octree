#include "benchmark_common.hpp"

#include <svo/Raycast.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

struct RaycastRun {
  svo_bench::TimingMetrics metrics;
  svo::TraversalStats stats;
  std::size_t hits = 0;
};

RaycastRun run_octree8(
    const svo::Octree& tree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    cudaStream_t stream,
    const svo_bench::BenchmarkConfig& config) {
  RaycastRun run;
  svo::RaycastOptions cpu_options;
  svo::TraversalStats cpu_stats;
  cpu_options.stats = config.profile ? &cpu_stats : nullptr;
  svo::RaycastBatch cpu_results;
  run.metrics.cpu_ms = svo_bench::time_wall_ms([&]() {
    cpu_results = svo::raycast_cpu(tree, origins, directions, cpu_options);
  });

  svo::DeviceBuffer<svo::NodeDescriptor> device_nodes(tree.nodes().size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_payload_indices(tree.leaf_payload_indices().size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_origins(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_directions(directions.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint8_t> device_hit_mask(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_leaf_ids(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_t(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_positions(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_depths(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<svo::TraversalStats> device_stats(1, svo::Device::CUDA);
  svo::TraversalStats zero_stats{};
  std::vector<std::uint8_t> hit_mask(origins.size());

  run.metrics.h2d_ms = svo_bench::time_cuda_ms(stream, [&]() {
    device_nodes.copy_from_host(tree.nodes().data(), tree.nodes().size(), stream);
    device_payload_indices.copy_from_host(tree.leaf_payload_indices().data(), tree.leaf_payload_indices().size(), stream);
    device_origins.copy_from_host(origins.data(), origins.size(), stream);
    device_directions.copy_from_host(directions.data(), directions.size(), stream);
    device_stats.copy_from_host(&zero_stats, 1, stream);
  });

  svo::RaycastOptions options;
  options.stats = config.profile ? device_stats.data() : nullptr;
  auto launch = [&]() {
    svo::raycast_cuda(
        device_nodes.data(),
        device_nodes.size(),
        device_payload_indices.data(),
        device_payload_indices.size(),
        tree.max_depth(),
        tree.root_bounds(),
        device_origins.data(),
        device_directions.data(),
        device_hit_mask.data(),
        device_leaf_ids.data(),
        device_t.data(),
        device_positions.data(),
        device_depths.data(),
        origins.size(),
        options,
        stream);
  };
  launch();
  svo_bench::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize raycast warmup");
  run.metrics.kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
    for (int iteration = 0; iteration < config.iterations; ++iteration) {
      launch();
    }
  }) / static_cast<double>(config.iterations);

  run.metrics.d2h_ms = svo_bench::time_cuda_ms(stream, [&]() {
    device_hit_mask.copy_to_host(hit_mask.data(), hit_mask.size(), stream);
    if (config.profile) {
      device_stats.copy_to_host(&run.stats, 1, stream);
    }
  });
  run.metrics.total_wall_ms = run.metrics.h2d_ms + run.metrics.kernel_ms + run.metrics.d2h_ms;
  if (hit_mask != cpu_results.hit_mask) {
    throw std::runtime_error("octree8 CUDA hit mask does not match CPU reference");
  }
  for (std::uint8_t value : hit_mask) {
    run.hits += value != 0u ? 1u : 0u;
  }
  return run;
}

RaycastRun run_wide4(
    const svo::Octree& tree,
    const std::vector<glm::vec3>& origins,
    const std::vector<glm::vec3>& directions,
    cudaStream_t stream,
    const svo_bench::BenchmarkConfig& config) {
  RaycastRun run;
  svo::RaycastOptions cpu_options;
  svo::TraversalStats cpu_stats;
  cpu_options.stats = config.profile ? &cpu_stats : nullptr;
  svo::RaycastBatch cpu_results;
  run.metrics.cpu_ms = svo_bench::time_wall_ms([&]() {
    cpu_results = svo::raycast_cpu(tree, origins, directions, cpu_options);
  });

  svo::DeviceBuffer<svo::WideNodeDescriptor> device_nodes(tree.wide_nodes().size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_payload_indices(tree.leaf_payload_indices().size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_origins(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_directions(directions.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint8_t> device_hit_mask(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_leaf_ids(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<float> device_t(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_positions(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_depths(origins.size(), svo::Device::CUDA);
  svo::DeviceBuffer<svo::TraversalStats> device_stats(1, svo::Device::CUDA);
  svo::TraversalStats zero_stats{};
  std::vector<std::uint8_t> hit_mask(origins.size());

  run.metrics.h2d_ms = svo_bench::time_cuda_ms(stream, [&]() {
    device_nodes.copy_from_host(tree.wide_nodes().data(), tree.wide_nodes().size(), stream);
    device_payload_indices.copy_from_host(tree.leaf_payload_indices().data(), tree.leaf_payload_indices().size(), stream);
    device_origins.copy_from_host(origins.data(), origins.size(), stream);
    device_directions.copy_from_host(directions.data(), directions.size(), stream);
    device_stats.copy_from_host(&zero_stats, 1, stream);
  });

  svo::RaycastOptions options;
  options.stats = config.profile ? device_stats.data() : nullptr;
  auto launch = [&]() {
    svo::raycast_wide_cuda(
        device_nodes.data(),
        device_nodes.size(),
        device_payload_indices.data(),
        device_payload_indices.size(),
        tree.max_depth(),
        tree.root_bounds(),
        device_origins.data(),
        device_directions.data(),
        device_hit_mask.data(),
        device_leaf_ids.data(),
        device_t.data(),
        device_positions.data(),
        device_depths.data(),
        origins.size(),
        options,
        stream);
  };
  launch();
  svo_bench::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize raycast wide warmup");
  run.metrics.kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
    for (int iteration = 0; iteration < config.iterations; ++iteration) {
      launch();
    }
  }) / static_cast<double>(config.iterations);

  run.metrics.d2h_ms = svo_bench::time_cuda_ms(stream, [&]() {
    device_hit_mask.copy_to_host(hit_mask.data(), hit_mask.size(), stream);
    if (config.profile) {
      device_stats.copy_to_host(&run.stats, 1, stream);
    }
  });
  run.metrics.total_wall_ms = run.metrics.h2d_ms + run.metrics.kernel_ms + run.metrics.d2h_ms;
  if (hit_mask != cpu_results.hit_mask) {
    throw std::runtime_error("wide4 CUDA hit mask does not match CPU reference");
  }
  for (std::uint8_t value : hit_mask) {
    run.hits += value != 0u ? 1u : 0u;
  }
  return run;
}

void print_result(
    const svo_bench::BenchmarkConfig& config,
    const svo::Octree& tree,
    const RaycastRun& run,
    std::string_view branching) {
  const double rays_per_second = (static_cast<double>(config.count) / run.metrics.kernel_ms) * 1000.0;
  std::cout << "raycast " << branching << '\n';
  std::cout << "  max_depth: " << tree.max_depth() << '\n';
  std::cout << "  nodes: " << svo_bench::total_nodes(tree) << '\n';
  std::cout << "  leaves: " << tree.num_leaves() << '\n';
  std::cout << "  rays: " << config.count << '\n';
  std::cout << "  hits: " << run.hits << '\n';
  std::cout << "  hit_rate: " << static_cast<double>(run.hits) / static_cast<double>(config.count) << '\n';
  std::cout << "  cpu_reference_wall_ms: " << run.metrics.cpu_ms << '\n';
  std::cout << "  h2d_ms: " << run.metrics.h2d_ms << '\n';
  std::cout << "  kernel_ms: " << run.metrics.kernel_ms << '\n';
  std::cout << "  d2h_ms: " << run.metrics.d2h_ms << '\n';
  std::cout << "  total_wall_ms: " << run.metrics.total_wall_ms << '\n';
  std::cout << "  rays_per_second: " << rays_per_second << '\n';
  if (config.profile) {
    svo_bench::print_stats(run.stats);
  }
  svo_bench::append_jsonl(config, tree, "raycast", run.metrics, rays_per_second, run.stats, "rays_per_second");
}

}  // namespace

int main(int argc, char** argv) {
  const svo_bench::BenchmarkConfig config = svo_bench::parse_config(argc, argv, 1u << 20u);
  svo_bench::print_common_header(config, "raycast");
  const std::vector<glm::ivec3> coordinates = svo_bench::make_scene(config);
  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> directions;
  svo_bench::make_rays(config.count, config.seed, origins, directions);
  svo_bench::CudaStream stream;

  if (config.branching == "octree8" || config.branching == "both") {
    const svo::Octree tree = svo_bench::build_tree(coordinates, config, "octree8");
    print_result(config, tree, run_octree8(tree, origins, directions, stream.get(), config), "octree8");
  }
  if (config.branching == "wide4" || config.branching == "both") {
    const svo::Octree tree = svo_bench::build_tree(coordinates, config, "wide4");
    print_result(config, tree, run_wide4(tree, origins, directions, stream.get(), config), "wide4");
  }
  return 0;
}
