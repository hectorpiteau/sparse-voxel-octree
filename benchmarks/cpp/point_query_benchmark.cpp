#include "benchmark_common.hpp"

#include <svo/Query.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

struct QueryRun {
  svo_bench::TimingMetrics metrics;
  svo::TraversalStats stats;
  std::size_t hits = 0;
};

template <typename NodeBuffer, typename Launcher>
QueryRun run_cuda_query(
    const svo::Octree& tree,
    const std::vector<glm::vec3>& points,
    const std::vector<std::int32_t>& cpu_results,
    cudaStream_t stream,
    const svo_bench::BenchmarkConfig& config,
    Launcher&& launcher) {
  NodeBuffer device_nodes(tree.branching() == svo::BranchingMode::Wide4 ? tree.wide_nodes().size() : tree.nodes().size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::uint32_t> device_leaf_payload_indices(tree.leaf_payload_indices().size(), svo::Device::CUDA);
  svo::DeviceBuffer<glm::vec3> device_points(points.size(), svo::Device::CUDA);
  svo::DeviceBuffer<std::int32_t> device_results(points.size(), svo::Device::CUDA);
  svo::DeviceBuffer<svo::TraversalStats> device_stats(1, svo::Device::CUDA);
  std::vector<std::int32_t> cuda_results(points.size());
  svo::TraversalStats zero_stats{};

  QueryRun run;
  run.metrics.h2d_ms = svo_bench::time_cuda_ms(stream, [&]() {
    if constexpr (std::is_same_v<typename NodeBuffer::value_type, svo::WideNodeDescriptor>) {
      device_nodes.copy_from_host(tree.wide_nodes().data(), tree.wide_nodes().size(), stream);
    } else {
      device_nodes.copy_from_host(tree.nodes().data(), tree.nodes().size(), stream);
    }
    device_leaf_payload_indices.copy_from_host(
        tree.leaf_payload_indices().data(), tree.leaf_payload_indices().size(), stream);
    device_points.copy_from_host(points.data(), points.size(), stream);
    device_stats.copy_from_host(&zero_stats, 1, stream);
  });

  svo::QueryOptions options;
  options.stats = config.profile ? device_stats.data() : nullptr;
  auto launch = [&]() {
    launcher(
        device_nodes.data(),
        device_nodes.size(),
        device_leaf_payload_indices.data(),
        device_leaf_payload_indices.size(),
        tree.max_depth(),
        tree.root_bounds(),
        device_points.data(),
        device_results.data(),
        device_results.size(),
        options,
        stream);
  };

  launch();
  svo_bench::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize query warmup");
  run.metrics.kernel_ms = svo_bench::time_cuda_ms(stream, [&]() {
    for (int iteration = 0; iteration < config.iterations; ++iteration) {
      launch();
    }
  }) / static_cast<double>(config.iterations);

  run.metrics.d2h_ms = svo_bench::time_cuda_ms(stream, [&]() {
    device_results.copy_to_host(cuda_results.data(), cuda_results.size(), stream);
    if (config.profile) {
      device_stats.copy_to_host(&run.stats, 1, stream);
    }
  });
  run.metrics.total_wall_ms = run.metrics.h2d_ms + run.metrics.kernel_ms + run.metrics.d2h_ms;

  if (cuda_results != cpu_results) {
    throw std::runtime_error("CUDA query results do not match CPU reference");
  }
  for (std::int32_t value : cuda_results) {
    run.hits += value >= 0 ? 1u : 0u;
  }
  return run;
}

QueryRun run_branch(
    const svo::Octree& tree,
    const std::vector<glm::vec3>& points,
    cudaStream_t stream,
    const svo_bench::BenchmarkConfig& config) {
  svo::QueryOptions cpu_options;
  svo::TraversalStats cpu_stats;
  cpu_options.stats = config.profile ? &cpu_stats : nullptr;
  std::vector<std::int32_t> cpu_results;
  QueryRun run;
  run.metrics.cpu_ms = svo_bench::time_wall_ms([&]() {
    cpu_results = svo::query_points(tree, points, cpu_options);
  });
  const double cpu_ms = run.metrics.cpu_ms;

  if (tree.branching() == svo::BranchingMode::Wide4) {
    run = run_cuda_query<svo::DeviceBuffer<svo::WideNodeDescriptor>>(
        tree,
        points,
        cpu_results,
        stream,
        config,
        svo::query_points_wide_cuda);
  } else {
    run = run_cuda_query<svo::DeviceBuffer<svo::NodeDescriptor>>(
        tree,
        points,
        cpu_results,
        stream,
        config,
        svo::query_points_cuda);
  }
  run.metrics.cpu_ms = cpu_ms;
  if (!config.profile) {
    run.stats = {};
  }
  return run;
}

void print_result(
    const svo_bench::BenchmarkConfig& config,
    const svo::Octree& tree,
    const QueryRun& run,
    std::string_view branching) {
  const double points_per_second = (static_cast<double>(config.count) / run.metrics.kernel_ms) * 1000.0;
  std::cout << "point query " << branching << '\n';
  std::cout << "  max_depth: " << tree.max_depth() << '\n';
  std::cout << "  nodes: " << svo_bench::total_nodes(tree) << '\n';
  std::cout << "  leaves: " << tree.num_leaves() << '\n';
  std::cout << "  points: " << config.count << '\n';
  std::cout << "  hits: " << run.hits << '\n';
  std::cout << "  cpu_reference_wall_ms: " << run.metrics.cpu_ms << '\n';
  std::cout << "  h2d_ms: " << run.metrics.h2d_ms << '\n';
  std::cout << "  kernel_ms: " << run.metrics.kernel_ms << '\n';
  std::cout << "  d2h_ms: " << run.metrics.d2h_ms << '\n';
  std::cout << "  total_wall_ms: " << run.metrics.total_wall_ms << '\n';
  std::cout << "  points_per_second: " << points_per_second << '\n';
  if (config.profile) {
    svo_bench::print_stats(run.stats);
  }
  svo_bench::append_jsonl(config, tree, "point_query", run.metrics, points_per_second, run.stats, "points_per_second");
}

}  // namespace

int main(int argc, char** argv) {
  const svo_bench::BenchmarkConfig config = svo_bench::parse_config(argc, argv, 1u << 20u);
  svo_bench::print_common_header(config, "point query");
  const std::vector<glm::ivec3> coordinates = svo_bench::make_scene(config);
  const std::vector<glm::vec3> points = svo_bench::make_points(config.count, config.grid_size, config.seed);
  svo_bench::CudaStream stream;

  if (config.branching == "octree8" || config.branching == "both") {
    const svo::Octree tree = svo_bench::build_tree(coordinates, config, "octree8");
    print_result(config, tree, run_branch(tree, points, stream.get(), config), "octree8");
  }
  if (config.branching == "wide4" || config.branching == "both") {
    const svo::Octree tree = svo_bench::build_tree(coordinates, config, "wide4");
    print_result(config, tree, run_branch(tree, points, stream.get(), config), "wide4");
  }
  return 0;
}
