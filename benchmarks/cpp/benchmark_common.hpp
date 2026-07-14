#pragma once

#include <svo/CoarseOccupancy.hpp>
#include <svo/DeviceBuffer.hpp>
#include <svo/Octree.hpp>
#include <svo/Serialization.hpp>

#include <cuda_runtime_api.h>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_int3.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svo_bench {

constexpr int kSceneVersion = 1;
constexpr std::uint32_t kDefaultSeed = 20260712u;

struct BenchmarkConfig {
  std::string operation = "default";
  std::string scene = "sparse_random";
  std::string scene_file;
  int grid_size = 64;
  std::string branching = "both";
  std::string render_strategy = "direct";
  std::string empty_space_accelerator = "none";
  int coarse_resolution = 32;
  std::uint32_t seed = kDefaultSeed;
  double density = 0.035;
  int iterations = 20;
  std::size_t count = 1u << 20u;
  std::string jsonl_path;
  bool profile = false;
};

struct TimingMetrics {
  double cpu_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_wall_ms = 0.0;
  double backward_kernel_ms = 0.0;
  double interval_build_ms = 0.0;
  std::uint64_t interval_count = 0;
  std::uint64_t max_intervals_per_ray = 0;
  std::uint64_t coarse_occupancy_bytes = 0;
};

inline void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

class CudaStream {
 public:
  CudaStream() { check(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
  ~CudaStream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { check(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() {
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

inline float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()), "cudaEventElapsedTime");
  return milliseconds;
}

template <typename F>
float time_cuda_ms(cudaStream_t stream, F&& function) {
  CudaEvent start;
  CudaEvent stop;
  check(cudaEventRecord(start.get(), stream), "cudaEventRecord start");
  function();
  check(cudaEventRecord(stop.get(), stream), "cudaEventRecord stop");
  check(cudaEventSynchronize(stop.get()), "cudaEventSynchronize stop");
  return elapsed_ms(start, stop);
}

template <typename F>
double time_wall_ms(F&& function) {
  const auto start = std::chrono::steady_clock::now();
  function();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

inline int parse_max_depth(int grid_size, std::string_view branching) {
  if (grid_size <= 0 || (grid_size & (grid_size - 1)) != 0) {
    throw std::runtime_error("--grid-size must be a positive power of two");
  }
  int depth = 0;
  for (int value = grid_size; value > 1; value >>= 1) {
    ++depth;
  }
  if (branching == "wide4" && (depth % 2) != 0) {
    throw std::runtime_error("--grid-size must produce an even max depth for wide4");
  }
  return depth;
}

inline std::string require_value(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string(argv[index]) + " requires a value");
  }
  ++index;
  return argv[index];
}

inline BenchmarkConfig parse_config(int argc, char** argv, std::size_t default_count) {
  BenchmarkConfig config;
  config.count = default_count;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--operation") {
      config.operation = require_value(index, argc, argv);
    } else if (flag == "--scene") {
      config.scene = require_value(index, argc, argv);
    } else if (flag == "--scene-file") {
      config.scene_file = require_value(index, argc, argv);
    } else if (flag == "--grid-size") {
      config.grid_size = std::stoi(require_value(index, argc, argv));
    } else if (flag == "--branching") {
      config.branching = require_value(index, argc, argv);
    } else if (flag == "--render-strategy") {
      config.render_strategy = require_value(index, argc, argv);
    } else if (flag == "--empty-space-accelerator") {
      config.empty_space_accelerator = require_value(index, argc, argv);
    } else if (flag == "--coarse-resolution") {
      config.coarse_resolution = std::stoi(require_value(index, argc, argv));
    } else if (flag == "--seed") {
      config.seed = static_cast<std::uint32_t>(std::stoul(require_value(index, argc, argv)));
    } else if (flag == "--density") {
      config.density = std::stod(require_value(index, argc, argv));
    } else if (flag == "--iterations") {
      config.iterations = std::stoi(require_value(index, argc, argv));
    } else if (flag == "--jsonl") {
      config.jsonl_path = require_value(index, argc, argv);
    } else if (flag == "--count" || flag == "--points" || flag == "--rays") {
      config.count = static_cast<std::size_t>(std::stoull(require_value(index, argc, argv)));
    } else if (flag == "--profile") {
      config.profile = true;
    } else if (index == 1 && !flag.empty() && flag[0] != '-') {
      config.count = static_cast<std::size_t>(std::stoull(flag));
    } else {
      throw std::runtime_error("unknown benchmark argument: " + flag);
    }
  }

  if (config.iterations <= 0) {
    throw std::runtime_error("--iterations must be positive");
  }
  if (config.count == 0) {
    throw std::runtime_error("--count must be positive");
  }
  if (config.branching != "octree8" && config.branching != "wide4" && config.branching != "both") {
    throw std::runtime_error("--branching must be octree8, wide4, or both");
  }
  if (config.render_strategy == "auto") {
    config.render_strategy = "direct";
  }
  if (config.render_strategy != "direct" && config.render_strategy != "intervals") {
    throw std::runtime_error("--render-strategy must be direct, intervals, or auto");
  }
  if (config.empty_space_accelerator != "none" && config.empty_space_accelerator != "coarse") {
    throw std::runtime_error("--empty-space-accelerator must be none or coarse");
  }
  if (config.empty_space_accelerator == "coarse" &&
      !svo::is_valid_coarse_occupancy_resolution(config.coarse_resolution)) {
    throw std::runtime_error("--coarse-resolution must be 16, 32, or 64");
  }
  if (config.density < 0.0 || config.density > 1.0) {
    throw std::runtime_error("--density must be in [0, 1]");
  }
  if (!config.scene_file.empty()) {
    config.scene = "file";
  } else {
    parse_max_depth(config.grid_size, config.branching == "wide4" ? "wide4" : "octree8");
  }
  return config;
}

inline std::uint64_t mix_seed(std::uint32_t seed, int grid_size, double density) {
  const auto density_bits = static_cast<std::uint64_t>(density * 1000000.0);
  std::uint64_t value = static_cast<std::uint64_t>(seed) ^ (static_cast<std::uint64_t>(kSceneVersion) << 32u);
  value ^= static_cast<std::uint64_t>(grid_size) * 0x9e3779b97f4a7c15ull;
  value ^= density_bits + 0xbf58476d1ce4e5b9ull + (value << 6u) + (value >> 2u);
  return value;
}

inline std::vector<glm::ivec3> make_scene(const BenchmarkConfig& config) {
  const int grid_size = config.grid_size;
  std::vector<glm::ivec3> coordinates;
  if (config.scene == "empty") {
    return coordinates;
  }
  if (config.scene == "single_voxel") {
    coordinates.push_back({grid_size / 2, grid_size / 2, grid_size / 2});
    return coordinates;
  }

  if (config.scene == "dense_cube") {
    const int begin = grid_size / 4;
    const int end = (3 * grid_size) / 4;
    coordinates.reserve(static_cast<std::size_t>(end - begin) * static_cast<std::size_t>(end - begin) *
                        static_cast<std::size_t>(end - begin));
    for (int z = begin; z < end; ++z) {
      for (int y = begin; y < end; ++y) {
        for (int x = begin; x < end; ++x) {
          coordinates.push_back({x, y, z});
        }
      }
    }
    return coordinates;
  }

  if (config.scene == "sphere") {
    const float radius = 0.28f;
    for (int z = 0; z < grid_size; ++z) {
      for (int y = 0; y < grid_size; ++y) {
        for (int x = 0; x < grid_size; ++x) {
          const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
          const float ny = (static_cast<float>(y) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
          const float nz = (static_cast<float>(z) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
          if (nx * nx + ny * ny + nz * nz <= radius * radius) {
            coordinates.push_back({x, y, z});
          }
        }
      }
    }
    return coordinates;
  }

  if (config.scene == "shell") {
    const float inner_radius = 0.24f;
    const float outer_radius = 0.30f;
    for (int z = 0; z < grid_size; ++z) {
      for (int y = 0; y < grid_size; ++y) {
        for (int x = 0; x < grid_size; ++x) {
          const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
          const float ny = (static_cast<float>(y) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
          const float nz = (static_cast<float>(z) + 0.5f) / static_cast<float>(grid_size) - 0.5f;
          const float r2 = nx * nx + ny * ny + nz * nz;
          if (r2 >= inner_radius * inner_radius && r2 <= outer_radius * outer_radius) {
            coordinates.push_back({x, y, z});
          }
        }
      }
    }
    return coordinates;
  }

  if (config.scene != "sparse_random") {
    throw std::runtime_error("--scene must be empty, single_voxel, dense_cube, sphere, shell, or sparse_random");
  }

  std::mt19937_64 rng(mix_seed(config.seed, config.grid_size, config.density));
  std::bernoulli_distribution keep(config.density);
  const int begin = grid_size / 4;
  const int end = (3 * grid_size) / 4;
  for (int z = begin; z < end; ++z) {
    for (int y = begin; y < end; ++y) {
      for (int x = begin; x < end; ++x) {
        if (keep(rng)) {
          coordinates.push_back({x, y, z});
        }
      }
    }
  }
  return coordinates;
}

inline svo::Octree build_tree(const std::vector<glm::ivec3>& coordinates, const BenchmarkConfig& config, std::string_view branching) {
  svo::BuildOptions options;
  options.max_depth = parse_max_depth(config.grid_size, branching);
  options.branching = branching == "wide4" ? svo::BranchingMode::Wide4 : svo::BranchingMode::Octree8;
  return svo::Octree::from_voxels_cpu(coordinates, options);
}

inline glm::vec3 voxel_center_point(int grid_size, const glm::ivec3& coordinate) {
  const float inv_grid = 1.0f / static_cast<float>(grid_size);
  return {
      (static_cast<float>(coordinate.x) + 0.5f) * inv_grid,
      (static_cast<float>(coordinate.y) + 0.5f) * inv_grid,
      (static_cast<float>(coordinate.z) + 0.5f) * inv_grid};
}

inline std::vector<glm::vec3> make_points(std::size_t count, int grid_size, std::uint32_t seed) {
  std::mt19937 rng(seed ^ 0x51f15eedu);
  std::uniform_int_distribution<int> coord_dist(0, grid_size - 1);
  std::vector<glm::vec3> points;
  points.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if ((index % 23u) == 0u) {
      points.push_back({1.0f, 0.5f, 0.5f});
    } else {
      points.push_back(voxel_center_point(grid_size, {coord_dist(rng), coord_dist(rng), coord_dist(rng)}));
    }
  }
  return points;
}

inline void make_rays(
    std::size_t count,
    std::uint32_t seed,
    std::vector<glm::vec3>& origins,
    std::vector<glm::vec3>& directions) {
  std::mt19937 rng(seed ^ 0x7217a57du);
  std::uniform_real_distribution<float> uv_dist(0.05f, 0.95f);
  std::uniform_real_distribution<float> jitter_dist(-0.12f, 0.12f);
  origins.clear();
  directions.clear();
  origins.reserve(count);
  directions.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const float y = uv_dist(rng);
    const float z = uv_dist(rng);
    origins.push_back({-0.35f, y, z});
    directions.push_back(glm::normalize(glm::vec3{1.0f, 0.5f - y + jitter_dist(rng), 0.5f - z + jitter_dist(rng)}));
  }
}

inline void make_rays_for_bounds(
    std::size_t count,
    std::uint32_t seed,
    const svo::RootBounds& bounds,
    std::vector<glm::vec3>& origins,
    std::vector<glm::vec3>& directions) {
  std::mt19937 rng(seed ^ 0x7217a57du);
  std::uniform_real_distribution<float> uv_dist(0.05f, 0.95f);
  std::uniform_real_distribution<float> jitter_dist(-0.12f, 0.12f);
  const glm::vec3 extent = bounds[1] - bounds[0];
  const glm::vec3 center = (bounds[0] + bounds[1]) * 0.5f;
  origins.clear();
  directions.clear();
  origins.reserve(count);
  directions.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const float y = bounds[0].y + uv_dist(rng) * extent.y;
    const float z = bounds[0].z + uv_dist(rng) * extent.z;
    origins.push_back({bounds[0].x - 0.35f * extent.x, y, z});
    directions.push_back(glm::normalize(glm::vec3{
        1.0f,
        (center.y - y) / extent.x + jitter_dist(rng),
        (center.z - z) / extent.x + jitter_dist(rng)}));
  }
}

inline void make_payloads(const svo::Octree& octree, std::vector<float>& sigma, std::vector<float>& color) {
  sigma.resize(octree.num_leaves());
  color.resize(octree.num_leaves() * 3u);
  for (std::size_t index = 0; index < octree.num_leaves(); ++index) {
    const float t = static_cast<float>(index % 97u) / 96.0f;
    sigma[index] = 0.25f + 2.0f * t;
    color[index * 3u + 0u] = 0.5f + 0.5f * std::sin(6.2831853f * t);
    color[index * 3u + 1u] = 0.5f + 0.5f * std::sin(6.2831853f * (t + 0.3333333f));
    color[index * 3u + 2u] = 0.5f + 0.5f * std::sin(6.2831853f * (t + 0.6666667f));
  }
}

inline std::string gpu_name() {
  int device = 0;
  cudaError_t result = cudaGetDevice(&device);
  if (result != cudaSuccess) {
    int count = 0;
    result = cudaGetDeviceCount(&count);
    if (result == cudaSuccess && count > 0) {
      device = 0;
      result = cudaSetDevice(device);
    }
  }
  if (result != cudaSuccess) {
    return std::string("unavailable: ") + cudaGetErrorString(result);
  }
  cudaDeviceProp properties{};
  result = cudaGetDeviceProperties(&properties, device);
  if (result != cudaSuccess) {
    return std::string("unavailable: ") + cudaGetErrorString(result);
  }
  return properties.name;
}

inline std::string cuda_runtime_version() {
  int version = 0;
  check(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
  std::ostringstream out;
  out << (version / 1000) << '.' << ((version % 1000) / 10);
  return out.str();
}

inline std::string build_type() {
#ifdef NDEBUG
  return "Release";
#else
  return "Debug";
#endif
}

inline std::uint64_t total_nodes(const svo::Octree& tree) {
  return tree.branching() == svo::BranchingMode::Wide4 ? tree.wide_nodes().size() : tree.nodes().size();
}

inline std::string json_escape(std::string_view value) {
  std::ostringstream out;
  for (char character : value) {
    if (character == '"' || character == '\\') {
      out << '\\' << character;
    } else if (character == '\n') {
      out << "\\n";
    } else {
      out << character;
    }
  }
  return out.str();
}

inline void write_json_field(std::ostream& out, bool& first, std::string_view name, std::string_view value) {
  if (!first) {
    out << ',';
  }
  first = false;
  out << '"' << name << "\":\"" << json_escape(value) << '"';
}

inline void write_json_field(std::ostream& out, bool& first, std::string_view name, const std::string& value) {
  write_json_field(out, first, name, std::string_view{value});
}

inline void write_json_field(std::ostream& out, bool& first, std::string_view name, const char* value) {
  write_json_field(out, first, name, std::string_view{value});
}

template <typename T>
void write_json_field(std::ostream& out, bool& first, std::string_view name, T value) {
  if (!first) {
    out << ',';
  }
  first = false;
  out << '"' << name << "\":" << value;
}

inline void write_stats_json(std::ostream& out, bool& first, const svo::TraversalStats& stats) {
  write_json_field(out, first, "stats_nodes_visited", stats.nodes_visited);
  write_json_field(out, first, "stats_child_candidates_tested", stats.child_candidates_tested);
  write_json_field(out, first, "stats_leaf_segments", stats.leaf_segments);
  write_json_field(out, first, "stats_early_terminations", stats.early_terminations);
  write_json_field(out, first, "stats_stack_pushes", stats.stack_pushes);
  write_json_field(out, first, "stats_stack_pops", stats.stack_pops);
  write_json_field(out, first, "stats_max_stack_depth", stats.max_stack_depth);
  write_json_field(out, first, "stats_macro_cells_tested", stats.macro_cells_tested);
  write_json_field(out, first, "stats_macro_cells_occupied", stats.macro_cells_occupied);
  write_json_field(out, first, "stats_macro_cells_skipped", stats.macro_cells_skipped);
  write_json_field(out, first, "stats_macro_tree_entries", stats.macro_tree_entries);
}

inline void append_jsonl(
    const BenchmarkConfig& config,
    const svo::Octree& tree,
    std::string_view operation,
    const TimingMetrics& metrics,
    double throughput,
    const svo::TraversalStats& stats,
    std::string_view throughput_name = "items_per_second") {
  if (config.jsonl_path.empty()) {
    return;
  }

  std::ofstream out(config.jsonl_path, std::ios::app);
  if (!out) {
    throw std::runtime_error("failed to open JSONL output path: " + config.jsonl_path);
  }
  bool first = true;
  out << '{';
  write_json_field(out, first, "schema", "svo_benchmark_v1");
  write_json_field(out, first, "operation", operation);
  write_json_field(out, first, "scene", config.scene);
  if (!config.scene_file.empty()) {
    write_json_field(out, first, "scene_file", config.scene_file);
  }
  write_json_field(out, first, "scene_version", kSceneVersion);
  write_json_field(out, first, "seed", config.seed);
  write_json_field(out, first, "density", config.density);
  write_json_field(out, first, "grid_size", config.grid_size);
  write_json_field(out, first, "branching", tree.branching() == svo::BranchingMode::Wide4 ? "wide4" : "octree8");
  write_json_field(out, first, "render_strategy", config.render_strategy);
  write_json_field(out, first, "empty_space_accelerator", config.empty_space_accelerator);
  write_json_field(out, first, "coarse_resolution", config.empty_space_accelerator == "coarse" ? config.coarse_resolution : 0);
  write_json_field(out, first, "max_depth", tree.max_depth());
  write_json_field(out, first, "nodes", total_nodes(tree));
  write_json_field(out, first, "leaves", tree.num_leaves());
  write_json_field(out, first, "iterations", config.iterations);
  write_json_field(out, first, "count", config.count);
  write_json_field(out, first, "build_type", build_type());
  write_json_field(out, first, "cuda_runtime", cuda_runtime_version());
  write_json_field(out, first, "gpu", gpu_name());
  write_json_field(out, first, "cpu_reference_wall_ms", metrics.cpu_ms);
  write_json_field(out, first, "h2d_ms", metrics.h2d_ms);
  write_json_field(out, first, "kernel_ms", metrics.kernel_ms);
  write_json_field(out, first, "interval_build_ms", metrics.interval_build_ms);
  write_json_field(out, first, "backward_kernel_ms", metrics.backward_kernel_ms);
  write_json_field(out, first, "d2h_ms", metrics.d2h_ms);
  write_json_field(out, first, "total_wall_ms", metrics.total_wall_ms);
  write_json_field(out, first, "interval_count", metrics.interval_count);
  write_json_field(out, first, "max_intervals_per_ray", metrics.max_intervals_per_ray);
  write_json_field(out, first, "coarse_occupancy_bytes", metrics.coarse_occupancy_bytes);
  write_json_field(out, first, throughput_name, throughput);
  write_stats_json(out, first, stats);
  out << "}\n";
}

inline void print_common_header(const BenchmarkConfig& config, std::string_view name) {
  std::cout << name << " benchmark\n";
  std::cout << "  scene: " << config.scene << " v" << kSceneVersion << '\n';
  if (!config.scene_file.empty()) {
    std::cout << "  scene_file: " << config.scene_file << '\n';
  }
  std::cout << "  grid_size: " << config.grid_size << '\n';
  std::cout << "  seed: " << config.seed << '\n';
  std::cout << "  density: " << config.density << '\n';
  std::cout << "  iterations: " << config.iterations << '\n';
  std::cout << "  render_strategy: " << config.render_strategy << '\n';
  std::cout << "  empty_space_accelerator: " << config.empty_space_accelerator << '\n';
  if (config.empty_space_accelerator == "coarse") {
    std::cout << "  coarse_resolution: " << config.coarse_resolution << '\n';
  }
  std::cout << "  gpu: " << gpu_name() << '\n';
  std::cout << "  cuda_runtime: " << cuda_runtime_version() << '\n';
}

inline void print_stats(const svo::TraversalStats& stats) {
  std::cout << "  stats.nodes_visited: " << stats.nodes_visited << '\n';
  std::cout << "  stats.child_candidates_tested: " << stats.child_candidates_tested << '\n';
  std::cout << "  stats.leaf_segments: " << stats.leaf_segments << '\n';
  std::cout << "  stats.early_terminations: " << stats.early_terminations << '\n';
  std::cout << "  stats.stack_pushes: " << stats.stack_pushes << '\n';
  std::cout << "  stats.stack_pops: " << stats.stack_pops << '\n';
  std::cout << "  stats.max_stack_depth: " << stats.max_stack_depth << '\n';
  std::cout << "  stats.macro_cells_tested: " << stats.macro_cells_tested << '\n';
  std::cout << "  stats.macro_cells_occupied: " << stats.macro_cells_occupied << '\n';
  std::cout << "  stats.macro_cells_skipped: " << stats.macro_cells_skipped << '\n';
  std::cout << "  stats.macro_tree_entries: " << stats.macro_tree_entries << '\n';
}

}  // namespace svo_bench
