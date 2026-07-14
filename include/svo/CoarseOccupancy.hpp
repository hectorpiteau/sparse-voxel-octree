#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <svo/Math.hpp>
#include <svo/Octree.hpp>

#if SVO_ENABLE_CUDA
#include <svo/DeviceBuffer.hpp>
#endif

namespace svo {

class CoarseOccupancyGrid {
 public:
  CoarseOccupancyGrid() = default;
  CoarseOccupancyGrid(int resolution, RootBounds root_bounds);

  static CoarseOccupancyGrid from_octree(const Octree& octree, int resolution);

  int resolution() const noexcept { return resolution_; }
  const RootBounds& root_bounds() const noexcept { return root_bounds_; }
  const std::vector<std::uint32_t>& words() const noexcept { return words_; }
  std::size_t word_count() const noexcept { return words_.size(); }
  std::size_t size_bytes() const noexcept { return words_.size() * sizeof(std::uint32_t); }

  bool occupied(int x, int y, int z) const;
  void mark(int x, int y, int z);
  CoarseOccupancyDeviceView device_view(const std::uint32_t* device_words) const noexcept;

 private:
  int resolution_ = 0;
  RootBounds root_bounds_ = default_root_bounds();
  std::vector<std::uint32_t> words_;
};

bool is_valid_coarse_occupancy_resolution(int resolution) noexcept;

#if SVO_ENABLE_CUDA
class DeviceCoarseOccupancyGrid {
 public:
  DeviceCoarseOccupancyGrid() = default;
  explicit DeviceCoarseOccupancyGrid(const CoarseOccupancyGrid& host_grid, CudaStreamHandle stream = nullptr);

  void upload(const CoarseOccupancyGrid& host_grid, CudaStreamHandle stream = nullptr);
  CoarseOccupancyDeviceView view() const noexcept;
  bool empty() const noexcept { return words_.empty(); }
  std::size_t size_bytes() const noexcept { return words_.size_bytes(); }

 private:
  DeviceBuffer<std::uint32_t> words_;
  int resolution_ = 0;
  RootBounds root_bounds_ = default_root_bounds();
};
#endif

}  // namespace svo
