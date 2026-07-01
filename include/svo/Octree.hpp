#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <svo/Math.hpp>

#ifndef SVO_HOST_DEVICE
#if defined(__CUDACC__)
#define SVO_HOST_DEVICE __host__ __device__
#else
#define SVO_HOST_DEVICE
#endif
#endif

namespace svo {

enum class Device : std::uint8_t {
  CPU = 0,
  CUDA = 1,
};

const char* device_name(Device device) noexcept;

using CudaStreamHandle = void*;

struct BuildOptions {
  int max_depth = 0;
  Device device = Device::CPU;
  RootBounds root_bounds = default_root_bounds();
};

struct QueryOptions {
  bool return_payload_indices = false;
};

struct RenderOptions {
  bool enable_empty_space_skipping = true;
};

class NodeDescriptor final {
 public:
  static constexpr std::uint64_t kChildMaskBits = 8;
  static constexpr std::uint64_t kLeafMaskBits = 8;
  static constexpr std::uint64_t kChildBaseBits = 24;
  static constexpr std::uint64_t kPayloadBaseBits = 24;

  static constexpr std::uint64_t kChildMaskShift = 0;
  static constexpr std::uint64_t kLeafMaskShift = kChildMaskShift + kChildMaskBits;
  static constexpr std::uint64_t kChildBaseShift = kLeafMaskShift + kLeafMaskBits;
  static constexpr std::uint64_t kPayloadBaseShift = kChildBaseShift + kChildBaseBits;

  static constexpr std::uint64_t kChildMaskMask = (1ull << kChildMaskBits) - 1ull;
  static constexpr std::uint64_t kLeafMaskMask = (1ull << kLeafMaskBits) - 1ull;
  static constexpr std::uint64_t kChildBaseMask = (1ull << kChildBaseBits) - 1ull;
  static constexpr std::uint64_t kPayloadBaseMask = (1ull << kPayloadBaseBits) - 1ull;

  SVO_HOST_DEVICE constexpr NodeDescriptor() noexcept = default;
  SVO_HOST_DEVICE explicit constexpr NodeDescriptor(std::uint64_t bits) noexcept : bits_(bits) {}

  SVO_HOST_DEVICE static constexpr NodeDescriptor pack(
      std::uint8_t child_mask,
      std::uint8_t leaf_mask,
      std::uint32_t child_base,
      std::uint32_t payload_base) noexcept {
    return NodeDescriptor{
        ((static_cast<std::uint64_t>(child_mask) & kChildMaskMask) << kChildMaskShift) |
        ((static_cast<std::uint64_t>(leaf_mask) & kLeafMaskMask) << kLeafMaskShift) |
        ((static_cast<std::uint64_t>(child_base) & kChildBaseMask) << kChildBaseShift) |
        ((static_cast<std::uint64_t>(payload_base) & kPayloadBaseMask) << kPayloadBaseShift)};
  }

  SVO_HOST_DEVICE constexpr std::uint64_t bits() const noexcept { return bits_; }

  SVO_HOST_DEVICE constexpr std::uint8_t child_mask() const noexcept {
    return static_cast<std::uint8_t>((bits_ >> kChildMaskShift) & kChildMaskMask);
  }

  SVO_HOST_DEVICE constexpr std::uint8_t leaf_mask() const noexcept {
    return static_cast<std::uint8_t>((bits_ >> kLeafMaskShift) & kLeafMaskMask);
  }

  SVO_HOST_DEVICE constexpr std::uint32_t child_base() const noexcept {
    return static_cast<std::uint32_t>((bits_ >> kChildBaseShift) & kChildBaseMask);
  }

  SVO_HOST_DEVICE constexpr std::uint32_t payload_base() const noexcept {
    return static_cast<std::uint32_t>((bits_ >> kPayloadBaseShift) & kPayloadBaseMask);
  }

  SVO_HOST_DEVICE constexpr std::uint8_t internal_child_mask() const noexcept {
    return static_cast<std::uint8_t>(child_mask() & static_cast<std::uint8_t>(~leaf_mask()));
  }

 private:
  std::uint64_t bits_ = 0;
};

class Octree {
 public:
  Octree() = default;
  Octree(
      int max_depth,
      Device device,
      RootBounds root_bounds,
      std::vector<NodeDescriptor> nodes,
      std::vector<std::uint32_t> leaf_payload_indices);

  static Octree from_voxels_cpu(
      const std::vector<glm::ivec3>& coordinates,
      const BuildOptions& options);

  int max_depth() const noexcept { return max_depth_; }
  std::int64_t num_nodes() const noexcept { return static_cast<std::int64_t>(nodes_.size()); }
  std::int64_t num_leaves() const noexcept {
    return static_cast<std::int64_t>(leaf_payload_indices_.size());
  }
  const RootBounds& root_bounds() const noexcept { return root_bounds_; }
  Device device() const noexcept { return device_; }
  const std::vector<NodeDescriptor>& nodes() const noexcept { return nodes_; }
  const std::vector<std::uint32_t>& leaf_payload_indices() const noexcept {
    return leaf_payload_indices_;
  }

  void validate() const;

 private:
  int max_depth_ = 0;
  Device device_ = Device::CPU;
  RootBounds root_bounds_ = default_root_bounds();
  std::vector<NodeDescriptor> nodes_;
  std::vector<std::uint32_t> leaf_payload_indices_;
};

void validate_octree(const Octree& octree);

}  // namespace svo
