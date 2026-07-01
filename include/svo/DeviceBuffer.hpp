#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <svo/Error.hpp>
#include <svo/Octree.hpp>

#ifndef SVO_ENABLE_CUDA
#define SVO_ENABLE_CUDA 0
#endif

#if SVO_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace svo {

using CudaStreamHandle = void*;

template <typename T>
class DeviceBuffer {
  static_assert(std::is_trivially_copyable<T>::value, "DeviceBuffer requires trivially copyable values");

 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t size, Device device = Device::CPU) {
    allocate(size, device);
  }

  ~DeviceBuffer() {
    release();
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept {
    move_from(std::move(other));
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  static DeviceBuffer from_host(
      const std::vector<T>& values,
      Device device = Device::CPU,
      CudaStreamHandle stream = nullptr) {
    DeviceBuffer buffer(values.size(), device);
    buffer.copy_from_host(values.data(), values.size(), stream);
    return buffer;
  }

  void allocate(std::size_t size, Device device = Device::CPU) {
    (void)checked_size_bytes(size);

    if (size == 0) {
      release();
      device_ = device;
      return;
    }

    switch (device) {
      case Device::CPU: {
        std::vector<T> storage(size);
        release();
        cpu_storage_ = std::move(storage);
        size_ = size;
        device_ = Device::CPU;
        break;
      }
      case Device::CUDA: {
        T* cuda_data = allocate_cuda_storage(size);
        release();
        cuda_data_ = cuda_data;
        size_ = size;
        device_ = Device::CUDA;
        break;
      }
    }
  }

  void release() noexcept {
    if (device_ == Device::CUDA && cuda_data_ != nullptr) {
      free_cuda_noexcept();
    }

    cpu_storage_.clear();
    cuda_data_ = nullptr;
    size_ = 0;
    device_ = Device::CPU;
  }

  void copy_from_host(const T* source, std::size_t count, CudaStreamHandle stream = nullptr) {
    validate_copy(source, count);

    if (count == 0) {
      return;
    }

    switch (device_) {
      case Device::CPU:
        std::copy(source, source + count, cpu_storage_.begin());
        break;
      case Device::CUDA:
        copy_host_to_cuda(source, count, stream);
        break;
    }
  }

  void copy_to_host(T* destination, std::size_t count, CudaStreamHandle stream = nullptr) const {
    validate_copy(destination, count);

    if (count == 0) {
      return;
    }

    switch (device_) {
      case Device::CPU:
        std::copy(cpu_storage_.begin(), cpu_storage_.begin() + static_cast<std::ptrdiff_t>(count), destination);
        break;
      case Device::CUDA:
        copy_cuda_to_host(destination, count, stream);
        break;
    }
  }

  std::vector<T> to_host(CudaStreamHandle stream = nullptr) const {
    std::vector<T> values(size_);
    copy_to_host(values.data(), values.size(), stream);
    return values;
  }

  T* data() noexcept {
    return device_ == Device::CPU ? cpu_storage_.data() : cuda_data_;
  }

  const T* data() const noexcept {
    return device_ == Device::CPU ? cpu_storage_.data() : cuda_data_;
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }
  bool empty() const noexcept { return size_ == 0; }
  Device device() const noexcept { return device_; }

 private:
  void validate_copy(const T* pointer, std::size_t count) const {
    if (count > size_) {
      throw ValidationError("copy count exceeds DeviceBuffer size");
    }
    if (count != 0 && pointer == nullptr) {
      throw ValidationError("copy pointer cannot be null when count is non-zero");
    }
  }

  void move_from(DeviceBuffer&& other) noexcept {
    device_ = other.device_;
    size_ = other.size_;
    cpu_storage_ = std::move(other.cpu_storage_);
    cuda_data_ = other.cuda_data_;

    other.device_ = Device::CPU;
    other.size_ = 0;
    other.cuda_data_ = nullptr;
  }

  static std::size_t checked_size_bytes(std::size_t size) {
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw ValidationError("DeviceBuffer byte size overflows size_t");
    }
    return size * sizeof(T);
  }

  static T* allocate_cuda_storage(std::size_t size) {
#if SVO_ENABLE_CUDA
    T* cuda_data = nullptr;
    const cudaError_t result = cudaMalloc(reinterpret_cast<void**>(&cuda_data), checked_size_bytes(size));
    if (result != cudaSuccess) {
      throw Error(std::string("cudaMalloc failed: ") + cudaGetErrorString(result));
    }
    return cuda_data;
#else
    (void)size;
    throw ValidationError("CUDA DeviceBuffer requested but SVO_ENABLE_CUDA is disabled");
#endif
  }

  void free_cuda_noexcept() noexcept {
#if SVO_ENABLE_CUDA
    (void)cudaFree(cuda_data_);
#endif
  }

  void copy_host_to_cuda(const T* source, std::size_t count, CudaStreamHandle stream) {
#if SVO_ENABLE_CUDA
    const cudaError_t result = cudaMemcpyAsync(
        cuda_data_,
        source,
        count * sizeof(T),
        cudaMemcpyHostToDevice,
        reinterpret_cast<cudaStream_t>(stream));
    if (result != cudaSuccess) {
      throw Error(std::string("cudaMemcpyAsync host-to-device failed: ") + cudaGetErrorString(result));
    }
    synchronize_if_default_stream(stream);
#else
    (void)source;
    (void)count;
    (void)stream;
    throw ValidationError("CUDA copy requested but SVO_ENABLE_CUDA is disabled");
#endif
  }

  void copy_cuda_to_host(T* destination, std::size_t count, CudaStreamHandle stream) const {
#if SVO_ENABLE_CUDA
    const cudaError_t result = cudaMemcpyAsync(
        destination,
        cuda_data_,
        count * sizeof(T),
        cudaMemcpyDeviceToHost,
        reinterpret_cast<cudaStream_t>(stream));
    if (result != cudaSuccess) {
      throw Error(std::string("cudaMemcpyAsync device-to-host failed: ") + cudaGetErrorString(result));
    }
    synchronize_if_default_stream(stream);
#else
    (void)destination;
    (void)count;
    (void)stream;
    throw ValidationError("CUDA copy requested but SVO_ENABLE_CUDA is disabled");
#endif
  }

  static void synchronize_if_default_stream(CudaStreamHandle stream) {
#if SVO_ENABLE_CUDA
    if (stream == nullptr) {
      const cudaError_t result = cudaStreamSynchronize(nullptr);
      if (result != cudaSuccess) {
        throw Error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(result));
      }
    }
#else
    (void)stream;
#endif
  }

  Device device_ = Device::CPU;
  std::size_t size_ = 0;
  std::vector<T> cpu_storage_;
  T* cuda_data_ = nullptr;
};

}  // namespace svo
