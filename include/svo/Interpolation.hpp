#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <svo/Octree.hpp>

namespace svo {

std::vector<float> sample_trilinear_float(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const float* payload,
    std::size_t payload_rows,
    std::size_t channels,
    float fill_value = 0.0f);

std::vector<double> sample_trilinear_double(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const double* payload,
    std::size_t payload_rows,
    std::size_t channels,
    double fill_value = 0.0);

std::vector<float> sample_trilinear_backward_float(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const float* grad_outputs,
    std::size_t payload_rows,
    std::size_t channels,
    float fill_value = 0.0f);

std::vector<double> sample_trilinear_backward_double(
    const Octree& octree,
    const std::vector<glm::vec3>& points,
    const double* grad_outputs,
    std::size_t payload_rows,
    std::size_t channels,
    double fill_value = 0.0);

#if SVO_ENABLE_CUDA
void sample_trilinear_cuda_float(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const float* payload,
    float* outputs,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    float fill_value = 0.0f,
    CudaStreamHandle stream = nullptr);

void sample_trilinear_cuda_double(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const double* payload,
    double* outputs,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    double fill_value = 0.0,
    CudaStreamHandle stream = nullptr);

void sample_trilinear_backward_cuda_float(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const float* grad_outputs,
    float* grad_payload,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    float fill_value = 0.0f,
    CudaStreamHandle stream = nullptr);

void sample_trilinear_backward_cuda_double(
    const NodeDescriptor* nodes,
    std::size_t num_nodes,
    const std::uint32_t* leaf_payload_indices,
    std::size_t num_leaves,
    int max_depth,
    const RootBounds& root_bounds,
    const glm::vec3* points,
    const double* grad_outputs,
    double* grad_payload,
    std::size_t count,
    std::size_t payload_rows,
    std::size_t channels,
    double fill_value = 0.0,
    CudaStreamHandle stream = nullptr);
#endif

}  // namespace svo
