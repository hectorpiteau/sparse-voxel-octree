#include <svo/Interpolation.hpp>
#include <svo/Octree.hpp>

#include <glm/ext/vector_int3.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void require_close(double actual, double expected, double tolerance, const std::string& message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
    std::exit(1);
  }
}

svo::Octree dense_depth_one_tree() {
  svo::BuildOptions options;
  options.max_depth = 1;
  return svo::Octree::from_voxels_cpu(
      {
          {0, 0, 0},
          {1, 0, 0},
          {0, 1, 0},
          {1, 1, 0},
          {0, 0, 1},
          {1, 0, 1},
          {0, 1, 1},
          {1, 1, 1},
      },
      options);
}

void test_leaf_centers_and_linear_field() {
  const svo::Octree octree = dense_depth_one_tree();
  const std::vector<float> payload{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  const std::vector<float> samples = svo::sample_trilinear_float(
      octree,
      {{0.25f, 0.25f, 0.25f}, {0.5f, 0.5f, 0.5f}},
      payload.data(),
      payload.size(),
      1,
      0.0f);

  require_close(samples[0], 0.0, 1e-6, "leaf center should return exact payload");
  require_close(samples[1], 3.5, 1e-6, "center should average the eight neighboring leaf values");
}

void test_feature_payload_and_custom_payload_indices() {
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu(
      {{0, 0, 0}, {1, 0, 0}},
      {1u, 0u},
      options);
  const std::vector<double> payload{10.0, 20.0, 30.0, 40.0};
  const std::vector<double> samples = svo::sample_trilinear_double(
      octree,
      {{0.25f, 0.25f, 0.25f}, {0.75f, 0.25f, 0.25f}},
      payload.data(),
      2,
      2,
      -1.0);

  require_close(samples[0], 30.0, 1e-12, "custom payload row should be used for first leaf channel 0");
  require_close(samples[1], 40.0, 1e-12, "custom payload row should be used for first leaf channel 1");
  require_close(samples[2], 10.0, 1e-12, "custom payload row should be used for second leaf channel 0");
  require_close(samples[3], 20.0, 1e-12, "custom payload row should be used for second leaf channel 1");
}

void test_missing_neighbors_use_fill_value() {
  svo::BuildOptions options;
  options.max_depth = 1;
  const svo::Octree octree = svo::Octree::from_voxels_cpu({{0, 0, 0}}, options);
  const std::vector<float> payload{8.0f};

  const std::vector<float> samples = svo::sample_trilinear_float(
      octree,
      {{0.25f, 0.25f, 0.25f}, {0.0f, 0.0f, 0.0f}, {-0.1f, 0.0f, 0.0f}},
      payload.data(),
      payload.size(),
      1,
      0.0f);

  require_close(samples[0], 8.0, 1e-6, "occupied center should return payload even in sparse tree");
  require_close(samples[1], 1.0, 1e-6, "root min should blend one occupied neighbor with seven fill values");
  require_close(samples[2], 0.0, 1e-6, "outside root should return fill value");
}

void test_backward_reference() {
  const svo::Octree octree = dense_depth_one_tree();
  const std::vector<float> grad_outputs{2.0f};
  const std::vector<float> grad_payload = svo::sample_trilinear_backward_float(
      octree,
      {{0.5f, 0.5f, 0.5f}},
      grad_outputs.data(),
      8,
      1,
      0.0f);

  require(grad_payload.size() == 8, "gradient payload size should match payload rows");
  for (float value : grad_payload) {
    require_close(value, 0.25, 1e-6, "center sample should distribute gradient equally");
  }
}

}  // namespace

int main() {
  test_leaf_centers_and_linear_field();
  test_feature_payload_and_custom_payload_indices();
  test_missing_neighbors_use_fill_value();
  test_backward_reference();
  return 0;
}
