#include <svo/Builder.hpp>
#include <svo/CoarseOccupancy.hpp>
#include <svo/Error.hpp>

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

void test_bitset_resolutions() {
  for (int resolution : {16, 32, 64}) {
    svo::CoarseOccupancyGrid grid(resolution, svo::default_root_bounds());
    require(grid.size_bytes() == static_cast<std::size_t>(resolution * resolution * resolution / 8), "unexpected bitset byte size");
    require(!grid.occupied(0, 0, 0), "new coarse cell should be empty");
    grid.mark(0, 0, 0);
    grid.mark(resolution - 1, resolution - 1, resolution - 1);
    require(grid.occupied(0, 0, 0), "marked first coarse cell should be occupied");
    require(grid.occupied(resolution - 1, resolution - 1, resolution - 1), "marked last coarse cell should be occupied");
  }
}

void test_invalid_resolution() {
  bool threw = false;
  try {
    (void)svo::CoarseOccupancyGrid(8, svo::default_root_bounds());
  } catch (const svo::ValidationError&) {
    threw = true;
  }
  require(threw, "invalid coarse resolution should fail");
}

void test_builder_octree8_single_voxel() {
  svo::BuildOptions options;
  options.max_depth = 4;
  options.branching = svo::BranchingMode::Octree8;
  const svo::Octree tree = svo::Octree::from_voxels_cpu({glm::ivec3{0, 0, 0}}, options);
  const svo::CoarseOccupancyGrid grid = svo::CoarseOccupancyGrid::from_octree(tree, 16);
  require(grid.occupied(0, 0, 0), "single voxel should mark the matching macro cell");
  require(!grid.occupied(15, 15, 15), "single voxel should not mark distant macro cells");
}

void test_builder_wide4_single_voxel() {
  svo::BuildOptions options;
  options.max_depth = 4;
  options.branching = svo::BranchingMode::Wide4;
  const svo::Octree tree = svo::Octree::from_voxels_cpu({glm::ivec3{15, 15, 15}}, options);
  const svo::CoarseOccupancyGrid grid = svo::CoarseOccupancyGrid::from_octree(tree, 16);
  require(grid.occupied(15, 15, 15), "wide4 single voxel should mark the matching macro cell");
  require(!grid.occupied(0, 0, 0), "wide4 single voxel should not mark distant macro cells");
}

void test_empty_tree_marks_no_cells() {
  svo::BuildOptions options;
  options.max_depth = 4;
  const svo::Octree tree = svo::Octree::from_voxels_cpu({}, options);
  const svo::CoarseOccupancyGrid grid = svo::CoarseOccupancyGrid::from_octree(tree, 16);
  for (int z = 0; z < 16; ++z) {
    for (int y = 0; y < 16; ++y) {
      for (int x = 0; x < 16; ++x) {
        require(!grid.occupied(x, y, z), "empty tree should not mark coarse cells");
      }
    }
  }
}

}  // namespace

int main() {
  test_bitset_resolutions();
  test_invalid_resolution();
  test_builder_octree8_single_voxel();
  test_builder_wide4_single_voxel();
  test_empty_tree_marks_no_cells();
  return 0;
}
