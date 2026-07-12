#include <svo/Builder.hpp>
#include <svo/Error.hpp>
#include <svo/Octree.hpp>

#include <glm/ext/vector_int3.hpp>

#include <cstdint>
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

void require_validation_error(
    const std::vector<glm::ivec3>& coordinates,
    const svo::BuildOptions& options,
    const std::string& expected_substring) {
  try {
    (void)svo::Octree::from_voxels_cpu(coordinates, options);
    std::cerr << "expected validation error containing: " << expected_substring << '\n';
    std::exit(1);
  } catch (const svo::ValidationError& error) {
    require(
        std::string(error.what()).find(expected_substring) != std::string::npos,
        "unexpected validation error: " + std::string(error.what()));
  }
}

}  // namespace

int main() {
  {
    svo::BuildOptions options;
    options.max_depth = 3;

    const svo::Octree octree = svo::Octree::from_voxels_cpu({}, options);
    require(octree.num_nodes() == 0, "empty input should produce zero nodes");
    require(octree.num_leaves() == 0, "empty input should produce zero leaves");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 1;

    const svo::Octree octree = svo::Octree::from_voxels_cpu({glm::ivec3{1, 0, 1}}, options);
    require(octree.num_nodes() == 1, "single voxel at depth 1 should produce one root node");
    require(octree.num_leaves() == 1, "single voxel should produce one leaf");
    require(octree.leaf_payload_indices() == std::vector<std::uint32_t>{0u}, "single leaf payload index should be zero");

    const svo::NodeDescriptor root = octree.nodes().front();
    require(root.child_mask() == 0b00100000u, "single voxel child bit is incorrect");
    require(root.leaf_mask() == 0b00100000u, "single voxel leaf bit is incorrect");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 1;

    const svo::Octree octree =
        svo::Octree::from_voxels_cpu({glm::ivec3{0, 0, 0}, glm::ivec3{1, 0, 0}}, options);
    require(octree.num_nodes() == 1, "sibling voxels at depth 1 should still fit in the root node");
    require(octree.num_leaves() == 2, "two occupied voxels should produce two leaves");
    require(
        octree.nodes().front().child_mask() == 0b00000011u,
        "sibling voxels should occupy child bits 0 and 1");
    require(
        octree.nodes().front().leaf_mask() == 0b00000011u,
        "sibling voxels at max depth should be marked as leaves");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 2;

    const svo::Octree octree =
        svo::Octree::from_voxels_cpu({glm::ivec3{0, 0, 0}, glm::ivec3{3, 3, 3}}, options);
    require(octree.num_nodes() == 3, "different root branches should produce root plus two child nodes");
    require(octree.num_leaves() == 2, "two voxels in different branches should still produce two leaves");
    require(
        octree.nodes().front().child_mask() == 0b10000001u,
        "root should reference child octants 0 and 7");
    require(octree.nodes().front().leaf_mask() == 0u, "non-terminal root should have no direct leaves");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 2;
    options.branching = svo::BranchingMode::Wide4;

    const svo::Octree octree =
        svo::Octree::from_voxels_cpu({glm::ivec3{0, 0, 0}, glm::ivec3{3, 3, 3}}, options);
    require(octree.branching() == svo::BranchingMode::Wide4, "wide tree should report wide4 branching");
    require(octree.nodes().empty(), "wide tree should not populate octree8 nodes");
    require(octree.wide_nodes().size() == 1, "depth 2 wide tree should fit in one root node");
    require(octree.num_nodes() == 1, "wide num_nodes should report wide node count");
    require(octree.num_leaves() == 2, "wide tree should keep two leaves");
    require(octree.wide_nodes().front().child_mask() == ((1ull << 0) | (1ull << 63)), "wide root child mask");
    require(octree.wide_nodes().front().leaf_mask() == ((1ull << 0) | (1ull << 63)), "wide root leaf mask");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 4;
    options.branching = svo::BranchingMode::Wide4;

    const svo::Octree wide = svo::Octree::from_voxels_cpu({glm::ivec3{0, 0, 0}, glm::ivec3{15, 15, 15}}, options);
    options.branching = svo::BranchingMode::Octree8;
    const svo::Octree octree = svo::Octree::from_voxels_cpu({glm::ivec3{0, 0, 0}, glm::ivec3{15, 15, 15}}, options);
    require(wide.num_nodes() < octree.num_nodes(), "wide nodes should reduce traversal node count for separated voxels");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 3;
    options.branching = svo::BranchingMode::Wide4;
    require_validation_error({glm::ivec3{0, 0, 0}}, options, "even max_depth");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 2;

    const svo::Octree octree = svo::Octree::from_voxels_cpu(
        {glm::ivec3{1, 1, 1}, glm::ivec3{1, 1, 1}, glm::ivec3{1, 1, 1}},
        options);
    require(octree.num_leaves() == 1, "duplicate voxels must be deduplicated");
    require(
        octree.leaf_payload_indices() == std::vector<std::uint32_t>{0u},
        "deduplicated tree should keep stable payload indexing");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 3;
    require_validation_error({glm::ivec3{-1, 0, 0}}, options, "non-negative");
    require_validation_error({glm::ivec3{8, 0, 0}}, options, "inside [0, 2^max_depth)");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 0;

    const svo::Octree octree = svo::Octree::from_voxels_cpu({glm::ivec3{0, 0, 0}}, options);
    require(octree.num_nodes() == 1, "max_depth=0 should produce a single descriptor");
    require(octree.num_leaves() == 1, "max_depth=0 should produce one leaf");
    require(
        octree.nodes().front().child_mask() == 0b00000001u &&
            octree.nodes().front().leaf_mask() == 0b00000001u,
        "max_depth=0 uses the reserved root-leaf descriptor convention");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 2;

    const std::vector<glm::ivec3> coordinates{
        glm::ivec3{3, 3, 3},
        glm::ivec3{0, 0, 0},
        glm::ivec3{2, 1, 0},
    };
    const std::vector<std::uint32_t> payload_indices{42u, 7u, 99u};
    const svo::Octree octree = svo::Octree::from_voxels_cpu(coordinates, payload_indices, options);

    require(octree.num_leaves() == 3, "custom payload tree should keep three leaves");
    require(
        octree.leaf_payload_indices() == std::vector<std::uint32_t>{7u, 99u, 42u},
        "custom payload indices should follow deterministic leaf order");
  }

  {
    svo::BuildOptions options;
    options.max_depth = 2;

    const svo::Octree octree = svo::Octree::from_voxels_cpu(
        {glm::ivec3{1, 1, 1}, glm::ivec3{1, 1, 1}},
        {5u, 5u},
        options);
    require(octree.num_leaves() == 1, "matching duplicate payload indices should deduplicate");
    require(
        octree.leaf_payload_indices() == std::vector<std::uint32_t>{5u},
        "deduplicated custom payload index should be preserved");

    try {
      (void)svo::Octree::from_voxels_cpu(
          {glm::ivec3{1, 1, 1}, glm::ivec3{1, 1, 1}},
          {5u, 6u},
          options);
      std::cerr << "expected conflicting duplicate payload indices to fail\n";
      return 1;
    } catch (const svo::ValidationError& error) {
      require(
          std::string(error.what()).find("matching payload indices") != std::string::npos,
          "unexpected duplicate payload validation error");
    }

    try {
      (void)svo::Octree::from_voxels_cpu(
          {glm::ivec3{0, 0, 0}},
          {1u, 2u},
          options);
      std::cerr << "expected payload index length mismatch to fail\n";
      return 1;
    } catch (const svo::ValidationError& error) {
      require(
          std::string(error.what()).find("same length") != std::string::npos,
          "unexpected payload length validation error");
    }
  }

  {
    svo::BuildOptions options;
    options.max_depth = 3;

    const std::vector<glm::ivec3> coordinates{
        glm::ivec3{7, 7, 7},
        glm::ivec3{0, 0, 0},
        glm::ivec3{3, 1, 4},
        glm::ivec3{3, 1, 4},
    };

    const svo::Octree tree_a = svo::Octree::from_voxels_cpu(coordinates, options);
    const svo::Octree tree_b = svo::Octree::from_voxels_cpu(coordinates, options);

    require(tree_a.num_nodes() == tree_b.num_nodes(), "build output must be deterministic");
    require(tree_a.leaf_payload_indices() == tree_b.leaf_payload_indices(), "payload ordering must be deterministic");

    for (std::size_t index = 0; index < tree_a.nodes().size(); ++index) {
      require(
          tree_a.nodes()[index].bits() == tree_b.nodes()[index].bits(),
          "descriptor bits must be deterministic");
    }
  }

  return 0;
}
