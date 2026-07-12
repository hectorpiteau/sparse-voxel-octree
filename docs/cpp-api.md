# C++ API

The C++ API is usable without Python. It uses GLM for vector math and stores the
octree topology in descriptor arrays.

## Headers

```cpp
#include <svo/Octree.hpp>
#include <svo/Builder.hpp>
#include <svo/Query.hpp>
#include <svo/Raycast.hpp>
#include <svo/Renderer.hpp>
#include <svo/Camera.hpp>
```

## Build and Query

```cpp
#include <svo/Builder.hpp>
#include <svo/Octree.hpp>
#include <svo/Query.hpp>

#include <cstdint>
#include <vector>

int main() {
    std::vector<glm::ivec3> coords = {
        {0, 0, 0},
        {1, 2, 3},
        {4, 4, 4},
    };

    svo::BuildOptions options;
    options.max_depth = 8;
    options.device = svo::Device::CPU;

    svo::Octree tree = svo::build_octree_cpu(coords, options);

    std::vector<glm::vec3> points = {
        {0.1f, 0.1f, 0.1f},
        {0.5f, 0.5f, 0.5f},
    };

    std::vector<std::int32_t> leaf_ids = svo::query_points(tree, points);
    return 0;
}
```

## Octree Properties

```cpp
class Octree {
public:
    int max_depth() const;
    int wide_depth() const;
    std::int64_t num_nodes() const;
    std::int64_t num_leaves() const;
    const RootBounds& root_bounds() const;
    Device device() const;
    BranchingMode branching() const;

    const std::vector<NodeDescriptor>& nodes() const;
    const std::vector<WideNodeDescriptor>& wide_nodes() const;
    const std::vector<std::uint32_t>& leaf_payload_indices() const;

    void validate() const;
};
```

## CMake Build

```bash
cmake -S . -B build-cpu -DSVO_ENABLE_CUDA=OFF -DSVO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release
ctest --test-dir build-cpu --output-on-failure
```

CUDA build:

```bash
cmake -S . -B build-cuda -DSVO_ENABLE_CUDA=ON -DSVO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release
ctest --test-dir build-cuda --output-on-failure
```
