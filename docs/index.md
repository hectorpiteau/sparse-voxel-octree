# Sparse Voxel Octree Documentation

This directory contains the project documentation in Markdown. The files render
directly on GitHub and are also served as a Docsify site from the same `docs/`
directory.

## Documentation Map

- [Installation](installation.md): local setup, source builds, CUDA builds, diagnostics, compatibility.
- [Quickstart](quickstart.md): build a tree, query points, raycast, render a small scene.
- [Python API](python-api.md): Python-facing `svo` objects, NumPy paths, CUDA/Torch paths.
- [C++ API](cpp-api.md): public C++ headers, CPU builder/query examples, CMake notes.
- [PyTorch Rendering](pytorch-rendering.md): CUDA tensor interop, `VolumeRenderer`, autograd behavior.
- [Architecture](architecture.md): topology/payload split, layering, repository layout, ownership rules.
- [Data Layout](data-layout.md): node descriptors, branching modes, payload indirection, future layout phases.
- [Differentiability](differentiability.md): what is differentiable, what is discrete, gradient scope.
- [Testing](testing.md): local tests, CUDA tests, numerical rules, CI expectations.
- [Packaging](packaging.md): runtime wheels, source distributions, release flow, versioning policy.
- [Troubleshooting](troubleshooting.md): diagnostics and common install/runtime errors.
- [Examples](examples.md): scripts and workflows included in the repository.
- [Documentation Workflow](documentation.md): how to update, preview, and publish the Docsify site.

## Current Scope

The project is a working pre-alpha C++20/CUDA sparse voxel octree library with
pybind11 Python bindings, CPU reference paths, CUDA traversal/rendering, and
optional PyTorch CUDA autograd integration.

The public API is still allowed to change before `1.0.0`. Changes should be
classified before release:

- Patch: fixes, documentation, tests, packaging, or compatible performance work.
- Minor: backward-compatible public features, or breaking changes before `1.0.0`.
- Major: breaking changes after `1.0.0`.