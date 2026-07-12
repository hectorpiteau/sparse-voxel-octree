# Sparse Voxel Octree Documentation

This directory contains the project documentation in Markdown. The files are
plain GitHub-rendered docs for now, and are structured so they can later become
a GitHub Pages site without moving the content again.

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

## Current Scope

The project is a working pre-alpha C++20/CUDA sparse voxel octree library with
pybind11 Python bindings, CPU reference paths, CUDA traversal/rendering, and
optional PyTorch CUDA autograd integration.

The public API is still allowed to change before `1.0.0`. Changes should be
classified before release:

- Patch: fixes, documentation, tests, packaging, or compatible performance work.
- Minor: backward-compatible public features, or breaking changes before `1.0.0`.
- Major: breaking changes after `1.0.0`.

## Future HTML Docs

When the project is ready for an online documentation site, the recommended next
step is to add a static documentation generator such as MkDocs or Sphinx and use
GitHub Pages to publish this directory. The content split here is intended to
make that migration mechanical.
