# Contributing

Thank you for helping improve Sparse Voxel Octree CUDA. Contributions are most
useful when they explain the problem being solved, preserve the CPU/CUDA and
C++/Python contracts, and leave enough documentation for the next person to
understand why the change matters.

## Before You Start

- Read the [architecture](docs/architecture.md), [data layout](docs/data-layout.md),
  and [testing](docs/testing.md) notes for changes to the core library.
- Check the [performance roadmap](docs/performance-roadmap.md) before proposing
  descriptor, traversal, rendering, or memory-layout work.
- For a large feature, breaking API change, or new storage layout, open an issue
  or discussion before investing in an implementation.
- Keep each contribution focused on one problem. Separate unrelated refactors,
  formatting changes, and dependency updates.

The project is pre-alpha, so APIs may change, but compatibility impact still
needs to be identified and justified.

## Version Numbering

The project follows [Semantic Versioning](https://semver.org/) using the
`MAJOR.MINOR.PATCH` (`0.0.0`) convention:

- Increment **PATCH** for compatible fixes, documentation, tests, packaging, or
  performance improvements.
- Increment **MINOR** for backward-compatible features and, while the project is
  below `1.0.0`, for breaking API changes.
- Increment **MAJOR** for breaking API changes after `1.0.0`.

Versions are derived from annotated Git tags named `vX.Y.Z`. Every PR should
state its expected patch, minor, or major impact; maintainers choose the final
release version.

## Development Setup

Create the development environment:

```bash
uv sync --extra test --extra lint
```

For CUDA builds and supported toolchains, follow the
[installation guide](docs/installation.md). Run `python -m svo.info` when
reporting build or device-specific behavior.

## Design Guidelines

- Keep topology separate from application payloads. Payload data belongs in
  external arrays, tensors, or user-managed buffers.
- Put important algorithms in the C++/CUDA core rather than implementing them
  only in Python.
- Make ownership explicit for host memory, device memory, NumPy arrays, and
  Torch tensors. Borrowed memory must remain alive for as long as it is used.
- Preserve device locality on CUDA hot paths; avoid implicit host transfers and
  synchronization.
- Validate shapes, dtypes, contiguity, devices, coordinates, and branching
  constraints at API boundaries, with actionable error messages.
- Treat topology construction, occupancy, leaf selection, and boundary
  decisions as discrete. The supported gradient scope is documented in
  [differentiability](docs/differentiability.md).
- Do not broaden packaging contents accidentally. Python wheels are
  runtime-only; C++ headers and libraries are distributed through source builds.
- Prefer measured performance changes. Record the scene, parameters, hardware,
  build type, and before/after results, and avoid mixing redesigns with tuning.

## Tests and Documentation

Every behavior change should include a test that fails without the change.
CUDA features should have CPU reference or parity coverage unless the PR clearly
explains why that is not possible. Use small deterministic numerical cases and
avoid rays exactly on voxel boundaries.

Update all affected documentation in the same PR. This includes public API
signatures, accepted inputs, device behavior, ownership or lifetime rules,
errors, examples, compatibility notes, and limitations. Documentation should
explain both what changed and why a user should care; avoid merely restating the
implementation.

If a documentation page is added, removed, or renamed, update `DOC_PAGES` in
`scripts/generate_docsify.py`, regenerate the Docsify files, and commit them:

```bash
./.venv/bin/python scripts/generate_docsify.py
./.venv/bin/python scripts/generate_docsify.py --check
```

## Local Checks

Run the checks relevant to your change and report exactly what you ran in the
PR. The common CPU checks are:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra lint ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run --extra test pytest tests/python -q
cmake -S . -B build-cpu -DSVO_ENABLE_CUDA=OFF -DSVO_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release -j2
ctest --test-dir build-cpu --output-on-failure
```

For packaging changes, also run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build
./.venv/bin/python scripts/check_package.py
```

For CUDA changes, build and run the CUDA C++ and Python tests described in
[testing](docs/testing.md). If you cannot run a relevant check, say why in the
PR instead of leaving its status ambiguous.

## Commit and Pull Request Guidelines

Before opening a PR:

- Rebase or merge the current `main` branch and resolve conflicts intentionally.
- Review the complete diff for generated files, debug output, large artifacts,
  accidental formatting, and unrelated edits.
- Use clear, imperative commit subjects and keep commits logically grouped.
- Do not commit build directories, virtual environments, benchmark output, or
  generated assets unless they are intentional project artifacts.
- Keep the PR small enough to review. Split independent changes into separate
  PRs and call out any deliberate follow-up work.

A good PR description should let a reviewer understand the contribution before
reading the code. Include:

1. **Problem and motivation:** what is missing or incorrect, who is affected,
   and why solving it matters.
2. **Approach:** the important design choices and why they fit the architecture.
3. **Scope:** what is intentionally included, excluded, or deferred.
4. **Compatibility:** effects on public C++/Python APIs, data layout, devices,
   gradients, performance, and packaging. Identify the expected patch, minor,
   or major release impact.
5. **Evidence:** tests run, CPU/CUDA parity results, benchmarks, screenshots, or
   minimal reproduction steps as appropriate.
6. **Documentation:** pages and examples updated, or a reason no documentation
   change is needed.
7. **Risks:** known limitations, tradeoffs, and areas that deserve careful
   review.

Link related issues with `Fixes #...` when the PR fully resolves them. Prefer a
draft PR while the design or validation is incomplete. Respond to review by
updating the relevant commit or adding a focused follow-up commit, and resolve
threads only after the concern has been addressed or agreement has been
recorded.

## Pull Request Checklist

- [ ] The PR explains the problem, why it matters, and the chosen approach.
- [ ] The diff is focused and contains no unrelated or accidental changes.
- [ ] Public API and compatibility impact are documented.
- [ ] Tests cover new behavior and relevant CPU/CUDA paths.
- [ ] Relevant local checks pass, and their commands/results are listed.
- [ ] User and contributor documentation is accurate and complete.
- [ ] Performance claims include reproducible before/after measurements.
- [ ] Known limitations, untested configurations, and follow-up work are explicit.
- [ ] Generated documentation and package artifacts were checked when affected.

By contributing, you agree that your contribution is provided under the
project's [MIT License](LICENSE). Do not copy code or assets from incompatible
sources; mention any third-party material and its license in the PR.
