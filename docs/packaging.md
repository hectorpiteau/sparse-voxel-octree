# Packaging

## Build Artifacts

The Python package uses:

- `uv`
- `scikit-build-core`
- CMake
- pybind11
- `setuptools_scm` for git-derived versions

Build:

```bash
uv build
```

Validate artifacts:

```bash
uv run python scripts/check_package.py
```

## Runtime Wheel Policy

The PyPI wheel is Python-runtime-only. It contains:

- `svo/*.py`
- `svo/_svo*.so` or platform equivalent
- package metadata
- license metadata

It does not contain C++ headers, GLM headers, CMake package files, or the static
C++ library. C++ users should build/install from source with CMake.

## Versioning

Versions are derived from git tags through `setuptools_scm`.

Release tags should be annotated and use `vX.Y.Z`:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

Before creating a release tag, decide whether the change is:

- Patch: fixes, docs, tests, packaging, or compatible performance work.
- Minor: backward-compatible features, or breaking changes before `1.0.0`.
- Major: breaking changes after `1.0.0`.

## CI and Publishing

Pull requests run:

- Ruff lint.
- Python tests on 3.10, 3.11, 3.12.
- CPU CMake tests.
- Wheel/sdist build and package smoke.

CUDA CI is optional and uses a self-hosted runner labeled `self-hosted`, `linux`,
`x64`, and `cuda`.

Publishing uses PyPI Trusted Publishing through GitHub Actions environments:

- `testpypi` for manual TestPyPI publishing.
- `pypi` for tag-based PyPI publishing.

## Release Checklist

1. Review API compatibility impact.
2. Choose patch/minor/major.
3. Run lint, tests, CMake tests, and package smoke.
4. Create an annotated `vX.Y.Z` tag.
5. Push the tag.
6. Validate TestPyPI if needed.
7. Publish to PyPI through the trusted workflow.
