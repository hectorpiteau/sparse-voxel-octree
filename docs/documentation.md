# Documentation Workflow

The documentation source lives in `docs/` as normal Markdown files. The Docsify
site shell is generated from `scripts/generate_docsify.py` so navigation stays
consistent and easy to review.

## Edit Documentation

1. Edit the relevant Markdown file in `docs/`.
2. If you add, rename, or remove a page, update `DOC_PAGES` in
   `scripts/generate_docsify.py`.
3. Regenerate the Docsify shell:

```bash
./.venv/bin/python scripts/generate_docsify.py
```

4. Check that generated files are current:

```bash
./.venv/bin/python scripts/generate_docsify.py --check
```

## Preview Locally

Docsify does not need a build step for normal preview. It serves Markdown files
directly in the browser:

```bash
npx docsify-cli serve docs
```

Then open the local URL printed by Docsify, usually `http://localhost:3000`.

If Node is not available, the Markdown files still render directly on GitHub.

## Publish Online

The GitHub Pages workflow in `.github/workflows/docs.yml` publishes the `docs/`
directory. Repository settings must allow Pages deployments from GitHub Actions.

Generated Docsify files that should be committed:

- `docs/index.html`
- `docs/_sidebar.md`
- `docs/_coverpage.md`
- `docs/.nojekyll`

Temporary static export folders are ignored by `.gitignore`.

## Current Pages

- [Home](index.md): Documentation map and current scope.
- [Installation](installation.md): Setup, builds, diagnostics, and compatibility.
- [Quickstart](quickstart.md): First tree, query, raycast, and render workflow.
- [Python API](python-api.md): Python objects, NumPy paths, and CUDA/Torch paths.
- [C++ API](cpp-api.md): Public C++ headers, examples, and CMake usage.
- [PyTorch Rendering](pytorch-rendering.md): CUDA tensor interop, VolumeRenderer, and autograd.
- [Architecture](architecture.md): Layering, ownership, repository layout, and design rules.
- [Data Layout](data-layout.md): Node descriptors, branching modes, and payload indirection.
- [Differentiability](differentiability.md): Gradient scope and discrete topology limits.
- [Performance Roadmap](performance-roadmap.md): Benchmark scenes, metrics, and acceleration decisions.
- [Testing](testing.md): Local tests, CUDA tests, numerical rules, and CI expectations.
- [Packaging](packaging.md): Runtime wheels, source distributions, and release flow.
- [Troubleshooting](troubleshooting.md): Common install, CUDA, Torch, and input issues.
- [Examples](examples.md): Runnable examples and planned examples.
- [Documentation Workflow](documentation.md): How to update, preview, and publish docs.
