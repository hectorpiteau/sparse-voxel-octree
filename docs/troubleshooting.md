# Troubleshooting

## Start With Diagnostics

```bash
python -m svo.info
```

Useful fields:

- `svo version`
- `C++ core version`
- extension path
- CUDA extension loaded
- Python executable
- platform and machine
- NumPy version
- Torch version
- Torch CUDA availability
- Torch CUDA device count

## CUDA Build Is Missing

Symptom:

```text
Octree.to('cuda') requires a Python extension built with SVO_ENABLE_CUDA=ON
```

Fix:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python --reinstall --no-cache -e . -C cmake.define.SVO_ENABLE_CUDA=ON
```

Then rerun:

```bash
python -m svo.info
```

## Torch CUDA Is Not Available

The SVO extension can be CUDA-enabled while PyTorch itself reports no CUDA
device. Check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Install a CUDA-enabled PyTorch build that matches your environment if needed.

## Bad Input Errors

Common validation failures:

- Coordinates must be `int32` or `int64`.
- Query/ray arrays must be floating point and C-contiguous.
- Torch CUDA paths require CUDA tensors.
- CPU and CUDA tensors cannot be mixed in one CUDA call.
- `wide4` requires an even `max_depth`.

## Package Build Issues

Run:

```bash
uv build
uv run python scripts/check_package.py
```

The package checker verifies that the wheel is runtime-only, the sdist contains
source files, and the wheel can be installed into a clean temporary environment.
