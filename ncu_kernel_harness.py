"""
Run under ``ncu`` to profile a KernelBench-style ``kernel.py`` (no ``__main__``).

Loads the module, builds ``Model(*get_init_inputs())``, calls ``.eval()``, then repeats:
``inputs = get_inputs()`` → forward, matching :func:`run_validation.run_forward_validation` call semantics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import traceback
from pathlib import Path


def _move_tensors_to(obj, device):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_tensors_to(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_tensors_to(v, device) for k, v in obj.items()}
    return obj


def _load_kernel(path: Path):
    mod_name = "ncu_harness_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import torch

    if len(sys.argv) < 2:
        print("Usage: ncu_kernel_harness.py <kernel.py> [iterations]", file=sys.stderr)
        return 2

    kernel_path = Path(sys.argv[1]).resolve()
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    if not torch.cuda.is_available():
        print("ncu_kernel_harness: CUDA is required.", file=sys.stderr)
        return 1

    if not kernel_path.is_file():
        print(f"ncu_kernel_harness: file not found: {kernel_path}", file=sys.stderr)
        return 1

    try:
        mod = _load_kernel(kernel_path)
    except Exception:
        print(f"ncu_kernel_harness: failed to import kernel:\n{traceback.format_exc()}", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    try:
        init_args = list(mod.get_init_inputs())
        model = mod.Model(*init_args).to(device)
        model.eval()
    except Exception:
        print(f"ncu_kernel_harness: Model init failed:\n{traceback.format_exc()}", file=sys.stderr)
        return 1

    try:
        for _ in range(iterations):
            inputs = mod.get_inputs()
            inputs_device = _move_tensors_to(inputs, device)
            if isinstance(inputs_device, torch.Tensor):
                _ = model(inputs_device)
            elif isinstance(inputs_device, (tuple, list)):
                _ = model(*inputs_device)
            else:
                _ = model(inputs_device)
            torch.cuda.synchronize()
    except Exception:
        print(f"ncu_kernel_harness: forward loop failed:\n{traceback.format_exc()}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
