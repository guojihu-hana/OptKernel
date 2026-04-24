"""
Reference vs generated kernel forward validation and lightweight timing.

Used by :class:`agent.KernelBenchAgent` after writing ``kernel.py`` for a round.
The agent calls :func:`run_forward_validation_subprocess` (fresh ``python`` process) so a bad
generated kernel is less likely to poison the parent’s CUDA context; :func:`run_forward_validation`
remains available for in-process use (tests, tools).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from subprocess import CompletedProcess
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch


def _cuda_context_error_dict(exc: BaseException) -> dict[str, Any]:
    """
    Stale or poisoned CUDA context (e.g. illegal memory access from a bad generated kernel)
    often surfaces at the *next* CUDA API call, not where the bug occurred.
    """
    return {
        "runnable": False,
        "status": "cuda_error",
        "cuda_error": {
            "message": (
                "CUDA runtime error—often illegal memory access from a previously executed "
                "generated kernel, reported asynchronously here. Restart this Python process "
                "before further GPU validation. To pinpoint the kernel: CUDA_LAUNCH_BLOCKING=1 "
                "or validate one round in a fresh process."
            ),
            "repr": repr(exc),
        },
    }


def import_kernelbench_file(path: Path, module_name: str) -> Any:
    path = path.resolve()
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _move_tensors_to(obj: Any, device: Any) -> Any:
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


# Max entries in numerical_error / metrics.json for per-element mismatch diagnostics.
MISMATCH_POSITION_SAMPLE_N = 100


def _flat_index_to_multidim(flat: int, shape: tuple[int, ...]) -> list[int]:
    if not shape:
        return []
    idx = [0] * len(shape)
    rem = flat
    for k in range(len(shape) - 1, -1, -1):
        s = int(shape[k])
        idx[k] = rem % s
        rem //= s
    return idx


def _tensor_mismatch_position_sample(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
    rtol: float,
    *,
    max_n: int = MISMATCH_POSITION_SAMPLE_N,
) -> tuple[int, list[dict[str, Any]]]:
    """
    List up to ``max_n`` element positions that fail :func:`torch.isclose` (same rule as
    :func:`torch.allclose` for well-behaved float tensors), plus the total count of
    failing elements. Flatten order matches ``reshape(-1)`` / C-order.
    """
    import torch

    mismatch = ~torch.isclose(
        a,
        b,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )
    n_mismatch = int(mismatch.sum().item())
    if n_mismatch == 0:
        return 0, []
    af, bf, mf = a.reshape(-1), b.reshape(-1), mismatch.reshape(-1)
    flat_i = torch.nonzero(mf, as_tuple=True)[0][: int(max_n)]
    shp = tuple(int(s) for s in a.shape)
    sample: list[dict[str, Any]] = []
    for t in range(flat_i.size(0)):
        fi = int(flat_i[t].item())
        r = float(af[fi].item())
        g = float(bf[fi].item())
        rec: dict[str, Any] = {
            "flat_index": fi,
            "ref": r,
            "gen": g,
            "abs_diff": abs(r - g),
        }
        if shp:
            rec["index"] = _flat_index_to_multidim(fi, shp)
        sample.append(rec)
    return n_mismatch, sample


def compare_outputs(
    out_ref: Any,
    out_gen: Any,
    atol: float,
    rtol: float,
) -> tuple[bool, Optional[dict[str, Any]]]:
    import torch

    if type(out_ref) is not type(out_gen):
        return False, {
            "reason": "type_mismatch",
            "type_ref": str(type(out_ref)),
            "type_gen": str(type(out_gen)),
        }

    if isinstance(out_ref, torch.Tensor):
        if out_ref.shape != out_gen.shape:
            return False, {
                "reason": "shape_mismatch",
                "shape_ref": list(out_ref.shape),
                "shape_gen": list(out_gen.shape),
                "dtype_ref": str(out_ref.dtype),
                "dtype_gen": str(out_gen.dtype),
            }
        if out_ref.dtype != out_gen.dtype:
            return False, {
                "reason": "dtype_mismatch",
                "dtype_ref": str(out_ref.dtype),
                "dtype_gen": str(out_gen.dtype),
            }
        # Compare on CPU: avoids extra GPU elementwise kernels; misaligned/illegal access from a
        # bad generated kernel often surfaces on the *next* CUDA op (previously the subtract here).
        a = out_ref.detach().float().cpu()
        b = out_gen.detach().float().cpu()
        diff = (a - b).abs().max().item()
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        out: dict[str, Any] = {
            "max_abs_diff": float(diff),
            "atol": float(atol),
            "rtol": float(rtol),
            "allclose": bool(ok),
        }
        if not ok:
            n_m, sm = _tensor_mismatch_position_sample(
                a, b, float(atol), float(rtol), max_n=MISMATCH_POSITION_SAMPLE_N
            )
            out["mismatching_elements_count"] = n_m
            if sm:
                out["mismatch_position_sample"] = sm
        return ok, out

    if isinstance(out_ref, (tuple, list)):
        if len(out_ref) != len(out_gen):
            return False, {
                "reason": "length_mismatch",
                "len_ref": len(out_ref),
                "len_gen": len(out_gen),
            }
        max_diff = 0.0
        for i, (a, b) in enumerate(zip(out_ref, out_gen)):
            ok_i, info = compare_outputs(a, b, atol, rtol)
            if not ok_i:
                info = info or {}
                info["output_index"] = i
                return False, info
            if info and "max_abs_diff" in info:
                max_diff = max(max_diff, info["max_abs_diff"])
        return True, {"max_abs_diff": max_diff, "allclose": True}

    if isinstance(out_ref, dict):
        if set(out_ref.keys()) != set(out_gen.keys()):
            return False, {"reason": "dict_keys_mismatch"}
        max_diff = 0.0
        for k in out_ref:
            ok_i, info = compare_outputs(out_ref[k], out_gen[k], atol, rtol)
            if not ok_i:
                info = info or {}
                info["dict_key"] = k
                return False, info
            if info and "max_abs_diff" in info:
                max_diff = max(max_diff, info["max_abs_diff"])
        return True, {"max_abs_diff": max_diff, "allclose": True}

    return out_ref == out_gen, {"reason": "non_tensor_equality", "equal": out_ref == out_gen}


# Mean forward latency after numerical match (reference vs generated).
BENCHMARK_FORWARD_ITERATIONS = 10
BENCHMARK_FORWARD_WARMUP = 10


def _forward_module(model: Any, inputs_device: Any) -> Any:
    """Single forward matching :func:`run_forward_validation` call semantics."""
    import torch

    if isinstance(inputs_device, torch.Tensor):
        return model(inputs_device)
    if isinstance(inputs_device, (tuple, list)):
        return model(*inputs_device)
    return model(inputs_device)


def _mean_forward_latency_seconds(
    model: Any,
    inputs_device: Any,
    *,
    iterations: int,
    warmup: int,
    cuda_sync: bool,
) -> float:
    import torch

    for _ in range(warmup):
        _forward_module(model, inputs_device)
        if cuda_sync:
            torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iterations):
        if cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _forward_module(model, inputs_device)
        if cuda_sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


def run_forward_validation(
    task_path: Path,
    kernel_path: Path,
    seed: int,
    atol: float,
    rtol: float,
    gen_module_name: str = "kernelbench_generated_uniq",
) -> dict[str, Any]:
    import torch

    ref_name = "kernelbench_reference_uniq"

    try:
        ref_mod = import_kernelbench_file(task_path, ref_name)
    except Exception:
        return {
            "runnable": False,
            "status": "compile_error",
            "compile_error": f"Failed to load reference file:\n{traceback.format_exc()}",
        }

    try:
        gen_mod = import_kernelbench_file(kernel_path, gen_module_name)
    except Exception:
        return {
            "runnable": False,
            "status": "compile_error",
            "compile_error": f"Failed to import generated kernel.py:\n{traceback.format_exc()}",
        }

    # Drain prior async CUDA errors (bad kernels from earlier rounds can poison the context).
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            return _cuda_context_error_dict(e)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as e:
        return _cuda_context_error_dict(e)

    try:
        inputs = ref_mod.get_inputs()
    except Exception:
        return {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": f"reference get_inputs() failed:\n{traceback.format_exc()}",
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        inputs_device = _move_tensors_to(inputs, device)
        init_args = list(ref_mod.get_init_inputs())
        model_ref = ref_mod.Model(*init_args).to(device)
        model_gen = gen_mod.Model(*init_args).to(device)
        # Both Model() use the *same* global RNG after get_inputs() but consume two
        # disjoint draws—without syncing, we would compare out_ref(x; W₁) to
        # out_gen(x; W₂) with W₁ ≠ W₂ and get meaningless numerical_error on every
        # non-trivial run. Load reference weights into the generated module so the
        # check is the same I/O and the same parameters.
        model_gen.load_state_dict(model_ref.state_dict(), strict=True)
    except Exception:
        return {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": f"Model construction or state_dict copy failed (architecture mismatch?):\n{traceback.format_exc()}",
        }

    try:
        if isinstance(inputs_device, torch.Tensor):
            out_ref = model_ref(inputs_device)
            out_gen = model_gen(inputs_device)
        elif isinstance(inputs_device, (tuple, list)):
            out_ref = model_ref(*inputs_device)
            out_gen = model_gen(*inputs_device)
        else:
            out_ref = model_ref(inputs_device)
            out_gen = model_gen(inputs_device)
        # Flush async CUDA work so bad generated kernels fail here, not on the next op (e.g. compare_outputs).
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        return {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": f"Forward pass failed:\n{traceback.format_exc()}",
        }

    try:
        ok, num_info = compare_outputs(out_ref, out_gen, atol, rtol)
    except Exception:
        return {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": (
                "Output comparison or host copy (.cpu()) failed—often a deferred CUDA error from the "
                "generated model forward, or a poisoned context. See traceback; try CUDA_LAUNCH_BLOCKING=1.\n"
                f"{traceback.format_exc()}"
            ),
        }
    if not ok:
        return {
            "runnable": False,
            "status": "numerical_error",
            "numerical_error": num_info
            or {"reason": "comparison_failed"},
        }

    benchmark_timing: dict[str, Any]
    try:
        cuda_sync = torch.cuda.is_available() and device.type == "cuda"
        mean_ref = _mean_forward_latency_seconds(
            model_ref,
            inputs_device,
            iterations=BENCHMARK_FORWARD_ITERATIONS,
            warmup=BENCHMARK_FORWARD_WARMUP,
            cuda_sync=cuda_sync,
        )
        mean_gen = _mean_forward_latency_seconds(
            model_gen,
            inputs_device,
            iterations=BENCHMARK_FORWARD_ITERATIONS,
            warmup=BENCHMARK_FORWARD_WARMUP,
            cuda_sync=cuda_sync,
        )
        speedup: Optional[float] = None
        if mean_gen > 0:
            speedup = mean_ref / mean_gen
        benchmark_timing = {
            "iterations": BENCHMARK_FORWARD_ITERATIONS,
            "warmup": BENCHMARK_FORWARD_WARMUP,
            "device": "cuda" if cuda_sync else "cpu",
            "mean_seconds_reference": round(mean_ref, 9),
            "mean_seconds_generated": round(mean_gen, 9),
            "speedup": round(speedup, 9) if speedup is not None else None,
        }
    except Exception:
        benchmark_timing = {
            "skipped": True,
            "reason": traceback.format_exc(),
        }

    return {
        "runnable": True,
        "status": "success",
        "numerical_check": num_info,
        "benchmark_timing": benchmark_timing,
    }


def _subprocess_io_capture(
    proc: CompletedProcess[str], cmd: list[str], *, max_chars: int = 16_000
) -> dict[str, Any]:
    """Structured stderr/stdout/cmd/returncode for metrics when a worker dies or misbehaves."""
    return {
        "returncode": proc.returncode,
        "command": cmd,
        "stderr": (proc.stderr or "")[:max_chars],
        "stdout": (proc.stdout or "")[:max_chars],
    }


def run_forward_validation_subprocess(
    task_path: Path,
    kernel_path: Path,
    seed: int,
    atol: float,
    rtol: float,
    gen_module_name: str = "kernelbench_generated_uniq",
    *,
    cuda_visible_device: Optional[str] = None,
    optkernel_worker_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run :func:`run_forward_validation` in a **fresh Python process** (CUDA isolation from the agent)
    *or* POST the same work to a remote :mod:`worker` HTTP service.

    If ``optkernel_worker_url`` is a non-empty string, calls :func:`worker_client.run_validation_via_worker`
    (see that module for path visibility requirements on the worker host).
    Otherwise runs a local child process as before.

    One local-child JSON object is printed to stdout by the child; this function parses and returns it.
    If the child exits before valid JSON (crash, OOM, SIGKILL), stderr/stdout are still captured in
    ``result["subprocess"]`` for :file:`metrics.json`.

    :param cuda_visible_device: if set (local child only), passed as ``CUDA_VISIBLE_DEVICES`` for the child
        (exclusive GPU in multi-tenant :mod:`worker` when not using HTTP).
    :param optkernel_worker_url: e.g. ``http://host:9876``; if set, use HTTP instead of a local child.
    """
    wu = (optkernel_worker_url or "").strip()
    if wu:
        from worker_client import run_validation_via_worker

        return run_validation_via_worker(
            wu,
            task_path,
            kernel_path,
            seed,
            atol,
            rtol,
            gen_module_name,
        )
    root = Path(__file__).resolve().parent
    script = root / "run_validation.py"
    cmd: list[str] = [
        sys.executable,
        str(script),
        "forward-validation",
        "--task-file",
        str(task_path.resolve()),
        "--kernel-file",
        str(kernel_path.resolve()),
        "--seed",
        str(seed),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
        "--gen-module-name",
        gen_module_name,
    ]
    _env = os.environ.copy()
    if cuda_visible_device is not None and str(cuda_visible_device).strip() != "":
        _env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(root),
        env=_env,
    )
    out = (proc.stdout or "").strip()
    if not out:
        cap = _subprocess_io_capture(proc, cmd)
        return {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": (
                f"validation worker process exited with code {proc.returncode} and no JSON on stdout. "
                "Check subprocess.stderr in metrics (e.g. tracebacks, OOM, signal)."
            ),
            "subprocess": cap,
        }
    try:
        parsed: dict[str, Any] = json.loads(out)
    except json.JSONDecodeError as e:
        cap = _subprocess_io_capture(proc, cmd)
        return {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": (
                f"validation worker: invalid JSON on stdout ({e!s}). "
                "See subprocess.stderr/stdout in metrics for the raw process output."
            ),
            "subprocess": cap,
        }
    if proc.returncode not in (0, None):
        # Worker should exit 0 once it prints; non-zero is unusual — attach diagnostics.
        parsed = dict(parsed)
        parsed["subprocess_exit_warning"] = _subprocess_io_capture(proc, cmd)
    return parsed


def _main_forward_validation_worker() -> int:
    """``python run_validation.py forward-validation --task-file ...`` — prints one JSON object to stdout."""
    import argparse

    p = argparse.ArgumentParser(description="Isolated run_forward_validation worker (for agent subprocess).")
    p.add_argument("--task-file", type=Path, required=True)
    p.add_argument("--kernel-file", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--gen-module-name", type=str, default="kernelbench_generated_uniq")
    args = p.parse_args()
    try:
        result: dict[str, Any] = run_forward_validation(
            args.task_file.resolve(),
            args.kernel_file.resolve(),
            args.seed,
            args.atol,
            args.rtol,
            gen_module_name=args.gen_module_name,
        )
    except Exception as e:  # noqa: BLE001 — worker must print JSON
        result = {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": f"run_forward_validation raised: {e!r}\n{traceback.format_exc()}",
        }
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "forward-validation":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        raise SystemExit(_main_forward_validation_worker())
    print("Usage: run_validation.py forward-validation --task-file T --kernel-file K [...]", file=sys.stderr)
    raise SystemExit(2)
