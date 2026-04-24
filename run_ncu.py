"""
Nsight Compute (ncu) profiling helpers for generated kernel modules.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# Defaults for ncu --launch-skip / --launch-count (profile window).
SKIP_K = 5
PROFILE_K = 10

# Forward iterations in ncu_kernel_harness.py (more launches → easier to profile after --launch-skip).
NCU_HARNESS_ITERATIONS = 20

_NCU_HARNESS_SCRIPT = Path(__file__).resolve().parent / "ncu_kernel_harness.py"

DEFAULT_NCU_METRICS: tuple[str, ...] = (
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "smsp__inst_executed_op_shared_st.sum",
)


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n // 2] + "\n...\n" + s[-(n // 2) :]


def nccu_bin(ncu_binary: str) -> str:
    return ncu_binary.split()[0] if ncu_binary else "ncu"


def effective_ncu_metrics(user_metrics: list[str]) -> list[str]:
    """Use explicit ``user_metrics`` when non-empty; otherwise :data:`DEFAULT_NCU_METRICS`."""
    return list(user_metrics) if user_metrics else list(DEFAULT_NCU_METRICS)


def _parse_ncu_value_cell(s: str) -> float | str:
    """Parse a Raw-page value cell (may include SI suffix or ``;``-separated instances)."""
    s = s.strip().strip('"')
    if not s:
        return s
    if ";" in s:
        parts: list[float] = []
        for chunk in s.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            tok = chunk.replace(",", "").split()
            if not tok:
                continue
            try:
                parts.append(float(tok[0]))
            except ValueError:
                return s
        return sum(parts) / len(parts) if parts else s
    tok = s.replace(",", "").split()
    if not tok:
        return s
    try:
        return float(tok[0])
    except ValueError:
        return s


def _cell_parses_as_float(s: str) -> bool:
    return isinstance(_parse_ncu_value_cell(s), float)


def _resolve_metric_row(row: list[str], name_idx: int) -> Optional[tuple[str, str, str]]:
    """
    Given ``row[name_idx]`` is the metric id cell, find unit + value.

    ncu Raw CSV is often ``..., name, unit, value`` with a numeric last field; some rows
    place another metric name after ``name`` (comparison / correlation). In that case
    ``name, other_metric_id, ...`` must not treat ``other_metric_id`` as the value: prefer
    the **last** column if it parses as a float.
    """
    name = row[name_idx]
    n = len(row)
    if name_idx >= n - 1:
        return None
    # Standard: name, unit, value (value is numeric)
    if name_idx + 2 < n and _cell_parses_as_float(row[name_idx + 2]):
        return name, row[name_idx + 1], row[name_idx + 2]
    # Wide row: value is last cell (e.g. PID, kernel, section, name, unit, value)
    last = row[-1]
    if name_idx < n - 1 and _cell_parses_as_float(last):
        if name_idx == n - 2:
            return name, "", last
        unit = row[-2] if n >= 2 else ""
        return name, unit, last
    # name, value only
    if name_idx + 1 < n and _cell_parses_as_float(row[name_idx + 1]):
        return name, "", row[name_idx + 1]
    return None


def _metric_row_name_unit_value(row: list[str]) -> Optional[tuple[str, str, str]]:
    """Detect Raw CSV row shape: ``name, unit, value`` or ``kernel, name, unit, value``."""
    row = [c.strip().strip('"') for c in row]
    if len(row) >= 4 and "__" in row[1]:
        return _resolve_metric_row(row, 1)
    if len(row) >= 3 and "__" in row[0]:
        return _resolve_metric_row(row, 0)
    return None


def _is_header_row(row: list[str]) -> bool:
    j = " ".join(row).lower()
    if "metric name" in j or ("metric" in j and "unit" in j and "value" in j):
        return True
    return j.strip().startswith("id,") and "kernel" in j


def _row_match_metric(
    row: list[str],
    only_names: Optional[set[str]],
) -> Optional[tuple[str, str, str]]:
    """
    Find (metric_name, unit, value) on one CSV row.

    Handles: table headers with ``Metric Name`` column; rows where the metric id appears in any column.
    """
    row = [c.strip().strip('"') for c in row]
    if not row:
        return None

    # Explicit metric column: any requested name appears as a full cell (common in ncu exports).
    if only_names:
        for name in only_names:
            if name not in row:
                continue
            i = row.index(name)
            got = _resolve_metric_row(row, i)
            if got:
                return got

    m = _metric_row_name_unit_value(row)
    if m:
        name, u, v = m
        if only_names is None or name in only_names:
            return name, u, v
        return None

    # Scan for a cell that looks like an Nsight metric id (contains __).
    for i, cell in enumerate(row):
        if "__" not in cell:
            continue
        if only_names is not None and cell not in only_names:
            continue
        got = _resolve_metric_row(row, i)
        if got:
            return got
    return None


def parse_ncu_raw_csv(
    text: str,
    only_names: Optional[set[str]] = None,
) -> dict[str, float | str]:
    """
    Parse ``ncu --import ... --page raw --csv`` stdout into metric -> scalar.

    If the same metric appears for multiple kernel launches, numeric values are **averaged**.
    If ``only_names`` is set, only metrics whose name is in that set are kept (exact match).
    """
    # Drop ncu profiler banner lines; parse full CSV (handles quoted commas).
    clean = "\n".join(
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("==")
    )
    if not clean.strip():
        return {}

    try:
        rows = list(csv.reader(io.StringIO(clean)))
    except csv.Error:
        return {}

    accum: dict[str, list[float | str]] = defaultdict(list)
    for row in rows:
        if not row or all(not (c or "").strip() for c in row):
            continue
        row = [c.strip().strip('"') for c in row]
        if _is_header_row(row):
            continue
        muv = _row_match_metric(row, only_names)
        if muv is None:
            continue
        name, _unit, val_cell = muv
        v = _parse_ncu_value_cell(val_cell)
        accum[name].append(v)

    out: dict[str, float | str] = {}
    for name, vals in accum.items():
        if not vals:
            continue
        if all(isinstance(x, float) for x in vals):
            out[name] = sum(vals) / len(vals)  # type: ignore[arg-type]
        else:
            out[name] = vals[-1]
    return out


def parse_ncu_wide_csv(
    text: str,
    only_names: Optional[set[str]],
) -> dict[str, float | str]:
    """
    Parse ncu ``--page raw --csv`` when the tool emits **one row per kernel** and each
    metric is a **column** (header row lists ``sm__...``, ``smsp__...``, ``Kernel Name``, etc.).

    This is the layout shown by recent Nsight Compute CLI; it is not the three-column
    ``name, unit, value`` layout that :func:`parse_ncu_raw_csv` handles.
    """
    if not only_names:
        return {}

    clean = "\n".join(
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("==")
    )
    if not clean.strip():
        return {}

    try:
        rows = list(csv.reader(io.StringIO(clean)))
    except csv.Error:
        return {}

    if len(rows) < 2:
        return {}

    header = [c.strip().strip('"') for c in rows[0]]
    lowered = [h.lower() for h in header]
    if "kernel name" not in lowered:
        return {}

    idx_of = {h: i for i, h in enumerate(header)}
    accum: dict[str, list[float | str]] = defaultdict(list)

    for row in rows[1:]:
        if not row or all(not (c or "").strip() for c in row):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        row = [c.strip().strip('"') for c in row[: len(header)]]
        for name in only_names:
            if name not in idx_of:
                continue
            j = idx_of[name]
            v = _parse_ncu_value_cell(row[j])
            accum[name].append(v)

    out: dict[str, float | str] = {}
    for name, vals in accum.items():
        if not vals:
            continue
        if all(isinstance(x, float) for x in vals):
            out[name] = sum(vals) / len(vals)  # type: ignore[arg-type]
        else:
            out[name] = vals[-1]
    return out


def import_ncu_report_metrics(
    ncu_binary: str,
    report_path: Path,
    metric_names: list[str],
) -> tuple[int, dict[str, float | str], str, str]:
    """
    Run ``ncu --import <report> --page raw --csv`` and parse metrics.

    Returns ``(returncode, metrics_dict, stderr, stdout)``.
    """
    cmd = [
        nccu_bin(ncu_binary),
        "--import",
        str(report_path.resolve()),
        "--page",
        "raw",
        "--csv",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    stdout = proc.stdout or ""
    stderr = (proc.stderr or "").strip()
    want = set(metric_names) if metric_names else None
    if proc.returncode != 0:
        return proc.returncode, {}, (proc.stderr or proc.stdout or "").strip(), stdout

    metrics = parse_ncu_raw_csv(stdout, want)
    # If layout differs, parse all rows then keep requested names only.
    if want and not metrics:
        metrics = {
            k: v for k, v in parse_ncu_raw_csv(stdout, None).items() if k in want
        }
    # Wide-table CSV: metric names are column headers, one row per kernel launch.
    if want and not metrics:
        metrics = parse_ncu_wide_csv(stdout, want)
    return 0, metrics, stderr, stdout


def run_ncu_profile(
    kernel_path: Path,
    work_dir: Path,
    ncu_binary: str,
    metrics_args: list[str],
    extra_args: list[str],
    metric_names: list[str],
    *,
    launch_skip: int = SKIP_K,
    launch_count: int = PROFILE_K,
) -> dict[str, Any]:
    report_path = work_dir / "kernel_profile.ncu-rep"
    cmd: list[str] = [
        nccu_bin(ncu_binary),
        "-f",
        "-o",
        str(report_path),
        "--target-processes",
        "all",
        "--profile-from-start",
        "on",
        f"--launch-skip={launch_skip}",
        f"--launch-count={launch_count}",
    ]
    cmd.extend(extra_args)
    cmd.extend(metrics_args)
    cmd.extend(
        [
            sys.executable,
            str(_NCU_HARNESS_SCRIPT),
            str(kernel_path.resolve()),
            str(NCU_HARNESS_ITERATIONS),
        ]
    )

    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=None,
    )
    out: dict[str, Any] = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": _truncate(proc.stdout or "", 12000),
        "stderr": _truncate(proc.stderr or "", 12000),
        "report_path": str(report_path) if report_path.exists() else None,
        "launch_skip": launch_skip,
        "launch_count": launch_count,
        "harness_script": str(_NCU_HARNESS_SCRIPT),
        "harness_kernel": str(kernel_path.resolve()),
        "harness_iterations": NCU_HARNESS_ITERATIONS,
    }

    if report_path.is_file():
        imp_rc, metrics_map, imp_err, imp_out = import_ncu_report_metrics(
            ncu_binary, report_path, metric_names
        )
        out["import_metrics_returncode"] = imp_rc
        if imp_err:
            out["import_metrics_stderr"] = _truncate(imp_err, 4000)
        if imp_rc == 0:
            # Always record ``metrics`` when import succeeds (may be {} if parser mismatched ncu's CSV).
            out["metrics"] = metrics_map
            if not metrics_map:
                out["import_csv_sample"] = _truncate(imp_out, 8000)
        else:
            out["metrics_import_error"] = _truncate(
                imp_err or "ncu --import failed",
                4000,
            )
    return out


def run_ncu_profile_subprocess(
    kernel_path: Path,
    work_dir: Path,
    ncu_binary: str,
    metrics_args: list[str],
    extra_args: list[str],
    *,
    launch_skip: int = SKIP_K,
    launch_count: int = PROFILE_K,
    cuda_visible_device: Optional[str] = None,
    optkernel_worker_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run :func:`run_ncu_profile` in a **fresh Python process** *or* POST the same work to
    a remote :mod:`worker` HTTP service.

    If ``optkernel_worker_url`` is non-empty, uses :func:`worker_client.run_ncu_via_worker``
    and ``metrics_args`` (``--metrics,comma...``) to build the request body.

    :param cuda_visible_device: if set (local child only), passed as ``CUDA_VISIBLE_DEVICES`` for the child
        (exclusive GPU in :mod:`worker` when not using HTTP).
    :param optkernel_worker_url: e.g. ``http://host:9876``; if set, use HTTP instead of a local child.
    """
    wu = (optkernel_worker_url or "").strip()
    if wu:
        from worker_client import run_ncu_via_worker

        metrics_comma = ""
        if len(metrics_args) >= 2 and str(metrics_args[0]).strip() == "--metrics":
            metrics_comma = str(metrics_args[1])
        if not (metrics_comma and str(metrics_comma).strip()):
            metrics_comma = ",".join(effective_ncu_metrics([]))
        return run_ncu_via_worker(
            wu,
            kernel_path,
            work_dir,
            ncu_binary,
            str(metrics_comma).strip(),
            extra_args,
            launch_skip=launch_skip,
            launch_count=launch_count,
        )
    root = Path(__file__).resolve().parent
    script = root / "run_ncu.py"
    metrics_joined = ""
    if len(metrics_args) >= 2 and str(metrics_args[0]).strip() == "--metrics":
        metrics_joined = str(metrics_args[1])
    cmd: list[str] = [
        sys.executable,
        str(script),
        "ncu-profile",
        "--kernel-file",
        str(Path(kernel_path).resolve()),
        "--work-dir",
        str(Path(work_dir).resolve()),
        "--ncu-binary",
        ncu_binary,
        "--metrics-joined",
        metrics_joined,
        "--extra-json",
        json.dumps(extra_args),
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        str(launch_count),
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
        rc = proc.returncode if proc.returncode is not None else -1
        return {
            "returncode": rc,
            "stdout": (proc.stdout or "")[:12_000],
            "stderr": f"ncu worker: no JSON on stdout. exit={rc}. See subprocess fields.",
            "subprocess": {
                "returncode": rc,
                "command": cmd,
                "stderr": (proc.stderr or "")[:16_000],
                "stdout": (proc.stdout or "")[:16_000],
            },
        }
    try:
        result: dict[str, Any] = json.loads(out)
    except json.JSONDecodeError as e:
        rc = proc.returncode if proc.returncode is not None else -1
        return {
            "returncode": rc,
            "stderr": f"ncu worker: invalid JSON ({e!s})",
            "subprocess": {
                "returncode": rc,
                "command": cmd,
                "stderr": (proc.stderr or "")[:16_000],
                "stdout": (out[:16_000] if out else (proc.stdout or ""))[:16_000],
            },
        }
    if proc.returncode not in (0, None) and "subprocess_exit_warning" not in result:
        result = dict(result)
        result["subprocess_exit_warning"] = {
            "returncode": proc.returncode,
            "command": cmd,
            "stderr": (proc.stderr or "")[:12_000],
            "stdout": (proc.stdout or "")[:8_000],
        }
    return result


def _main_ncu_profile_worker() -> int:
    """``python run_ncu.py ncu-profile --kernel-file ...`` — one JSON on stdout."""
    import argparse

    p = argparse.ArgumentParser(description="Isolated ncu run_ncu_profile worker (for agent subprocess).")
    p.add_argument("--kernel-file", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--ncu-binary", type=str, default="ncu")
    p.add_argument("--metrics-joined", type=str, default="")
    p.add_argument("--extra-json", type=str, default="[]")
    p.add_argument("--launch-skip", type=int, default=SKIP_K)
    p.add_argument("--launch-count", type=int, default=PROFILE_K)
    args = p.parse_args()
    try:
        names = [x.strip() for x in (args.metrics_joined or "").split(",") if x.strip()]
        metric_names = effective_ncu_metrics(names)
        metrics_args = ["--metrics", ",".join(metric_names)]
        extra: Any = json.loads(args.extra_json or "[]")
        if not isinstance(extra, list):
            extra = []
        extra_s = [str(x) for x in extra]
        out = run_ncu_profile(
            args.kernel_file.resolve(),
            args.work_dir.resolve(),
            args.ncu_binary,
            metrics_args,
            extra_s,
            metric_names,
            launch_skip=args.launch_skip,
            launch_count=args.launch_count,
        )
    except Exception as e:  # noqa: BLE001
        out = {
            "returncode": -1,
            "worker_error": f"{e!r}\n{traceback.format_exc()}",
        }
    print(json.dumps(out, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ncu-profile":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        raise SystemExit(_main_ncu_profile_worker())
    print("Usage: run_ncu.py ncu-profile --kernel-file K --work-dir D [...]", file=sys.stderr)
    raise SystemExit(2)
