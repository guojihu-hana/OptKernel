#!/usr/bin/env python3
"""
Test parsing metrics from an Nsight Compute ``.ncu-rep`` file (same path as ``run_ncu.import_ncu_report_metrics``).

Example::

    python check_ncu_rep_metrics.py \\
      /mnt/shared-storage-user/ailab-sys/guojihu/OptKernel/runs/kgen_vllm8/round_000/kernel_profile.ncu-rep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_ncu import DEFAULT_NCU_METRICS, import_ncu_report_metrics, _truncate


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run ncu --import on a .ncu-rep and print parsed metrics (uses run_ncu parser).",
    )
    p.add_argument(
        "report",
        nargs="?",
        default=str(
            _ROOT
            / "runs/kgen_vllm8/round_000/kernel_profile.ncu-rep"
        ),
        type=str,
        help="Path to kernel_profile.ncu-rep",
    )
    p.add_argument(
        "--ncu-binary",
        default="ncu",
        help="ncu executable (default: ncu on PATH)",
    )
    p.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metric names to keep (default: run_ncu.DEFAULT_NCU_METRICS)",
    )
    p.add_argument(
        "--raw-head",
        type=int,
        default=12000,
        help="When metrics dict is empty, print this many chars of ncu stdout (0=skip)",
    )
    args = p.parse_args()

    report = Path(args.report).resolve()
    if not report.is_file():
        print(f"File not found: {report}", file=sys.stderr)
        return 1

    names = (
        [x.strip() for x in args.metrics.split(",") if x.strip()]
        if args.metrics.strip()
        else list(DEFAULT_NCU_METRICS)
    )

    try:
        rc, metrics, err, out = import_ncu_report_metrics(args.ncu_binary, report, names)
    except FileNotFoundError:
        print(
            f"Could not run ncu binary: {args.ncu_binary!r} (not on PATH?). "
            f"Example: --ncu-binary /usr/local/cuda/NsightCompute-*/ncu",
            file=sys.stderr,
        )
        return 127

    print(f"report: {report}")
    print(f"ncu --import returncode: {rc}")
    if err:
        print(f"ncu stderr: {err}")
    print(f"parsed metric count: {len(metrics)}")
    print("metrics (JSON):")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if rc == 0 and not metrics and args.raw_head > 0:
        print("\n--- ncu --import stdout sample (parser matched no rows; compare format) ---")
        print(_truncate(out, args.raw_head))

    return 0 if rc == 0 and metrics else (1 if rc != 0 else 2)


if __name__ == "__main__":
    raise SystemExit(main())
