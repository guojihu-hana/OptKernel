#!/usr/bin/env python3
"""
Used by ``run_spec_*_batch.sh`` when ``PARALLEL_MODE=rounds``.

1. Lists every ``spec_act_runs/.../round_*`` numeric index under the task mirror.
2. Takes the **last N rounds** by index (``N = --parallel-window``, e.g. ``MAX_PARALLEL=10``
   ⇒ the ten **largest** round numbers that appear in the directory, not necessarily
   consecutive).
3. From **smallest to largest among those N**, picks the **first** round that still needs
   work: same head list as ``speculative_agent --sweep`` (from ``runs/…/round_*/llm_output.txt``
   token count); a head is ``done`` if ``head<t>/metrics.json`` exists under the spec mirror.
4. Prints that round index for ``--start-round``. Actual **「接着往后的 head」** is done inside
   ``speculative_agent`` with ``--sweep-skip-complete-heads`` (no directory deletes here).

If every audited round is fully done ⇒ prints ``hi + 1`` where ``hi`` is the global max index.

If spec tree has no ``round_*`` yet ⇒ prints ``0``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _sorted_spec_round_indices(spec_parent: Path) -> list[int]:
    idx: list[int] = []
    if not spec_parent.is_dir():
        return idx
    for p in spec_parent.glob("round_*"):
        if not p.is_dir():
            continue
        base = p.name.removeprefix("round_")
        if not base.isdigit():
            continue
        idx.append(int(base, 10))
    idx.sort()
    return idx


def main(argv: list[str] | None = None) -> int:
    from speculative_agent import (
        _count_tokens,
        _head_sizes_through_full,
        _read_text,
        _resolve_output_dir,
    )

    ap = argparse.ArgumentParser(description="Resume --start_round for spec batch rounds mode.")
    ap.add_argument("runs_task_dir", type=Path, help="Path to …/runs/…/<task>/")
    ap.add_argument("--head-step", type=int, required=True, dest="head_step")
    ap.add_argument(
        "--parallel-window",
        type=int,
        required=True,
        dest="parallel_window",
        help="Scan the last K round indices present under spec_act_runs (batch: MAX_PARALLEL).",
    )
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Optional HF tokenizer (match speculative_agent --tokenizer).",
    )
    args = ap.parse_args(argv)

    step = max(1, int(args.head_step))
    window = max(1, int(args.parallel_window))
    tokenizer = (args.tokenizer or "").strip() or None

    runs_task = args.runs_task_dir.resolve()
    spec_parent = _resolve_output_dir(runs_task)

    indices = _sorted_spec_round_indices(spec_parent)
    if not indices:
        print(0, end="")
        return 0

    hi = indices[-1]
    audit = indices[-window:]

    def expected_heads(rr: Path) -> list[int] | None:
        llm = rr / "llm_output.txt"
        if not llm.is_file():
            return None
        total = _count_tokens(_read_text(llm), tokenizer)
        hs = _head_sizes_through_full(total, step)
        return hs if hs else None

    def incomplete(r: int, hs: list[int], spec_round_out: Path) -> bool:
        for h in hs:
            if not (spec_round_out / f"head{h}" / "metrics.json").is_file():
                return True
        return False

    for r in sorted(audit):
        runs_rr = runs_task / f"round_{r:03d}"
        if not runs_rr.is_dir():
            continue
        spec_rr_out = _resolve_output_dir(runs_rr)
        hs = expected_heads(runs_rr)
        if hs is None:
            print(r, end="")
            return 0
        if not spec_rr_out.is_dir() or incomplete(r, hs, spec_rr_out):
            print(r, end="")
            return 0

    print(hi + 1, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
