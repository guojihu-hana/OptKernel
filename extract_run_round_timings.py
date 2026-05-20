#!/usr/bin/env python3
"""
Mirror OptKernel ``runs/<stamp>/level1/<task>/round_* metrics timing rows into

``data/<stamp>/level1/<task>/timings.json``

Uses GPU-queue-exclusive durations via ``queue_timing.execute_s`` when present:
validation/profile excludes worker queue ``wait_s``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent

_ROUND_DIR = re.compile(r"^round_(\d+)$", re.IGNORECASE)


def _maybe_phase_secs_exclude_queue(block: Any, *, allow_wall_fallback: bool) -> float | None:
    """Prefer GPU-runner execution slice ``execute_s``; else optionally fallback to phase ``seconds``."""
    if not isinstance(block, dict):
        return None
    qt = block.get("queue_timing")
    if isinstance(qt, dict):
        exe = qt.get("execute_s")
        if isinstance(exe, (int, float)) and exe >= 0:
            return float(exe)
        exe_legacy = qt.get("execute_seconds")  # defensive alias if introduced later
        if isinstance(exe_legacy, (int, float)) and exe_legacy >= 0:
            return float(exe_legacy)
    wall = block.get("seconds")
    if allow_wall_fallback and isinstance(wall, (int, float)) and wall >= 0:
        return float(wall)
    return None


def _llm_seconds(block: Any) -> float | None:
    if not isinstance(block, dict):
        return None
    s = block.get("seconds")
    return float(s) if isinstance(s, (int, float)) and s >= 0 else None


def _profile_seconds_exclude_queue(ncu_block: Any) -> float | None:
    if not isinstance(ncu_block, dict) or ncu_block.get("skipped") is True:
        return None
    # Prefer GPU worker measured exec slice identical semantics as validation.
    return _maybe_phase_secs_exclude_queue(ncu_block, allow_wall_fallback=True)


def _round_dirs_sorted(level_task_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for rd in level_task_dir.iterdir():
        if not rd.is_dir():
            continue
        m = _ROUND_DIR.match(rd.name)
        if not m:
            continue
        pairs.append((int(m.group(1)), rd))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _metrics_timing_record(metrics_path: Path, metrics_relative_under_stamp: str) -> dict[str, Any]:
    raw_text = metrics_path.read_text(encoding="utf-8")
    data = json.loads(raw_text)

    et = data.get("eval_timing") if isinstance(data, dict) else None

    llm_b = et.get("llm") if isinstance(et, dict) else None
    val_b = et.get("validation") if isinstance(et, dict) else None
    prof_b = et.get("ncu") if isinstance(et, dict) else None

    rn = data.get("round") if isinstance(data.get("round"), int) else None

    rec: dict[str, Any] = {
        "round": rn,
        "round_segment": metrics_path.parent.name if metrics_path.parent != metrics_path else "",
        "metrics_relative_path": metrics_relative_under_stamp,
        "llm_s": _llm_seconds(llm_b),
        "validation_s_no_queue": (
            _maybe_phase_secs_exclude_queue(val_b, allow_wall_fallback=True) if val_b is not None else None
        ),
        "profile_s_no_queue": _profile_seconds_exclude_queue(prof_b),
    }

    status = data.get("status") if isinstance(data.get("status"), str) else None
    if status:
        rec["metrics_status"] = status

    return rec


def _mirror_stamp(run_root: Path, data_root: Path) -> tuple[int, int]:
    """Return (tasks_written, rounds_written_total)."""
    stamp = run_root.resolve().name
    level1 = run_root / "level1"
    if not level1.is_dir():
        raise SystemExit(f"No level1/ under run root: {run_root}")

    tasks_written = 0
    rounds_total = 0

    tasks = sorted([p for p in level1.iterdir() if p.is_dir()], key=lambda p: p.name)

    dest_level1 = data_root / stamp / "level1"

    for td in tasks:
        round_pairs = _round_dirs_sorted(td)
        rows: list[dict[str, Any]] = []
        for _, rd in round_pairs:
            mp = rd / "metrics.json"
            if not mp.is_file():
                continue
            rel_under_stamp = mp.relative_to(run_root.resolve()).as_posix()
            rows.append(_metrics_timing_record(mp, rel_under_stamp))

        if not rows:
            continue

        model_name: str | None = None
        for _, rd in round_pairs:
            fmp = rd / "metrics.json"
            if not fmp.is_file():
                continue
            fm_data = json.loads(fmp.read_text(encoding="utf-8"))
            mn = fm_data.get("model_name")
            if isinstance(mn, str):
                model_name = mn
            break

        payload: dict[str, Any] = {
            "run_stamp": stamp,
            "source_runs_root": str(run_root.resolve()),
            "task_dir_name": td.name,
            "model_name": model_name,
            "timing_fields": {
                "llm_s": "eval_timing.llm.seconds",
                "validation_s_no_queue": (
                    "eval_timing.validation.queue_timing.execute_s if present, "
                    "else validation.seconds (local / no queue slice)"
                ),
                "profile_s_no_queue": (
                    "eval_timing.ncu.queue_timing.execute_s if ncu phase ran; skipped rounds null"
                ),
            },
            "rounds": rows,
        }

        dest_task = dest_level1 / td.name
        dest_task.mkdir(parents=True, exist_ok=True)
        out_fp = dest_task / "timings.json"
        out_fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tasks_written += 1
        rounds_total += len(rows)

    return tasks_written, rounds_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_roots",
        nargs="+",
        type=Path,
        help="One or more .../runs/<stamp> directories (must contain level1/).",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=_REPO_ROOT / "data",
        help=f"Output root (default: {_REPO_ROOT / 'data'})",
    )
    args = ap.parse_args()
    data_root = args.data_root.expanduser().resolve()

    grand_tasks = 0
    grand_rounds = 0
    for rr in args.run_roots:
        r = rr.expanduser().resolve()
        if not r.is_dir():
            print(f"Skip (not a dir): {r}", flush=True)
            continue
        tw, rw = _mirror_stamp(r, data_root)
        print(f"{r.name}: tasks={tw} rounds={rw} -> {data_root / r.name / 'level1'}", flush=True)
        grand_tasks += tw
        grand_rounds += rw

    print(f"Total: tasks={grand_tasks} rounds={grand_rounds}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
