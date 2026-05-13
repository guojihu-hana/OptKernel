from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROUND_DIR_RE = re.compile(r"^round_(\d{3})$")


@dataclass(frozen=True)
class Task:
    task_file: Path
    work_dir: Path


def _discover_tasks(
    level_dir: Path,
    output_root: Path,
    only_task_stems: Optional[list[str]] = None,
) -> list[Task]:
    files = sorted(level_dir.glob("*.py"))
    if not files:
        raise FileNotFoundError(f"No .py tasks found under {level_dir}")
    if only_task_stems:
        seen: set[str] = set()
        stems_ordered: list[str] = []
        for raw in only_task_stems:
            s = raw.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            stems_ordered.append(s)
        want = set(stems_ordered)
        files = [f for f in files if f.stem in want]
        missing = want - {f.stem for f in files}
        if missing:
            raise FileNotFoundError(
                f"No .py for task stem(s) under {level_dir}: {sorted(missing)}"
            )
        order = {s: i for i, s in enumerate(stems_ordered)}
        files.sort(key=lambda f: (order.get(f.stem, 10**9), f.name))
    out: list[Task] = []
    for f in files:
        out.append(Task(task_file=f.resolve(), work_dir=(output_root / f.stem).resolve()))
    return out


def _detect_last_round_index(work_dir: Path) -> int:
    """Highest k such that ``work_dir/round_kkk`` exists; -1 if none or work_dir missing."""
    if not work_dir.is_dir():
        return -1
    best = -1
    for p in work_dir.iterdir():
        if not p.is_dir():
            continue
        m = _ROUND_DIR_RE.match(p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def _resume_start_round(work_dir: Path) -> int:
    """First round index when resuming: redo from the highest existing ``round_*`` (or 0 if none)."""
    last = _detect_last_round_index(work_dir)
    return 0 if last < 0 else last


def _effective_max_rounds(
    start_round: int,
    rounds_per_task: int,
    max_round_exclusive: Optional[int],
) -> int:
    """
    Agent ``--max-rounds`` is an **exclusive** end index: runs ``range(start, max_rounds)``.

    When ``max_round_exclusive`` is set (same cap for every task), it matches that convention:
    ``exclusive_end = min(start + rounds_per_task, max_round_exclusive)``.
    Last round index run is ``exclusive_end - 1`` (e.g. ``--max-round 100`` → through ``round_099``).
    """
    planned = int(start_round) + int(rounds_per_task)
    if max_round_exclusive is None:
        return planned
    return min(planned, int(max_round_exclusive))


def _build_agent_cmd(
    args: argparse.Namespace,
    task: Task,
    start_round: int,
    max_rounds: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "agent.py").resolve()),
        "--task-file",
        str(task.task_file),
        "--work-dir",
        str(task.work_dir),
        "--start-round",
        str(int(start_round)),
        "--max-rounds",
        str(max_rounds),
        "--server-type",
        str(args.server_type),
        "--server-address",
        str(args.server_address),
        "--server-port",
        str(int(args.server_port)),
        "--model",
        str(args.model),
        "--api-key",
        str(args.api_key),
        "--max-tokens",
        str(int(args.max_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--gpu-type",
        str(args.gpu_type),
    ]

    if int(args.max_context_length) > 0:
        cmd += ["--max-context-length", str(int(args.max_context_length))]
    if args.no_ncu:
        cmd.append("--no-ncu")
    if args.worker_url:
        cmd += ["--worker-url", str(args.worker_url)]
    if args.no_reasoning:
        cmd.append("--no-reasoning")
    if args.reasoning_only_rounds:
        cmd += ["--reasoning-only-rounds", str(args.reasoning_only_rounds)]
    if args.reasoning_except_rounds:
        cmd += ["--reasoning-except-rounds", str(args.reasoning_except_rounds)]
    return cmd


def _run_one(
    args: argparse.Namespace,
    task: Task,
    start_round: int,
    max_rounds: int,
) -> tuple[Task, int]:
    task.work_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_agent_cmd(args, task, start_round, max_rounds)
    env = os.environ.copy()
    if args.extra_env:
        for kv in args.extra_env:
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
    p = subprocess.run(cmd, env=env)
    return task, int(p.returncode)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run all tasks in a KernelBench level with bounded parallelism. "
            "Each task runs exactly N rounds, then scheduler moves to next task."
        )
    )
    p.add_argument("--level-dir", type=Path, required=True, help="e.g. ./KernelBench/level1")
    p.add_argument("--rounds-per-task", type=int, required=True, help="N rounds per task")
    p.add_argument("--parallel", type=int, default=10, help="Max concurrent tasks (default 10)")
    p.add_argument("--start-round", type=int, default=0, help="Start round index for every task (ignored when --resume)")
    p.add_argument("--output-root", type=Path, default=None, help="Default: ./runs/<timestamp>/<level_name>")
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume under --output-root: for each task, set --start-round to the highest existing "
            "round_NNN under that task's work_dir (re-run that round), then run --rounds-per-task "
            "rounds inclusive from there (indices [start, start+N)). Requires --output-root."
        ),
    )
    p.add_argument(
        "--only-tasks",
        type=str,
        default="",
        help=(
            "Comma-separated KernelBench task stems (basename without .py). If set, only those "
            "tasks from --level-dir are run, in list order. Default: all *.py under level-dir."
        ),
    )
    p.add_argument(
        "--max-round",
        type=int,
        default=None,
        metavar="E",
        help=(
            "Exclusive cap on agent --max-rounds for every task (same as agent.py): runs "
            "range(start, min(start+rounds_per_task, E)). E.g. --max-round 100 runs through "
            "round_099 only. Tasks with start_round >= E are skipped. Omit for no cap."
        ),
    )

    # Agent passthrough (core subset)
    p.add_argument("--server-type", type=str, default=os.environ.get("KERNEL_AGENT_SERVER", "vllm"))
    p.add_argument("--server-address", type=str, default="localhost")
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--model", type=str, default=os.environ.get("KERNEL_AGENT_MODEL", "glm-5.1"))
    p.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--gpu-type", type=str, default=os.environ.get("KERNEL_AGENT_GPU_TYPE", "H200"))
    p.add_argument("--max-tokens", type=int, default=65536)
    p.add_argument("--max-context-length", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--worker-url", type=str, default=os.environ.get("OPTKERNEL_WORKER_URL", ""))
    p.add_argument("--no-ncu", action="store_true")
    p.add_argument("--no-reasoning", action="store_true")
    p.add_argument("--reasoning-only-rounds", type=str, default="")
    p.add_argument("--reasoning-except-rounds", type=str, default="")
    p.add_argument(
        "--extra-env",
        action="append",
        default=[],
        help="Extra env vars for child agent, format KEY=VALUE (repeatable)",
    )
    args = p.parse_args()

    if args.rounds_per_task <= 0:
        raise ValueError("--rounds-per-task must be > 0")
    if args.parallel <= 0:
        raise ValueError("--parallel must be > 0")
    if args.resume and args.output_root is None:
        raise ValueError("--resume requires --output-root (existing run directory)")
    if args.max_round is not None and int(args.max_round) < 0:
        raise ValueError("--max-round must be >= 0 (exclusive end; use 0 only to force no work)")

    level_dir = args.level_dir.resolve()
    if not level_dir.is_dir():
        raise FileNotFoundError(f"Level dir not found: {level_dir}")

    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_root = (Path.cwd() / "runs" / stamp / level_dir.name).resolve()
    else:
        output_root = args.output_root.resolve()
    if args.resume and not output_root.is_dir():
        print(
            f"Warning: --resume but output root does not exist yet: {output_root} "
            "(all tasks will start from round 0).",
            file=sys.stderr,
        )
    output_root.mkdir(parents=True, exist_ok=True)

    only_list: Optional[list[str]] = None
    if (args.only_tasks or "").strip():
        only_list = [x.strip() for x in str(args.only_tasks).split(",")]
    tasks = _discover_tasks(level_dir, output_root, only_task_stems=only_list)

    task_starts: list[int]
    if args.resume:
        task_starts = [_resume_start_round(t.work_dir) for t in tasks]
        if args.start_round != 0:
            print(
                "Note: --start-round is ignored with --resume (per-task start from last round_* on disk).",
                file=sys.stderr,
            )
    else:
        task_starts = [int(args.start_round)] * len(tasks)

    cap_excl: Optional[int] = int(args.max_round) if args.max_round is not None else None
    work_items: list[tuple[Task, int, int]] = []
    skipped_cap = 0
    for t, sr in zip(tasks, task_starts):
        mr = _effective_max_rounds(sr, int(args.rounds_per_task), cap_excl)
        if mr <= int(sr):
            skipped_cap += 1
            print(
                f"[SKIP] {t.task_file.name}: start_round={sr} >= capped max_rounds={mr} "
                f"(start >= --max-round exclusive {cap_excl})",
                file=sys.stderr,
            )
            continue
        work_items.append((t, int(sr), mr))

    print(f"Discovered {len(tasks)} tasks under {level_dir}")
    print(f"Output root: {output_root}")
    print(f"Parallel workers: {args.parallel}, rounds/task: {args.rounds_per_task}")
    if cap_excl is not None:
        print(f"--max-round exclusive end cap: {cap_excl} (last round index <= {cap_excl - 1})")
    if skipped_cap:
        print(f"Skipped {skipped_cap} task(s) due to --max-round cap (no rounds to run).", file=sys.stderr)
    if args.resume:
        for t, sr in zip(tasks, task_starts):
            last = _detect_last_round_index(t.work_dir)
            if last < 0:
                print(f"  resume {t.task_file.name}: start_round={sr} (no round_* yet)")
            else:
                print(f"  resume {t.task_file.name}: start_round={sr} (re-run from round_{sr:03d})")

    if not work_items:
        print("No tasks to run (all skipped or empty list).", file=sys.stderr)
        return 0

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.parallel)) as ex:
        futs = {ex.submit(_run_one, args, t, sr, mr): t for t, sr, mr in work_items}
        for fut in concurrent.futures.as_completed(futs):
            t = futs[fut]
            try:
                _, rc = fut.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {t.task_file.name}: exception={e}")
                continue
            if rc == 0:
                print(f"[OK]   {t.task_file.name} -> {t.work_dir}")
            else:
                failed += 1
                print(f"[FAIL] {t.task_file.name}: returncode={rc} -> {t.work_dir}")

    print(f"Done. ran={len(work_items)} skipped={skipped_cap} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

