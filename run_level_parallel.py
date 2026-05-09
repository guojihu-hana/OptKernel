from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Task:
    task_file: Path
    work_dir: Path


def _discover_tasks(level_dir: Path, output_root: Path) -> list[Task]:
    files = sorted(level_dir.glob("*.py"))
    if not files:
        raise FileNotFoundError(f"No .py tasks found under {level_dir}")
    out: list[Task] = []
    for f in files:
        out.append(Task(task_file=f.resolve(), work_dir=(output_root / f.stem).resolve()))
    return out


def _build_agent_cmd(args: argparse.Namespace, task: Task) -> list[str]:
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "agent.py").resolve()),
        "--task-file",
        str(task.task_file),
        "--work-dir",
        str(task.work_dir),
        "--start-round",
        str(int(args.start_round)),
        "--max-rounds",
        str(int(args.start_round) + int(args.rounds_per_task)),
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


def _run_one(args: argparse.Namespace, task: Task) -> tuple[Task, int]:
    task.work_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_agent_cmd(args, task)
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
    p.add_argument("--start-round", type=int, default=0, help="Start round index for every task")
    p.add_argument("--output-root", type=Path, default=None, help="Default: ./runs/<timestamp>/<level_name>")

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

    level_dir = args.level_dir.resolve()
    if not level_dir.is_dir():
        raise FileNotFoundError(f"Level dir not found: {level_dir}")

    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_root = (Path.cwd() / "runs" / stamp / level_dir.name).resolve()
    else:
        output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = _discover_tasks(level_dir, output_root)
    print(f"Discovered {len(tasks)} tasks under {level_dir}")
    print(f"Output root: {output_root}")
    print(f"Parallel workers: {args.parallel}, rounds/task: {args.rounds_per_task}")

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.parallel)) as ex:
        futs = {ex.submit(_run_one, args, t): t for t in tasks}
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

    print(f"Done. total={len(tasks)} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

